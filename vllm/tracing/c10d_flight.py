# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""torch.distributed collective tracing via PyTorch's NCCL flight recorder.

PyTorch already records every c10d collective (op name, sizes, seq id,
scheduled/started/completed state, timestamps) in a per-process ring buffer
when ``TORCH_NCCL_TRACE_BUFFER_SIZE`` is set. This module turns those
recordings into something actionable for multi-rank hang debugging:

- ``dump_flight_recorder()``: snapshot the ring buffer to a JSON file per
  rank (callable from a signal handler or debug endpoint while hung).
- ``export_flight_recorder_otel()``: replay the ring buffer as OpenTelemetry
  spans through vllm.tracing.otel, one span per collective, so the timeline
  is browsable in the existing trace backend.
- CLI (``python -m vllm.tracing.c10d_flight <dump-dir>``): align dumps from
  all ranks by (process group, seq id) and report the first divergence:
  the op some rank issued that its peer never matched.

Enable recording with (must be set before process group init):
    TORCH_NCCL_TRACE_BUFFER_SIZE=20000
    TORCH_NCCL_DUMP_ON_TIMEOUT=1

For cross-rank shape/op mismatch detection at issue time (abort instead of
hang), additionally set TORCH_DISTRIBUTED_DEBUG=DETAIL: torch wraps every
process group and fingerprints each collective across ranks.

No new torch.distributed calls are made by this module; it only reads what
the library already records.
"""

import json
import os
import pickle
import signal
import sys
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_DUMP_ENV = "VLLM_C10D_FLIGHT_DUMP_DIR"


def flight_recorder_enabled() -> bool:
    return int(os.environ.get("TORCH_NCCL_TRACE_BUFFER_SIZE", "0")) > 0


def _raw_trace(include_stacktraces: bool = False) -> dict[str, Any]:
    from torch._C._distributed_c10d import _dump_nccl_trace

    return pickle.loads(
        _dump_nccl_trace(
            includeCollectives=True,
            includeStackTraces=include_stacktraces,
            onlyActive=False,
        )
    )


def dump_flight_recorder(
    dump_dir: str | None = None,
    rank: int | None = None,
    include_stacktraces: bool = False,
) -> str | None:
    """Write this process's flight-recorder ring buffer to
    ``<dump_dir>/c10d_flight_rank<rank>.json`` and return the path."""
    if not flight_recorder_enabled():
        logger.warning(
            "Flight recorder disabled; set TORCH_NCCL_TRACE_BUFFER_SIZE "
            "before process group init to enable c10d tracing."
        )
        return None
    if dump_dir is None:
        dump_dir = os.environ.get(_DUMP_ENV, "/tmp/c10d-flight")
    if rank is None:
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else os.getpid()
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, f"c10d_flight_rank{rank}.json")
    trace = _raw_trace(include_stacktraces)
    with open(path, "w") as f:
        json.dump(trace, f, default=str)
    logger.info(
        "c10d flight recorder dumped %d entries to %s",
        len(trace.get("entries", [])),
        path,
    )
    return path


def install_dump_signal_handler(signum: int = signal.SIGUSR2) -> None:
    """Dump the flight recorder on a signal, so a hung rank can be probed
    from outside (``kill -USR2 <pid>``) without attaching a debugger."""

    def _handler(_signum, _frame):
        dump_flight_recorder()

    signal.signal(signum, _handler)


def export_flight_recorder_otel(instrumenting_module_name: str = "vllm.c10d") -> int:
    """Replay recorded collectives as OTel spans via the fork's existing
    tracer. Returns the number of spans exported. Requires
    OTEL_EXPORTER_OTLP_TRACES_ENDPOINT (same contract as vllm.tracing.otel).
    """
    from vllm.tracing.otel import init_otel_tracer, is_otel_available

    if not is_otel_available():
        logger.warning("OTel unavailable; skipping c10d span export")
        return 0
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    if not endpoint:
        logger.warning("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT unset; skipping")
        return 0
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else -1
    tracer = init_otel_tracer(
        instrumenting_module_name, endpoint, {"c10d.rank": str(rank)}
    )
    entries = _raw_trace().get("entries", [])
    exported = 0
    for e in entries:
        start_ns = e.get("time_created_ns")
        if start_ns is None:
            continue
        duration_ms = e.get("duration_ms")
        end_ns = start_ns + int(duration_ms * 1e6) if duration_ms else start_ns + 1
        span = tracer.start_span(
            e.get("profiling_name", "c10d"), start_time=start_ns
        )
        span.set_attribute("c10d.seq_id", e.get("seq_id", -1))
        span.set_attribute("c10d.record_id", e.get("record_id", -1))
        span.set_attribute("c10d.pg_name", str(e.get("process_group", "")))
        span.set_attribute("c10d.state", str(e.get("state", "")))
        span.set_attribute("c10d.input_sizes", str(e.get("input_sizes", "")))
        span.set_attribute("c10d.output_sizes", str(e.get("output_sizes", "")))
        span.set_attribute("c10d.rank", rank)
        span.end(end_time=end_ns)
        exported += 1
    logger.info("exported %d c10d spans to %s", exported, endpoint)
    return exported


def _load_dumps(dump_dir: str) -> dict[int, list[dict[str, Any]]]:
    dumps: dict[int, list[dict[str, Any]]] = {}
    for name in sorted(os.listdir(dump_dir)):
        if not (name.startswith("c10d_flight_rank") and name.endswith(".json")):
            continue
        rank = int(name[len("c10d_flight_rank") : -len(".json")])
        with open(os.path.join(dump_dir, name)) as f:
            dumps[rank] = json.load(f).get("entries", [])
    return dumps


def analyze_divergence(dump_dir: str) -> list[str]:
    """Align per-rank dumps by (pg, seq) and report, per process group, the
    highest completed seq on each rank plus every op that is scheduled or
    started but not completed: the frontier where a hang lives."""
    dumps = _load_dumps(dump_dir)
    lines: list[str] = []
    if not dumps:
        return ["no c10d_flight_rank*.json dumps found in " + dump_dir]
    pgs: set[str] = set()
    for entries in dumps.values():
        pgs.update(str(e.get("process_group", "")) for e in entries)
    for pg in sorted(pgs):
        lines.append(f"process group {pg}:")
        for rank, entries in sorted(dumps.items()):
            mine = [e for e in entries if str(e.get("process_group", "")) == pg]
            done = [e for e in mine if e.get("state") == "completed"]
            pending = [e for e in mine if e.get("state") != "completed"]
            top = max((e.get("seq_id", -1) for e in done), default=-1)
            lines.append(f"  rank {rank}: last completed seq {top}")
            for e in pending:
                lines.append(
                    f"    PENDING seq {e.get('seq_id')} "
                    f"{e.get('profiling_name')} in={e.get('input_sizes')} "
                    f"out={e.get('output_sizes')} state={e.get('state')}"
                )
    return lines


def main() -> None:
    if len(sys.argv) != 2:
        print(
            "usage: python -m vllm.tracing.c10d_flight <dump-dir>\n"
            "Analyzes c10d_flight_rank*.json dumps for cross-rank divergence."
        )
        raise SystemExit(2)
    for line in analyze_divergence(sys.argv[1]):
        print(line)


if __name__ == "__main__":
    main()
