# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

from vllm.tracing.c10d_flight import analyze_divergence, flight_recorder_enabled


def _write_dump(tmp_path, rank, entries):
    path = tmp_path / f"c10d_flight_rank{rank}.json"
    path.write_text(json.dumps({"entries": entries}))


def test_flight_recorder_disabled_by_default(monkeypatch):
    monkeypatch.delenv("TORCH_NCCL_TRACE_BUFFER_SIZE", raising=False)
    assert not flight_recorder_enabled()
    monkeypatch.setenv("TORCH_NCCL_TRACE_BUFFER_SIZE", "20000")
    assert flight_recorder_enabled()


def test_analyze_divergence_empty_dir(tmp_path):
    lines = analyze_divergence(str(tmp_path))
    assert len(lines) == 1
    assert "no c10d_flight_rank" in lines[0]


def test_analyze_divergence_reports_pending_frontier(tmp_path):
    pg = ["0", "default"]
    _write_dump(
        tmp_path,
        0,
        [
            {
                "process_group": pg,
                "seq_id": 991,
                "state": "completed",
                "profiling_name": "nccl:send",
            },
            {
                "process_group": pg,
                "seq_id": 992,
                "state": "started",
                "profiling_name": "nccl:send",
                "input_sizes": [[262144]],
            },
        ],
    )
    _write_dump(
        tmp_path,
        1,
        [
            {
                "process_group": pg,
                "seq_id": 990,
                "state": "completed",
                "profiling_name": "nccl:recv",
            }
        ],
    )
    lines = analyze_divergence(str(tmp_path))
    text = "\n".join(lines)
    assert "rank 0: last completed seq 991" in text
    assert "rank 1: last completed seq 990" in text
    assert "PENDING seq 992 nccl:send" in text


def test_analyze_divergence_groups_by_process_group(tmp_path):
    _write_dump(
        tmp_path,
        0,
        [
            {
                "process_group": ["0", "default"],
                "seq_id": 5,
                "state": "completed",
                "profiling_name": "nccl:all_reduce",
            },
            {
                "process_group": ["1", "pp"],
                "seq_id": 3,
                "state": "scheduled",
                "profiling_name": "nccl:send",
            },
        ],
    )
    lines = analyze_divergence(str(tmp_path))
    text = "\n".join(lines)
    assert "process group ['0', 'default']" in text
    assert "process group ['1', 'pp']" in text
    assert "PENDING seq 3 nccl:send" in text
