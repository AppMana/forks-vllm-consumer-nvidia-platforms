#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end DSpark speculative-decode correctness smoke test.

Builds a shrunk-but-real DeepSeek-V4-Flash-DSpark checkpoint (see
dsv4_dspark_shrunk_config.py) and runs vLLM with `--load-format dummy`
(deterministic random weights, seed=1234) twice: once with DSpark speculative
decoding OFF, once ON. At temperature=0 (greedy), a speculative decoder's
accept/reject step is a mathematical no-op on the OUTPUT distribution: an
accepted draft token is, by construction, the same token target-only greedy
decoding would have produced, and a rejected step falls back to the target's
own greedy sample. So spec-ON output must be BIT-IDENTICAL to spec-OFF output,
token-for-token, for the full generation length. This is the actual
correctness invariant under test -- not "does it run", "does it match".

Usage:
    python tools/ampere/dsv4_dspark_spec_decode_smoke.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.ampere.dsv4_dspark_shrunk_config import (  # noqa: E402
    build_shrunk_config,
    find_real_dspark_config,
    write_shrunk_checkpoint_dir,
)


def _choose_gpu() -> str | None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    rows = [line.split(",") for line in proc.stdout.splitlines() if line.strip()]
    rows = [(idx.strip(), int(used.strip())) for idx, used in rows]
    if not rows:
        return None
    rows.sort(key=lambda r: r[1])
    return rows[0][0]


COMMON_KWARGS = dict(
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=2048,
    max_num_batched_tokens=2048,
    max_num_seqs=1,
    kv_cache_dtype="fp8",
    load_format="dummy",
    enforce_eager=True,
    gpu_memory_utilization=0.7,
    tensor_parallel_size=1,
    kv_cache_memory_bytes=10 * 1024 * 1024 * 1024,
)

PROMPTS = ["Hello world", "The capital of France is", "def fibonacci(n):"]
MAX_TOKENS = 12


def _generate(model_dir: str, speculative_config: dict | None) -> list[list[int]]:
    from vllm import LLM, SamplingParams

    kwargs = dict(COMMON_KWARGS)
    if speculative_config is not None:
        kwargs["speculative_config"] = speculative_config

    llm = LLM(model=model_dir, **kwargs)
    params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)
    outputs = llm.generate(PROMPTS, params)
    return [list(o.outputs[0].token_ids) for o in outputs]


def main() -> int:
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES") or _choose_gpu()
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    real_config_path = find_real_dspark_config()
    cfg = build_shrunk_config(real_config_path, num_layers=6, n_routed_experts=8)
    model_dir = "/tmp/dsv4_dspark_spec_decode_smoke_ckpt"
    write_shrunk_checkpoint_dir(Path(model_dir), cfg, real_config_path)

    # Two separate LLM() processes (not one process running both) so the spec-
    # OFF and spec-ON runs cannot leak KV-cache/CUDA-graph/scheduler state into
    # each other -- the only thing that must match between them is the
    # deterministic dummy-weight init (same seed=1234, same tensor shapes).
    spec_off = json.loads(
        subprocess.run(
            [sys.executable, __file__, "--role", "spec-off", "--model-dir", model_dir],
            check=True,
            text=True,
            capture_output=True,
            env=os.environ,
        ).stdout.strip().splitlines()[-1]
    )
    spec_on = json.loads(
        subprocess.run(
            [sys.executable, __file__, "--role", "spec-on", "--model-dir", model_dir],
            check=True,
            text=True,
            capture_output=True,
            env=os.environ,
        ).stdout.strip().splitlines()[-1]
    )

    all_match = spec_off == spec_on
    print(
        "DSV4_DSPARK_SPEC_DECODE_SMOKE "
        + json.dumps(
            {
                "spec_off_tokens": spec_off,
                "spec_on_tokens": spec_on,
                "bit_identical": all_match,
            },
            sort_keys=True,
        )
    )
    return 0 if all_match else 1


def _subprocess_role() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["spec-off", "spec-on"], required=True)
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()

    if args.role == "spec-off":
        tokens = _generate(args.model_dir, speculative_config=None)
    else:
        tokens = _generate(
            args.model_dir,
            speculative_config={
                "method": "dspark",
                "num_speculative_tokens": 5,
                "draft_sample_method": "greedy",
            },
        )
    print(json.dumps(tokens))
    return 0


if __name__ == "__main__":
    if "--role" in sys.argv:
        raise SystemExit(_subprocess_role())
    raise SystemExit(main())
