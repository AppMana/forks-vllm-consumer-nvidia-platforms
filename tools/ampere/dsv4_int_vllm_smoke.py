#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a local vLLM smoke test for a dsv4_int checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time


def _gpu_snapshot() -> list[dict[str, object]]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,display_active",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    out = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        out.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_mib": int(parts[2]),
                "memory_total_mib": int(parts[3]),
                "utilization_gpu_pct": int(parts[4]),
                "display_active": parts[5],
            }
        )
    return out


def _choose_non_display_gpu() -> str | None:
    candidates = [
        gpu
        for gpu in _gpu_snapshot()
        if str(gpu.get("display_active", "")).lower() not in ("enabled", "yes", "1")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda gpu: int(gpu["memory_used_mib"]))
    return str(candidates[0]["index"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Hello, my name is")
    parser.add_argument("--num-prompts", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--kv-cache-dtype", default="fp8")
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--load-format", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument(
        "--allow-display-gpu",
        action="store_true",
        help="Do not auto-select a non-display GPU when CUDA_VISIBLE_DEVICES is unset.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use vLLM compile/cudagraph defaults instead of enforce_eager=True.",
    )
    parser.add_argument(
        "--allow-sparse-mla-warmup",
        action="store_true",
        help="Keep DeepSeek V4 sparse MLA warmup enabled.",
    )
    parser.add_argument(
        "--allow-mhc-warmup",
        action="store_true",
        help="Keep DeepSeek V4 MHC warmup enabled.",
    )
    args = parser.parse_args()

    cuda_visible_devices = args.cuda_visible_devices
    if (
        cuda_visible_devices is None
        and "CUDA_VISIBLE_DEVICES" not in os.environ
        and not args.allow_display_gpu
    ):
        cuda_visible_devices = _choose_non_display_gpu()
        if cuda_visible_devices is not None:
            print(
                "DSV4_INT_SMOKE_SELECTED_GPU "
                + json.dumps(
                    {
                        "cuda_visible_devices": cuda_visible_devices,
                        "reason": "avoid_display_gpu",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    if not args.allow_sparse_mla_warmup:
        os.environ["VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP"] = "0"
    if not args.allow_mhc_warmup:
        os.environ["VLLM_ENABLE_DEEPSEEK_V4_MHC_WARMUP"] = "0"

    from vllm import LLM, SamplingParams

    before = _gpu_snapshot()
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        kv_cache_dtype=args.kv_cache_dtype,
        enforce_eager=not args.compile,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        pipeline_parallel_size=args.pipeline_parallel_size,
        load_format=args.load_format,
    )

    prompts = [args.prompt for _ in range(args.num_prompts)]
    params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)
    start = time.perf_counter()
    outputs = llm.generate(prompts, params)
    elapsed = time.perf_counter() - start

    output_tokens = sum(len(request.outputs[0].token_ids) for request in outputs)
    summary = {
        "model": args.model,
        "num_prompts": args.num_prompts,
        "max_tokens": args.max_tokens,
        "output_tokens": output_tokens,
        "elapsed_s": round(elapsed, 3),
        "output_tokens_per_s": round(output_tokens / elapsed, 3)
        if elapsed > 0
        else None,
        "before_gpus": before,
        "after_gpus": _gpu_snapshot(),
        "first_output": outputs[0].outputs[0].text if outputs else "",
    }
    print("DSV4_INT_SMOKE_SUMMARY " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
