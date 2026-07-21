#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduce the int8_ds_mla misaligned-address crash locally.

Mirrors the crashing cluster config (JobSet dsv4-m2-int4-int8kv-001) at
local scale: int8_ds_mla KV cache, chunked prefill (max_num_batched_tokens
1024), long prompt so the gather/dequant paths engage, optional cudagraph
FULL_DECODE_ONLY.
"""

from __future__ import annotations

import argparse
import json
import os
import time


def _install_triton_sync_probe() -> None:
    """Synchronize after every Triton kernel launch and name the faulting one.

    Triton launches via the CUDA driver API, which ignores
    CUDA_LAUNCH_BLOCKING; this probe makes Triton faults synchronous.
    Enable with DSV4_SYNC_TRITON=1 (requires VLLM_ENABLE_V1_MULTIPROCESSING=0).
    """
    import torch
    from triton.runtime.jit import JITFunction

    original_run = JITFunction.run

    def run(self, *args, **kwargs):
        result = original_run(self, *args, **kwargs)
        try:
            torch.cuda.synchronize()
        except Exception:
            print(f"TRITON_SYNC_PROBE_FAULT kernel={self.fn.__name__}", flush=True)
            raise
        return result

    JITFunction.run = run


def _install_kv_insert_probe() -> None:
    """Log cache geometry at every fused qnorm/rope/kv-insert call."""
    from vllm.models.deepseek_v4 import attention as dsv4_attention

    original = dsv4_attention.DeepseekV4Attention._fused_qnorm_rope_kv_insert

    def wrapper(self, q, kv, positions, attn_metadata):
        cache = self.swa_cache_layer.kv_cache
        if cache.numel():
            print(
                "KV_INSERT_PROBE "
                f"layer={self.prefix} kv_cache_dtype={self.kv_cache_dtype} "
                f"shape={tuple(cache.shape)} strides={cache.stride()} "
                f"dtype={cache.dtype} data_ptr_mod16={cache.data_ptr() % 16} "
                f"storage_offset={cache.storage_offset()}",
                flush=True,
            )
        return original(self, q, kv, positions, attn_metadata)

    dsv4_attention.DeepseekV4Attention._fused_qnorm_rope_kv_insert = wrapper


def main() -> int:
    if os.environ.get("DSV4_SYNC_TRITON") == "1":
        _install_triton_sync_probe()
    if os.environ.get("DSV4_KV_INSERT_PROBE") == "1":
        _install_kv_insert_probe()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/administrator/Documents/dsv4-debug/ckpt")
    parser.add_argument("--prompt-tokens", type=int, default=16000)
    parser.add_argument("--num-prompts", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=17408)
    parser.add_argument("--kv-cache-dtype", default="int8_ds_mla")
    parser.add_argument(
        "--hf-overrides",
        default=None,
        help='JSON dict merged into the HF config, e.g. \'{"vllm": {...}}\' '
        "to activate the unified kernel-config block on a checkpoint that "
        "predates it.",
    )
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument(
        "--load-format",
        default="dummy",
        help='"dummy" (default) for shape-only repros; "auto" to load the real '
        "trimmed weights so routing/topk value distributions match production.",
    )
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--long-prefill-token-threshold", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=1 << 30)
    parser.add_argument("--compile", action="store_true",
                        help="cudagraph FULL_DECODE_ONLY instead of enforce_eager")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    kwargs = {}
    if args.hf_overrides:
        kwargs["hf_overrides"] = json.loads(args.hf_overrides)
    if args.compile:
        kwargs["compilation_config"] = {
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4, 8],
        }

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        kv_cache_dtype=args.kv_cache_dtype,
        enforce_eager=not args.compile,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        tensor_parallel_size=1,
        pipeline_parallel_size=args.pipeline_parallel_size,
        load_format=args.load_format,
        max_num_seqs=2,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=True,
        long_prefill_token_threshold=args.long_prefill_token_threshold,
        enable_prefix_caching=False,
        **kwargs,
    )

    import random

    random.seed(0)
    prompts = [
        {
            "prompt_token_ids": [
                random.randint(10, 20000) for _ in range(args.prompt_tokens)
            ]
        }
        for _ in range(args.num_prompts)
    ]
    params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    start = time.perf_counter()
    outputs = llm.generate(prompts, params)
    elapsed = time.perf_counter() - start

    output_tokens = sum(len(r.outputs[0].token_ids) for r in outputs)
    summary = {
        "kv_cache_dtype": args.kv_cache_dtype,
        "prompt_tokens": args.prompt_tokens,
        "output_tokens": output_tokens,
        "elapsed_s": round(elapsed, 3),
        "first_output_token_ids": list(outputs[0].outputs[0].token_ids)[:16],
        "first_output_text": outputs[0].outputs[0].text[:200],
    }
    print("DSV4_INT8KV_REPRO_SUMMARY " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
