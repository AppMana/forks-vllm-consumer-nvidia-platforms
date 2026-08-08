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
        # Synchronizing inside CUDA graph capture is illegal and would report
        # a phantom fault for whatever kernel happened to launch first.
        if torch.cuda.is_current_stream_capturing():
            return result
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


def _stub_vllm_flash_attn() -> None:
    """Satisfy vllm.vllm_flash_attn's import without its CUDA extensions.

    The fa2/fa3 .so files are built against a specific torch ABI; in a
    testbed venv running a different torch they fail to import, and the
    package __init__ raises. DSV4 attends via flash_mla and Triton, never
    through flash-attn, so for this harness the module only needs to exist.
    Every callable raises so an unexpected real use fails loudly.
    """
    import sys
    import types

    def _unavailable(*args, **kwargs):
        raise RuntimeError("vllm_flash_attn stubbed out by dsv4_int8kv_repro")

    mod = types.ModuleType("vllm.vllm_flash_attn")
    mod.FA2_AVAILABLE = False
    mod.FA3_AVAILABLE = False
    mod.flash_attn_varlen_func = _unavailable
    mod.get_scheduler_metadata = _unavailable
    mod.compile_flash_attn_varlen_func_from_specs = _unavailable
    mod.fa_version_unsupported_reason = lambda v: "stubbed"
    mod.is_fa_version_supported = lambda v: False
    sys.modules["vllm.vllm_flash_attn"] = mod


def main() -> int:
    if os.environ.get("DSV4_STUB_VLLM_FA") == "1":
        _stub_vllm_flash_attn()
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
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
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
    parser.add_argument(
        "--prod-compile",
        action="store_true",
        help="The serving manifest's compilation config: mode 3 with "
        "FULL_AND_PIECEWISE cudagraphs, capture sizes 1,2,4,8,16.",
    )
    parser.add_argument(
        "--async-scheduling",
        action="store_true",
        help="Matches the serving manifest; the scheduler finish path "
        "differs from the synchronous one.",
    )
    parser.add_argument(
        "--speculative-config",
        default=None,
        help='e.g. \'{"method":"dspark","num_speculative_tokens":5}\'',
    )
    parser.add_argument(
        "--stop-finish",
        action="store_true",
        help="Make every round finish with finish_reason=stop instead of "
        "length: a short probe generation records the greedy continuation, "
        "then the round re-runs with the probe's last token as a stop token. "
        "The production crash follows EOS-finished generations under dspark; "
        "length-capped ones are known not to trigger it.",
    )
    parser.add_argument(
        "--eos-finish",
        action="store_true",
        help="Alternate rounds: one request whose sampler may only emit the "
        "EOS token (finish via the real EOS detection path, invalidating "
        "in-flight dspark drafts), then a normal request -- the alternation "
        "that kills production.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Sequential generate() calls. A single batch never frees a block, "
        "so recycling -- and the block zeroing that rides on it -- only "
        "happens from round 2 on.",
    )
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    kwargs = {}
    if args.hf_overrides:
        kwargs["hf_overrides"] = json.loads(args.hf_overrides)
    if args.speculative_config:
        kwargs["speculative_config"] = json.loads(args.speculative_config)
    if args.compile:
        kwargs["compilation_config"] = {
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4, 8],
        }
    if args.prod_compile:
        kwargs["compilation_config"] = {
            "mode": 3,
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16],
        }
    if args.async_scheduling:
        kwargs["async_scheduling"] = True

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        kv_cache_dtype=args.kv_cache_dtype,
        enforce_eager=not (args.compile or args.prod_compile),
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        # Matches production, and dodges flashinfer.comm's cuda_ipc import,
        # which dies on tilelang's libcudart_stub shadowing real cudart.
        disable_custom_all_reduce=True,
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
    for round_idx in range(args.rounds):
        # Fresh token ids per round: identical prompts would be served from the
        # prefix cache and never allocate a recycled block.
        round_prompts = [
            {
                "prompt_token_ids": [
                    random.randint(10, 20000) for _ in range(args.prompt_tokens)
                ]
            }
            for _ in range(args.num_prompts)
        ]
        active = round_prompts if round_idx else prompts
        round_params = params
        if args.eos_finish and round_idx % 2 == 0:
            eos_id = 1
            round_params = SamplingParams(
                max_tokens=args.max_tokens,
                temperature=0.0,
                allowed_token_ids=[eos_id],
            )
        elif args.stop_finish:
            probe = llm.generate(
                active,
                SamplingParams(max_tokens=24, temperature=0.0),
            )
            stop_ids = list(
                {p.outputs[0].token_ids[-1] for p in probe if p.outputs[0].token_ids}
            )
            round_params = SamplingParams(
                max_tokens=args.max_tokens,
                temperature=0.0,
                stop_token_ids=stop_ids,
            )
        print(f"DSV4_INT8KV_REPRO_ROUND {round_idx}", flush=True)
        outputs = llm.generate(active, round_params)
        reasons = [r.outputs[0].finish_reason for r in outputs]
        ntok = [len(r.outputs[0].token_ids) for r in outputs]
        print(
            f"DSV4_INT8KV_REPRO_ROUND_OK {round_idx} "
            f"finish={reasons} tokens={ntok}",
            flush=True,
        )
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
