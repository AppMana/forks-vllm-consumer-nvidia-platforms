#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Profile a single DeepSeek-V4 decode step at 16k context on one GPU.

Uses dummy weights + the real-shape (4-layer) debug checkpoint, chunked
prefill to 16k, then captures steady-state DECODE forwards only by stepping
torch.profiler from a hook on the top-level model forward (skips prefill
chunks). Per-component CUDA time is collected via record_function ranges
wrapped around curated submodules, and the raw per-kernel key_averages table
is dumped for kernel->component mapping.

Scratch tool; not on any production import path.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/administrator/Documents/dsv4-debug/ckpt")
    parser.add_argument("--prompt-tokens", type=int, default=16000)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--max-model-len", type=int, default=17408)
    parser.add_argument("--kv-cache-dtype", default="fp8_ds_mla")
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=2 << 30)
    parser.add_argument("--skip-first", type=int, default=16)
    parser.add_argument("--active", type=int, default=16)
    parser.add_argument("--trace-out", default=None)
    parser.add_argument("--shapes", action="store_true",
                        help="record_shapes and print per-shape mm/bmm/mv breakdown")
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import torch
    from torch.profiler import ProfilerActivity, profile, record_function, schedule

    from vllm import LLM, SamplingParams
    from vllm.models.deepseek_v4.nvidia import model as dsv4_model

    # ---- component labeling by submodule suffix / class name ----
    def classify(name: str, mod) -> str | None:
        cls = type(mod).__name__
        if cls in ("MHCPreOp", "MHCPostOp", "MHCFusedPostPreOp"):
            return "MHC"
        if name.endswith(".attn.indexer"):
            return "INDEXER"
        if name.endswith(".attn"):
            return "ATTN(incl indexer)"
        if name.endswith(".ffn.experts"):
            return "MOE_routed_experts"
        if name.endswith(".ffn.shared_experts"):
            return "MOE_shared_expert"
        if name.endswith(".ffn.gate"):
            return "MOE_gate"
        if name.endswith(".ffn"):
            return "FFN(incl experts+shared+gate)"
        if name.endswith(".attn_norm") or name.endswith(".ffn_norm"):
            return "RMSNorm(attn/ffn)"
        return None

    _prof_holder: dict = {}
    _hooked = {"done": False}
    _stack_depth = {"n": 0}

    def install_component_hooks(model: torch.nn.Module) -> None:
        for name, mod in model.named_modules():
            label = classify(name, mod)
            if label is None:
                continue
            rf_stack: list = []

            def pre_hook(m, inp, _label=label, _stack=rf_stack):
                rf = record_function(_label)
                rf.__enter__()
                _stack.append(rf)

            def post_hook(m, inp, out, _stack=rf_stack):
                if _stack:
                    _stack.pop().__exit__(None, None, None)

            mod.register_forward_pre_hook(pre_hook)
            mod.register_forward_hook(post_hook)

    # Patch the top-level model forward to (a) install component hooks on first
    # call and (b) step the profiler each decode/prefill forward.
    orig_forward = dsv4_model.DeepseekV4Model.forward

    def patched_forward(self, *a, **kw):
        if not _hooked["done"]:
            install_component_hooks(self)
            _hooked["done"] = True
        out = orig_forward(self, *a, **kw)
        prof = _prof_holder.get("prof")
        if prof is not None:
            prof.step()
        return out

    dsv4_model.DeepseekV4Model.forward = patched_forward

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        kv_cache_dtype=args.kv_cache_dtype,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        load_format="dummy",
        max_num_seqs=2,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=True,
        long_prefill_token_threshold=args.max_num_batched_tokens,
        enable_prefix_caching=False,
    )

    import random

    random.seed(0)
    prompts = [
        {"prompt_token_ids": [random.randint(10, 20000) for _ in range(args.prompt_tokens)]}
    ]
    params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    sched = schedule(skip_first=args.skip_first, wait=0, warmup=2, active=args.active, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        record_shapes=args.shapes,
        with_stack=False,
    ) as prof:
        _prof_holder["prof"] = prof
        llm.generate(prompts, params)

    # ---- per-component (record_function) CUDA totals ----
    ka = prof.key_averages()
    n_active = args.active
    comp_labels = {
        "MHC", "INDEXER", "ATTN(incl indexer)", "MOE_routed_experts",
        "MOE_shared_expert", "MOE_gate", "FFN(incl experts+shared+gate)",
        "RMSNorm(attn/ffn)",
    }
    comp = {}
    for e in ka:
        if e.key in comp_labels:
            cuda_us = getattr(e, "cuda_time_total", None)
            if cuda_us is None:
                cuda_us = getattr(e, "device_time_total", 0.0)
            comp[e.key] = {
                "count": e.count,
                "cuda_total_us": round(cuda_us, 1),
                "cuda_per_active_step_us": round(cuda_us / n_active, 1),
                "cuda_per_layer_per_step_us": round(cuda_us / n_active / 4.0, 2),
            }

    # ---- top kernels by self CUDA time ----
    kernels = []
    for e in ka:
        self_cuda = getattr(e, "self_cuda_time_total", None)
        if self_cuda is None:
            self_cuda = getattr(e, "self_device_time_total", 0.0)
        if self_cuda and self_cuda > 0 and e.key not in comp_labels:
            kernels.append((e.key, e.count, self_cuda))
    kernels.sort(key=lambda x: -x[2])

    total_self_cuda = sum(k[2] for k in kernels)
    print("DSV4_DECODE_PROFILE_COMPONENTS " + json.dumps(comp, sort_keys=True))
    print(f"DSV4_DECODE_TOTAL_SELF_CUDA_US_ALL_ACTIVE {round(total_self_cuda,1)} "
          f"per_step_us={round(total_self_cuda/n_active,1)} n_active={n_active}")
    print("=== TOP KERNELS (self CUDA us: total over active steps | per-step | per-layer/step) ===")
    for key, count, self_cuda in kernels[:45]:
        print(f"{round(self_cuda,1):>10} | {round(self_cuda/n_active,1):>8} | "
              f"{round(self_cuda/n_active/4.0,2):>7} | cnt={count:>4} | {key[:90]}")

    if args.shapes:
        print("=== MM/BMM BY INPUT SHAPE (self CUDA us: per-step | per-layer/step) ===")
        ka_s = prof.key_averages(group_by_input_shape=True)
        rows = []
        for e in ka_s:
            if e.key not in ("aten::mm", "aten::bmm", "aten::mv", "aten::addmm",
                             "aten::matmul", "aten::linear"):
                continue
            self_cuda = getattr(e, "self_cuda_time_total", None)
            if self_cuda is None:
                self_cuda = getattr(e, "self_device_time_total", 0.0)
            cuda_tot = getattr(e, "cuda_time_total", None)
            if cuda_tot is None:
                cuda_tot = getattr(e, "device_time_total", 0.0)
            rows.append((e.key, str(e.input_shapes), e.count, self_cuda, cuda_tot))
        rows.sort(key=lambda x: -x[4])
        for key, shp, count, self_c, cuda_c in rows[:30]:
            print(f"{round(cuda_c/n_active,1):>8} | {round(cuda_c/n_active/4.0,2):>7} | "
                  f"cnt={count:>4} | {key:<12} | {shp[:80]}")

    if args.trace_out:
        prof.export_chrome_trace(args.trace_out)
        print(f"trace written to {args.trace_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
