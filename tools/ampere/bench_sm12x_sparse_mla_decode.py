# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GB10 sparse-MLA decode microbench: sparkinfer compressed_mla (fp8-Q CuTe
kernel) vs the fork's portable Triton decode (bf16-Q, same fp8_ds_mla cache).

Answers two Phase-6 questions that gate kernel policy on sm12x:
1. how much faster is the fp8-Q CuTe kernel than the Triton bf16-Q fallback
   at the real decode footprints (SWA 128 + top-k 512, heads 32 = TP2)?
2. what would a BF16-Q compute mode cost — bounded below by the Triton
   number, so "is implementing ComputeMode.BF16 in the sparkinfer fork worth
   it" gets a data-driven answer.

Runs standalone on a spark (no vLLM install): the Triton kernel module chain
(vllm.triton_utils -> fp8e4m3_arith -> nvidia_sm86.triton_kernels) is stubbed
and loaded by file path. Both kernels see identical caches, indices, lengths.

Usage (from a source rsync):  python tools/ampere/bench_sm12x_sparse_mla_decode.py
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
import types

import torch

REPO = pathlib.Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Load the fork's Triton decode kernel without a vllm install.
# ---------------------------------------------------------------------------


def _load_by_path(dotted: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(dotted, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_triton_decode():
    try:
        from vllm.models.deepseek_v4.nvidia_sm86.triton_kernels import (
            decode_sparse_attention_triton,
        )

        return decode_sparse_attention_triton
    except ImportError:
        pass
    import triton
    import triton.language as tl

    if "vllm" not in sys.modules:
        pkg = types.ModuleType("vllm")
        pkg.__path__ = [str(REPO / "vllm")]
        sys.modules["vllm"] = pkg
    tu = types.ModuleType("vllm.triton_utils")
    tu.triton = triton
    tu.tl = tl
    tu.LOG2E = math.log2(math.e)
    sys.modules["vllm.triton_utils"] = tu
    _load_by_path(
        "vllm.models.deepseek_v4.common.ops.fp8e4m3_arith",
        REPO / "vllm/models/deepseek_v4/common/ops/fp8e4m3_arith.py",
    )
    mod = _load_by_path(
        "vllm.models.deepseek_v4.nvidia_sm86.triton_kernels",
        REPO / "vllm/models/deepseek_v4/nvidia_sm86/triton_kernels.py",
    )
    return mod.decode_sparse_attention_triton


# ---------------------------------------------------------------------------
# Fixtures: realistic fp8_ds_mla pools at the padding-free block sizes.
# ---------------------------------------------------------------------------

_HEAD_DIM = 512
_SWA_PAGE = 288
_EXTRA_PAGE = 72
_SWA_WIDTH = 128  # DSV4-Flash sliding window
_TOPK = 512  # DSV4-Flash index_topk
_SM_SCALE = 1.0 / math.sqrt(_HEAD_DIM)


def _fill_pool(num_pages: int, page_size: int, seed: int) -> torch.Tensor:
    from sparkinfer.attention._shared.mla.compressed_reference import (
        pack_compressed_mla_kv_cache_reference,
    )

    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    tokens = num_pages * page_size
    k_nope = (
        torch.randn((tokens, 448), generator=gen, dtype=torch.float32, device="cuda")
        / 4
    )
    k_rope = (
        torch.randn((tokens, 64), generator=gen, dtype=torch.float32, device="cuda") / 4
    )
    return pack_compressed_mla_kv_cache_reference(
        k_nope, k_rope.to(torch.bfloat16), page_size=page_size, num_pages=num_pages
    )


def _indices(rows: int, width: int, num_slots: int, seed: int):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    idx = torch.stack(
        [
            torch.randperm(num_slots, generator=gen, device="cuda")[:width]
            .sort()
            .values
            for _ in range(rows)
        ]
    ).to(torch.int32)
    lens = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    return idx, lens


def _time_cuda(fn, iters: int = 100, warmup: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters  # us


def _time_graph(fn, iters: int = 100, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return _time_cuda(graph.replay, iters=iters)


def main() -> None:
    assert torch.cuda.is_available()
    cap = torch.cuda.get_device_capability(0)
    assert cap[0] == 12, f"sm12x required, got {cap}"
    from sparkinfer.attention import compressed_mla

    assert compressed_mla.is_supported()
    triton_decode = _load_triton_decode()

    # Pools: 64 seqs of 8k context -> 8k swa slots is plenty (window is 128);
    # compressed C4 slots: 64 * 8192/4 = 131072.
    swa_pool = _fill_pool(32, _SWA_PAGE, seed=1)  # 9216 slots
    extra_pool = _fill_pool(1824, _EXTRA_PAGE, seed=2)  # 131328 slots
    swa_3d = swa_pool.view(swa_pool.shape[0], _SWA_PAGE, 584)
    extra_3d = extra_pool.view(extra_pool.shape[0], _EXTRA_PAGE, 584)

    print(
        f"# device={torch.cuda.get_device_name(0)} cap={cap} "
        f"swa_pool={tuple(swa_pool.shape)} extra_pool={tuple(extra_pool.shape)}"
    )
    header = (
        f"{'rows':>5} {'heads':>5} | {'sparkinfer(us)':>14} {'graph(us)':>10} "
        f"| {'triton(us)':>10} | {'speedup':>7}"
    )
    print(header)
    print("-" * len(header))

    for heads in (32, 64):
        for rows in (1, 2, 4, 8, 16, 32, 64):
            gen = torch.Generator(device="cuda")
            gen.manual_seed(rows * 100 + heads)
            q = (
                torch.randn(
                    rows, heads, _HEAD_DIM, device="cuda", dtype=torch.float32,
                    generator=gen,
                )
                / 4
            ).to(torch.bfloat16)
            swa_idx, swa_lens = _indices(rows, _SWA_WIDTH, 9216, seed=rows)
            ex_idx, ex_lens = _indices(rows, _TOPK, 131328, seed=rows + 7)
            sink = torch.linspace(
                -0.2, 0.2, heads, dtype=torch.float32, device="cuda"
            )
            out = torch.empty(
                rows, heads, _HEAD_DIM, device="cuda", dtype=torch.bfloat16
            )

            plan = compressed_mla.plan(
                compressed_mla.Caps(
                    device=q.device,
                    dtype=torch.bfloat16,
                    kv_dtype=torch.uint8,
                    num_q_heads=heads,
                    max_width=_SWA_WIDTH + _TOPK,
                    max_q_rows=rows,
                    max_batch=rows,
                    max_kv_rows=rows * (_SWA_WIDTH + _TOPK),
                )
            )
            (spec,) = plan.scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
            binding = plan.bind(
                scratch=scratch,
                q=q,
                swa_indices=swa_idx,
                swa_lengths=swa_lens,
                indexed_indices=ex_idx,
                indexed_lengths=ex_lens,
            )
            binding.scratch.use_cuda_graph = True

            def spark_run():
                compressed_mla.run(
                    swa_k_cache=swa_pool,
                    binding=binding,
                    swa_page_size=_SWA_PAGE,
                    indexed_k_cache=extra_pool,
                    indexed_page_size=_EXTRA_PAGE,
                    attn_sink=sink,
                    sm_scale=_SM_SCALE,
                    out=out,
                )

            def triton_run():
                triton_decode(
                    q=q,
                    swa_cache=swa_3d,
                    swa_indices=swa_idx,
                    swa_lens=swa_lens,
                    scale=_SM_SCALE,
                    attn_sink=sink,
                    out=out,
                    extra_cache=extra_3d,
                    extra_indices=ex_idx,
                    extra_lens=ex_lens,
                )

            t_spark = _time_cuda(spark_run)
            t_graph = _time_graph(spark_run)
            t_triton = _time_cuda(triton_run)
            print(
                f"{rows:>5} {heads:>5} | {t_spark:>14.1f} {t_graph:>10.1f} "
                f"| {t_triton:>10.1f} | {t_triton / t_graph:>6.2f}x"
            )


if __name__ == "__main__":
    main()
