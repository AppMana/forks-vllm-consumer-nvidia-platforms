# SPDX-License-Identifier: Apache-2.0
"""Standalone DSV4 INT8 indexer prefill benchmark.

This deliberately avoids model construction.  Its tensors have the exact
production contract: Q [M, H, 128] s8, gathered K [N, 128] s8, one fp32 K
scale per row, fp32 folded Q weights [M, H], causal row bounds, and top-512.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from vllm import _custom_ops as ops
from vllm.models.deepseek_v4.nvidia_imma.triton_kernels import (
    mqa_logits_workspace_triton,
)


def timed(fn, *, warmup: int, repeats: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    return statistics.median(samples), min(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--context-rows", type=int, default=16384)
    parser.add_argument("--slab-rows", type=int, default=16384)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(7)
    device = torch.device("cuda")
    m, n, slab, topk, heads = (
        args.rows,
        args.context_rows,
        args.slab_rows,
        args.topk,
        args.heads,
    )
    q = torch.randint(-32, 33, (m, heads, 128), dtype=torch.int8, device=device)
    k = torch.randint(-32, 33, (n, 128), dtype=torch.int8, device=device)
    k_scale = torch.rand(n, dtype=torch.float32, device=device) * 0.02 + 0.001
    weights = torch.rand(m, heads, dtype=torch.float32, device=device) * 0.02
    row_start = torch.zeros(m, dtype=torch.int32, device=device)
    row_end = torch.linspace(max(1, n - m // 4), n, m, device=device).to(torch.int32)

    one_indices = torch.empty((m, topk), dtype=torch.int32, device=device)
    slab_indices = [
        torch.empty((m, 2 * topk), dtype=torch.int32, device=device) for _ in range(2)
    ]
    slab_values = [
        torch.empty((m, 2 * topk), dtype=torch.float32, device=device) for _ in range(2)
    ]

    def logits_only():
        return mqa_logits_workspace_triton(
            q, (k, k_scale), weights, row_start, row_end, qk_int8=True
        )

    def oneshot():
        logits = logits_only()
        ops.top_k_per_row_prefill(
            logits,
            row_start,
            row_end,
            one_indices,
            m,
            logits.stride(0),
            logits.stride(1),
            topk,
        )
        return one_indices

    def streaming():
        current = 0
        first = True
        for n0 in range(0, n, slab):
            n1 = min(n0 + slab, n)
            width = n1 - n0
            starts = torch.clamp(row_start - n0, 0, width)
            ends = torch.clamp(row_end - n0, 0, width)
            logits = mqa_logits_workspace_triton(
                q,
                (k[n0:n1], k_scale[n0:n1]),
                weights,
                starts,
                ends,
                qk_int8=True,
            )
            half = 0 if first else topk
            ops.top_k_per_row_prefill_candidates(
                logits,
                starts,
                ends,
                slab_indices[current][:, half : half + topk],
                slab_values[current][:, half : half + topk],
                topk,
                n0,
            )
            if first:
                first = False
            else:
                other = 1 - current
                ops.top_k_per_row_merge_candidates(
                    slab_values[current],
                    slab_indices[current],
                    slab_indices[other][:, :topk],
                    slab_values[other][:, :topk],
                    topk,
                )
                current = other
        return slab_indices[current][:, :topk]

    cases = (("logits", logits_only), ("oneshot", oneshot), ("streaming", streaming))
    print(
        f"gpu={torch.cuda.get_device_name()} "
        f"capability={torch.cuda.get_device_capability()} "
        f"M={m} N={n} H={heads} D=128 topk={topk} slab={slab}",
        flush=True,
    )
    for name, fn in cases:
        started = time.perf_counter()
        median_ms, best_ms = timed(fn, warmup=args.warmup, repeats=args.repeats)
        print(
            f"{name:9s} median_ms={median_ms:.3f} best_ms={best_ms:.3f} "
            f"wall_s={time.perf_counter() - started:.3f}",
            flush=True,
        )

    expected = oneshot().sort(dim=1).values
    actual = streaming().sort(dim=1).values
    mismatch = int((expected != actual).sum().item())
    print(f"sorted_index_mismatches={mismatch}", flush=True)
    if mismatch:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
