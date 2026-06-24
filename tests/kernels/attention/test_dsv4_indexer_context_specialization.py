# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSV4 indexer logits Triton kernel must NOT specialize (recompile) on context.

The indexer logits kernel declared its output row-stride `stride_lm` (==
seq_len_kv, the context length) as `tl.constexpr`. Triton compiles a separate
kernel for every distinct constexpr value, so the kernel recompiles on every new
context length. During a long-context (16k) prefill this stacks into a
multi-minute compile stall that wedges the whole PP chain (one rank pinned at
100% GPU inside Triton's launch/compile path, every other rank blocked on the
collective).

This test reproduces the symptom directly: calling the kernel at a *new* context
length must not pay a fresh compilation. A Triton compile is ~100ms+; a cached
launch is a few ms. We isolate the compile cost from the (context-dependent)
execution time by comparing the first call at a context length against a cached
repeat at the SAME length -- apples-to-apples. If the kernel specializes on
context length, the first call is far slower than the cached one.
"""
import os
import tempfile

# Force a cold, isolated Triton compile cache BEFORE any kernel compiles, so this
# test reproduces the fresh-pod (cold-cache) condition deterministically. A warm
# ~/.triton cache from a prior run would reload the per-context specializations
# from disk and mask the recompile. Must be set before triton is first used.
os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="triton_dsv4_ctx_")

import time  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from vllm.models.deepseek_v4.nvidia_sm86.triton_kernels import (  # noqa: E402
    mqa_logits_workspace_triton,
)


def _call(num_rows: int, seq_len_kv: int, heads: int = 64, dim: int = 128):
    dev = "cuda"
    q = torch.randint(-8, 8, (num_rows, heads, dim), dtype=torch.int8, device=dev)
    k = torch.randint(-8, 8, (seq_len_kv, dim), dtype=torch.int8, device=dev)
    k_scale = torch.rand(seq_len_kv, dtype=torch.float32, device=dev)
    weights = torch.rand(num_rows, heads, dtype=torch.float32, device=dev)
    ks = torch.zeros(num_rows, dtype=torch.int32, device=dev)
    ke = torch.full((num_rows,), seq_len_kv, dtype=torch.int32, device=dev)
    mqa_logits_workspace_triton(q, (k, k_scale), weights, ks, ke, qk_int8=True)
    torch.cuda.synchronize()


def _time_call(num_rows: int, seq_len_kv: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _call(num_rows, seq_len_kv)
    return time.perf_counter() - t0


# A Triton compilation of this kernel is ~100-700ms; a cached launch (even at 16k
# context) is a few ms. 50ms cleanly separates "recompiled" from "cached".
_COMPILE_LATENCY_S = 0.05


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mqa_logits_workspace_does_not_recompile_per_context():
    # Pay the initial one-time compilation at a first context length.
    _call(64, 1024)

    offenders = []
    # Include context lengths that are NOT multiples of 16. Triton specializes a
    # kernel on the *divisibility-by-16* of its runtime int args (num_rows,
    # seq_len_kv, the output row-stride), even when they are plain i32/i64 and not
    # tl.constexpr. During decode the context grows by 1 token per step, so it
    # constantly crosses the ÷16 boundary -> a fresh compile every ~16 tokens,
    # which stacks into the multi-minute PP-chain wedge. A test that only probes
    # ÷16 lengths (2048, 4096, ...) never leaves the "divisible" specialization
    # and masks the bug. The fix is do_not_specialize on those args.
    for n in (2048, 2049, 4096, 4097, 8192, 8193, 8208, 8209, 16000, 15999):
        first = _time_call(64, n)               # compiles here iff specializing
        cached = min(_time_call(64, n) for _ in range(2))  # same n -> cached
        if first - cached > _COMPILE_LATENCY_S:
            offenders.append((n, first, cached))

    assert not offenders, (
        "indexer logits kernel paid a fresh compile on new context length(s): "
        + ", ".join(
            f"N={n}: first {f * 1e3:.0f}ms vs cached {c * 1e3:.0f}ms"
            for n, f, c in offenders
        )
        + " -- a stride/size declared tl.constexpr is specializing on the "
        "context length; make it a runtime kernel argument."
    )
