# SPDX-License-Identifier: Apache-2.0
"""Peak-memory and kernel-time comparison: one-shot indexer prefill top-k vs
activation-chunked streaming top-k, at the DSV4 prefill-step shape
(chunk=1024 queries, 1M-token window, CSA-4 -> N=262400 compressed rows)."""

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.sparse_attn_indexer import (
    oneshot_prefill_topk_reference,
    streaming_prefill_topk,
)
from vllm.models.deepseek_v4.nvidia_sm86.triton_kernels import (
    mqa_logits_workspace_triton,
)

M, H, D, K = 1024, 64, 128, 2048
WINDOW = 1 << 20  # 1M tokens
C = 4  # CSA-4
N = (WINDOW + M) // C  # 262400 compressed rows
DEV = "cuda"

print(f"M={M} N={N} H={H} D={D} K={K}")
print(f"one-shot logits [M,N] fp32 = {M * N * 4 / 2**30:.3f} GiB")

torch.manual_seed(0)
q = torch.randint(-16, 16, (M, H, D), dtype=torch.int8, device=DEV)
k = torch.randint(-16, 16, (N, D), dtype=torch.int8, device=DEV)
k_scale = torch.rand(N, dtype=torch.float32, device=DEV) + 0.5
w = torch.randn(M, H, dtype=torch.float32, device=DEV) * 0.05
ks = torch.zeros(M, dtype=torch.int32, device=DEV)
ke = (torch.arange(M, dtype=torch.int32, device=DEV) + (N - M) + 1).clamp_(max=N)
out = torch.full((M, K), -1, dtype=torch.int32, device=DEV)


def one_shot():
    logits = mqa_logits_workspace_triton(q, (k, k_scale), w, ks, ke, qk_int8=True)
    ops.top_k_per_row_prefill(
        logits, ks, ke, out, M, logits.stride(0), logits.stride(1), K
    )


def one_shot_looplike():
    """Reproduce the production chunk loop's allocation pattern: the previous
    sub-chunk's logits stays referenced while the next one is computed."""
    prev = None
    for _ in range(2):
        logits = mqa_logits_workspace_triton(
            q, (k, k_scale), w, ks, ke, qk_int8=True
        )
        ops.top_k_per_row_prefill(
            logits, ks, ke, out, M, logits.stride(0), logits.stride(1), K
        )
        prev = logits  # noqa: F841  (mirrors `logits` living across iterations)


def streaming(slab):
    streaming_prefill_topk(
        q, (k, k_scale), w, ks, ke, out, K, slab_rows=slab, qk_int8=True
    )


def measure(fn, label, iters=10):
    fn()  # warmup / triton compile
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(stop) / iters
    print(f"{label:34s} peak {peak / 2**20:9.1f} MiB   {ms:8.2f} ms")
    return peak, ms


# Correctness of the exact benchmark shapes first.
ref = oneshot_prefill_topk_reference(q, (k, k_scale), w, ks, ke, K, qk_int8=True)
chk = torch.empty(M, K, dtype=torch.int32, device=DEV)
streaming_prefill_topk(q, (k, k_scale), w, ks, ke, chk, K, slab_rows=16384, qk_int8=True)
assert torch.equal(chk, ref), "streaming != one-shot at benchmark shape"
print("bit-identical at benchmark shape: OK")

_, base_ms = measure(one_shot, "one-shot (logits + topk kernel)")
measure(one_shot_looplike, "one-shot, loop pattern (2 live)")
for slab in (8192, 16384, 32768, 65536):
    p, ms = measure(lambda s=slab: streaming(s), f"streaming slab={slab}")
    print(f"{'':34s} overhead vs one-shot: {(ms / base_ms - 1) * 100:+.1f}%")
