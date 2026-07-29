# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep the DSV4 sparse-MLA attention over query-width T = 0..13 and compare
the two implementations a spec-verify step could use:

  * flash_sparse_mla_decode  -- the s_q-parameterized "spec-as-decode" kernel
  * flash_sparse_mla_prefill -- the causal multi-query path

For each T we check CORRECTNESS (cos-diff of each flash path vs the Triton
sparse-attention reference, per query row) and LATENCY (median over N iters),
then recommend, per T, the faster path among the correct ones.

This is the microbenchmark behind the routing decision for DSpark verify
steps (whose per-request query width is 1 + parallel*num_spec, up to 13).

Run on Ampere (sm_8x); picks the least-busy GPU unless CUDA_VISIBLE_DEVICES
is set. fp8_ds_mla cache (the path with both flash decode+prefill kernels).
"""

import math
import os
import statistics
import time

import torch

import flash_mla
from flash_mla import flash_sparse_mla_decode, flash_sparse_mla_prefill

from vllm.models.deepseek_v4.nvidia_imma.triton_kernels import (
    decode_sparse_attention_triton,
)

_FP8_DIM = 448
_ROPE_DIM = 64
_SCALE_DIM = 8
_TOKEN_DATA_SIZE = _FP8_DIM + _ROPE_DIM * 2  # 576
_HEAD_DIM = 512


def _write_fp8_ds_mla_token(k_cache, slot, block_size):
    block_idx = slot // block_size
    block_offset = slot % block_size
    values = (
        (torch.arange(_FP8_DIM, device=k_cache.device, dtype=torch.float32) % 17) - 8
    ) / 16.0 + float(slot) / 32.0
    scale_exponents = torch.tensor(
        [-2, -1, 0, 1, 2, -2, 1], device=k_cache.device, dtype=torch.float32
    )
    scale_per_dim = torch.exp2(scale_exponents).repeat_interleave(64)
    fp8_values = (values / scale_per_dim).to(torch.float8_e4m3fn)
    rope = (
        torch.linspace(-1.0, 1.0, _ROPE_DIM, device=k_cache.device) + float(slot) / 16.0
    ).to(torch.bfloat16)
    flat = k_cache[block_idx].view(-1)
    ds = block_offset * _TOKEN_DATA_SIZE
    ss = block_size * _TOKEN_DATA_SIZE + block_offset * _SCALE_DIM
    flat[ds : ds + _FP8_DIM] = fp8_values.view(torch.uint8)
    flat[ds + _FP8_DIM : ds + _TOKEN_DATA_SIZE] = rope.view(torch.uint8)
    enc = (scale_exponents.to(torch.int32) + 127).to(torch.uint8)
    flat[ss : ss + enc.numel()] = enc
    flat[ss + enc.numel() : ss + _SCALE_DIM] = 127


def _row_cos_diff(x, y):
    # Per-row (per query token) cosine distance; returns the WORST row so a
    # single degenerate/collapsed row is visible even if others agree.
    x, y = x.double(), y.double()
    num = (x * y).sum(dim=(-1, -2))
    den = (x * x).sum(dim=(-1, -2)) + (y * y).sum(dim=(-1, -2))
    cd = 1 - 2 * num / den.clamp_min(1e-12)
    return float(cd.max())


def _collapsed(out):
    # Are all T query rows identical (the degenerate signature)?
    if out.shape[0] < 2:
        return False
    first = out[0]
    return bool((out == first).all())


def _pick_gpu():
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    try:
        import subprocess

        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            check=True, text=True, capture_output=True,
        )
        rows = [ln.split(",") for ln in r.stdout.splitlines() if ln.strip()]
        rows = [(i.strip(), int(m)) for i, m in rows]
        rows.sort(key=lambda x: x[1])
        os.environ["CUDA_VISIBLE_DEVICES"] = rows[0][0]
    except Exception:
        pass


def _time(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(ts)


def run(topk, prefix_len, H=64, block_size=32, iters=50):
    dev = "cuda"
    scale = 1.0 / math.sqrt(_HEAD_DIM)
    # Sequence has `prefix_len` valid KV slots (short-seq stress: prefix < topk).
    num_slots = max(topk + 64, prefix_len + 16)
    nb = (num_slots + block_size - 1) // block_size
    cache = torch.zeros(
        nb, block_size, _TOKEN_DATA_SIZE + _SCALE_DIM, dtype=torch.uint8, device=dev
    )
    for slot in range(num_slots):
        _write_fp8_ds_mla_token(cache, slot, block_size)
    sink = torch.randn(H, device=dev, dtype=torch.float32) * 0.1

    print(
        f"\n=== topk={topk} prefix_len={prefix_len} (short-seq={'Y' if prefix_len < topk else 'N'}) ==="
    )
    print(
        f"{'T':>3} {'dec_cd':>9} {'pre_cd':>9} {'dec_collapse':>12} "
        f"{'dec_us':>8} {'pre_us':>8} {'winner':>8}"
    )
    for T in range(0, 14):
        if T == 0:
            print(f"{0:>3} {'(no-op)':>9}")
            continue
        torch.manual_seed(1234 + T)
        q = torch.randn(T, H, _HEAD_DIM, device=dev, dtype=torch.bfloat16)
        # Per-token CAUSAL lens: query row i attends to prefix_len - (T-1) + i
        # valid slots (the verify's staircase), capped at topk.
        base = max(1, prefix_len - (T - 1))
        lens = torch.tensor(
            [min(base + i, topk) for i in range(T)], dtype=torch.int32, device=dev
        )
        idx = torch.stack(
            [
                torch.randperm(num_slots, device=dev)[:topk].to(torch.int32)
                for _ in range(T)
            ]
        )

        def _dec():
            return flash_sparse_mla_decode(
                q=q, swa_cache=cache, swa_indices=idx, swa_lens=lens,
                scale=scale, attn_sink=sink,
            )

        def _pre():
            return flash_sparse_mla_prefill(
                q=q, swa_cache=cache, swa_indices=idx, swa_lens=lens,
                scale=scale, attn_sink=sink,
            )

        ref = torch.empty(T, H, _HEAD_DIM, device=dev, dtype=torch.bfloat16)
        decode_sparse_attention_triton(
            q=q, swa_cache=cache, swa_indices=idx, swa_lens=lens,
            scale=scale, attn_sink=sink, out=ref,
        )
        dec = _dec()
        pre = _pre()
        dec_cd = _row_cos_diff(dec.float(), ref.float())
        pre_cd = _row_cos_diff(pre.float(), ref.float())
        dec_col = _collapsed(dec)

        dec_ok = dec_cd < 1e-3 and not dec_col
        pre_ok = pre_cd < 1e-3
        dec_us = _time(_dec, iters=iters)
        pre_us = _time(_pre, iters=iters)
        if dec_ok and pre_ok:
            winner = "decode" if dec_us <= pre_us else "prefill"
        elif pre_ok:
            winner = "prefill*"  # decode incorrect
        elif dec_ok:
            winner = "decode*"
        else:
            winner = "NEITHER"
        print(
            f"{T:>3} {dec_cd:>9.2e} {pre_cd:>9.2e} "
            f"{'YES' if dec_col else '.':>12} {dec_us:>8.1f} {pre_us:>8.1f} {winner:>8}"
        )


def main():
    _pick_gpu()
    print("flash_mla:", getattr(flash_mla, "__file__", "?"))
    print("device:", torch.cuda.get_device_name(0))
    # Short-seq (degeneracy regime: prefix < topk) and normal-seq.
    for topk, prefix_len in [(256, 27), (256, 512), (512, 27), (512, 2048)]:
        run(topk, prefix_len)


if __name__ == "__main__":
    raise SystemExit(main())
