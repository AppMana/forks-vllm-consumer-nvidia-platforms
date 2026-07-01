# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sm_8x torch fallbacks for V4 fp8_ds_mla KV-cache kernels.

The Triton kernels in `cache_utils.py` use `tl.float8e4nv` casts that
sm_8x cannot lower. The fallbacks emulate the kernels via PyTorch's
software-emulated `torch.float8_e4m3fn` cast and produce a binary-equal
KV-cache layout.
"""
from __future__ import annotations

import pytest
import torch

from vllm.models.deepseek_v4.common.ops.cache_utils import (
    dequantize_and_gather_k_cache,
    _dequantize_and_gather_k_cache_torch,
    _dequantize_global_slots_k_cache_torch,
    _quantize_and_insert_k_cache_torch,
    _supports_fp8e4nv_in_triton,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required",
)


def _make_empty_k_cache(num_blocks: int, block_size: int) -> torch.Tensor:
    """Allocate a fp8_ds_mla cache: (num_blocks, block_stride) uint8."""
    fp8_dim = 448
    bf16_dim = 64
    scale_dim = 8
    token_data_size = fp8_dim + bf16_dim * 2  # 576
    # block_stride = 64 tokens × 576 bytes + 64 × 8 bytes scales = 37376 bytes,
    # rounded to multiple of 576 (the kernel allows extra padding bytes).
    block_stride = block_size * token_data_size + block_size * scale_dim
    return torch.zeros(num_blocks, block_stride, dtype=torch.uint8, device="cuda")


def test_round_trip_quant_dequant() -> None:
    """Store K, then gather it back: per-block UE8M0 dequant should match."""
    block_size = 64
    num_blocks = 4
    num_tokens = 16
    torch.manual_seed(0)

    k = (torch.randn(num_tokens, 512, dtype=torch.bfloat16, device="cuda")) * 1.5
    k_cache = _make_empty_k_cache(num_blocks, block_size)

    # Pack `num_tokens` tokens into the first block, contiguous slots.
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device="cuda")
    _quantize_and_insert_k_cache_torch(k, k_cache, slot_mapping, block_size)

    # Gather all `num_tokens` back via the gather fallback (1 sequence).
    out = torch.zeros(1, num_tokens, 512, dtype=torch.bfloat16, device="cuda")
    seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device="cuda")
    gather_lens = torch.tensor([num_tokens], dtype=torch.int32, device="cuda")
    block_table = torch.zeros(1, 1, dtype=torch.int32, device="cuda")  # block 0
    _dequantize_and_gather_k_cache_torch(
        out, k_cache, seq_lens, gather_lens, block_table, block_size, offset=0
    )

    recovered = out[0]
    fp8_part_orig = k[:, :448].float()
    fp8_part_recv = recovered[:, :448].float()
    bf16_part_orig = k[:, 448:].float()
    bf16_part_recv = recovered[:, 448:].float()

    # BF16 portion is byte-copied; should match exactly.
    assert torch.equal(bf16_part_orig, bf16_part_recv), (
        "BF16 tail must round-trip exactly"
    )

    # FP8 portion: per-64-block UE8M0 quant. Error bounded by ~max/8 (E4M3
    # mantissa precision after block-scaling).
    err = (fp8_part_orig - fp8_part_recv).abs()
    blocks = fp8_part_orig.view(num_tokens, 7, 64)
    block_max = blocks.abs().amax(dim=-1, keepdim=True).expand_as(blocks).reshape_as(
        fp8_part_orig
    )
    tol = block_max / 8.0 + 1e-3
    assert (err <= tol).all(), (
        f"FP8 round-trip exceeded UE8M0/E4M3 tolerance: "
        f"max_err={err.max().item():.4f}, max_ref={block_max.max().item():.4f}"
    )


def test_global_slots_dequant_matches_gather() -> None:
    """Global-slot dequant must produce the same bf16 as the gather path."""
    block_size = 64
    num_blocks = 4
    num_tokens = 8
    torch.manual_seed(1)

    k = torch.randn(num_tokens, 512, dtype=torch.bfloat16, device="cuda")
    k_cache = _make_empty_k_cache(num_blocks, block_size)
    # Distribute tokens across two blocks — exercises block_idx arithmetic.
    slot_mapping = torch.tensor(
        [0, 1, 2, 3, 64, 65, 66, 67], dtype=torch.int64, device="cuda"
    )
    _quantize_and_insert_k_cache_torch(k, k_cache, slot_mapping, block_size)

    # Mark a few invalid (-1) slots, mix with valid ones.
    slot_ids = torch.tensor(
        [[0, 64, -1], [3, 67, 1]], dtype=torch.int64, device="cuda"
    )
    out = torch.empty(2, 3, 512, dtype=torch.bfloat16, device="cuda")
    _dequantize_global_slots_k_cache_torch(out, k_cache, slot_ids, block_size)

    # Slot -1 must be all-zero bf16.
    assert torch.equal(
        out[0, 2], torch.zeros(512, dtype=torch.bfloat16, device="cuda")
    )

    # Valid slots: dequantized values should be within UE8M0 tol of original.
    pairs = [(0, 0, 0), (0, 1, 4), (1, 0, 3), (1, 1, 7), (1, 2, 1)]
    for out_row, out_col, k_idx in pairs:
        ref_bf16 = recovered = out[out_row, out_col]
        orig = k[k_idx]
        err = (orig.float()[:448] - recovered.float()[:448]).abs()
        block_max = (
            orig.float()[:448]
            .view(7, 64)
            .abs()
            .amax(dim=-1, keepdim=True)
            .expand(7, 64)
            .reshape(448)
        )
        tol = block_max / 8.0 + 1e-3
        assert (err <= tol).all(), (
            f"global-slot dequant diverged for slot mapping ({out_row},{out_col})"
        )
        # bf16 tail exact.
        assert torch.equal(orig[448:], recovered[448:])


def test_round_trip_with_padded_cache_stride() -> None:
    """Real KV-cache blocks have padded stride > shape[1].

    Reproduces the cluster failure where `_gather_token_bytes` did
    `k_cache.view(-1)` on a non-contiguous tensor and crashed with
    'view size is not compatible with input tensor's size and stride'.
    """
    block_size = 64
    num_blocks = 4
    num_tokens = 4
    fp8_dim = 448
    bf16_dim = 64
    scale_dim = 8
    token_data_size = fp8_dim + bf16_dim * 2  # 576
    needed = block_size * token_data_size + block_size * scale_dim
    # Allocate with extra outer-dim padding so stride(0) > shape[1].
    pad = 1024
    backing = torch.zeros(num_blocks, needed + pad, dtype=torch.uint8, device="cuda")
    k_cache = backing[:, :needed]
    assert k_cache.stride(0) > k_cache.shape[1], "expected padded stride for repro"

    torch.manual_seed(2)
    k = torch.randn(num_tokens, 512, dtype=torch.bfloat16, device="cuda")
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device="cuda")
    _quantize_and_insert_k_cache_torch(k, k_cache, slot_mapping, block_size)

    out = torch.zeros(1, num_tokens, 512, dtype=torch.bfloat16, device="cuda")
    seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device="cuda")
    block_table = torch.zeros(1, 1, dtype=torch.int32, device="cuda")
    _dequantize_and_gather_k_cache_torch(
        out, k_cache, seq_lens, None, block_table, block_size, offset=0
    )
    # bf16 tail must match exactly.
    assert torch.equal(out[0, :, 448:], k[:, 448:])


def test_dispatch_skips_triton_on_ampere() -> None:
    """Confirm the gate routes Ampere to the torch fallback."""
    cap = torch.cuda.get_device_capability()
    if (cap[0], cap[1]) >= (8, 9):
        pytest.skip("requires sm_8x (Ampere, sm_8.0–8.6)")
    assert _supports_fp8e4nv_in_triton() is False


def test_ampere_dispatch_uses_native_dequant_gather(monkeypatch) -> None:
    """Ampere should use the native fp8_ds_mla gather before the torch fallback."""
    called = False

    def fake_native(out, k_cache, seq_lens, gather_lens, block_table, block_size, offset):
        nonlocal called
        called = True

    monkeypatch.setattr(
        torch.ops._C,
        "deepseek_v4_fp8_ds_mla_dequantize_and_gather_k_cache",
        fake_native,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm.models.deepseek_v4.common.ops.cache_utils._supports_fp8e4nv_in_triton",
        lambda: False,
    )

    out = torch.empty(1, 1, 512, dtype=torch.bfloat16, device="cuda")
    k_cache = _make_empty_k_cache(1, 64)
    seq_lens = torch.tensor([0], dtype=torch.int32, device="cuda")
    block_table = torch.zeros(1, 1, dtype=torch.int32, device="cuda")
    dequantize_and_gather_k_cache(out, k_cache, seq_lens, None, block_table, 64, 0)
    assert called


def test_native_dequant_gather_matches_torch_fallback() -> None:
    native_op = getattr(
        torch.ops._C, "deepseek_v4_fp8_ds_mla_dequantize_and_gather_k_cache", None
    )
    if native_op is None:
        pytest.skip("native fp8_ds_mla dequant gather op is not built")

    block_size = 64
    num_blocks = 8
    seq_lens_host = [65, 17]
    gather_lens_host = [33, 9]
    max_gather_len = max(gather_lens_host)
    torch.manual_seed(3)
    k = torch.randn(sum(seq_lens_host), 512, dtype=torch.bfloat16, device="cuda")
    k_cache = _make_empty_k_cache(num_blocks, block_size)

    block_table = torch.tensor([[0, 2], [5, -1]], dtype=torch.int32, device="cuda")
    slot_mapping = torch.empty(k.size(0), dtype=torch.int64, device="cuda")
    start = 0
    for req_id, seq_len in enumerate(seq_lens_host):
        logical_pos = torch.arange(seq_len, dtype=torch.int64, device="cuda")
        block_idx = block_table[req_id, logical_pos // block_size].to(torch.int64)
        slot_mapping[start : start + seq_len] = block_idx * block_size + (
            logical_pos % block_size
        )
        start += seq_len
    _quantize_and_insert_k_cache_torch(k, k_cache, slot_mapping, block_size)

    seq_lens = torch.tensor(seq_lens_host, dtype=torch.int32, device="cuda")
    gather_lens = torch.tensor(gather_lens_host, dtype=torch.int32, device="cuda")
    ref = torch.empty(2, max_gather_len + 2, 512, dtype=torch.bfloat16, device="cuda")
    actual = torch.empty_like(ref)
    _dequantize_and_gather_k_cache_torch(
        ref, k_cache, seq_lens, gather_lens, block_table, block_size, offset=1
    )
    native_op(actual, k_cache, seq_lens, gather_lens, block_table, block_size, 1)
    torch.cuda.synchronize()
    for req_id, gather_len in enumerate(gather_lens_host):
        torch.testing.assert_close(
            actual[req_id, 1 : 1 + gather_len],
            ref[req_id, 1 : 1 + gather_len],
            rtol=0,
            atol=0,
        )
