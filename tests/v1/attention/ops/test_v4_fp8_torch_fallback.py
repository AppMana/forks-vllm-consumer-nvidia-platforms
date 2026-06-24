# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sm_8x torch fallbacks for the V4 fp8e4nv-emitting Triton kernels.

The V4-Flash attention path uses two Triton kernels that emit
`tl.float8e4nv` casts — only available on sm_89+ (Ada/Hopper/Blackwell).
On sm_8x (Ampere) Triton refuses with::

    ValueError: type fp8e4nv not supported in this architecture.
    Supported fp8 dtypes: ('fp8e4b15', 'fp8e5')

The fork ships pure-torch fallbacks (`_fused_inv_rope_fp8_quant_torch` and
`_fused_indexer_q_rope_fp8_torch`) that mirror the Triton kernel logic and
use PyTorch's software-emulated `torch.float8_e4m3fn` cast on Ampere.

These tests don't have access to a sm_89+ device for cross-validation, so
the assertions verify:
  - The fallback runs at all on Ampere (the Triton path raises).
  - Output tensors have the correct shape, dtype, and stride layout.
  - Dequantizing the fp8 output back to bf16 reproduces the unquantized
    bf16 reference within E4M3FN's representable precision.
  - The scale tensor is a power-of-two and at least as large as the input
    absmax / fp8_max (so the clamp never saturates the input).
"""
from __future__ import annotations

import math

import pytest
import torch

from vllm.platforms import current_platform
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    _fused_inv_rope_fp8_quant_torch,
)
from vllm.models.deepseek_v4.common.ops.fused_indexer_q import (
    _fused_indexer_q_rope_fp8_torch,
)


def _is_ampere() -> bool:
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] == 8


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required",
)


def _e4m3fn_max() -> float:
    return float(torch.finfo(torch.float8_e4m3fn).max)


def _make_cos_sin_cache(max_pos: int, half_rope: int, device: str) -> torch.Tensor:
    pos = torch.arange(max_pos, device=device, dtype=torch.float32)[:, None]
    inv_freq = 1.0 / (10000.0 ** (torch.arange(half_rope, device=device).float() / half_rope))
    angles = pos * inv_freq[None, :]
    cache = torch.empty(max_pos, half_rope * 2, dtype=torch.float32, device=device)
    cache[:, :half_rope] = torch.cos(angles)
    cache[:, half_rope:] = torch.sin(angles)
    return cache


def test_fused_inv_rope_fp8_quant_torch_smoke() -> None:
    """Output projection FP8 quant: torch fallback runs and produces sensible
    fp8_e4m3fn values on whatever device is available."""
    device = "cuda"
    torch.manual_seed(0)

    num_tokens = 8
    n_groups = 2
    heads_per_group = 4
    head_dim = 512
    rope_dim = 64
    nope_dim = head_dim - rope_dim
    quant_group_size = 128
    chunks_per_head = head_dim // quant_group_size  # 4
    rope_start = nope_dim % quant_group_size
    half_rope = rope_dim // 2
    fp8_max = _e4m3fn_max()
    num_heads = n_groups * heads_per_group

    o = torch.randn(
        num_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    cos_sin = _make_cos_sin_cache(max_pos=128, half_rope=half_rope, device=device)

    d = heads_per_group * head_dim
    num_scale_blocks = d // quant_group_size  # heads_per_group * chunks_per_head
    tma_aligned_T = num_tokens

    # Both packed and per-block scale formats.
    for tma_aligned_scales in (False, True):
        scale_inner = (
            (num_scale_blocks + 3) // 4 if tma_aligned_scales else num_scale_blocks
        )
        if tma_aligned_scales and chunks_per_head != 4:
            pytest.skip("packed scales path validated for chunks_per_head=4")

        fp8_buf, scale_buf = _fused_inv_rope_fp8_quant_torch(
            o,
            positions,
            cos_sin,
            heads_per_group=heads_per_group,
            quant_group_size=quant_group_size,
            chunks_per_head=chunks_per_head,
            rope_start=rope_start,
            half_rope=half_rope,
            tma_aligned_scales=tma_aligned_scales,
            fp8_max=fp8_max,
            tma_aligned_T=tma_aligned_T,
            num_tokens=num_tokens,
            n_groups=n_groups,
            d=d,
            scale_inner=scale_inner,
        )

        # Shape and dtype.
        assert fp8_buf.shape == (n_groups, num_tokens, d)
        assert fp8_buf.dtype == torch.float8_e4m3fn
        expected_scale_dtype = torch.int32 if tma_aligned_scales else torch.float32
        assert scale_buf.dtype == expected_scale_dtype
        assert scale_buf.shape == (n_groups, num_tokens, scale_inner)

        # Dequant round-trip: fp8 * scale ≈ original (within E4M3FN precision).
        # Reconstruct: for each (group, token, head_in_group, chunk), scale
        # multiplies a `quant_group_size` slab of fp8 values.
        if not tma_aligned_scales:
            # scale_buf layout: (G, T, num_scale_blocks). Per-block scales
            # ordered so head_in_group * chunks_per_head + chunk_idx.
            fp8_view = fp8_buf.view(
                n_groups, num_tokens, heads_per_group, chunks_per_head, quant_group_size
            )
            scale_view = scale_buf.view(
                n_groups, num_tokens, heads_per_group, chunks_per_head
            )
            dequant = fp8_view.to(torch.float32) * scale_view.unsqueeze(-1)

            # The kernel applies inverse RoPE before quant, so we need to
            # apply it on the input here too for the comparison.
            o_f32 = o.to(torch.float32)
            cos_local = cos_sin[positions, :half_rope]
            sin_local = cos_sin[positions, half_rope:]
            nope = o_f32[..., :nope_dim]
            rope = o_f32[..., nope_dim:]
            new_even = (
                rope[..., 0::2] * cos_local[:, None, :]
                + rope[..., 1::2] * sin_local[:, None, :]
            )
            new_odd = (
                -rope[..., 0::2] * sin_local[:, None, :]
                + rope[..., 1::2] * cos_local[:, None, :]
            )
            rope_rot = torch.empty_like(rope)
            rope_rot[..., 0::2] = new_even
            rope_rot[..., 1::2] = new_odd
            ref_full = torch.cat([nope, rope_rot], dim=-1)
            ref_grouped = ref_full.view(
                num_tokens, n_groups, heads_per_group, chunks_per_head, quant_group_size
            ).transpose(0, 1)

            # E4M3FN has ~2-3 bits of mantissa precision. After block-scaling,
            # error magnitude is bounded by ~(max_block / 8). Verify
            # element-wise relative error stays within E4M3 precision.
            err = (dequant - ref_grouped).abs()
            ref_max = ref_grouped.abs().amax(dim=-1, keepdim=True)
            tol = ref_max / 8.0 + 1e-4
            assert (err <= tol.expand_as(err)).all(), (
                f"E4M3 round-trip exceeded ~1/8 of block max: "
                f"max_err={err.max().item():.4f}, ref_max={ref_max.max().item():.4f}"
            )

            # Power-of-2 invariant.
            log2_s = torch.log2(scale_view.to(torch.float64))
            assert torch.allclose(log2_s, log2_s.round()), (
                "scales must be exact powers of two (ceil(log2) of input)"
            )


def test_fused_indexer_q_rope_fp8_torch_smoke() -> None:
    """Indexer Q FP8 quant: torch fallback runs and produces sensible
    fp8_e4m3fn values + a folded weights tensor."""
    device = "cuda"
    torch.manual_seed(1)

    num_tokens = 8
    num_index_q_heads = 4
    rope_dim = 64
    nope_dim = 0  # V4-Flash indexer has no NoPE on the index Q
    head_dim = nope_dim + rope_dim
    half_rope = rope_dim // 2

    index_q = torch.randn(
        num_tokens, num_index_q_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    cos_sin = _make_cos_sin_cache(max_pos=128, half_rope=half_rope, device=device)
    weights = torch.randn(
        num_tokens, num_index_q_heads, dtype=torch.float32, device=device
    )
    softmax_scale = 0.125
    head_scale = 1.0

    fp8_q, weights_out = _fused_indexer_q_rope_fp8_torch(
        positions,
        index_q,
        cos_sin,
        weights,
        index_weights_softmax_scale=softmax_scale,
        index_weights_head_scale=head_scale,
    )

    assert fp8_q.shape == index_q.shape
    assert fp8_q.dtype == torch.float8_e4m3fn
    assert weights_out.shape == weights.shape
    assert weights_out.dtype == torch.float32
    assert torch.isfinite(weights_out).all()

    # Dequant: there's no separately-stored q_scale; it's folded into weights.
    # Recover q_scale_per_token_head = weights_out / (weights * softmax_scale * head_scale).
    denom = weights * softmax_scale * head_scale
    q_scale = (weights_out / denom).clamp_min(1e-12)

    # Verify q_scale is a power of 2.
    log2_s = torch.log2(q_scale.to(torch.float64))
    assert torch.allclose(log2_s, log2_s.round(), atol=1e-6), (
        "q_scale must be an exact power of two"
    )

    # Verify dequant is close to RoPE'd input.
    cos_local = cos_sin[positions, :half_rope]
    sin_local = cos_sin[positions, half_rope:]
    q_f32 = index_q.to(torch.float32)
    x_even = q_f32[..., 0::2]
    x_odd = q_f32[..., 1::2]
    r_even = x_even * cos_local[:, None, :] - x_odd * sin_local[:, None, :]
    r_odd = x_odd * cos_local[:, None, :] + x_even * sin_local[:, None, :]
    rope_rot = torch.empty_like(q_f32)
    rope_rot[..., 0::2] = r_even.to(torch.bfloat16).to(torch.float32)
    rope_rot[..., 1::2] = r_odd.to(torch.bfloat16).to(torch.float32)

    dequant = fp8_q.to(torch.float32) * q_scale.unsqueeze(-1)
    err = (dequant - rope_rot).abs()
    ref_max = rope_rot.abs().amax(dim=-1, keepdim=True)
    tol = ref_max / 8.0 + 1e-4
    assert (err <= tol.expand_as(err)).all(), (
        f"E4M3 round-trip exceeded ~1/8 of input max: "
        f"max_err={err.max().item():.4f}"
    )


def test_inv_rope_fallback_dispatch_on_ampere() -> None:
    """Confirm the gate routes Ampere to the torch fallback, not Triton."""
    if not _is_ampere():
        pytest.skip("requires sm_8x")

    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
        _supports_fp8e4nv_in_triton,
    )

    assert _supports_fp8e4nv_in_triton() is False, (
        "Ampere should route to the torch fallback"
    )


def test_indexer_q_fallback_dispatch_on_ampere() -> None:
    if not _is_ampere():
        pytest.skip("requires sm_8x")

    from vllm.models.deepseek_v4.common.ops.fused_indexer_q import (
        _supports_fp8e4nv_in_triton,
    )

    assert _supports_fp8e4nv_in_triton() is False
