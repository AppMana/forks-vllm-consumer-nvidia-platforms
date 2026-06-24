# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sm_8x torch fallback for `deepseek_v4_sm12x_fp8_einsum`.

The SM12x Triton FP8 einsum kernel uses `tl.float8e4nv` casts inside
`tl.dot(a, b, ...)`, which Triton refuses on Ampere::

    ValueError: type fp8e4nv not supported in this architecture.

`_deepseek_v4_fp8_einsum_torch` mirrors the kernel using fp32 dequant +
torch.bmm, so V4-Flash's output projection (wo_a einsum) works on sm_8x.
"""
from __future__ import annotations

import pytest
import torch

from vllm.models.deepseek_v4.common.ops.fp8_einsum import (
    _deepseek_v4_fp8_einsum_torch,
    _supports_fp8e4nv_in_triton,
    deepseek_v4_sm12x_fp8_einsum,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required",
)


def _is_ampere() -> bool:
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return (cap[0], cap[1]) < (8, 9)


def _make_block_quant_tensor(
    shape: tuple[int, ...], block_dim: int, block_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create an fp8_e4m3fn tensor with random per-block fp32 scales.

    The reference dequant is `fp8 * scale`. ``shape`` is the unpacked
    shape; the scale tensor has ``shape[block_dim] / block_size`` along
    that axis and full size along the others.
    """
    ref_bf16 = torch.randn(shape, dtype=torch.bfloat16, device="cuda") * 0.5
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)

    # Compute per-block max → fp32 scale, quantize, store.
    block_count = shape[block_dim] // block_size
    new_shape = list(shape)
    new_shape[block_dim] = block_count
    new_shape.insert(block_dim + 1, block_size)
    blocked = ref_bf16.view(*new_shape).to(torch.float32)
    block_amax = blocked.abs().amax(dim=block_dim + 1, keepdim=True).clamp_min(1e-8)
    block_scale = block_amax / fp8_max
    fp8_view = (blocked / block_scale).to(torch.float8_e4m3fn)
    fp8 = fp8_view.view(*shape)
    scale = block_scale.squeeze(block_dim + 1).to(torch.float32)
    return fp8, scale


def test_torch_fallback_matches_dequant_reference() -> None:
    """Pure-torch fallback ≈ fp32-dequant matmul reference."""
    num_tokens = 16
    num_groups = 8
    hidden_size = 256  # 2 hidden blocks
    out_rank = 1024  # 8 out blocks

    # Build a, b with realistic block-quant layouts.
    a_fp8, a_scale = _make_block_quant_tensor(
        (num_tokens, num_groups, hidden_size), block_dim=2
    )

    # b shape = (G, R, H); the scale is per (G, R/128, H/128). Easiest path:
    # build b as bf16, block-quant per (R/128, H/128) tile.
    b_bf16 = (
        torch.randn(num_groups, out_rank, hidden_size, dtype=torch.bfloat16, device="cuda")
        * 0.5
    )
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    b_blocks = b_bf16.view(num_groups, out_rank // 128, 128, hidden_size // 128, 128).to(
        torch.float32
    )
    b_amax = b_blocks.abs().amax(dim=(2, 4), keepdim=True).clamp_min(1e-8)
    b_scale = (b_amax / fp8_max).squeeze(4).squeeze(2).to(torch.float32)
    b_quant = (b_blocks / b_amax * fp8_max).to(torch.float8_e4m3fn)
    b_fp8 = b_quant.view(num_groups, out_rank, hidden_size).contiguous()

    out = torch.empty(
        num_tokens, num_groups, out_rank, dtype=torch.bfloat16, device="cuda"
    )
    _deepseek_v4_fp8_einsum_torch(a_fp8, a_scale, b_fp8, b_scale, out)

    # Reference: dequant both sides, then einsum exactly.
    a_dq = (
        a_fp8.to(torch.float32).view(num_tokens, num_groups, hidden_size // 128, 128)
        * a_scale.unsqueeze(-1)
    ).view(num_tokens, num_groups, hidden_size)
    b_dq = (
        b_fp8.to(torch.float32).view(
            num_groups, out_rank // 128, 128, hidden_size // 128, 128
        )
        * b_scale.unsqueeze(2).unsqueeze(-1)
    ).view(num_groups, out_rank, hidden_size)
    ref = torch.einsum("tgh,gdh->tgd", a_dq, b_dq).to(torch.bfloat16)

    err = (out.float() - ref.float()).abs()
    # The fallback IS the dequant-then-bmm path, so it should match the
    # reference to within bf16 store precision.
    assert err.max().item() < 1e-2, (
        f"torch fallback diverged from dequant reference: "
        f"max_err={err.max().item():.4e}"
    )


def test_dispatch_uses_torch_fallback_on_ampere() -> None:
    """`deepseek_v4_sm12x_fp8_einsum` must NOT invoke Triton on sm_8x."""
    if not _is_ampere():
        pytest.skip("requires sm_8x")
    assert _supports_fp8e4nv_in_triton() is False

    num_tokens, num_groups, hidden_size, out_rank = 8, 4, 128, 256
    a_fp8, a_scale = _make_block_quant_tensor(
        (num_tokens, num_groups, hidden_size), block_dim=2
    )
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    b_bf16 = torch.randn(
        num_groups, out_rank, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    b_blocks = b_bf16.view(num_groups, out_rank // 128, 128, hidden_size // 128, 128).to(
        torch.float32
    )
    b_amax = b_blocks.abs().amax(dim=(2, 4), keepdim=True).clamp_min(1e-8)
    b_scale = (b_amax / fp8_max).squeeze(4).squeeze(2).to(torch.float32)
    b_fp8 = (b_blocks / b_amax * fp8_max).to(torch.float8_e4m3fn).view(
        num_groups, out_rank, hidden_size
    ).contiguous()

    out = torch.empty(
        num_tokens, num_groups, out_rank, dtype=torch.bfloat16, device="cuda"
    )
    # If this raised `ValueError: type fp8e4nv not supported`, the dispatch
    # would still be invoking the Triton kernel.
    deepseek_v4_sm12x_fp8_einsum(a_fp8, a_scale, b_fp8, b_scale, out)
    assert torch.isfinite(out.float()).all()
