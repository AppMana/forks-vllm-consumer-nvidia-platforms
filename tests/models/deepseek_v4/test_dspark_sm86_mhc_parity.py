# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel-level parity for the sm86-routed MHC path (MHCPreOp/MHCPostOp/
HCHeadOp) vs a pure-torch fp32 oracle.

This is the test that makes commit fcf42a8acf's fix ("dspark sm86: route MHC
through MHCPostOp/HCHeadOp, not bare tilelang calls") load-bearing: it proves
the Triton path DSpark now reaches on sm_8x (verified live in
test_dspark_sm86_characterization.py) numerically agrees with the reference
math, not just that it runs without crashing.

Oracle = vllm.model_executor.kernels.mhc.torch.{mhc_pre_torch,mhc_post_torch}
(mhc_pre/post already ship a pure-torch reference for this exact purpose) and
a local fp32 re-derivation of the hc_head reduction (mirrors
mhc.py::_hc_head_fused_reference, re-derived here rather than imported so the
oracle doesn't share a bug with the module under test).

Tolerance: rtol/atol 2e-2 (this fork's established kernel-parity convention,
see e6ac87ecb1 / 1be9094d15) plus a cosine-similarity check at <8e-5 relative
delta on the flattened outputs (this fork's tighter end-to-end tolerance).
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and current_platform.is_cuda()),
    reason="requires a CUDA device",
)


def _cosine_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    cos_sim = F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()
    return abs(1.0 - cos_sim)


def _hc_head_oracle_fp32(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """Independent fp32 re-derivation of the hc_head reduction (does not
    import mhc.py's own reference, so the oracle can't share its bug)."""
    x_float = hs_flat.flatten(-2).float()
    rstd = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + rms_eps)
    x_normed = x_float * rstd
    mixes = F.linear(x_normed, fn.float())
    gate = torch.sigmoid(mixes * hc_scale.float() + hc_base.float()) + hc_eps
    return torch.sum(gate.unsqueeze(-1) * hs_flat.float(), dim=1)


@pytest.mark.parametrize("hc_mult,hidden_size,num_tokens", [(4, 64, 8), (2, 128, 5)])
def test_mhc_pre_sm86_triton_matches_fp32_oracle(
    hc_mult: int, hidden_size: int, num_tokens: int, default_vllm_config
) -> None:
    from vllm.model_executor.kernels.mhc.torch import mhc_pre_torch
    from vllm.model_executor.layers.mhc import MHCPreOp, _MHC_PRE_TRITON, _MHC_TORCH_FALLBACK

    assert _MHC_TORCH_FALLBACK and _MHC_PRE_TRITON, (
        "test assumes the sm86 Triton path is live (see characterization test)"
    )

    torch.manual_seed(0)
    device = "cuda"
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    residual = torch.randn(
        num_tokens, hc_mult, hidden_size, device=device, dtype=torch.bfloat16
    )
    fn = torch.randn(hc_mult3, hc_mult * hidden_size, device=device, dtype=torch.float32)
    hc_scale = torch.rand(3, device=device, dtype=torch.float32) + 0.5
    hc_base = torch.randn(hc_mult3, device=device, dtype=torch.float32) * 0.1
    rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value = 1e-6, 1e-3, 1e-6, 2.0
    sinkhorn_repeat = 2

    op = MHCPreOp()
    post_mix, comb_mix, layer_input = op.forward_cuda(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )

    exp_post_mix, exp_comb_mix, exp_layer_input = mhc_pre_torch(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )

    torch.testing.assert_close(post_mix.float(), exp_post_mix.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(comb_mix.float(), exp_comb_mix.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        layer_input.float(), exp_layer_input.float(), rtol=2e-2, atol=2e-2
    )
    assert _cosine_delta(layer_input, exp_layer_input) < 8e-5


@pytest.mark.parametrize("hc_mult,hidden_size,num_tokens", [(4, 64, 8), (2, 128, 5)])
def test_mhc_post_sm86_triton_matches_fp32_oracle(
    hc_mult: int, hidden_size: int, num_tokens: int, default_vllm_config
) -> None:
    from vllm.model_executor.kernels.mhc.torch import mhc_post_torch
    from vllm.model_executor.layers.mhc import (
        MHCPostOp,
        _MHC_POST_TRITON,
        _MHC_TORCH_FALLBACK,
    )

    assert _MHC_TORCH_FALLBACK and _MHC_POST_TRITON

    torch.manual_seed(1)
    device = "cuda"
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    residual = torch.randn(
        num_tokens, hc_mult, hidden_size, device=device, dtype=torch.bfloat16
    )
    post_layer_mix = torch.rand(num_tokens, hc_mult, 1, device=device, dtype=torch.float32)
    comb_res_mix = torch.rand(
        num_tokens, hc_mult, hc_mult, device=device, dtype=torch.float32
    )

    op = MHCPostOp()
    out = op.forward_cuda(x, residual, post_layer_mix, comb_res_mix)
    exp = mhc_post_torch(x, residual, post_layer_mix, comb_res_mix)

    torch.testing.assert_close(out.float(), exp.float(), rtol=2e-2, atol=2e-2)
    assert _cosine_delta(out, exp) < 8e-5


@pytest.mark.parametrize("hc_mult,hidden_size,num_tokens", [(4, 64, 8), (2, 128, 5)])
def test_hc_head_sm86_triton_matches_fp32_oracle(
    hc_mult: int, hidden_size: int, num_tokens: int, default_vllm_config
) -> None:
    from vllm.model_executor.layers.mhc import (
        HCHeadOp,
        _MHC_HEAD_TRITON,
        _MHC_TORCH_FALLBACK,
    )

    assert _MHC_TORCH_FALLBACK and _MHC_HEAD_TRITON

    torch.manual_seed(2)
    device = "cuda"
    hs_flat = torch.randn(
        num_tokens, hc_mult, hidden_size, device=device, dtype=torch.bfloat16
    )
    # hc_fn projects the flattened (hc_mult * hidden_size) HC residual down to
    # hc_mult gate logits: shape (hc_mult, hc_mult * hidden_size), matching
    # F.linear(x.flatten(-2), hc_fn) in both the Triton kernel and the oracle.
    fn = torch.randn(hc_mult, hc_mult * hidden_size, device=device, dtype=torch.float32)
    hc_scale = torch.tensor(1.3, device=device, dtype=torch.float32)
    hc_base = torch.zeros(hc_mult, device=device, dtype=torch.float32)
    rms_eps, hc_eps = 1e-6, 1e-3

    op = HCHeadOp()
    out = op.forward_cuda(hs_flat, fn, hc_scale, hc_base, rms_eps, hc_eps)
    exp = _hc_head_oracle_fp32(hs_flat, fn, hc_scale, hc_base, hidden_size, rms_eps, hc_eps)

    torch.testing.assert_close(out.float(), exp.float(), rtol=2e-2, atol=2e-2)
    assert _cosine_delta(out, exp) < 8e-5
