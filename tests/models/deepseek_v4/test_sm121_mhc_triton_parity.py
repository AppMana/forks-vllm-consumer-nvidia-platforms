# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical parity for the GB10 DSV4 Triton mHC path.

The deployed DSV4 shape is hidden_size=7168, hc_mult=4, with at most 8192
scheduled tokens.  Keep those shapes here so a small kernel-only job can
exclude mHC before paying the cost of loading the full checkpoint.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

import vllm
import vllm.model_executor.kernels.mhc  # noqa: F401 -- registers torch ops
from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_torch
from vllm.platforms import current_platform


def _requires_sm121() -> bool:
    capability = current_platform.get_device_capability()
    return not (
        torch.cuda.is_available()
        and capability is not None
        and capability.major == 12
        and capability.minor == 1
    )


pytestmark = pytest.mark.skipif(_requires_sm121(), reason="requires an SM121 GPU")

HIDDEN_SIZE = 7168
HC_MULT = 4
HC_MULT3 = HC_MULT * (HC_MULT + 2)
RMS_EPS = 1.0e-6
HC_EPS = 1.0e-6
HC_POST_ALPHA = 2.0
SINKHORN_ITERS = 20


def _load_sparkinfer_adapter():
    """Load the adapter without importing the heavyweight model package."""
    path = (
        Path(vllm.__file__).parent
        / "models"
        / "deepseek_v4"
        / "nvidia_sm12x"
        / "mhc.py"
    )
    spec = importlib.util.spec_from_file_location("dsv4_sparkinfer_mhc", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(num_tokens: int):
    generator = torch.Generator(device="cuda").manual_seed(20260731 + num_tokens)
    residual = torch.randn(
        num_tokens,
        HC_MULT,
        HIDDEN_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    x = torch.randn(
        num_tokens,
        HIDDEN_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    fn = (
        torch.randn(
            HC_MULT3,
            HC_MULT * HIDDEN_SIZE,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 1.0e-4
    )
    hc_scale = torch.tensor([1.0, 1.0, 1.0], device="cuda")
    hc_base = (
        torch.randn(
            HC_MULT3, device="cuda", dtype=torch.float32, generator=generator
        )
        * 0.1
    )
    post = torch.rand(
        num_tokens,
        HC_MULT,
        1,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    comb = torch.rand(
        num_tokens,
        HC_MULT,
        HC_MULT,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    comb /= comb.sum(dim=-2, keepdim=True)
    return x, residual, post, comb, fn, hc_scale, hc_base


@pytest.mark.parametrize("num_tokens", [1, 8192])
def test_sm121_fused_post_pre_triton_matches_torch(num_tokens: int) -> None:
    x, residual, post, comb, fn, hc_scale, hc_base = _inputs(num_tokens)

    actual = torch.ops.vllm.mhc_fused_post_pre_triton(
        x,
        residual,
        post,
        comb,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_EPS,
        HC_EPS,
        HC_POST_ALPHA,
        SINKHORN_ITERS,
        1,
    )
    expected_residual = mhc_post_torch(x, residual, post, comb)
    expected_pre = mhc_pre_torch(
        expected_residual,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_EPS,
        HC_EPS,
        HC_POST_ALPHA,
        SINKHORN_ITERS,
        1,
    )
    expected = (expected_residual, *expected_pre)

    torch.cuda.synchronize()
    for result, reference in zip(actual, expected, strict=True):
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, reference, rtol=2.0e-2, atol=2.0e-2)


@pytest.mark.parametrize("num_tokens", [1, 8])
def test_sm121_hc_head_triton_matches_torch(num_tokens: int) -> None:
    _, residual, _, _, _, _, _ = _inputs(num_tokens)
    generator = torch.Generator(device="cuda").manual_seed(20260801 + num_tokens)
    fn = (
        torch.randn(
            HC_MULT,
            HC_MULT * HIDDEN_SIZE,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 1.0e-4
    )
    hc_scale = torch.tensor(1.0, device="cuda")
    hc_base = torch.zeros(HC_MULT, device="cuda")
    actual = torch.empty(
        num_tokens, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    )

    torch.ops.vllm.hc_head_triton(
        residual,
        fn,
        hc_scale,
        hc_base,
        actual,
        HIDDEN_SIZE,
        RMS_EPS,
        HC_EPS,
        HC_MULT,
    )
    flat = residual.flatten(-2).float()
    normalized = flat * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + RMS_EPS
    )
    gates = torch.sigmoid(
        torch.nn.functional.linear(normalized.to(torch.bfloat16).float(), fn)
        * hc_scale
        + hc_base
    ) + HC_EPS
    expected = (gates.unsqueeze(-1) * residual.float()).sum(dim=1).to(torch.bfloat16)

    torch.cuda.synchronize()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2.0e-2, atol=2.0e-2)


def _apply_rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    normalized = x.float() * torch.rsqrt(
        x.float().square().mean(dim=-1, keepdim=True) + RMS_EPS
    )
    return (normalized * weight.float()).to(torch.bfloat16)


@pytest.mark.parametrize("num_tokens", [1, 8192])
def test_sm121_sparkinfer_post_pre_matches_torch(num_tokens: int) -> None:
    sparkinfer = pytest.importorskip("sparkinfer")
    del sparkinfer
    adapter = _load_sparkinfer_adapter()

    reason = adapter.sparkinfer_mhc_unavailable_reason()
    if reason is not None:
        pytest.skip(reason)
    x, residual, post, comb, fn, hc_scale, hc_base = _inputs(num_tokens)
    norm_weight = torch.rand(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16) + 0.5

    actual = adapter.sparkinfer_mhc_post_pre(
        x,
        residual,
        post,
        comb,
        fn,
        hc_scale,
        hc_base,
        rms_eps=RMS_EPS,
        hc_eps=HC_EPS,
        sinkhorn_iters=SINKHORN_ITERS,
        norm_weight=norm_weight,
        norm_eps=RMS_EPS,
    )
    expected_residual = mhc_post_torch(x, residual, post, comb)
    expected_post, expected_comb, expected_y = mhc_pre_torch(
        expected_residual,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_EPS,
        HC_EPS,
        HC_POST_ALPHA,
        SINKHORN_ITERS,
        1,
    )
    expected = (
        expected_residual,
        expected_post,
        expected_comb,
        _apply_rms_norm(expected_y, norm_weight),
    )

    torch.cuda.synchronize()
    for result, reference in zip(actual, expected, strict=True):
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, reference, rtol=2.0e-2, atol=2.0e-2)


@pytest.mark.parametrize("num_tokens", [1, 8])
def test_sm121_sparkinfer_first_layer_broadcast_matches_torch(
    num_tokens: int,
) -> None:
    sparkinfer = pytest.importorskip("sparkinfer")
    del sparkinfer
    adapter = _load_sparkinfer_adapter()

    reason = adapter.sparkinfer_mhc_unavailable_reason()
    if reason is not None:
        pytest.skip(reason)
    _, _, _, _, full_fn, hc_scale, hc_base = _inputs(num_tokens)
    generator = torch.Generator(device="cuda").manual_seed(20260802 + num_tokens)
    x = torch.randn(
        num_tokens,
        HIDDEN_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    fn_broadcast = full_fn.view(HC_MULT3, HC_MULT, HIDDEN_SIZE).sum(dim=1)
    norm_weight = torch.rand(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16) + 0.5

    actual = adapter.sparkinfer_mhc_pre_broadcast(
        x,
        fn_broadcast,
        hc_scale,
        hc_base,
        rms_eps=RMS_EPS,
        hc_eps=HC_EPS,
        sinkhorn_iters=SINKHORN_ITERS,
        norm_weight=norm_weight,
        norm_eps=RMS_EPS,
    )
    expected_residual = x[:, None, :].expand(-1, HC_MULT, -1).contiguous()
    expected_post, expected_comb, expected_y = mhc_pre_torch(
        expected_residual,
        full_fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_EPS,
        HC_EPS,
        HC_POST_ALPHA,
        SINKHORN_ITERS,
        1,
    )
    expected = (
        expected_residual,
        expected_post,
        expected_comb,
        _apply_rms_norm(expected_y, norm_weight),
    )

    torch.cuda.synchronize()
    for result, reference in zip(actual, expected, strict=True):
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, reference, rtol=2.0e-2, atol=2.0e-2)
