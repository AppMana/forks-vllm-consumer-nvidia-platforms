# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.nvidia_sm12x import mhc as adapter


def _install_fake_sparkinfer(monkeypatch, mhc) -> None:
    package = types.ModuleType("sparkinfer")
    norm = types.ModuleType("sparkinfer.norm")
    norm.mhc = mhc
    package.norm = norm
    monkeypatch.setitem(sys.modules, "sparkinfer", package)
    monkeypatch.setitem(sys.modules, "sparkinfer.norm", norm)


def test_pre_broadcast_preserves_vllm_mix_shapes_and_fuses_norm(monkeypatch):
    calls = []

    def run_pre(x, fn, scale, base, **kwargs):
        calls.append((x, fn, scale, base, kwargs))
        tokens, hidden = x.shape
        return (
            x[:, None, :].expand(-1, 4, -1).contiguous(),
            torch.zeros(tokens, 4),
            torch.zeros(tokens, 4, 4),
            x,
        )

    _install_fake_sparkinfer(
        monkeypatch,
        SimpleNamespace(run_pre=run_pre),
    )
    x = torch.zeros(3, 8)
    fn = torch.zeros(24, 8)
    scale = torch.zeros(3)
    base = torch.zeros(24)
    norm_weight = torch.ones(8)

    residual, post, comb, y = adapter.sparkinfer_mhc_pre_broadcast(
        x,
        fn,
        scale,
        base,
        rms_eps=1.0e-6,
        hc_eps=1.0e-6,
        sinkhorn_iters=20,
        norm_weight=norm_weight,
        norm_eps=1.0e-6,
    )

    assert residual.shape == (3, 4, 8)
    assert post.shape == (3, 4, 1)
    assert comb.shape == (3, 4, 4)
    assert y.shape == (3, 8)
    assert calls[0][4]["norm_weight"] is norm_weight
    assert "binding" not in calls[0][4]


def test_post_pre_preserves_vllm_mix_shapes_and_fuses_norm(monkeypatch):
    calls = []

    def run_post_pre(x, residual, post, comb, fn, scale, base, **kwargs):
        calls.append(kwargs)
        tokens, _, hidden = residual.shape
        return (
            residual,
            torch.zeros(tokens, 4),
            torch.zeros(tokens, 4, 4),
            torch.zeros(tokens, hidden),
        )

    _install_fake_sparkinfer(
        monkeypatch,
        SimpleNamespace(run_post_pre=run_post_pre),
    )
    x = torch.zeros(2, 8)
    residual = torch.zeros(2, 4, 8)
    norm_weight = torch.ones(8)
    outputs = adapter.sparkinfer_mhc_post_pre(
        x,
        residual,
        torch.zeros(2, 4, 1),
        torch.zeros(2, 4, 4),
        torch.zeros(24, 32),
        torch.zeros(3),
        torch.zeros(24),
        rms_eps=1.0e-6,
        hc_eps=1.0e-6,
        sinkhorn_iters=20,
        norm_weight=norm_weight,
        norm_eps=1.0e-6,
    )

    assert [tuple(output.shape) for output in outputs] == [
        (2, 4, 8),
        (2, 4, 1),
        (2, 4, 4),
        (2, 8),
    ]
    assert calls[0]["norm_weight"] is norm_weight
    assert "binding" not in calls[0]


def test_contract_validation_fails_closed(monkeypatch):
    monkeypatch.setattr(
        adapter,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            is_device_capability_family=lambda family: family == 120,
        ),
    )
    _install_fake_sparkinfer(
        monkeypatch,
        SimpleNamespace(is_supported=lambda: True),
    )

    adapter.validate_sparkinfer_mhc_contract(
        hidden_size=7168,
        hc_mult=4,
        rms_eps=1.0e-6,
        hc_eps=1.0e-6,
        hc_post_alpha=2.0,
        sinkhorn_iters=20,
    )
    with pytest.raises(ValueError, match="sinkhorn_iters"):
        adapter.validate_sparkinfer_mhc_contract(
            hidden_size=7168,
            hc_mult=4,
            rms_eps=1.0e-6,
            hc_eps=1.0e-6,
            hc_post_alpha=2.0,
            sinkhorn_iters=3,
        )


def test_hf_kernel_role_overrides_diagnostic_env(monkeypatch):
    from vllm.model_executor.layers import mhc
    from vllm.transformers_utils.configs.dsv4 import kernel_config

    resolved = kernel_config.resolve_kernel_config(
        {"kernels": [kernel_config.MHC_SPARKINFER]}
    )
    monkeypatch.setattr(kernel_config, "_ACTIVE_CONFIG", resolved)
    monkeypatch.setattr(
        mhc,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: True),
    )
    monkeypatch.setenv("VLLM_MHC_CUDA_BACKEND", "triton")

    assert mhc.mhc_uses_sparkinfer()
    assert not mhc.mhc_uses_tilelang()


def test_default_hf_kernel_role_also_overrides_diagnostic_env(monkeypatch):
    from vllm.model_executor.layers import mhc
    from vllm.transformers_utils.configs.dsv4 import kernel_config

    resolved = kernel_config.resolve_kernel_config(
        {"kernels": [kernel_config.MHC_VLLM_AUTO]}
    )
    monkeypatch.setattr(kernel_config, "_ACTIVE_CONFIG", resolved)
    monkeypatch.setenv("VLLM_MHC_CUDA_BACKEND", "sparkinfer")

    assert mhc._selected_mhc_cuda_backend() == "auto"
    assert not mhc.mhc_uses_sparkinfer()


def test_unlisted_default_role_keeps_diagnostic_env_compatibility(monkeypatch):
    from vllm.model_executor.layers import mhc
    from vllm.transformers_utils.configs.dsv4 import kernel_config

    resolved = kernel_config.resolve_kernel_config({"kernels": []})
    monkeypatch.setattr(kernel_config, "_ACTIVE_CONFIG", resolved)
    monkeypatch.setenv("VLLM_MHC_CUDA_BACKEND", "sparkinfer")

    assert mhc._selected_mhc_cuda_backend() == "sparkinfer"


def test_compiled_mhc_dispatch_uses_refreshed_constants(monkeypatch):
    from vllm.model_executor.layers import mhc

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(mhc, "_MHC_SPARKINFER", True)
    monkeypatch.setattr(mhc, "_MHC_TORCH_FALLBACK", True)
    monkeypatch.setattr(mhc, "_MHC_PRE_TRITON", True)
    monkeypatch.setattr(mhc, "_MHC_POST_TRITON", False)
    monkeypatch.setattr(mhc, "_MHC_HEAD_TRITON", True)
    monkeypatch.setattr(
        mhc,
        "_selected_mhc_cuda_backend",
        lambda: pytest.fail("compiled dispatch re-entered Python configuration"),
    )
    monkeypatch.setattr(
        mhc,
        "_should_use_mhc_torch_fallback",
        lambda: pytest.fail("compiled dispatch re-entered platform detection"),
    )

    assert mhc.mhc_uses_sparkinfer()
    assert not mhc.mhc_uses_tilelang()
    assert mhc._use_mhc_torch_fallback()
    assert mhc._use_mhc_pre_triton()
    assert not mhc._use_mhc_post_triton()
    assert mhc._use_mhc_head_triton()


def test_refresh_mhc_dispatch_runs_after_kernel_config_activation(monkeypatch):
    from vllm.model_executor.layers import mhc

    monkeypatch.setattr(mhc, "_selected_mhc_cuda_backend", lambda: "sparkinfer")
    monkeypatch.setattr(mhc, "_should_use_mhc_torch_fallback", lambda: True)
    monkeypatch.setattr(mhc.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    mhc.refresh_mhc_backend_selection()

    assert mhc._MHC_SPARKINFER
    assert mhc._MHC_TORCH_FALLBACK
    assert mhc._MHC_PRE_TRITON
    assert mhc._MHC_POST_TRITON
    assert mhc._MHC_HEAD_TRITON


def test_mhc_torch_fallback_does_not_synchronize_by_default(monkeypatch):
    from vllm.model_executor.layers import mhc

    monkeypatch.delenv("VLLM_MHC_TORCH_FALLBACK_SYNCHRONIZE", raising=False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda: pytest.fail("default MHC fallback synchronized the CUDA stream"),
    )

    assert not mhc._mhc_torch_fallback_synchronize()
    mhc._synchronize_mhc_torch_fallback()


def test_mhc_torch_fallback_synchronizes_when_explicitly_enabled(monkeypatch):
    from vllm.model_executor.layers import mhc

    calls = []
    stream = SimpleNamespace(synchronize=lambda: calls.append("stream"))
    monkeypatch.setenv("VLLM_MHC_TORCH_FALLBACK_SYNCHRONIZE", "1")
    monkeypatch.delenv("VLLM_MHC_TORCH_FALLBACK_SYNC_MODE", raising=False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)

    assert mhc._mhc_torch_fallback_synchronize()
    mhc._synchronize_mhc_torch_fallback()

    assert calls == ["stream"]


def test_uncompiled_raw_cuda_graph_capture_fails_closed(monkeypatch):
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="compiled functional custom-op"):
        adapter._fail_on_uncompiled_cuda_graph(SimpleNamespace(is_cuda=True))


def test_sparkinfer_selection_keeps_hc_head_on_triton(monkeypatch):
    from vllm.model_executor.layers import mhc

    calls = []
    monkeypatch.setattr(mhc, "_use_mhc_torch_fallback", lambda: True)
    monkeypatch.setattr(mhc, "_use_mhc_head_triton", lambda: True)

    def fake_head(**kwargs):
        calls.append(kwargs)
        kwargs["out"].zero_()

    monkeypatch.setattr(mhc, "_hc_head_fused_kernel", fake_head)
    result = mhc.HCHeadOp.forward_cuda(
        None,
        torch.zeros(2, 4, 8, dtype=torch.bfloat16),
        torch.zeros(4, 32),
        torch.zeros(1),
        torch.zeros(4),
        1.0e-6,
        1.0e-6,
    )

    assert result.shape == (2, 8)
    assert len(calls) == 1


def test_mtp_has_no_direct_tilelang_mhc_calls():
    import inspect

    from vllm.models.deepseek_v4.nvidia import mtp

    source = inspect.getsource(mtp)
    assert "mhc_post_tilelang" not in source
    assert "hc_head_fused_kernel_tilelang" not in source
    assert "MHCPostOp" in source
    assert "HCHeadOp" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compiled_post_pre_is_cuda_graph_replayable():
    pytest.importorskip("sparkinfer")
    if not adapter.current_platform.is_device_capability_family(120):
        pytest.skip("requires SM120/SM121")

    tokens, hidden = 2, 4096
    device = torch.device("cuda")
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device)
    residual = torch.randn(tokens, 4, hidden, dtype=torch.bfloat16, device=device)
    post = torch.randn(tokens, 4, 1, dtype=torch.float32, device=device)
    comb = torch.randn(tokens, 4, 4, dtype=torch.float32, device=device)
    fn = torch.randn(24, 4 * hidden, dtype=torch.float32, device=device) / 64
    scale = torch.ones(3, dtype=torch.float32, device=device)
    base = torch.zeros(24, dtype=torch.float32, device=device)
    norm_weight = torch.ones(hidden, dtype=torch.bfloat16, device=device)

    def run():
        return adapter.sparkinfer_mhc_post_pre(
            x,
            residual,
            post,
            comb,
            fn,
            scale,
            base,
            rms_eps=1.0e-6,
            hc_eps=1.0e-6,
            sinkhorn_iters=20,
            norm_weight=norm_weight,
            norm_eps=1.0e-6,
        )

    compiled = torch.compile(run, backend="eager", fullgraph=True)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = compiled()
    pointers = tuple(output.data_ptr() for output in outputs)
    graph.replay()
    torch.cuda.synchronize()

    assert tuple(output.data_ptr() for output in outputs) == pointers
    assert all(torch.isfinite(output).all() for output in outputs)

    from vllm.model_executor.kernels.mhc.torch import mhc_post_torch

    residual_ref = mhc_post_torch(x, residual, post, comb)
    flat = residual_ref.flatten(1).float()
    mixes = torch.nn.functional.linear(flat, fn) * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + 1.0e-6
    )
    pre_ref = torch.sigmoid(mixes[:, :4] * scale[0] + base[:4]) + 1.0e-6
    post_ref = 2 * torch.sigmoid(mixes[:, 4:8] * scale[1] + base[4:8])
    comb_ref = mixes[:, 8:].view(-1, 4, 4) * scale[2] + base[8:].view(4, 4)
    comb_ref = torch.softmax(comb_ref, dim=-1) + 1.0e-6
    comb_ref = comb_ref / (comb_ref.sum(dim=-2, keepdim=True) + 1.0e-6)
    for _ in range(19):
        comb_ref = comb_ref / (comb_ref.sum(dim=-1, keepdim=True) + 1.0e-6)
        comb_ref = comb_ref / (comb_ref.sum(dim=-2, keepdim=True) + 1.0e-6)
    y_raw = (pre_ref.unsqueeze(-1) * residual_ref.float()).sum(dim=1)
    y_bf16 = y_raw.to(torch.bfloat16)
    y_ref = (
        y_bf16.float()
        * torch.rsqrt(y_bf16.float().square().mean(dim=-1, keepdim=True) + 1.0e-6)
        * norm_weight.float()
    ).to(torch.bfloat16)

    torch.testing.assert_close(outputs[0], residual_ref, rtol=0.0, atol=2e-2)
    torch.testing.assert_close(outputs[1], post_ref.unsqueeze(-1), rtol=2e-6, atol=1e-5)
    torch.testing.assert_close(outputs[2], comb_ref, rtol=2e-6, atol=1e-5)
    torch.testing.assert_close(outputs[3], y_ref, rtol=0.0, atol=2e-2)
