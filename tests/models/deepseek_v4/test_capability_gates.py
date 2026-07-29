# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capability gates for the consumer-NVIDIA DSV4 platforms (sm_86, sm_121).

These pin gate *scope*: which architectures a gate admits, and that gates
answering the same question agree. They are pure-Python and need no GPU --
the device capability is monkeypatched.
"""

import pytest

from vllm.platforms.interface import DeviceCapability

# (major, minor) -> whether Triton can lower tl.float8e4nv / tl.dot(fp8, fp8).
# sm_89 (Ada) is the floor and is the case the two `major != 8` copies used
# to get wrong.
FP8E4NV_CASES = [
    ((8, 0), False),  # A100
    ((8, 6), False),  # RTX 3090 / A5000
    ((8, 9), True),  # L40S / RTX 4090
    ((9, 0), True),  # H100
    ((10, 0), True),  # B200
    ((12, 1), True),  # GB10 DGX Spark
]


@pytest.fixture
def patch_capability(monkeypatch):
    """Force current_platform.get_device_capability() to a fixed value."""

    def _apply(major: int, minor: int):
        from vllm.platforms import current_platform

        # current_platform is an instance; get_device_capability is a
        # classmethod on its class and is_cuda a plain instance method.
        monkeypatch.setattr(
            type(current_platform),
            "get_device_capability",
            classmethod(lambda cls, device_id=0: DeviceCapability(major, minor)),
        )
        monkeypatch.setattr(type(current_platform), "is_cuda", lambda self: True)

    return _apply


@pytest.mark.parametrize("capability,expected", FP8E4NV_CASES)
def test_fp8e4nv_gate_floor_is_sm89(capability, expected, patch_capability):
    from vllm.models.deepseek_v4.common.ops.fp8e4m3_arith import (
        cuda_supports_fp8e4nv_in_triton,
    )

    patch_capability(*capability)
    assert cuda_supports_fp8e4nv_in_triton(unknown=False) is expected
    # `unknown` only decides the no-capability case, never a known one.
    assert cuda_supports_fp8e4nv_in_triton(unknown=True) is expected


@pytest.mark.parametrize("capability,expected", FP8E4NV_CASES)
def test_every_fp8e4nv_call_site_agrees(capability, expected, patch_capability):
    """The four dispatch sites must not drift apart again."""
    import importlib

    from vllm.models.deepseek_v4.common.ops import (
        cache_utils,
        fp8_einsum,
        fused_indexer_q,
    )

    # ops/__init__ re-exports a *function* named fused_inv_rope_fp8_quant,
    # which shadows the submodule of the same name -- importing it from the
    # package yields the function, whose _supports_fp8e4nv_in_triton lookup
    # raises AttributeError. Resolve the module by path so this fourth call
    # site is actually checked rather than silently erroring out.
    fused_inv_rope_fp8_quant = importlib.import_module(
        "vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant"
    )

    patch_capability(*capability)
    for module in (
        cache_utils,
        fp8_einsum,
        fused_indexer_q,
        fused_inv_rope_fp8_quant,
    ):
        assert module._supports_fp8e4nv_in_triton() is expected, module.__name__


def test_fp8e4nv_gate_without_capability(monkeypatch):
    """No readable capability: each side picks what is safe for its fallback."""
    from vllm.models.deepseek_v4.common.ops.fp8e4m3_arith import (
        cuda_supports_fp8e4nv_in_triton,
    )
    from vllm.platforms import current_platform

    monkeypatch.setattr(
        type(current_platform),
        "get_device_capability",
        classmethod(lambda cls, device_id=0: None),
    )
    assert cuda_supports_fp8e4nv_in_triton(unknown=True) is True
    assert cuda_supports_fp8e4nv_in_triton(unknown=False) is False


@pytest.mark.parametrize(
    "capability,expected",
    [
        ((8, 0), True),
        ((8, 6), True),  # the arch DeepseekV4TritonSM86Attention serves
        ((9, 0), True),
        ((10, 0), True),
        ((12, 1), True),  # GB10: IMMA is available, so int8_ds_mla is servable
        ((7, 5), False),
    ],
)
def test_dsv4_flashmla_backend_admits_ampere(capability, expected):
    """The backend must not reject the arch its own subclass runs on.

    The gate is a floor, not an enumeration: listing majors excluded sm_12x,
    where these int8 kernels run fine, and made an int8_ds_mla checkpoint
    unservable on GB10.
    """
    from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLABackend

    assert (
        DeepseekV4FlashMLABackend.supports_compute_capability(
            DeviceCapability(*capability)
        )
        is expected
    )


def test_every_dsv4_backend_is_warmed():
    """Each DSV4 sparse-MLA backend reaches the same three Triton helper
    kernels, so each must be in the warmup's backend-name set; one missing
    means that arch JIT-compiles them on the first real request."""
    from vllm.model_executor.warmup.sparse_mla_triton_warmup import (
        _DEEPSEEK_V4_SPARSE_MLA_BACKENDS,
    )
    from vllm.models.deepseek_v4.nvidia_sm12x.attention import (
        DeepseekV4SparkInferMLABackend,
    )
    from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLABackend

    for backend in (DeepseekV4FlashMLABackend, DeepseekV4SparkInferMLABackend):
        assert backend.get_name() in _DEEPSEEK_V4_SPARSE_MLA_BACKENDS


def test_mhc_cuda_backend_can_force_triton_on_sm121(monkeypatch, patch_capability):
    """The INT4/INT8 deployment can explicitly exclude TileLang mHC."""
    from vllm.model_executor.layers import mhc

    patch_capability(12, 1)
    monkeypatch.setenv("VLLM_MHC_CUDA_BACKEND", "triton")
    monkeypatch.setattr(mhc, "HAS_TILELANG_MHC", True)

    assert mhc._should_use_mhc_torch_fallback() is True


def test_mhc_cuda_backend_auto_keeps_sm121_tilelang(monkeypatch, patch_capability):
    """The new deployment override must not change the default policy."""
    from vllm.model_executor.layers import mhc

    patch_capability(12, 1)
    monkeypatch.setenv("VLLM_MHC_CUDA_BACKEND", "auto")
    monkeypatch.setattr(mhc, "HAS_TILELANG_MHC", True)

    assert mhc._should_use_mhc_torch_fallback() is False


def test_mhc_cuda_backend_rejects_unknown_value(monkeypatch, patch_capability):
    """A misspelled backend must fail at startup instead of using TileLang."""
    from vllm.model_executor.layers import mhc

    patch_capability(12, 1)
    monkeypatch.setenv("VLLM_MHC_CUDA_BACKEND", "trtion")

    with pytest.raises(ValueError, match="Invalid VLLM_MHC_CUDA_BACKEND"):
        mhc._should_use_mhc_torch_fallback()


def test_sparkinfer_backend_is_selectable_by_flag():
    """The sm_12x backend needs a registry entry or --attention-backend
    cannot name it."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    assert AttentionBackendEnum.SPARKINFER_MLA_SPARSE_DSV4.value.endswith(
        "DeepseekV4SparkInferMLABackend"
    )


@pytest.mark.parametrize(
    "capability,cache_dtype,expected_qualname",
    [
        # int8_ds_mla selects the class that reads int8 pages, at any arch that
        # has IMMA -- including sm_12x, where arch-first dispatch used to send
        # it to sparkinfer and fail.
        ((8, 6), "int8_ds_mla", "DeepseekV4TritonSM86Attention"),
        ((12, 1), "int8_ds_mla", "DeepseekV4TritonSM86Attention"),
        # fp8_ds_mla keeps the per-arch default.
        ((12, 1), "fp8_ds_mla", "DeepseekV4SparkInferSM12xAttention"),
        ((8, 6), "fp8_ds_mla", "DeepseekV4TritonSM86Attention"),
    ],
)
def test_attn_cls_dispatch_follows_cache_dtype(
    patch_capability, capability, cache_dtype, expected_qualname
):
    """The attention class is chosen by what the checkpoint asks for, with
    capability as a floor -- not by arch alone.

    sparkinfer's kernels read the 584-byte fp8 page byte-for-byte, so the KV
    cache dtype, not the device, decides which class can serve. Selecting on
    arch first made int8_ds_mla unreachable on sm_12x.
    """
    from vllm.models.deepseek_v4.nvidia.model import _select_dsv4_attn_cls

    patch_capability(*capability)

    class _StubVllmConfig:
        attention_config = type("_A", (), {"backend": None})()
        cache_config = type("_C", (), {"cache_dtype": cache_dtype})()

    assert _select_dsv4_attn_cls(_StubVllmConfig()).__qualname__ == expected_qualname
