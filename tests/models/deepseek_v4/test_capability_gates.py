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
    from vllm.models.deepseek_v4.common.ops import (
        cache_utils,
        fp8_einsum,
        fused_indexer_q,
        fused_inv_rope_fp8_quant,
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
        ((7, 5), False),
    ],
)
def test_dsv4_flashmla_backend_admits_ampere(capability, expected):
    """The backend must not reject the arch its own subclass runs on."""
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


def test_sparkinfer_backend_is_selectable_by_flag():
    """The sm_12x backend needs a registry entry or --attention-backend
    cannot name it."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    assert AttentionBackendEnum.SPARKINFER_MLA_SPARSE_DSV4.value.endswith(
        "DeepseekV4SparkInferMLABackend"
    )
