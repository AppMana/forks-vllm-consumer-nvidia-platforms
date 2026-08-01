# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect

import pytest

from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.model_executor.kernels.mhc.tilelang import mhc_pre_broadcast_tilelang
from vllm.models.deepseek_v4.attention import (
    DeepseekV4Indexer,
    use_compilation_safe_attn_gemm_overlap,
)
from vllm.models.deepseek_v4.compressor import DeepseekCompressor
from vllm.models.deepseek_v4.nvidia.dspark import DSparkDeepseekV4Model
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4Model
from vllm.models.deepseek_v4.nvidia.mtp import DeepSeekV4MTP
from vllm.v1.attention.backends.mla.indexer import _prepare_uniform_decode_kernel


def test_deepseek_v4_models_register_for_torch_compile():
    assert TorchCompileWithNoGuardsWrapper in DeepseekV4Model.__bases__
    assert TorchCompileWithNoGuardsWrapper in DeepSeekV4MTP.__bases__
    assert TorchCompileWithNoGuardsWrapper in DSparkDeepseekV4Model.__bases__


def test_compressor_forward_has_no_runtime_platform_dispatch():
    source = inspect.getsource(DeepseekCompressor.forward)
    assert "current_platform" not in source
    assert "has_cutedsl" not in source


def test_indexer_forward_has_no_runtime_backend_resolution():
    source = inspect.getsource(DeepseekV4Indexer.forward)
    assert "current_platform" not in source
    assert "has_cutedsl" not in source
    assert "indexer_imma_enabled" not in source


def test_uniform_decode_length_does_not_create_jit_variants():
    assert "max_decode_len" in _prepare_uniform_decode_kernel.do_not_specialize


def test_first_layer_tilelang_mhc_is_an_opaque_custom_op():
    source = inspect.getsource(mhc_pre_broadcast_tilelang)
    assert "torch.ops.vllm.mhc_pre_broadcast_tilelang" in source
    assert "tf32_hc_prenorm_gemm" not in source


@pytest.mark.parametrize(("is_compiling", "expected"), [(False, True), (True, False)])
def test_attn_input_gemm_overlap_is_disabled_while_compiling(
    monkeypatch, is_compiling: bool, expected: bool
):
    monkeypatch.setattr("torch.compiler.is_compiling", lambda: is_compiling)
    monkeypatch.setattr(
        "vllm.models.deepseek_v4.attention.envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD",
        1024,
    )
    assert use_compilation_safe_attn_gemm_overlap(6) is expected
