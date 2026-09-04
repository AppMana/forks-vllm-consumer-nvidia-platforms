# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace

import pytest

from vllm.compilation.backends import wrap_with_cudagraph_if_needed
from vllm.config.compilation import CUDAGraphMode
from vllm.config.vllm import VllmConfig


def _config(architecture: str):
    return SimpleNamespace(model_config=SimpleNamespace(architectures=[architecture]))


@pytest.mark.parametrize(
    "architecture",
    [
        "DeepseekV4ForCausalLM",
        "DeepSeekV4MTPModel",
        "DSparkDraftModel",
    ],
)
def test_deepseek_v4_does_not_auto_enable_breakable_cudagraph(architecture):
    # DSV4 attention is an opaque splitting op; standard piecewise CUDA
    # graphs cover it, and lazy breakable capture stalls decode on a full
    # unified-memory pool.
    assert not VllmConfig._uses_breakable_cudagraph_by_default(_config(architecture))


@pytest.mark.parametrize(
    "architecture",
    [
        "InklingForCausalLM",
        "InklingForConditionalGeneration",
        "MiniMaxM3SparseForCausalLM",
        "MiniMaxM3SparseForConditionalGeneration",
    ],
)
def test_unsupported_compile_architecture_auto_enables_breakable_cudagraph(
    architecture,
):
    assert VllmConfig._uses_breakable_cudagraph_by_default(_config(architecture))


def test_deepseek_v4_compile_mode_defaults_to_standard_piecewise(monkeypatch):
    monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)

    breakable_enabled = VllmConfig._maybe_enable_breakable_cudagraph(
        _config("DeepseekV4ForCausalLM")
    )

    assert not breakable_enabled
    # Nothing opted the architecture in, so the env stays unset and the model
    # body keeps its standard piecewise compilation.
    assert "VLLM_USE_BREAKABLE_CUDAGRAPH" not in os.environ


def test_deepseek_v4_explicit_breakable_opt_in_is_preserved(monkeypatch):
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")

    assert VllmConfig._maybe_enable_breakable_cudagraph(
        _config("DeepseekV4ForCausalLM")
    )


def test_deepseek_v4_explicit_breakable_opt_out_preserves_compile(monkeypatch):
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")

    assert not VllmConfig._maybe_enable_breakable_cudagraph(
        _config("DeepseekV4ForCausalLM")
    )


def test_breakable_cudagraph_does_not_nest_piecewise_wrappers(monkeypatch):
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    backend = object()
    compilation_config = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        use_inductor_graph_partition=False,
    )

    wrapped = wrap_with_cudagraph_if_needed(
        backend,
        SimpleNamespace(),
        compilation_config,
        is_first_graph=True,
        is_last_graph=True,
    )

    assert wrapped is backend
