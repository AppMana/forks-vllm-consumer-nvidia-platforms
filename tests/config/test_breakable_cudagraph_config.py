# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.compilation.backends import wrap_with_cudagraph_if_needed
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.config.vllm import (
    _configure_breakable_cudagraph,
    _should_auto_enable_breakable_cudagraph,
)


def _model_config(architecture: str):
    return SimpleNamespace(architectures=[architecture])


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
    assert not _should_auto_enable_breakable_cudagraph(_model_config(architecture))


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
    assert _should_auto_enable_breakable_cudagraph(_model_config(architecture))


def test_deepseek_v4_compile_mode_defaults_to_standard_piecewise(monkeypatch):
    monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)
    compilation_config = SimpleNamespace(mode=CompilationMode.VLLM_COMPILE)

    auto_enabled, breakable_enabled = _configure_breakable_cudagraph(
        _model_config("DeepseekV4ForCausalLM"),
        compilation_config,
    )

    assert not auto_enabled
    assert not breakable_enabled
    assert compilation_config.mode == CompilationMode.VLLM_COMPILE


def test_deepseek_v4_explicit_breakable_opt_in_is_preserved(monkeypatch):
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    compilation_config = SimpleNamespace(mode=CompilationMode.VLLM_COMPILE)

    auto_enabled, breakable_enabled = _configure_breakable_cudagraph(
        _model_config("DeepseekV4ForCausalLM"),
        compilation_config,
    )

    assert not auto_enabled
    assert breakable_enabled
    assert compilation_config.mode == CompilationMode.VLLM_COMPILE


def test_deepseek_v4_explicit_breakable_opt_out_preserves_compile(monkeypatch):
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")
    compilation_config = SimpleNamespace(mode=CompilationMode.VLLM_COMPILE)

    auto_enabled, breakable_enabled = _configure_breakable_cudagraph(
        _model_config("DeepseekV4ForCausalLM"),
        compilation_config,
    )

    assert not auto_enabled
    assert not breakable_enabled
    assert compilation_config.mode == CompilationMode.VLLM_COMPILE


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
