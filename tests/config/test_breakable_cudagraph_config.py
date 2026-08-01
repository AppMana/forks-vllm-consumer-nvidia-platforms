# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config.compilation import CompilationMode
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
def test_deepseek_v4_auto_enables_breakable_cudagraph(architecture):
    assert _should_auto_enable_breakable_cudagraph(_model_config(architecture))


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


def test_deepseek_v4_compile_mode_uses_breakable_cudagraph(monkeypatch):
    monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)
    compilation_config = SimpleNamespace(mode=CompilationMode.VLLM_COMPILE)

    auto_enabled, breakable_enabled = _configure_breakable_cudagraph(
        _model_config("DeepseekV4ForCausalLM"),
        compilation_config,
    )

    assert auto_enabled
    assert breakable_enabled
    assert compilation_config.mode == CompilationMode.NONE


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
