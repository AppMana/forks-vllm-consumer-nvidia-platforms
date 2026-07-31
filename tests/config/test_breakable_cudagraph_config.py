# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config.vllm import _should_auto_enable_breakable_cudagraph


def _model_config(architecture: str):
    return SimpleNamespace(architectures=[architecture])


@pytest.mark.parametrize(
    "architecture",
    ["DeepseekV4ForCausalLM", "DeepSeekV4MTPModel"],
)
def test_deepseek_v4_does_not_auto_enable_breakable_cudagraph(architecture):
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
