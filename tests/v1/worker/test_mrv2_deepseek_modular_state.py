# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.v1.worker.gpu.model_states import init_model_state
from vllm.v1.worker.gpu.model_states import default as default_state


class FakeDefaultModelState:
    def __init__(self, vllm_config, model, encoder_cache, device):
        self.vllm_config = vllm_config
        self.model = model
        self.encoder_cache = encoder_cache
        self.device = device


class FakeCustomModelState(FakeDefaultModelState):
    pass


class FakeModel(nn.Module):
    pass


class FakeCustomStateModel(nn.Module):
    @staticmethod
    def get_model_state_cls():
        return FakeCustomModelState


def _vllm_config_for_arch(arch: str):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=[arch],
            is_hybrid=False,
        )
    )


@pytest.mark.parametrize(
    "arch",
    [
        "Qwen3ForCausalLM",
        "DeepseekV4ForCausalLM",
    ],
)
def test_mrv2_qwen3_and_deepseek_v4_use_default_model_state(
    monkeypatch, arch: str
):
    monkeypatch.setattr(default_state, "DefaultModelState", FakeDefaultModelState)

    model_state = init_model_state(
        _vllm_config_for_arch(arch),
        FakeModel(),
        encoder_cache=None,
        device=torch.device("cpu"),
    )

    assert type(model_state) is FakeDefaultModelState
    assert model_state.vllm_config.model_config.architectures == [arch]


def test_mrv2_custom_model_state_still_wins(monkeypatch):
    monkeypatch.setattr(default_state, "DefaultModelState", FakeDefaultModelState)

    model_state = init_model_state(
        _vllm_config_for_arch("DeepseekV4ForCausalLM"),
        FakeCustomStateModel(),
        encoder_cache=None,
        device=torch.device("cpu"),
    )

    assert type(model_state) is FakeCustomModelState
