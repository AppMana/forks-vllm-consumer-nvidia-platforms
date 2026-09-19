# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The vision warmup runs one synthetic image through the tower on its rank only."""

from types import SimpleNamespace

import torch

from vllm.model_executor.warmup.kernel_warmup import _deepseek_v4_vision_warmup


class FakeTower(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        proj = SimpleNamespace(
            weight=torch.zeros(4, 588, dtype=torch.bfloat16), input_size=588
        )
        self.patch_embed = SimpleNamespace(proj=proj)
        self.calls: list[tuple[torch.Size, int, int]] = []

    def forward(self, patches: torch.Tensor, n_vit_h: int, n_vit_w: int):
        self.calls.append((patches.shape, n_vit_h, n_vit_w))
        return torch.zeros(patches.shape[0], 4, dtype=patches.dtype)


class FakeAligner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.downsample_ratio = 2
        self.calls: list[tuple[int, int]] = []

    def forward(self, x: torch.Tensor, n_vit_h: int, n_vit_w: int):
        self.calls.append((n_vit_h, n_vit_w))
        return x


def _worker(model, architectures=("DeepseekV4ForConditionalGeneration",)):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=list(architectures)),
            dtype=torch.bfloat16,
        ),
        model_runner=object(),
        get_model=lambda: model,
    )


def test_vision_warmup_runs_one_image_through_tower_and_aligner(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda device=None: None)
    model = SimpleNamespace(
        vision=FakeTower(), aligner=FakeAligner(), compute_dtype=torch.bfloat16
    )
    _deepseek_v4_vision_warmup(_worker(model))
    assert model.vision.calls == [(torch.Size([16, 588]), 4, 4)]
    assert model.aligner.calls == [(4, 4)]


def test_vision_warmup_skips_ranks_without_the_tower_and_other_models(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda device=None: None)
    tower = FakeTower()
    _deepseek_v4_vision_warmup(_worker(SimpleNamespace(vision=None, aligner=None)))
    _deepseek_v4_vision_warmup(
        _worker(
            SimpleNamespace(vision=tower, aligner=FakeAligner()), ("LlamaForCausalLM",)
        )
    )
    assert tower.calls == []
