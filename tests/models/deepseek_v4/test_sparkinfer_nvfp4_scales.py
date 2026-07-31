# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts.sparkinfer_moe import (
    _modelopt_activation_gscales,
)
from vllm.model_executor.layers.fused_moe.oracle import nvfp4


def test_modelopt_activation_scales_preserve_released_expert_axis():
    num_experts = 256
    w1_input_scale = torch.linspace(0.0023832775, 0.0029529390, num_experts)
    w13_input_scale = torch.stack((w1_input_scale, w1_input_scale), dim=-1)
    w2_input_scale = torch.linspace(0.0036039806, 0.0083240330, num_experts)
    w13_weight_scale = torch.linspace(0.00012207031, 0.00024414062, num_experts)
    w2_weight_scale = torch.linspace(0.00018310547, 0.00030517578, num_experts)

    a1_gscale, a2_gscale = _modelopt_activation_gscales(
        w13_input_scale,
        w2_input_scale,
    )

    assert a1_gscale.shape == (num_experts,)
    assert a2_gscale.shape == (num_experts,)
    torch.testing.assert_close(a1_gscale, torch.reciprocal(w1_input_scale))
    torch.testing.assert_close(a2_gscale, torch.reciprocal(w2_input_scale))
    torch.testing.assert_close(
        w13_weight_scale / a1_gscale,
        w13_weight_scale * w1_input_scale,
    )
    torch.testing.assert_close(
        w2_weight_scale / a2_gscale,
        w2_weight_scale * w2_input_scale,
    )
    assert a1_gscale.dtype == torch.float32
    assert a2_gscale.dtype == torch.float32
    assert a1_gscale.is_contiguous()
    assert a2_gscale.is_contiguous()

    collapsed_alpha = w13_weight_scale * w1_input_scale.max()
    per_expert_alpha = w13_weight_scale / a1_gscale
    relative_logit_error = (collapsed_alpha[0] / per_expert_alpha[0]) - 1
    assert relative_logit_error > 0.23


def test_modelopt_fc1_rejects_distinct_gate_and_up_input_scales():
    w13_input_scale = torch.ones(256, 2)
    w13_input_scale[17] = torch.tensor([0.0023832775, 0.0029529390])

    with pytest.raises(ValueError, match="gate and up input scales must match"):
        _modelopt_activation_gscales(w13_input_scale, torch.ones(256))


@pytest.mark.parametrize("bad_scale", [0.0, -1.0, float("inf"), float("nan")])
def test_modelopt_activation_scales_reject_invalid_values(bad_scale):
    with pytest.raises(ValueError, match="finite and positive"):
        _modelopt_activation_gscales(
            torch.tensor([[bad_scale, bad_scale]]),
            torch.ones(1),
        )


def test_sparkinfer_layout_reorders_gate_up_before_swizzling(monkeypatch):
    w13 = torch.arange(8, dtype=torch.uint8).reshape(1, 4, 2)
    w13_scale = torch.arange(4, dtype=torch.uint8).reshape(1, 4, 1)
    checkpoint_w13 = w13.clone()
    checkpoint_w13_scale = w13_scale.clone()
    w2 = torch.arange(4, dtype=torch.uint8).reshape(1, 2, 2)
    w2_scale = torch.arange(2, dtype=torch.uint8).reshape(1, 2, 1)
    w13_scale_2 = torch.ones(1)
    w2_scale_2 = torch.ones(1)
    a13_scale = torch.ones(1, 2)
    a2_scale = torch.ones(1)
    monkeypatch.setattr(nvfp4, "swizzle_blockscale", lambda scale: scale + 10)

    converted = nvfp4.convert_to_nvfp4_moe_kernel_format(
        nvfp4_backend=nvfp4.NvFp4MoeBackend.SPARKINFER,
        layer=None,
        w13=w13,
        w13_scale=w13_scale,
        w13_scale_2=w13_scale_2,
        a13_scale=a13_scale,
        w2=w2,
        w2_scale=w2_scale,
        w2_scale_2=w2_scale_2,
        a2_scale=a2_scale,
        is_act_and_mul=True,
    )

    expected_w13 = torch.cat(
        [checkpoint_w13[:, 2:], checkpoint_w13[:, :2]],
        dim=1,
    )
    expected_w13_scale = torch.cat(
        [checkpoint_w13_scale[:, 2:], checkpoint_w13_scale[:, :2]],
        dim=1,
    )
    torch.testing.assert_close(converted[0], expected_w13)
    torch.testing.assert_close(converted[1], expected_w13_scale + 10)
    torch.testing.assert_close(converted[4], w2)
    torch.testing.assert_close(converted[5], w2_scale + 10)
    assert converted[2] is w13_scale_2
    assert converted[3] is a13_scale
    assert converted[6] is w2_scale_2
    assert converted[7] is a2_scale
