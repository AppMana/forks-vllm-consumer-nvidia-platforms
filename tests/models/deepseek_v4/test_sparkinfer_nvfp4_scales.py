# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts.sparkinfer_moe import (
    _modelopt_activation_gscales,
)
from vllm.model_executor.layers.fused_moe.oracle import nvfp4


def test_modelopt_activation_scales_follow_sparkinfer_reciprocal_contract():
    # These stand in for ModelOpt checkpoint `*.input_scale` fields. FC1 has
    # separate gate/up values; the fused activation quantizer uses their max.
    w13_input_scale = torch.tensor([[0.5, 0.25], [0.125, 0.25]])
    w2_input_scale = torch.tensor([0.2, 0.1])
    w13_weight_scale = torch.tensor([0.125, 0.25])
    w2_weight_scale = torch.tensor([0.5, 0.75])

    a1_gscale, a2_gscale = _modelopt_activation_gscales(
        w13_input_scale,
        w2_input_scale,
    )

    torch.testing.assert_close(a1_gscale, torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(a2_gscale, torch.tensor([5.0, 10.0]))
    # SparkInfer's modelopt_nvfp4 preparation computes weight/a_reciprocal.
    torch.testing.assert_close(
        w13_weight_scale / a1_gscale,
        w13_weight_scale * w13_input_scale.max(dim=-1).values,
    )
    torch.testing.assert_close(
        w2_weight_scale / a2_gscale,
        w2_weight_scale * w2_input_scale,
    )
    assert a1_gscale.dtype == torch.float32
    assert a2_gscale.dtype == torch.float32
    assert a1_gscale.is_contiguous()
    assert a2_gscale.is_contiguous()


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
