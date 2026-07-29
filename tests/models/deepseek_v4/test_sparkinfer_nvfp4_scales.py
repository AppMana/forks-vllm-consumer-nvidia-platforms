# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts.sparkinfer_moe import (
    _modelopt_activation_gscales,
)


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
