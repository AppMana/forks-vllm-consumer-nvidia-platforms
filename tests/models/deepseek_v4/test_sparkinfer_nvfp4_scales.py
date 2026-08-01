# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import sys
import types
from pathlib import Path

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.sparkinfer_moe import (
    SparkInferExperts,
    _modelopt_activation_gscales,
)
from vllm.model_executor.layers.fused_moe.oracle import nvfp4
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    merge_nvfp4_gate_up_input_scales,
    require_uniform_nvfp4_expert_scale,
)

_SCALE_FIXTURE = Path(__file__).parent / "fixtures" / "nvfp4_layer0_input_scales.json"
_NVFP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def _load_released_scales() -> tuple[torch.Tensor, torch.Tensor]:
    payload = json.loads(_SCALE_FIXTURE.read_text())
    assert payload["revision"] == "e3cd60e7de98e9867116860d522499a728de1cf9"
    assert payload["num_experts"] == 256
    w1 = torch.tensor(payload["w1_input_scale"], dtype=torch.float32)
    w2 = torch.tensor(payload["w2_input_scale"], dtype=torch.float32)
    w3 = torch.tensor(payload["w3_input_scale"], dtype=torch.float32)
    torch.testing.assert_close(w1, w3, rtol=0, atol=0)
    return torch.stack((w1, w3), dim=-1), w2


def _torch_nvfp4_quant_dequant(
    x: torch.Tensor, checkpoint_scale: torch.Tensor
) -> torch.Tensor:
    """Dequantize Torch NVFP4 blocks using ModelOpt's per-expert scale."""
    x_f32 = x.float().reshape(x.shape[0], -1, 16)
    global_scale = checkpoint_scale.reciprocal().reshape(-1, 1)
    block_scale = (x_f32.abs().amax(dim=-1) * global_scale / 6.0).clamp(
        max=torch.finfo(torch.float8_e4m3fn).max
    )
    block_scale = block_scale.to(torch.float8_e4m3fn).float()
    dequant_scale = block_scale / global_scale
    normalized = x_f32 / dequant_scale.unsqueeze(-1).clamp(min=1e-30)
    sign = normalized.sign()
    magnitude = normalized.abs().unsqueeze(-1)
    fp4 = _NVFP4_VALUES[(magnitude - _NVFP4_VALUES).abs().argmin(dim=-1)] * sign
    return (fp4 * dequant_scale.unsqueeze(-1)).reshape_as(x).to(torch.bfloat16)


def test_modelopt_activation_scales_preserve_released_expert_axis():
    w13_input_scale, w2_input_scale = _load_released_scales()

    a1_gscale, a2_gscale = _modelopt_activation_gscales(
        w13_input_scale,
        w2_input_scale,
    )

    assert a1_gscale.shape == (256,)
    assert a2_gscale.shape == (256,)
    torch.testing.assert_close(a1_gscale, torch.reciprocal(w13_input_scale[:, 0]))
    torch.testing.assert_close(a2_gscale, torch.reciprocal(w2_input_scale))
    assert a1_gscale.dtype == torch.float32
    assert a2_gscale.dtype == torch.float32
    assert a1_gscale.is_contiguous()
    assert a2_gscale.is_contiguous()

    ramp = torch.linspace(-1.0, 1.0, 32).repeat(256, 1)
    calibrated_amax = w2_input_scale[:, None] * (448.0 * 6.0)
    hidden_states = (ramp * calibrated_amax * 0.91).to(torch.bfloat16)
    reference = _torch_nvfp4_quant_dequant(hidden_states, w2_input_scale)
    collapsed = _torch_nvfp4_quant_dequant(
        hidden_states,
        w2_input_scale.max().expand_as(w2_input_scale),
    )
    mismatch = torch.nonzero(reference != collapsed)
    assert mismatch.numel() > 0
    assert tuple(mismatch[0].tolist()) == (0, 0)


def test_modelopt_fc1_rejects_distinct_gate_and_up_input_scales():
    w13_input_scale = torch.ones(256, 2)
    w13_input_scale[17] = torch.tensor([0.0023832775, 0.0029529390])

    with pytest.raises(ValueError, match="gate and up input scales must match"):
        _modelopt_activation_gscales(w13_input_scale, torch.ones(256))


def test_shared_scale_backends_reject_released_expert_axis():
    w13_input_scale, w2_input_scale = _load_released_scales()

    torch.testing.assert_close(
        merge_nvfp4_gate_up_input_scales(w13_input_scale),
        w13_input_scale[:, 0],
    )
    with pytest.raises(ValueError, match="differs across experts"):
        require_uniform_nvfp4_expert_scale(
            w2_input_scale,
            num_local_experts=256,
            name="a2_scale",
        )


def test_sparkinfer_marks_checkpoint_activation_scales_static(monkeypatch):
    captured = {}
    fused_moe = types.ModuleType("sparkinfer.moe.fused_moe")

    def bind(plan, **kwargs):
        captured.update(kwargs)
        return object()

    fused_moe.bind = bind
    fused_moe.run = lambda *, binding: captured["output"]
    moe_module = types.ModuleType("sparkinfer.moe")
    moe_module.fused_moe = fused_moe
    package = types.ModuleType("sparkinfer")
    package.moe = moe_module
    monkeypatch.setitem(sys.modules, "sparkinfer", package)
    monkeypatch.setitem(sys.modules, "sparkinfer.moe", moe_module)

    experts = object.__new__(SparkInferExperts)
    experts._experts = object()
    experts._weight_plan = object()
    experts._get_plan_scratch = lambda num_tokens, device: (object(), torch.empty(1))
    hidden_states = torch.zeros(2, 16, dtype=torch.bfloat16)
    output = torch.empty_like(hidden_states)
    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=torch.empty(0),
        w2=torch.empty(0),
        topk_weights=torch.ones(2, 1),
        topk_ids=torch.zeros(2, 1, dtype=torch.int64),
        activation=MoEActivation.SILU,
        global_num_experts=256,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=None,
        workspace2=None,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    assert captured["input_scales_static"] is True


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
