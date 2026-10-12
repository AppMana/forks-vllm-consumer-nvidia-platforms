# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serialized scale loading must follow the same fused/TP shards as weights."""

import pytest
import torch

import vllm.model_executor.layers.linear as linear
import vllm.model_executor.parameter as parameter
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.layers.quantization.qwen_int8 import (
    QwenInt8Config,
    QwenVisionInt8LinearMethod,
)
from vllm.model_executor.models.utils import WeightsMapper


@pytest.mark.parametrize("rank", [0, 1])
def test_fused_qkvz_channel_weights_and_scales_follow_tp_shards(monkeypatch, rank):
    for module in (parameter, linear):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: rank)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 2)
    prefix = "language_model.model.layers.0.linear_attn"
    config = QwenInt8Config(
        {
            f"{prefix}.in_proj_qkv": "channel_uint8",
            f"{prefix}.in_proj_z": "channel_uint8",
        }
    )
    config.packed_modules_mapping = {"in_proj_qkvz": ["in_proj_qkv", "in_proj_z"]}
    sizes = [256, 256, 768, 768]
    with torch.device("cpu"), set_current_vllm_config(VllmConfig()):
        layer = MergedColumnParallelLinear(
            16,
            sizes,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=config,
            prefix=f"{prefix}.in_proj_qkvz",
        )
    weight = torch.arange(sum(sizes) * 16).remainder(256).to(torch.uint8).view(-1, 16)
    scales = torch.arange(1, sum(sizes) + 1).to(torch.bfloat16)
    split = sum(sizes[:3])
    for name, tensor in (("weight", weight), ("weight_scale_inv", scales)):
        param = getattr(layer, name)
        param.weight_loader(param, tensor[:split], (0, 1, 2))
        param.weight_loader(param, tensor[split:], 3)
        expected = torch.cat([part.chunk(2)[rank] for part in tensor.split(sizes)])
        torch.testing.assert_close(param, expected, atol=0, rtol=0)


def test_manifest_mapping_preserves_mtp_and_does_not_double_apply_prefix():
    config = QwenInt8Config(
        {
            "model.language_model.layers.0.mlp.gate_proj": "channel_uint8",
            "mtp.fc": "channel_uint8",
        }
    )
    mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "mtp.": None,
        }
    )
    config.apply_vllm_mapper(mapper)
    expected = dict(config.quantized_modules)
    config.apply_vllm_mapper(mapper)
    assert config.quantized_modules == expected
    assert config._encoding("mtp.fc") == "channel_uint8"
    assert config._encoding("language_model.lm_head") is None


def test_partial_quantization_of_fused_layer_is_rejected():
    config = QwenInt8Config({"layer.gate_proj": "channel_uint8"})
    config.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    with pytest.raises(ValueError, match="Mixed/missing"):
        config._encoding("layer.gate_up_proj")


def test_vision_groups_crossing_tp_boundary_are_rejected_before_allocation():
    with pytest.raises(ValueError, match="mm-encoder-tp-mode data"):
        QwenVisionInt8LinearMethod().create_weights(
            torch.nn.Module(),
            2152,
            [1152],
            4304,
            1152,
            torch.bfloat16,
        )
