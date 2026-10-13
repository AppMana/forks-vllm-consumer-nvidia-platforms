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
    QwenInt8LinearMethod,
    QwenInt8PrefillLinearMethod,
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


@pytest.mark.parametrize(
    ("prefix", "native_prefill"),
    [
        ("language_model.model.layers.0.mlp.gate_up_proj", True),
        ("language_model.model.layers.0.mlp.down_proj", False),
        ("language_model.model.layers.0.self_attn.qkv_proj", False),
        ("language_model.model.layers.0.linear_attn.in_proj_qkvz", False),
        ("language_model.model.layers.0.linear_attn.out_proj", False),
        ("mtp.fc", False),
        ("mtp.layers.0.mlp.gate_up_proj", True),
    ],
)
def test_mlp_prefill_keeps_attention_and_mtp_fc_at_bf16_activation_precision(
    prefix, native_prefill
):
    config = QwenInt8Config(
        {prefix: "channel_uint8"}, linear_backend="consumer_int8_mlp_prefill"
    )
    layer = object.__new__(MergedColumnParallelLinear)
    method = config.get_quant_method(layer, prefix)
    expected = QwenInt8PrefillLinearMethod if native_prefill else QwenInt8LinearMethod
    assert type(method) is expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefill_dispatch_survives_guard_free_graph_reuse_for_decode():
    """A graph traced at M=1024 must still use W8A16 for smaller runtime M."""
    from vllm import _custom_ops as ops

    if torch.cuda.get_device_capability() != (8, 6):
        pytest.skip("consumer INT8 kernels require SM86")
    torch.manual_seed(911)
    n, k = 128, 256
    codes = torch.randint(0, 256, (k, n), device="cuda", dtype=torch.uint8)
    scales = torch.full((1, n), 0.01, device="cuda", dtype=torch.bfloat16)
    weight, scale, _ = ops.allspark_repack_weight(codes, scales, None, False)
    config = QwenInt8Config({"layer": "channel_uint8"})

    class Program(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Module()
            self.layer.weight = torch.nn.Parameter(weight, requires_grad=False)
            self.layer.weight_scale_inv = torch.nn.Parameter(scale, requires_grad=False)
            self.layer.output_size_per_partition = n
            self.layer._dsv4_int_allspark = True
            self.layer._dsv4_int_allspark_args = {
                "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
                "sm_version": 86,
                "cublas_m_threshold": 1024,
            }
            self.method = QwenInt8PrefillLinearMethod(config, "layer")
            self.method.process_weights_after_loading(self.layer)

        def forward(self, x):
            return self.method.apply(self.layer, x)

    program = Program()
    sample = torch.randn(1024, k, device="cuda", dtype=torch.bfloat16)
    # vLLM reuses an AOT graph over its compile range without Python guards.
    graph = torch._dynamo.export(program)(sample).graph_module
    reference = QwenInt8LinearMethod(config, "layer")
    for rows in (1, 4, 229, 916):
        x = sample[:rows]
        expected = reference.apply(program.layer, x)
        torch.testing.assert_close(graph(x), expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("width", [8704, 17408])
def test_silu_prefill_dispatch_preserves_decode_after_1024_row_trace(width):
    from vllm import _custom_ops as ops

    if torch.cuda.get_device_capability() != (8, 6):
        pytest.skip("consumer INT8 kernels require SM86")
    torch.manual_seed(113)
    n = 128
    raw = torch.randint(0, 256, (width, n), device="cuda", dtype=torch.uint8)
    scales = torch.full((1, n), 0.01, device="cuda", dtype=torch.bfloat16)
    weight, scales, _ = ops.allspark_repack_weight(raw, scales, None, False)
    sms = torch.cuda.get_device_properties(0).multi_processor_count

    class FusedProgram(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", weight)
            self.register_buffer("scales", scales)

        def forward(self, x):
            return torch.ops.vllm.qwen_int8_silu_prefill_linear(
                x, self.weight, self.scales, n, sms, 86, 1024
            )

    fused = FusedProgram()

    @torch.compile(fullgraph=True, dynamic=True)
    def activation(x):
        return torch.nn.functional.silu(x[:, :width]) * x[:, width:]

    sample = torch.randn(1024, 2 * width, device="cuda", dtype=torch.bfloat16)
    graph = torch._dynamo.export(fused)(sample).graph_module
    for rows in (1, 4, 229, 916, 1024):
        x = sample[:rows]
        reference = torch.ops.vllm.qwen_int8_prefill_linear(
            activation(x), weight, scales, n, sms, 86, 1024
        )
        torch.testing.assert_close(graph(x), reference, atol=0, rtol=0)
