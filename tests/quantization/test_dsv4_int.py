# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from tools.ampere.dsv4_checkpoint_audit import classify_tensor, matched_scale_name
from tools.ampere.dsv4_requant_checkpoint import convert_checkpoint
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.dsv4_int import (
    Dsv4Int4MoEMethod,
    Dsv4Int8LinearMethod,
    Dsv4IntConfig,
    Dsv4Mxfp4Int8Config,
    _e2m1_nibble_to_fp32,
    _e8m0_to_fp32_scale,
    _unpack_int4_pairs,
    dequantize_allspark_uint8_w8a16,
    dequantize_int4_w4a16,
    dequantize_int8_w8a16,
    dequantize_uint4_asym_w4a16,
    quantize_fp32_to_uint4_asym_w4a16,
    requantize_fp8_to_allspark_uint8_w8a16,
    requantize_fp8_to_int8_w8a16,
    requantize_mxfp4_to_int4_w4a16,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_act_int8_process_scales,
    marlin_make_workspace_new,
    marlin_moe_permute_scales,
)
from vllm.models.deepseek_v4.nvidia.model import (
    _get_deepseek_v4_scale_fmt,
    _make_deepseek_v4_weights_mapper,
)
from vllm.models.deepseek_v4.nvidia.mtp import _mtp_scale_can_be_mapped_later
from vllm.scalar_type import scalar_types


def _snr_db(reference: torch.Tensor, actual: torch.Tensor) -> float:
    noise = (reference.float() - actual.float()).norm()
    return (20 * torch.log10(reference.float().norm() / noise)).item()


def _pack_nibbles(nibbles: torch.Tensor) -> torch.Tensor:
    low = nibbles[..., 0::2]
    high = nibbles[..., 1::2]
    return (((high & 0x0F) << 4) | (low & 0x0F)).view(torch.int8)


def test_dsv4_int_quantization_config_registered():
    assert get_quantization_config("dsv4_int") is Dsv4IntConfig
    assert get_quantization_config("dsv4_mxfp4_int8") is Dsv4Mxfp4Int8Config
    cfg = Dsv4IntConfig.from_config({"quant_method": "dsv4_int"})
    assert cfg.get_name() == "dsv4_int"
    assert cfg.weight_block_size == (128, 128)

    channel_cfg = Dsv4IntConfig.from_config(
        {
            "quant_method": "dsv4_int",
            "config_groups": {
                "linears_w8a16": {
                    "weights": {
                        "num_bits": 8,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "channel",
                    }
                }
            },
        }
    )
    assert channel_cfg.int8_weight_strategy == "channel"
    assert channel_cfg.weight_block_size is None

    hybrid_cfg = Dsv4Mxfp4Int8Config.from_config(
        {
            "quant_method": "dsv4_mxfp4_int8",
            "config_groups": {
                "linears_w8a16": {
                    "weights": {
                        "num_bits": 8,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "channel",
                    }
                }
            },
        }
    )
    assert hybrid_cfg.get_name() == "dsv4_mxfp4_int8"
    assert hybrid_cfg.int8_weight_strategy == "channel"


def test_deepseek_v4_nvidia_scale_fmt_defaults_for_dsv4_int():
    class Config:
        quantization_config = {"quant_method": "dsv4_int"}

    assert _get_deepseek_v4_scale_fmt(Config()) == "ue8m0"


def test_deepseek_v4_nvidia_scale_fmt_preserves_explicit_value():
    class Config:
        quantization_config = {
            "quant_method": "deepseek_v4_fp8",
            "scale_fmt": "float32",
        }

    assert _get_deepseek_v4_scale_fmt(Config()) == "float32"


def test_dsv4_quant_configs_use_int8_for_remapped_mtp_projections():
    class FakeLinear(LinearBase):
        pass

    layer = FakeLinear.__new__(FakeLinear)
    prefixes = [
        "model.layers.61.e_proj",
        "model.layers.61.h_proj",
        "model.layers.62.e_proj",
        "model.layers.62.h_proj",
    ]
    for config_cls in (Dsv4IntConfig, Dsv4Mxfp4Int8Config):
        cfg = config_cls.from_config({"quant_method": config_cls.QUANT_METHOD_NAME})
        for prefix in prefixes:
            assert isinstance(
                cfg.get_quant_method(layer, prefix), Dsv4Int8LinearMethod
            )


def test_deepseek_v4_mtp_loader_defers_remapped_scales():
    stacked_params_mapping = [
        ("gate_up_proj", "w1", 0),
        ("gate_up_proj", "w3", 1),
        ("fused_wqa_wkv", "attn.wq_a", 0),
        ("fused_wqa_wkv", "attn.wkv", 1),
    ]

    assert _mtp_scale_can_be_mapped_later(
        "model.layers.43.mtp_block.attn.wq_a.weight_scale_inv",
        stacked_params_mapping,
    )
    assert _mtp_scale_can_be_mapped_later(
        "model.layers.43.mtp_block.attn.wkv.weight_scale_inv",
        stacked_params_mapping,
    )
    assert _mtp_scale_can_be_mapped_later(
        "model.layers.43.mtp_block.ffn.shared_experts.w1.weight_scale_inv",
        stacked_params_mapping,
    )
    assert _mtp_scale_can_be_mapped_later(
        "model.layers.43.mtp_block.ffn.shared_experts.w2.weight_scale_inv",
        stacked_params_mapping,
    )
    assert not _mtp_scale_can_be_mapped_later(
        "model.layers.43.e_proj.weight_scale_inv",
        stacked_params_mapping,
    )


def test_mxfp4_to_int4_requant_roundtrip():
    torch.manual_seed(0)
    rows = 16
    cols = 256
    nibbles = torch.randint(0, 16, (rows, cols), dtype=torch.uint8)
    packed = _pack_nibbles(nibbles)
    scale_bytes = torch.randint(120, 134, (rows, cols // 32), dtype=torch.uint8)

    result = requantize_mxfp4_to_int4_w4a16(packed, scale_bytes)
    int4_dequant = dequantize_int4_w4a16(
        result["qweight_packed"], result["scales"], group_size=32
    )

    fp4 = _e2m1_nibble_to_fp32(_unpack_int4_pairs(packed))
    scale = _e8m0_to_fp32_scale(scale_bytes)
    fp4_truth = (fp4.reshape(rows, -1, 32) * scale.unsqueeze(-1)).reshape(rows, cols)

    assert _snr_db(fp4_truth, int4_dequant) > 17.0


def test_mxfp4_to_int4_accepts_absmax8_scale_mode():
    nibbles = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7] * 4],
        dtype=torch.uint8,
    )
    packed = _pack_nibbles(nibbles)
    scale_bytes = torch.full((1, 1), 127, dtype=torch.uint8)

    absmax7 = requantize_mxfp4_to_int4_w4a16(
        packed,
        scale_bytes,
        scale_mode="absmax7",
    )
    absmax8 = requantize_mxfp4_to_int4_w4a16(
        packed,
        scale_bytes,
        scale_mode="absmax8",
    )

    fp4_truth = _e2m1_nibble_to_fp32(nibbles)
    dequant7 = dequantize_int4_w4a16(
        absmax7["qweight_packed"],
        absmax7["scales"],
        group_size=32,
    )
    dequant8 = dequantize_int4_w4a16(
        absmax8["qweight_packed"],
        absmax8["scales"],
        group_size=32,
    )

    assert torch.isfinite(fp4_truth).all()
    assert torch.isfinite(dequant7).all()
    assert torch.isfinite(dequant8).all()
    assert torch.allclose(absmax8["scales"], absmax7["scales"] * 7.0 / 8.0)


def test_mxfp4_to_int4_mse_scale_mode_beats_absmax7():
    torch.manual_seed(0)
    rows = 64
    cols = 1024
    nibbles = torch.randint(0, 16, (rows, cols), dtype=torch.uint8)
    packed = _pack_nibbles(nibbles)
    scale_bytes = torch.randint(120, 134, (rows, cols // 32), dtype=torch.uint8)

    fp4 = _e2m1_nibble_to_fp32(_unpack_int4_pairs(packed))
    scale = _e8m0_to_fp32_scale(scale_bytes)
    fp4_truth = (fp4.reshape(rows, -1, 32) * scale.unsqueeze(-1)).reshape(rows, cols)

    snrs = {}
    for mode in ("absmax7", "mse"):
        result = requantize_mxfp4_to_int4_w4a16(packed, scale_bytes, scale_mode=mode)
        dequant = dequantize_int4_w4a16(
            result["qweight_packed"], result["scales"], group_size=32
        )
        snrs[mode] = _snr_db(fp4_truth, dequant)

    # Measured +5.5 to +6.1 dB on real V4-Flash shards. Uniform-random
    # synthetic nibbles exercise the search less (+1.7 dB measured), so the
    # regression floor here is +1.0 dB.
    assert snrs["mse"] > snrs["absmax7"] + 1.0, snrs


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "float8_e8m0fnu"),
    reason="requires torch float8 dtypes",
)
def test_fp8_to_int8_requant_roundtrip():
    torch.manual_seed(1)
    n = 256
    k = 256
    source = (torch.randn(n, k) * 0.5).clamp(-4, 4)
    weight_fp8 = source.to(torch.float8_e4m3fn)
    scale_bytes = torch.randint(123, 131, (2, 2), dtype=torch.uint8)
    scale_e8m0 = scale_bytes.view(torch.float8_e8m0fnu)

    result = requantize_fp8_to_int8_w8a16(weight_fp8, scale_e8m0)
    int8_dequant = dequantize_int8_w8a16(
        result["qweight"], result["scales"], block_size=(128, 128)
    )

    scale = _e8m0_to_fp32_scale(scale_e8m0)
    scale_full = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    fp8_truth = weight_fp8.to(torch.float32) * scale_full[:n, :k]

    assert _snr_db(fp8_truth, int8_dequant) > 30.0


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "float8_e8m0fnu"),
    reason="requires torch float8 dtypes",
)
def test_fp8_to_allspark_uint8_channel_requant_roundtrip():
    torch.manual_seed(11)
    n = 256
    k = 256
    source = (torch.randn(n, k) * 0.5).clamp(-4, 4)
    weight_fp8 = source.to(torch.float8_e4m3fn)
    scale_e8m0 = torch.randint(123, 131, (2, 2), dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )

    result = requantize_fp8_to_allspark_uint8_w8a16(weight_fp8, scale_e8m0)
    int8_dequant = dequantize_allspark_uint8_w8a16(
        result["qweight"], result["scales"]
    )

    scale = _e8m0_to_fp32_scale(scale_e8m0)
    scale_full = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    fp8_truth = weight_fp8.to(torch.float32) * scale_full[:n, :k]

    assert result["qweight"].dtype is torch.uint8
    assert result["scales"].shape == (n,)
    assert _snr_db(fp8_truth, int8_dequant) > 30.0


def test_asymmetric_uint4_improves_biased_groups():
    x = torch.linspace(-0.1, 1.0, 256, dtype=torch.float32).reshape(8, 32)

    asym = quantize_fp32_to_uint4_asym_w4a16(x, group_size=32)
    asym_dequant = dequantize_uint4_asym_w4a16(
        asym["qweight_packed"],
        asym["scales"],
        asym["zero_points"],
        group_size=32,
    )

    scale = x.abs().amax(dim=-1, keepdim=True).clamp(
        min=torch.finfo(torch.float32).tiny
    ) / 7.0
    sym_dequant = torch.round(x / scale).clamp(-8, 7) * scale

    asym_rmse = torch.sqrt(torch.mean((x - asym_dequant.float()) ** 2))
    sym_rmse = torch.sqrt(torch.mean((x - sym_dequant) ** 2))
    assert asym_rmse < sym_rmse


def test_deepseek_v4_int4_mapper_keeps_expert_scale_suffix():
    mapper = _make_deepseek_v4_weights_mapper("int4")
    names = mapper.apply_list(
        [
            "layers.0.ffn.experts.0.w1.scale",
            "layers.0.attn.wq_a.scale",
        ]
    )

    assert names == [
        "model.layers.0.ffn.experts.0.w1.weight_scale",
        "model.layers.0.attn.wq_a.weight_scale_inv",
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dsv4_channel_int8_linear_method_prefers_allspark_on_ampere():
    props = torch.cuda.get_device_properties()
    sm_version = props.major * 10 + props.minor
    if sm_version < 80 or sm_version >= 90:
        pytest.skip("Ampere W8A16 channel paths only run on sm_8x")

    torch.manual_seed(14)
    m = 12
    n = 256
    k = 256
    dtype = torch.bfloat16
    device = torch.device("cuda")
    weight = torch.randn(n, k, device=device, dtype=torch.float32) * 0.02
    scale = weight.abs().amax(dim=1).clamp(min=torch.finfo(torch.float32).tiny) / 127.0
    q_signed = torch.round(weight / scale.unsqueeze(1)).clamp(-128, 127)
    q_biased = (q_signed.to(torch.int16) + 128).to(torch.uint8)

    class FakeLayer(torch.nn.Module):
        pass

    layer = FakeLayer()
    layer.input_size_per_partition = k
    layer.output_size_per_partition = n
    layer.weight = torch.nn.Parameter(q_biased.contiguous(), requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(
        scale.to(dtype).contiguous(), requires_grad=False
    )

    cfg = Dsv4IntConfig.from_config(
        {
            "quant_method": "dsv4_int",
            "config_groups": {
                "linears_w8a16": {
                    "weights": {
                        "num_bits": 8,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "channel",
                    }
                }
            },
        }
    )
    method = Dsv4Int8LinearMethod(cfg, "model.layers.0.e_proj")
    method.process_weights_after_loading(layer)
    allspark_available = hasattr(torch.ops, "_C") and hasattr(
        torch.ops._C, "allspark_w8a16_gemm"
    )
    if allspark_available:
        # AllSpark is the sole Ampere channel-W8A16 kernel (the Triton channel
        # path was removed: identical alignment gate, 2-5x slower). When the op
        # is present and dims are aligned, AllSpark takes the layer.
        assert layer._dsv4_int_allspark
    else:
        # Without the AllSpark op the layer falls through to the bf16 dequant.
        assert layer._dsv4_int_dequanted

    x = torch.randn(m, k, device=device, dtype=dtype) * 0.02
    actual = method.apply(layer, x)
    ref_weight = dequantize_allspark_uint8_w8a16(q_biased, scale.to(dtype)).to(dtype)
    reference = F.linear(x, ref_weight)
    torch.cuda.synchronize()

    assert _snr_db(reference, actual) > 40.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_allspark_channel_int8_linear_method_matches_dequant_reference():
    if not hasattr(torch.ops, "_C") or not hasattr(
        torch.ops._C, "allspark_w8a16_gemm"
    ):
        pytest.skip("AllSpark W8A16 op is not available")
    props = torch.cuda.get_device_properties()
    sm_version = props.major * 10 + props.minor
    if sm_version < 80 or sm_version >= 90:
        pytest.skip("AllSpark Ampere path only runs on sm_8x")

    torch.manual_seed(12)
    m = 12
    n = 1024
    k = 1024
    dtype = torch.bfloat16
    device = torch.device("cuda")
    weight = torch.randn(n, k, device=device, dtype=torch.float32) * 0.02
    scale = weight.abs().amax(dim=1).clamp(min=torch.finfo(torch.float32).tiny) / 127.0
    q_signed = torch.round(weight / scale.unsqueeze(1)).clamp(-128, 127)
    q_biased = (q_signed.to(torch.int16) + 128).to(torch.uint8)

    class FakeLayer(torch.nn.Module):
        pass

    layer = FakeLayer()
    layer.input_size_per_partition = k
    layer.output_size_per_partition = n
    layer.weight = torch.nn.Parameter(q_biased, requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(scale.to(dtype), requires_grad=False)

    cfg = Dsv4IntConfig.from_config(
        {
            "quant_method": "dsv4_int",
            "config_groups": {
                "linears_w8a16": {
                    "weights": {
                        "num_bits": 8,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "channel",
                    }
                }
            },
        }
    )
    method = Dsv4Int8LinearMethod(cfg, "model.layers.0.attn.wq_a")
    assert method._try_process_allspark(layer)

    x = torch.randn(m, k, device=device, dtype=dtype) * 0.02
    actual = method.apply(layer, x)
    ref_weight = dequantize_allspark_uint8_w8a16(q_biased, scale.to(dtype)).to(dtype)
    reference = torch.nn.functional.linear(x, ref_weight)
    torch.cuda.synchronize()

    assert _snr_db(reference, actual) > 45.0


def test_checkpoint_audit_classifies_deepseek_v4_precision_roles():
    assert classify_tensor("layers.2.ffn.experts.0.w1.weight", "I8") == (
        "routed_expert_mxfp4_weight",
        "quantize_asym_int4_awq_candidate",
    )
    assert classify_tensor("layers.2.attn.indexer.wq_b.weight", "F8_E4M3") == (
        "indexer_qk_fp8_weight",
        "measure_recall_then_quantize",
    )
    assert classify_tensor("layers.2.attn.compressor.wkv.weight", "BF16") == (
        "preserved_precision_tensor",
        "preserve",
    )
    assert classify_tensor("mtp.0.h_proj.scale", "F8_E8M0") == (
        "mtp_fp8_scale",
        "quantize_int8_w8a16_candidate",
    )
    assert classify_tensor("mtp.1.e_proj.weight", "F8_E4M3") == (
        "mtp_fp8_weight",
        "quantize_int8_w8a16_candidate",
    )
    assert (
        matched_scale_name("layers.0.attn.wq_a.weight")
        == "layers.0.attn.wq_a.scale"
    )


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "float8_e8m0fnu"),
    reason="requires torch float8 dtypes",
)
def test_requant_checkpoint_rewrites_remapped_layers_and_quant_config(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()

    shard_name = "model-00001-of-00001.safetensors"
    expert_nibbles = torch.randint(0, 16, (4, 64), dtype=torch.uint8)
    expert_packed = _pack_nibbles(expert_nibbles)
    fp8_weight = torch.randn(130, 129).clamp(-2, 2).to(torch.float8_e4m3fn)
    tensors = {
        "layers.0.ffn.experts.0.w1.weight": expert_packed,
        "layers.0.ffn.experts.0.w1.scale": torch.full(
            (4, 2), 127, dtype=torch.uint8
        ),
        "layers.42.attn.wq_a.weight": fp8_weight,
        "layers.42.attn.wq_a.scale": torch.full(
            (2, 2), 127, dtype=torch.uint8
        ).view(torch.float8_e8m0fnu),
        "layers.42.attn.attn_sink": torch.ones(4, dtype=torch.bfloat16),
        "mtp.1.e_proj.weight": fp8_weight.clone(),
        "mtp.1.e_proj.scale": torch.full(
            (2, 2), 127, dtype=torch.uint8
        ).view(torch.float8_e8m0fnu),
    }
    save_file(tensors, str(src / shard_name))
    (src / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV4ForCausalLM"],
                "num_hidden_layers": 2,
                "expert_dtype": "fp4",
            }
        )
    )
    (src / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": "0"},
                "weight_map": {name: shard_name for name in tensors},
            }
        )
    )

    convert_checkpoint(
        src,
        dst,
        device="cpu",
        out_scale_dtype=torch.bfloat16,
        overwrite=False,
        layer_remap=None,
    )

    cfg = json.loads((dst / "config.json").read_text())
    assert cfg["expert_dtype"] == "int4"
    assert cfg["quantization_config"]["quant_method"] == "dsv4_int"
    assert (
        cfg["quantization_config"]["config_groups"]["experts_w4a16"]["weights"][
            "scale_mode"
        ]
        == "absmax7"
    )
    assert cfg["num_hidden_layers"] == 2

    index = json.loads((dst / "model.safetensors.index.json").read_text())
    assert "layers.42.attn.wq_a.weight" not in index["weight_map"]
    assert "layers.1.attn.wq_a.weight" in index["weight_map"]

    with safe_open(dst / shard_name, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        assert "layers.42.attn.wq_a.weight" not in keys
        assert "layers.1.attn.wq_a.weight" in keys
        assert "layers.1.attn.wq_a.scale" in keys
        assert "layers.1.attn.attn_sink" in keys
        assert "mtp.1.e_proj.weight" in keys
        assert "mtp.1.e_proj.scale" in keys
        assert handle.get_tensor("layers.0.ffn.experts.0.w1.weight").dtype is torch.int8
        assert (
            handle.get_tensor("layers.0.ffn.experts.0.w1.scale").dtype
            is torch.bfloat16
        )
        assert handle.get_tensor("layers.1.attn.wq_a.weight").dtype is torch.int8
        assert handle.get_tensor("layers.1.attn.wq_a.scale").dtype is torch.bfloat16
        assert handle.get_tensor("mtp.1.e_proj.weight").dtype is torch.int8
        assert handle.get_tensor("mtp.1.e_proj.scale").dtype is torch.bfloat16


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "float8_e8m0fnu"),
    reason="requires torch float8 dtypes",
)
def test_requant_checkpoint_can_preserve_mxfp4_experts_with_int8_dense(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()

    shard_name = "model-00001-of-00001.safetensors"
    expert_packed = _pack_nibbles(torch.randint(0, 16, (4, 64), dtype=torch.uint8))
    expert_scale = torch.full((4, 2), 127, dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    fp8_weight = torch.randn(128, 128).clamp(-2, 2).to(torch.float8_e4m3fn)
    tensors = {
        "layers.0.ffn.experts.0.w1.weight": expert_packed,
        "layers.0.ffn.experts.0.w1.scale": expert_scale,
        "layers.0.attn.wq_a.weight": fp8_weight,
        "layers.0.attn.wq_a.scale": torch.full(
            (1, 1), 127, dtype=torch.uint8
        ).view(torch.float8_e8m0fnu),
    }
    save_file(tensors, str(src / shard_name))
    (src / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV4ForCausalLM"],
                "num_hidden_layers": 1,
                "expert_dtype": "fp4",
            }
        )
    )
    (src / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": "0"},
                "weight_map": {name: shard_name for name in tensors},
            }
        )
    )

    convert_checkpoint(
        src,
        dst,
        device="cpu",
        out_scale_dtype=torch.bfloat16,
        overwrite=False,
        layer_remap=None,
        dense_int8_strategy="channel",
        expert_format="mxfp4",
    )

    cfg = json.loads((dst / "config.json").read_text())
    assert cfg["expert_dtype"] == "fp4"
    assert cfg["quantization_config"]["quant_method"] == "dsv4_mxfp4_int8"
    assert (
        cfg["quantization_config"]["config_groups"]["experts_w4a16"]["weights"][
            "scale_mode"
        ]
        == "native_e8m0"
    )

    with safe_open(dst / shard_name, framework="pt", device="cpu") as handle:
        assert torch.equal(
            handle.get_tensor("layers.0.ffn.experts.0.w1.weight"),
            expert_packed,
        )
        assert torch.equal(
            handle.get_tensor("layers.0.ffn.experts.0.w1.scale").view(torch.uint8),
            expert_scale.view(torch.uint8),
        )
        assert handle.get_tensor("layers.0.attn.wq_a.weight").dtype is torch.uint8
        assert handle.get_tensor("layers.0.attn.wq_a.scale").dtype is torch.bfloat16


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "float8_e8m0fnu"),
    reason="requires torch float8 dtypes",
)
def test_requant_checkpoint_can_reshard_layer_contiguous(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()

    shard_name = "model-00001-of-00001.safetensors"
    tensors = {
        "embed.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        "layers.0.attn.wq_a.weight": torch.randn(128, 128)
        .clamp(-2, 2)
        .to(torch.float8_e4m3fn),
        "layers.0.attn.wq_a.scale": torch.ones(1, 1, dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        ),
        "layers.1.attn.wq_a.weight": torch.randn(128, 128)
        .clamp(-2, 2)
        .to(torch.float8_e4m3fn),
        "layers.1.attn.wq_a.scale": torch.ones(1, 1, dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        ),
        "norm.weight": torch.ones(2, dtype=torch.bfloat16),
    }
    save_file(tensors, str(src / shard_name))
    (src / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV4ForCausalLM"],
                "num_hidden_layers": 2,
                "expert_dtype": "fp4",
            }
        )
    )
    (src / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": "0"},
                "weight_map": {name: shard_name for name in tensors},
            }
        )
    )

    convert_checkpoint(
        src,
        dst,
        device="cpu",
        out_scale_dtype=torch.bfloat16,
        overwrite=False,
        layer_remap=None,
        num_output_shards=4,
    )

    index = json.loads((dst / "model.safetensors.index.json").read_text())
    assert index["metadata"]["num_output_shards"] == "4"
    assert index["weight_map"]["embed.weight"] == "model-00001-of-00004.safetensors"
    assert (
        index["weight_map"]["layers.0.attn.wq_a.weight"]
        == "model-00002-of-00004.safetensors"
    )
    assert (
        index["weight_map"]["layers.1.attn.wq_a.weight"]
        == "model-00003-of-00004.safetensors"
    )
    assert index["weight_map"]["norm.weight"] == "model-00004-of-00004.safetensors"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_int4_moe_marlin_repack_smoke():
    torch.manual_seed(2)
    num_experts = 2
    size_k = 128
    size_n = 256
    weight = torch.randint(
        0,
        256,
        (num_experts, size_n, size_k // 2),
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.int8)

    repacked = Dsv4Int4MoEMethod._repack_int4_for_marlin(
        weight,
        size_n=size_n,
        size_k=size_k,
    )
    torch.cuda.synchronize()

    assert repacked.shape[0] == num_experts
    assert repacked.dtype == torch.int32
    assert repacked.is_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("input_dtype", [None, torch.int8])
def test_int4_moe_marlin_matches_dequant_reference(input_dtype):
    torch.manual_seed(3)
    num_experts = 2
    hidden_size = 128
    intermediate_size = 128
    group_size = 32
    m = 9
    dtype = torch.bfloat16
    device = torch.device("cuda")
    is_a_8bit = input_dtype is not None

    def make_quant_weight(shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        nibbles = torch.randint(0, 16, shape, dtype=torch.uint8, device=device)
        qweight = _pack_nibbles(nibbles).to(device)
        scales = (
            torch.rand(
                shape[0],
                shape[1] // group_size,
                dtype=torch.float32,
                device=device,
            )
            * 0.02
            + 0.001
        ).to(dtype)
        return qweight, scales

    w1_q, w1_s = zip(
        *[
            make_quant_weight((intermediate_size, hidden_size))
            for _ in range(num_experts)
        ]
    )
    w2_q, w2_s = zip(
        *[
            make_quant_weight((hidden_size, intermediate_size))
            for _ in range(num_experts)
        ]
    )
    w3_q, w3_s = zip(
        *[
            make_quant_weight((intermediate_size, hidden_size))
            for _ in range(num_experts)
        ]
    )
    w1_q = torch.stack(list(w1_q))
    w2_q = torch.stack(list(w2_q))
    w3_q = torch.stack(list(w3_q))
    w1_s = torch.stack(list(w1_s))
    w2_s = torch.stack(list(w2_s))
    w3_s = torch.stack(list(w3_s))

    w13_q = torch.cat([w1_q, w3_q], dim=1).contiguous()
    w13_s = torch.cat([w1_s, w3_s], dim=1).contiguous()
    w13_marlin = Dsv4Int4MoEMethod._repack_int4_for_marlin(
        w13_q,
        size_n=2 * intermediate_size,
        size_k=hidden_size,
        is_a_8bit=is_a_8bit,
    )
    w2_marlin = Dsv4Int4MoEMethod._repack_int4_for_marlin(
        w2_q,
        size_n=hidden_size,
        size_k=intermediate_size,
        is_a_8bit=is_a_8bit,
    )
    w13_scale = marlin_moe_permute_scales(
        w13_s.transpose(1, 2).contiguous(),
        size_k=hidden_size,
        size_n=2 * intermediate_size,
        group_size=group_size,
        is_a_8bit=is_a_8bit,
    )
    w2_scale = marlin_moe_permute_scales(
        w2_s.transpose(1, 2).contiguous(),
        size_k=intermediate_size,
        size_n=hidden_size,
        group_size=group_size,
        is_a_8bit=is_a_8bit,
    )

    w13_input_global_scale = w2_input_global_scale = None
    if input_dtype == torch.int8:
        w13_scale, w13_input_global_scale = marlin_act_int8_process_scales(w13_scale)
        w2_scale, w2_input_global_scale = marlin_act_int8_process_scales(w2_scale)

    x = torch.randn(m, hidden_size, dtype=dtype, device=device) * 0.1
    score = torch.randn(m, num_experts, dtype=torch.float32, device=device)
    topk_weights, topk_ids = torch.topk(torch.softmax(score, dim=-1), k=2)
    topk_ids = topk_ids.to(torch.int32)
    actual = fused_marlin_moe(
        x,
        w13_marlin,
        w2_marlin,
        None,
        None,
        w13_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        quant_type_id=scalar_types.uint4b8.id,
        global_num_experts=num_experts,
        g_idx1=torch.empty(num_experts, 0, dtype=torch.int32, device=device),
        g_idx2=torch.empty(num_experts, 0, dtype=torch.int32, device=device),
        sort_indices1=torch.empty(num_experts, 0, dtype=torch.int32, device=device),
        sort_indices2=torch.empty(num_experts, 0, dtype=torch.int32, device=device),
        workspace=marlin_make_workspace_new(device, 4),
        is_k_full=True,
        input_dtype=input_dtype,
        input_global_scale1=w13_input_global_scale,
        input_global_scale2=w2_input_global_scale,
    )

    reference = torch.zeros_like(actual)
    for token in range(m):
        for choice in range(2):
            expert = int(topk_ids[token, choice])
            w1 = dequantize_int4_w4a16(w1_q[expert], w1_s[expert]).to(dtype)
            w2 = dequantize_int4_w4a16(w2_q[expert], w2_s[expert]).to(dtype)
            w3 = dequantize_int4_w4a16(w3_q[expert], w3_s[expert]).to(dtype)
            gate = F.linear(x[token : token + 1], w1)
            up = F.linear(x[token : token + 1], w3)
            expert_out = F.linear(F.silu(gate) * up, w2)
            reference[token : token + 1] += topk_weights[token, choice] * expert_out

    torch.cuda.synchronize()
    # INT8 activations add per-token quant noise on top of the INT4 weight
    # error, so the W4A8 floor is lower than the W4A16 floor.
    assert _snr_db(reference, actual) > (30.0 if is_a_8bit else 45.0)
