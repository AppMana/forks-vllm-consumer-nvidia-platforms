# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 AOT INT4/INT8 quantization for Ampere.

This runtime path is for checkpoints converted from the native DeepSeek V4
FP4/FP8 format:

* routed experts: MXFP4 -> signed INT4 W4A16, group size 32
* attention and shared experts: FP8 -> signed INT8 W8A16, 128x128 blocks

The conversion is intentionally ahead-of-time. Runtime code should only load
the packed tensors, repack routed experts for Marlin, and run regular BF16
linear math for the smaller INT8 blocks until a W8A16 linear kernel is wired in.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from vllm.logger import init_logger as _dsv4_init_logger
_dsv4_logger = _dsv4_init_logger(__name__)
_DSV4_KERNEL_PATHS: dict = {}

def _dsv4_log_path(path: str) -> None:
    _DSV4_KERNEL_PATHS[path] = _DSV4_KERNEL_PATHS.get(path, 0) + 1
    _dsv4_logger.info("DSV4KERNEL dense path=%s running_counts=%s", path, _DSV4_KERNEL_PATHS)

from vllm import _custom_ops as ops
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEMethodBase,
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    int4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization import (
    QuantizationMethods,
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from vllm.model_executor.layers.quantization.utils.allspark_utils import (
    ALLSPARK_AMPERE_M_CUBLAS_THRESHOLD,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_input_dtype,
    marlin_act_int8_process_scales,
    marlin_make_workspace_new,
    marlin_moe_permute_scales,
)
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    ChannelQuantScaleParameter,
    ModelWeightParameter,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.scalar_type import scalar_types
from vllm.utils.platform_utils import num_compute_units

_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)


def _unpack_int4_pairs(packed: torch.Tensor) -> torch.Tensor:
    """Unpack low/even, high/odd int4 byte pairs into uint8 nibbles."""
    u = packed.view(torch.uint8) if packed.dtype == torch.int8 else packed
    low = u & 0x0F
    high = (u >> 4) & 0x0F
    out_shape = list(u.shape)
    out_shape[-1] *= 2
    out = torch.empty(out_shape, dtype=torch.uint8, device=u.device)
    out[..., 0::2] = low
    out[..., 1::2] = high
    return out


def _pack_int4_pairs(unpacked: torch.Tensor) -> torch.Tensor:
    """Pack uint8 nibbles as low/even, high/odd byte pairs."""
    assert unpacked.shape[-1] % 2 == 0
    low = unpacked[..., 0::2]
    high = unpacked[..., 1::2]
    return ((high & 0x0F) << 4) | (low & 0x0F)


def _e2m1_nibble_to_fp32(nibble: torch.Tensor) -> torch.Tensor:
    sign_bit = (nibble >> 3) & 1
    magnitude = (nibble & 0x07).to(torch.long)
    sign = 1.0 - 2.0 * sign_bit.to(torch.float32)
    values = _E2M1_VALUES.to(nibble.device)
    mag = values[magnitude.reshape(-1)].reshape(magnitude.shape)
    return sign * mag


def _e8m0_to_fp32_scale(scale_e8m0: torch.Tensor) -> torch.Tensor:
    if scale_e8m0.dtype in (torch.float8_e8m0fnu, torch.int8):
        u = scale_e8m0.view(torch.uint8)
    elif scale_e8m0.dtype == torch.uint8:
        u = scale_e8m0
    else:
        raise TypeError(f"Unsupported e8m0 scale dtype: {scale_e8m0.dtype}")
    return torch.exp2(u.to(torch.float32) - 127.0)


def requantize_mxfp4_to_int4_w4a16(
    weight_packed: torch.Tensor,
    scale_e8m0: torch.Tensor,
    *,
    scale_mode: str = "absmax7",
    out_scale_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor | int]:
    """Convert one MXFP4 tensor to INT4 W4A16 with group size 32."""
    nibble = _unpack_int4_pairs(weight_packed)
    fp4 = _e2m1_nibble_to_fp32(nibble)
    scale = _e8m0_to_fp32_scale(scale_e8m0)
    last_dim = fp4.shape[-1]
    if last_dim != scale.shape[-1] * 32:
        raise ValueError(
            f"weight last dim {last_dim} != scale groups {scale.shape[-1]} * 32"
        )

    grouped = fp4.reshape(*fp4.shape[:-1], -1, 32) * scale.unsqueeze(-1)
    abs_max = grouped.abs().amax(dim=-1)
    abs_max = abs_max.clamp(min=torch.finfo(torch.float32).tiny)
    if scale_mode == "absmax7":
        new_scale = abs_max / 7.0
    elif scale_mode == "absmax8":
        # MXFP4's largest magnitude is usually an outlier level (6.0).
        # Dividing by 8 sacrifices the positive +6 endpoint, but aligns the
        # common 1.5/3.0 levels better for signed INT4's -8..7 codebook.
        new_scale = abs_max / 8.0
    elif scale_mode == "mse":
        # Per-group scale search minimizing round-trip MSE against the
        # dequantized MXFP4 values. On real V4-Flash expert tensors this
        # measured +5.5 to +6.1 dB SNR over absmax7 (20.1 -> 25.6-26.2 dB,
        # tools/ampere/dsv4_rotation_snr.py, 2026-06-10) with no format
        # change: same signed INT4 group-32 layout and Marlin kernels.
        best_scale = abs_max / 7.0
        best_err = None
        for div in torch.linspace(5.0, 9.5, 19, device=grouped.device):
            cand = (abs_max / div).to(out_scale_dtype).to(torch.float32)
            q = torch.round(grouped / cand.unsqueeze(-1)).clamp(-8, 7)
            err = (q * cand.unsqueeze(-1) - grouped).pow(2).sum(dim=-1)
            if best_err is None:
                best_err = err
                best_scale = cand
            else:
                mask = err < best_err
                best_err = torch.where(mask, err, best_err)
                best_scale = torch.where(mask, cand, best_scale)
        new_scale = best_scale.clamp(min=torch.finfo(torch.float32).tiny)
    else:
        raise ValueError(f"unsupported MXFP4->INT4 scale mode: {scale_mode}")

    int4_signed = torch.round(grouped / new_scale.unsqueeze(-1)).clamp(-8, 7)
    unsigned = (int4_signed + 8).to(torch.uint8)
    packed = _pack_int4_pairs(unsigned.reshape(*fp4.shape)).view(torch.int8)
    return {
        "qweight_packed": packed,
        "scales": new_scale.to(out_scale_dtype),
        "group_size": 32,
    }


def quantize_fp32_to_uint4_asym_w4a16(
    weight: torch.Tensor,
    *,
    group_size: int = 32,
    out_scale_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor | int]:
    """Quantize FP weights to affine UINT4 W4A16 along the last dimension.

    This is the quality-search representation for the next DeepSeek V4 expert
    format. It is intentionally separate from the current symmetric Marlin
    runtime path because AWQ/GPTQ-compatible kernels consume explicit
    zero-points, not the implicit ``u4 - 8`` convention.
    """
    if weight.shape[-1] % group_size != 0:
        raise ValueError(
            f"weight last dim {weight.shape[-1]} is not divisible by {group_size}"
        )

    grouped = weight.to(torch.float32).reshape(*weight.shape[:-1], -1, group_size)
    group_min = torch.minimum(
        grouped.amin(dim=-1), torch.zeros((), device=grouped.device)
    )
    group_max = torch.maximum(
        grouped.amax(dim=-1), torch.zeros((), device=grouped.device)
    )
    scale = (group_max - group_min).clamp(min=torch.finfo(torch.float32).tiny) / 15.0
    zero_point = torch.round(-group_min / scale).clamp(0, 15).to(torch.uint8)
    q = torch.round(grouped / scale.unsqueeze(-1) + zero_point.unsqueeze(-1))
    q = q.clamp(0, 15).to(torch.uint8)
    packed = _pack_int4_pairs(q.reshape(*weight.shape)).view(torch.int8)
    return {
        "qweight_packed": packed,
        "scales": scale.to(out_scale_dtype),
        "zero_points": zero_point,
        "group_size": group_size,
    }


def quantize_fp32_to_uint4_affine_w4a16(
    weight: torch.Tensor,
    *,
    group_size: int = 32,
    out_scale_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor | int]:
    """Quantize FP weights to affine UINT4 with explicit per-group bias.

    This mirrors MLX-style affine quantization more closely than the zero-point
    form: dequantized weights are ``q * scale + bias``.
    """
    if weight.shape[-1] % group_size != 0:
        raise ValueError(
            f"weight last dim {weight.shape[-1]} is not divisible by {group_size}"
        )

    grouped = weight.to(torch.float32).reshape(*weight.shape[:-1], -1, group_size)
    bias = grouped.amin(dim=-1)
    group_max = grouped.amax(dim=-1)
    scale = (group_max - bias).clamp(min=torch.finfo(torch.float32).tiny) / 15.0
    q = torch.round((grouped - bias.unsqueeze(-1)) / scale.unsqueeze(-1))
    q = q.clamp(0, 15).to(torch.uint8)
    packed = _pack_int4_pairs(q.reshape(*weight.shape)).view(torch.int8)
    return {
        "qweight_packed": packed,
        "scales": scale.to(out_scale_dtype),
        "biases": bias.to(out_scale_dtype),
        "group_size": group_size,
    }


def requantize_fp8_to_int8_w8a16(
    weight_fp8: torch.Tensor,
    scale_e8m0: torch.Tensor,
    *,
    block_size: tuple[int, int] = (128, 128),
    out_scale_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor | tuple[int, int]]:
    """Convert one FP8 e4m3 tensor to INT8 W8A16 with 2D block scales."""
    if weight_fp8.dtype != torch.float8_e4m3fn:
        raise TypeError(f"weight must be float8_e4m3fn, got {weight_fp8.dtype}")

    bn, bk = block_size
    n, k = weight_fp8.shape
    gn = (n + bn - 1) // bn
    gk = (k + bk - 1) // bk
    if scale_e8m0.shape != (gn, gk):
        raise ValueError(f"expected scale shape {(gn, gk)}, got {scale_e8m0.shape}")

    dequant = weight_fp8.to(torch.float32)
    scale = _e8m0_to_fp32_scale(scale_e8m0)
    scale_full = scale.repeat_interleave(bn, dim=0).repeat_interleave(bk, dim=1)
    dequant = dequant * scale_full[:n, :k]

    padded = F.pad(dequant, (0, gk * bk - k, 0, gn * bn - n))
    blocked = padded.reshape(gn, bn, gk, bk).permute(0, 2, 1, 3)
    abs_max = blocked.abs().amax(dim=(-2, -1))
    abs_max = abs_max.clamp(min=torch.finfo(torch.float32).tiny)
    new_scale = abs_max / 127.0

    new_scale_full = new_scale.repeat_interleave(bn, dim=0).repeat_interleave(
        bk, dim=1
    )
    qweight = torch.round(dequant / new_scale_full[:n, :k]).clamp(-128, 127)
    return {
        "qweight": qweight.to(torch.int8),
        "scales": new_scale.to(out_scale_dtype),
        "block_size": block_size,
    }


def requantize_fp8_to_allspark_uint8_w8a16(
    weight_fp8: torch.Tensor,
    scale_e8m0: torch.Tensor,
    *,
    block_size: tuple[int, int] = (128, 128),
    out_scale_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor | str]:
    """Convert one FP8 e4m3 tensor to AllSpark channelwise UINT8 W8A16.

    AllSpark's Ampere kernel consumes per-output-channel scales and stores
    signed INT8 values in uint8 form with an implicit +128 bias.
    """
    if weight_fp8.dtype != torch.float8_e4m3fn:
        raise TypeError(f"weight must be float8_e4m3fn, got {weight_fp8.dtype}")

    bn, bk = block_size
    n, k = weight_fp8.shape
    gn = (n + bn - 1) // bn
    gk = (k + bk - 1) // bk
    if scale_e8m0.shape != (gn, gk):
        raise ValueError(f"expected scale shape {(gn, gk)}, got {scale_e8m0.shape}")

    dequant = weight_fp8.to(torch.float32)
    scale = _e8m0_to_fp32_scale(scale_e8m0)
    scale_full = scale.repeat_interleave(bn, dim=0).repeat_interleave(bk, dim=1)
    dequant = dequant * scale_full[:n, :k]

    channel_scale = dequant.abs().amax(dim=1)
    channel_scale = channel_scale.clamp(min=torch.finfo(torch.float32).tiny) / 127.0
    q_signed = torch.round(dequant / channel_scale.unsqueeze(1)).clamp(-128, 127)
    q_biased = (q_signed.to(torch.int16) + 128).to(torch.uint8)
    return {
        "qweight": q_biased,
        "scales": channel_scale.to(out_scale_dtype),
        "strategy": "channel",
    }


def dequantize_int4_w4a16(
    weight_packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    group_size: int = 32,
) -> torch.Tensor:
    nibble = _unpack_int4_pairs(weight_packed)
    int4 = (nibble.to(torch.int8) - 8).to(torch.float32)
    last = int4.shape[-1]
    grouped = int4.reshape(*int4.shape[:-1], -1, group_size)
    out = grouped * scale.to(torch.float32).unsqueeze(-1)
    return out.reshape(*int4.shape[:-1], last).to(torch.bfloat16)


def dequantize_uint4_asym_w4a16(
    weight_packed: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    *,
    group_size: int = 32,
) -> torch.Tensor:
    nibble = _unpack_int4_pairs(weight_packed).to(torch.float32)
    last = nibble.shape[-1]
    grouped = nibble.reshape(*nibble.shape[:-1], -1, group_size)
    out = (grouped - zero_point.to(torch.float32).unsqueeze(-1)) * scale.to(
        torch.float32
    ).unsqueeze(-1)
    return out.reshape(*nibble.shape[:-1], last).to(torch.bfloat16)


def dequantize_uint4_affine_w4a16(
    weight_packed: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    *,
    group_size: int = 32,
) -> torch.Tensor:
    nibble = _unpack_int4_pairs(weight_packed).to(torch.float32)
    last = nibble.shape[-1]
    grouped = nibble.reshape(*nibble.shape[:-1], -1, group_size)
    out = grouped * scale.to(torch.float32).unsqueeze(-1)
    out = out + bias.to(torch.float32).unsqueeze(-1)
    return out.reshape(*nibble.shape[:-1], last).to(torch.bfloat16)


def dequantize_int8_w8a16(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_size: tuple[int, int] = (128, 128),
) -> torch.Tensor:
    bn, bk = block_size
    n, k = weight.shape
    scale_full = scale.to(torch.float32).repeat_interleave(bn, dim=0)
    scale_full = scale_full.repeat_interleave(bk, dim=1)[:n, :k]
    return (weight.to(torch.float32) * scale_full).to(torch.bfloat16)


def dequantize_allspark_uint8_w8a16(
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Dequantize AllSpark's biased UINT8 W8A16 tensor for CPU/fallback paths."""
    signed = weight.to(torch.float32) - 128.0
    return (signed * scale.to(torch.float32).reshape(-1, 1)).to(torch.bfloat16)


@register_quantization_config("dsv4_int")
class Dsv4IntConfig(QuantizationConfig):
    """Quantization config for AOT-requantized DeepSeek V4 INT checkpoints."""

    QUANT_METHOD_NAME = "dsv4_int"
    INT8_PARENT_PATTERNS = (
        ".attn.fused_wqa_wkv",
        ".attn.wq_b",
        ".attn.wo_a",
        ".attn.wo_b",
        ".attn.indexer.wq_b",
        ".attn.indexer.weights_proj",
        ".attn.compressor.fused_wkv_wgate",
        ".attn.indexer.compressor.fused_wkv_wgate",
        ".ffn.shared_experts.gate_up_proj",
        ".ffn.shared_experts.down_proj",
        ".e_proj",
        ".h_proj",
    )

    def __init__(
        self,
        config_groups: dict[str, Any] | None = None,
        ignore_patterns: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.config_groups = config_groups or {}
        self.ignore_patterns = ignore_patterns or []
        linears = self.config_groups.get("linears_w8a16", {})
        weights = linears.get("weights", {})
        self.int8_weight_strategy = weights.get("strategy", "block")
        if self.int8_weight_strategy not in ("block", "channel"):
            raise ValueError(
                "dsv4_int linears_w8a16 weights.strategy must be "
                f"'block' or 'channel', got {self.int8_weight_strategy!r}"
            )
        self.weight_block_size = (
            tuple(weights.get("block_size", [128, 128]))
            if self.int8_weight_strategy == "block"
            else None
        )
        self.activation_scheme = "dynamic"

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "dsv4_int"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Dsv4IntConfig:
        return cls(
            config_groups=config.get("config_groups", {}),
            ignore_patterns=config.get("ignore", []),
        )

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> QuantizationMethods | None:
        if hf_quant_cfg.get("quant_method") == cls.QUANT_METHOD_NAME:
            return cls.QUANT_METHOD_NAME
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, RoutedExperts):
            return Dsv4Int4MoEMethod(self, layer.moe_config)
        if isinstance(layer, LinearBase):
            if any(pattern in prefix for pattern in self.INT8_PARENT_PATTERNS):
                return Dsv4Int8LinearMethod(self, prefix)
            return UnquantizedLinearMethod()
        if isinstance(layer, Attention):
            return BaseKVCacheMethod(self)
        return None


@register_quantization_config("dsv4_mxfp4_int8")
class Dsv4Mxfp4Int8Config(Dsv4IntConfig):
    """DeepSeek V4 hybrid path: native MXFP4 routed experts + INT8 dense linears.

    This keeps the DeepSeek/MLX-style routed expert representation intact
    (E2M1 packed weights with E8M0 group scales) while using the Ampere INT8
    dense linear path for FP8 checkpoint tensors. It is the apples-to-apples
    comparison point for ``mxfp4+fp8`` versus ``mxfp4+int8``.
    """

    QUANT_METHOD_NAME = "dsv4_mxfp4_int8"

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return cls.QUANT_METHOD_NAME

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Dsv4Mxfp4Int8Config:
        return cls(
            config_groups=config.get("config_groups", {}),
            ignore_patterns=config.get("ignore", []),
        )

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> QuantizationMethods | None:
        if hf_quant_cfg.get("quant_method") == cls.QUANT_METHOD_NAME:
            return cls.QUANT_METHOD_NAME
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, RoutedExperts):
            return Mxfp4MoEMethod(layer.moe_config)
        if isinstance(layer, LinearBase):
            if any(pattern in prefix for pattern in self.INT8_PARENT_PATTERNS):
                return Dsv4Int8LinearMethod(self, prefix)
            return UnquantizedLinearMethod()
        if isinstance(layer, Attention):
            return BaseKVCacheMethod(self)
        return None


@register_weight_loader_v2_supported_method
class Dsv4Int8LinearMethod(LinearMethodBase):
    """INT8 W8A16 linear method for DeepSeek V4 dense FP8 replacements."""

    BLOCK_SIZE = (128, 128)

    def __init__(self, quant_config: Dsv4IntConfig, prefix: str) -> None:
        self.quant_config = quant_config
        self.strategy = quant_config.int8_weight_strategy
        # WO_A is consumed by a custom inverse-RoPE einsum path whose FP8 helper
        # expects block scales. Keep this one on the BF16 reference path even
        # when the rest of the dense INT8 linears use AllSpark.
        self.force_dequant = ".attn.wo_a" in prefix

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        parameter_attrs = dict(extra_weight_attrs)
        parameter_attrs.pop("weight_loader", None)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.params_dtype = params_dtype
        layer.weight_block_size = self.BLOCK_SIZE if self.strategy == "block" else None
        layer._dsv4_int8_strategy = self.strategy

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.int8 if self.strategy == "block" else torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, parameter_attrs)

        if self.strategy == "block":
            weight_scale = BlockQuantScaleParameter(
                data=torch.empty(
                    (output_size_per_partition + self.BLOCK_SIZE[0] - 1)
                    // self.BLOCK_SIZE[0],
                    (input_size_per_partition + self.BLOCK_SIZE[1] - 1)
                    // self.BLOCK_SIZE[1],
                    dtype=params_dtype,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
        else:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty(output_size_per_partition, dtype=params_dtype),
                output_dim=0,
                weight_loader=weight_loader,
            )
        layer.register_parameter("weight_scale_inv", weight_scale)
        set_weight_attrs(weight_scale, parameter_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_dsv4_int_dequanted", False) or getattr(
            layer, "_dsv4_int_allspark", False
        ):
            return
        if self.strategy == "channel":
            # AllSpark is the only Ampere channel-W8A16 kernel: the 2026-06-10
            # A5000 GEMM matrix measured it 2-5x faster than the (now removed)
            # Triton channel kernel at every M, and its alignment gate (% 16 on
            # both dims) is identical to Triton's, so any layer Triton could
            # take AllSpark already takes. Layers AllSpark cannot take (op
            # unavailable, unaligned dims) fall through to the bf16 dequant.
            if not self.force_dequant and self._try_process_allspark(layer):
                _dsv4_log_path("allspark")
                return
            weight = dequantize_allspark_uint8_w8a16(
                layer.weight.data,
                layer.weight_scale_inv.data,
            )
            replace_parameter(layer, "weight", weight.contiguous())
            layer._dsv4_int_dequanted = True
            _dsv4_log_path("dequant_channel_bf16" + ("_wo_a" if self.force_dequant else ""))
            return

        weight = dequantize_int8_w8a16(
            layer.weight.data,
            layer.weight_scale_inv.data,
            block_size=self.BLOCK_SIZE,
        )
        replace_parameter(layer, "weight", weight.contiguous())
        layer._dsv4_int_dequanted = True
        _dsv4_log_path("dequant_block_bf16")

    def _try_process_allspark(self, layer: torch.nn.Module) -> bool:
        if not layer.weight.is_cuda:
            return False
        if not hasattr(torch.ops, "_C") or not hasattr(
            torch.ops._C, "allspark_w8a16_gemm"
        ):
            return False

        device = layer.weight.device
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        sm_version = properties.major * 10 + properties.minor
        if sm_version < 80 or sm_version >= 90:
            return False
        if (
            layer.input_size_per_partition % 16 != 0
            or layer.output_size_per_partition % 16 != 0
        ):
            return False

        qweight_kn = layer.weight.data.t().contiguous()
        scales_1n = layer.weight_scale_inv.data.reshape(1, -1).contiguous()
        qweight_reorder, scale_reorder, _ = ops.allspark_repack_weight(
            qweight_kn,
            scales_1n,
            None,
            False,
        )
        replace_parameter(layer, "weight", qweight_reorder)
        replace_parameter(layer, "weight_scale_inv", scale_reorder)
        layer._dsv4_int_allspark = True
        layer._dsv4_int_allspark_args = {
            "sm_count": num_compute_units(device_index),
            "sm_version": sm_version,
        }
        return True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(layer, "_dsv4_int_allspark", False):
            reshaped_x = x.reshape(-1, x.shape[-1]).contiguous()
            args = layer._dsv4_int_allspark_args
            output = ops.allspark_w8a16_gemm(
                a=reshaped_x,
                b_qweight=layer.weight,
                b_scales=layer.weight_scale_inv,
                b_qzeros=None,
                n=layer.output_size_per_partition,
                group_size=-1,
                sm_count=args["sm_count"],
                sm_version=args["sm_version"],
                CUBLAS_M_THRESHOLD=ALLSPARK_AMPERE_M_CUBLAS_THRESHOLD,
                has_zp=False,
                n32k16_reorder=True,
            )
            if bias is not None:
                output.add_(bias)
            return output.reshape(x.shape[:-1] + (layer.output_size_per_partition,))
        return F.linear(x, layer.weight, bias)


class Dsv4Int4MoEMethod(FusedMoEMethodBase):
    """Routed expert INT4 W4A16 method using Marlin on Ampere."""

    GROUP_SIZE = 32

    def __init__(self, quant_config: Dsv4IntConfig, moe: FusedMoEConfig) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        self.num_experts = 0
        self.hidden_size = 0
        self.intermediate_size = 0
        # W4A8: VLLM_MARLIN_INPUT_DTYPE=int8 routes the expert GEMMs through
        # the sm80 s8 integer-MMA Marlin kernels (2x FP16 peak on GA102;
        # 20-25% prefill GEMM win in the 2026-06-10 A5000 matrix). Unset env
        # keeps the W4A16 behavior bit-for-bit.
        self.input_dtype = get_marlin_input_dtype()

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size_per_partition
        layer.num_experts = num_experts
        layer.params_dtype = params_dtype

        w13 = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13)
        set_weight_attrs(w13, extra_weight_attrs)

        w13_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.GROUP_SIZE,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, extra_weight_attrs)
        w13_scale.quant_method = "block"

        w2 = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2)
        set_weight_attrs(w2, extra_weight_attrs)

        w2_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.GROUP_SIZE,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, extra_weight_attrs)
        w2_scale.quant_method = "block"

    @staticmethod
    def _repack_int4_for_marlin(
        weight: torch.Tensor,
        *,
        size_n: int,
        size_k: int,
        is_a_8bit: bool = False,
    ) -> torch.Tensor:
        num_experts = weight.shape[0]
        device = weight.device
        perm = torch.empty(0, dtype=torch.int, device=device)

        def pack_one(expert_weight: torch.Tensor) -> torch.Tensor:
            gptq_weight = expert_weight.view(torch.uint8).view(torch.int32)
            gptq_weight = gptq_weight.t().contiguous()
            return ops.gptq_marlin_repack(
                gptq_weight,
                perm,
                size_k,
                size_n,
                4,
                is_a_8bit=is_a_8bit,
            )

        first = pack_one(weight[0])
        out = torch.empty(
            num_experts,
            *first.shape,
            dtype=first.dtype,
            device=first.device,
        )
        out[0].copy_(first)
        for expert in range(1, num_experts):
            out[expert].copy_(pack_one(weight[expert]))
        return out

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        hidden_size = self.hidden_size
        intermediate = self.intermediate_size
        device = layer.w13_weight.device
        is_a_8bit = self.input_dtype is not None and self.input_dtype.itemsize == 1

        w13 = self._repack_int4_for_marlin(
            layer.w13_weight.data,
            size_n=2 * intermediate,
            size_k=hidden_size,
            is_a_8bit=is_a_8bit,
        )
        replace_parameter(layer, "w13_weight", w13)

        w2 = self._repack_int4_for_marlin(
            layer.w2_weight.data,
            size_n=hidden_size,
            size_k=intermediate,
            is_a_8bit=is_a_8bit,
        )
        replace_parameter(layer, "w2_weight", w2)

        w13_scale = layer.w13_weight_scale.data.transpose(1, 2).contiguous()
        w13_scale = marlin_moe_permute_scales(
            w13_scale,
            size_k=hidden_size,
            size_n=2 * intermediate,
            group_size=self.GROUP_SIZE,
            is_a_8bit=is_a_8bit,
        )
        w2_scale = layer.w2_weight_scale.data.transpose(1, 2).contiguous()
        w2_scale = marlin_moe_permute_scales(
            w2_scale,
            size_k=intermediate,
            size_n=hidden_size,
            group_size=self.GROUP_SIZE,
            is_a_8bit=is_a_8bit,
        )

        if self.input_dtype == torch.int8:
            # Group scales become int16-quantized relative values; the global
            # factor folds into the per-token activation scales inside
            # fused_marlin_moe (input_global_scale1/2).
            w13_scale, w13_input_global_scale = marlin_act_int8_process_scales(
                w13_scale
            )
            w2_scale, w2_input_global_scale = marlin_act_int8_process_scales(w2_scale)
            layer.w13_input_global_scale = torch.nn.Parameter(
                w13_input_global_scale, requires_grad=False
            )
            layer.w2_input_global_scale = torch.nn.Parameter(
                w2_input_global_scale, requires_grad=False
            )

        replace_parameter(layer, "w13_weight_scale", w13_scale)
        replace_parameter(layer, "w2_weight_scale", w2_scale)

        empty_g_idx = torch.empty(self.num_experts, 0, dtype=torch.int32, device=device)
        layer.w13_weight_g_idx = torch.nn.Parameter(empty_g_idx, requires_grad=False)
        layer.w2_weight_g_idx = torch.nn.Parameter(
            empty_g_idx.clone(), requires_grad=False
        )
        layer.w13_g_idx_sort_indices = torch.nn.Parameter(
            empty_g_idx.clone(), requires_grad=False
        )
        layer.w2_g_idx_sort_indices = torch.nn.Parameter(
            empty_g_idx.clone(), requires_grad=False
        )
        layer.workspace = marlin_make_workspace_new(device, 4)
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return int4_w4a16_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w1_zp=None,
            w2_zp=None,
            block_shape=[0, self.GROUP_SIZE],
        )

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: Any = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        return fused_marlin_moe(
            x,
            layer.w13_weight,
            layer.w2_weight,
            None,
            None,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            topk_weights,
            topk_ids,
            quant_type_id=scalar_types.uint4b8.id,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            activation=layer.activation,
            expert_map=layer.expert_map,
            g_idx1=layer.w13_weight_g_idx,
            g_idx2=layer.w2_weight_g_idx,
            sort_indices1=layer.w13_g_idx_sort_indices,
            sort_indices2=layer.w2_g_idx_sort_indices,
            workspace=layer.workspace,
            is_k_full=True,
            input_dtype=self.input_dtype,
            input_global_scale1=getattr(layer, "w13_input_global_scale", None),
            input_global_scale2=getattr(layer, "w2_input_global_scale", None),
        )
