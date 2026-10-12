# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consumer INT8 checkpoint adapter for Qwen hybrid attention models.

Weights use the DSV4 channel-UINT8 and vision group32 formats. AllSpark
W8A16 is the baseline for language layers; native W8A8 candidates live in
the separate consumer-int8-kernels package and require quality/performance
validation before replacing it.
"""

from typing import Any, Literal

import torch

from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.dsv4_int import Dsv4Int8LinearMethod
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.models.utils import WeightsMapper
from vllm.models.deepseek_v4.common.vision_int8 import VisionInt8LinearMethod
from vllm.platforms import current_platform


class QwenInt8Config(QuantizationConfig):
    def __init__(
        self,
        quantized_modules: dict[str, str],
        gdn_decode_backend: str = "triton",
        kv_cache_backend: str = "triton",
        linear_backend: str = "allspark",
    ) -> None:
        super().__init__()
        if not quantized_modules or any(
            encoding not in ("channel_uint8", "group32_int8")
            for encoding in quantized_modules.values()
        ):
            raise ValueError("qwen_int8 requires an explicit valid module manifest")
        self.source_modules = dict(quantized_modules)
        self.quantized_modules = dict(quantized_modules)
        self.int8_weight_strategy = "channel"
        for backend in (gdn_decode_backend, kv_cache_backend):
            if backend not in ("triton", "consumer_int8"):
                raise ValueError(f"Unsupported qwen_int8 backend: {backend}")
        self.gdn_decode_backend = gdn_decode_backend
        self.kv_cache_backend = kv_cache_backend
        if linear_backend not in (
            "allspark",
            "consumer_int8",
            "consumer_int8_hybrid",
            "consumer_int8_prefill",
            "consumer_int8_mlp_prefill",
        ):
            raise ValueError(f"Unsupported qwen_int8 linear backend: {linear_backend}")
        self.linear_backend = linear_backend

    @classmethod
    def get_name(cls) -> Literal["qwen_int8"]:
        return "qwen_int8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QwenInt8Config":
        if (
            config.get("format_version") != 1
            or config.get("weight_strategy") != "channel_uint8"
            or config.get("vision_weight_strategy") != "group32_int8"
        ):
            raise ValueError("Unsupported qwen_int8 checkpoint format")
        return cls(
            config.get("quantized_modules", {}),
            config.get("gdn_decode_backend", "triton"),
            config.get("kv_cache_backend", "triton"),
            config.get("linear_backend", "allspark"),
        )

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        # Start from the serialized names so repeated model initialization
        # cannot apply a prefix rename twice. MTP names must survive mappings
        # that exclude draft weights from the target model's loader.
        self.quantized_modules = hf_to_vllm_mapper.get_rename_mapper().apply_dict(
            self.source_modules
        )

    def _encoding(self, prefix: str) -> str | None:
        direct = self.quantized_modules.get(prefix)
        if direct is not None:
            return direct
        parent, _, leaf = prefix.rpartition(".")
        components = self.packed_modules_mapping.get(leaf)
        if not components:
            return None
        encodings = [
            self.quantized_modules.get(f"{parent}.{name}" if parent else name)
            for name in components
        ]
        if all(value is None for value in encodings):
            return None
        if len(set(encodings)) != 1:
            raise ValueError(f"Mixed/missing INT8 encodings for fused layer {prefix}")
        return encodings[0]

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            encoding = self._encoding(prefix)
            if encoding == "channel_uint8":
                if self.linear_backend == "consumer_int8_prefill" or (
                    self.linear_backend == "consumer_int8_mlp_prefill"
                    and ".mlp." in prefix
                ):
                    return QwenInt8PrefillLinearMethod(self, prefix)
                if self.linear_backend in ("consumer_int8", "consumer_int8_hybrid"):
                    return QwenInt8CUTLASSLinearMethod(self, prefix)
                return QwenInt8LinearMethod(self, prefix)
            if encoding == "group32_int8":
                return QwenVisionInt8LinearMethod()
            return UnquantizedLinearMethod()
        if isinstance(layer, Attention):
            return QwenInt8KVCacheMethod(self)
        return None


class QwenInt8KVCacheMethod(BaseKVCacheMethod):
    quant_config: QwenInt8Config

    def create_weights(self, layer: torch.nn.Module) -> None:
        super().create_weights(layer)
        layer._consumer_int8_cache_write = None
        if self.quant_config.kv_cache_backend == "consumer_int8":
            if not (
                current_platform.is_cuda()
                and current_platform.is_device_capability(86)
                and layer.head_size == 256
                and layer.kv_cache_dtype == "int8_per_token_head"
            ):
                raise ValueError(
                    "consumer_int8 KV writer requires SM86, D256 "
                    "and INT8 per-head cache"
                )
            # Import during construction, before graph capture or serving.
            from consumer_int8.ops import kv_cache_update

            layer._consumer_int8_cache_write = kv_cache_update


@register_weight_loader_v2_supported_method
class QwenInt8LinearMethod(Dsv4Int8LinearMethod):
    quant_config: QwenInt8Config

    def checkpoint_required_params(self, layer: torch.nn.Module) -> frozenset[str]:
        return frozenset({"weight", "weight_scale_inv"})

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_dsv4_int_allspark", False):
            return
        if (
            layer.weight.dtype != torch.uint8
            or layer.weight_scale_inv.dtype != torch.bfloat16
        ):
            raise ValueError("qwen_int8 requires UINT8 weights and BF16 channel scales")
        if not self._try_process_allspark(layer):
            # Expanding 27B weights to BF16 would silently exceed the intended
            # memory budget. Unsupported builds/shapes must fail at load time.
            raise ValueError("qwen_int8 requires AllSpark W8A16 and 16-aligned shards")


@register_weight_loader_v2_supported_method
class QwenInt8PrefillLinearMethod(QwenInt8LinearMethod):
    """Keep packed decode weights; restore one transient layer for INT8 prefill."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not current_platform.is_device_capability(86):
            raise ValueError("consumer_int8 prefill requires SM86")
        super().process_weights_after_loading(layer)
        from consumer_int8.ops import cutlass_channel_mm, quantize_rows, unpack_allspark

        self._unpack = unpack_allspark
        self._quantize = quantize_rows
        self._mm = cutlass_channel_mm

    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        shape = x.shape[:-1]
        m = x.numel() // x.shape[-1]
        if m < 1024:
            return super().apply(layer, x, bias)
        x = x.reshape(m, -1).contiguous()
        k, n = x.shape[1], layer.output_size_per_partition
        weight = torch.empty((n, k), dtype=torch.int8, device=x.device)
        scales = torch.empty(n, dtype=torch.bfloat16, device=x.device)
        self._unpack(layer.weight, layer.weight_scale_inv.flatten(), weight, scales)
        codes = torch.empty_like(x, dtype=torch.int8)
        row_scales = torch.empty(m, dtype=torch.float32, device=x.device)
        workspace = torch.empty((m, n), dtype=torch.int32, device=x.device)
        out = torch.empty((m, n), dtype=x.dtype, device=x.device)
        self._quantize(x, codes, row_scales)
        self._mm(codes, weight, row_scales, scales, workspace, out)
        if bias is not None:
            out.add_(bias)
        return out.reshape(*shape, n)


@register_weight_loader_v2_supported_method
class QwenInt8CUTLASSLinearMethod(Dsv4Int8LinearMethod):
    """Explicit W8A8 candidate; shares the serialized W8A16 weight format."""

    quant_config: QwenInt8Config

    def checkpoint_required_params(self, layer: torch.nn.Module) -> frozenset[str]:
        return frozenset({"weight", "weight_scale_inv"})

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_consumer_int8_processed", False):
            return
        if not (
            current_platform.is_cuda()
            and current_platform.is_device_capability(86)
            and layer.weight.dtype == torch.uint8
            and layer.weight_scale_inv.dtype == torch.bfloat16
            and layer.input_size_per_partition % 32 == 0
            and layer.output_size_per_partition % 4 == 0
        ):
            raise ValueError(
                "consumer_int8 linears require SM86 and aligned UINT8 weights"
            )
        from consumer_int8.ops import cutlass_channel_mm, quantize_rows

        self._quantize_rows = quantize_rows
        self._channel_mm = cutlass_channel_mm
        self._gemv = None
        if self.quant_config.linear_backend == "consumer_int8_hybrid":
            from consumer_int8.ops import channel_gemv

            self._gemv = channel_gemv
        # Lossless reinterpretation of the signed-plus-128 storage. Keep only
        # one weight copy per layer, rather than doubling model memory to hold
        # both AllSpark's reordered and CUTLASS's row-major representations.
        layer.weight = torch.nn.Parameter(
            (layer.weight.to(torch.int16) - 128).to(torch.int8).contiguous(),
            requires_grad=False,
        )
        layer.weight_scale_inv = torch.nn.Parameter(
            layer.weight_scale_inv.contiguous(),
            requires_grad=False,
        )
        layer._consumer_int8_processed = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1]).contiguous()
        m, k = x.shape
        n = layer.output_size_per_partition
        out = torch.empty((m, n), dtype=x.dtype, device=x.device)
        if m == 0:
            return out.reshape(*shape, n)
        if self._gemv is not None and m <= 8:
            self._gemv(x, layer.weight, layer.weight_scale_inv, out)
            if bias is not None:
                out.add_(bias)
            return out.reshape(*shape, n)
        # The engine owns these tensors; the AOT operators never allocate.
        # PyTorch's graph memory pool retains their captured addresses.
        codes = torch.empty((m, k), dtype=torch.int8, device=x.device)
        scales = torch.empty((m,), dtype=torch.float32, device=x.device)
        workspace = torch.empty((m, n), dtype=torch.int32, device=x.device)
        self._quantize_rows(x, codes, scales)
        self._channel_mm(
            codes, layer.weight, scales, layer.weight_scale_inv, workspace, out
        )
        if bias is not None:
            out.add_(bias)
        return out.reshape(*shape, n)


@register_weight_loader_v2_supported_method
class QwenVisionInt8LinearMethod(VisionInt8LinearMethod):
    def checkpoint_required_params(self, layer: torch.nn.Module) -> frozenset[str]:
        return frozenset({"weight", "weight_scale"})

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        if input_size_per_partition != input_size and input_size_per_partition % 32:
            raise ValueError(
                "Qwen vision INT8 groups cross TP shard boundaries; "
                "use --mm-encoder-tp-mode data"
            )
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )
