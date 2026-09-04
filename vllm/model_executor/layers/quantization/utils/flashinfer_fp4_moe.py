# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility helpers for NVFP4 + FlashInfer fused-MoE path"""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    align_fp4_moe_weights_for_fi,
    align_trtllm_fp4_moe_hidden_dim_for_fi,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    swizzle_blockscale,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    amax_for_moe_activation_quant,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        NvFp4MoeBackend,
    )

logger = init_logger(__name__)

__all__ = [
    "merge_nvfp4_gate_up_input_scales",
    "reorder_w1w3_to_w3w1",
    "require_uniform_nvfp4_expert_scale",
]


def merge_nvfp4_gate_up_input_scales(scale: torch.Tensor) -> torch.Tensor:
    """Return one input scale per expert without collapsing experts."""
    if scale.ndim != 2 or scale.shape[1] != 2:
        raise ValueError(
            "NVFP4 gate/up input scale must have shape (num_experts, 2), got "
            f"{tuple(scale.shape)}"
        )
    scale = scale.to(torch.float32)
    if not torch.all(torch.isfinite(scale) & (scale > 0)):
        raise ValueError("NVFP4 gate/up input scale must be finite and positive")
    if not torch.equal(scale[:, 0], scale[:, 1]):
        raise ValueError("NVFP4 gate and up input scales must match per expert")
    return scale[:, 0].contiguous()


def require_uniform_nvfp4_expert_scale(
    scale: torch.Tensor,
    *,
    num_local_experts: int,
    name: str,
) -> torch.Tensor:
    """Fail closed when a backend cannot represent per-expert input scales."""
    if scale.ndim != 1:
        raise ValueError(f"{name} must have shape (num_experts,), got {scale.shape}")
    scale = scale.to(torch.float32)
    if not torch.all(torch.isfinite(scale) & (scale > 0)):
        raise ValueError(f"{name} must be finite and positive")
    if not torch.equal(scale, scale[:1].expand_as(scale)):
        raise ValueError(
            f"{name} differs across experts, but this NVFP4 backend only supports "
            "one activation scale"
        )
    return scale[:1].expand(num_local_experts).contiguous()


def reorder_w1w3_to_w3w1(
    weight: torch.Tensor, scale: torch.Tensor, dim: int = -2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-order concatenated `[w1, w3]` tensors to `[w3, w1]` in-place.

    `weight` and `scale` must be contiguous; they remain contiguous on return.
    """
    assert weight.is_contiguous(), "weight must be contiguous"
    assert scale.is_contiguous(), "scale must be contiguous"
    size = weight.size(dim)
    assert size % 2 == 0, f"Expected even size in dim {dim}, got {size}"
    half = size // 2
    d = dim % weight.dim()

    # 64 MB transient cap
    bytes_per_row = max(
        weight.numel() // size * weight.element_size(),
        scale.numel() // size * scale.element_size(),
    )
    chunk = max(1, min(half, (64 << 20) // max(bytes_per_row, 1)))

    fa, fb = [slice(None)] * weight.dim(), [slice(None)] * weight.dim()
    for off in range(0, half, chunk):
        end = min(off + chunk, half)
        fa[d], fb[d] = slice(off, end), slice(half + off, half + end)
        a, b = tuple(fa), tuple(fb)
        for t in (weight, scale):
            tmp = t[b].clone()
            t[b] = t[a]
            t[a] = tmp

    return weight, scale


def interleave_linear_and_gate(
    x: torch.Tensor,
    group_size: int = 64,
    dim: int = -1,
) -> torch.Tensor:
    """Interleave gate and linear weight rows for CuteDSL wrapper."""
    sizes = x.size()
    dim = dim % x.dim()
    assert sizes[dim] % (group_size * 2) == 0, (
        f"dim {dim} size {sizes[dim]} must be divisible by {group_size * 2}"
    )
    prev_sizes = sizes[:dim]
    post_sizes = sizes[dim + 1 :]
    x = x.view(*prev_sizes, 2, sizes[dim] // (group_size * 2), group_size, *post_sizes)
    x = x.transpose(dim, dim + 1).contiguous().view(*sizes)
    return x


def reorder_w13_to_w31_for_flashinfer_cutedsl(
    activation: MoEActivation,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize gated w13 rows to the [up; gate] order used by FlashInfer."""
    if activation == MoEActivation.SWIGLUOAI:
        # gpt-oss checkpoints store w13 interleaved as [gate0, up0, gate1, ...].
        gate, up = w13[:, 0::2], w13[:, 1::2]
        gate_scale, up_scale = w13_scale[:, 0::2], w13_scale[:, 1::2]
        return (
            torch.cat([up, gate], dim=1).contiguous(),
            torch.cat([up_scale, gate_scale], dim=1).contiguous(),
        )

    half = w13.shape[1] // 2
    return (
        torch.cat([w13[:, half:], w13[:, :half]], dim=1).contiguous(),
        torch.cat([w13_scale[:, half:], w13_scale[:, :half]], dim=1).contiguous(),
    )


def prepare_nvfp4_moe_layer_for_flashinfer_cutedsl(
    layer: "RoutedExperts",
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_scale_2: torch.Tensor,
    a13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_scale_2: torch.Tensor,
    a2_scale: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Prepare weights for the CuteDSL wrapper-based NvFP4 MoE backend.

    Converts weight scale factors to MMA layout expected by CuteDslMoEWrapper,
    and interleaves w13 gate/linear rows for gated activations. Non-gated
    activations use a single w13 projection and keep its row order unchanged.
    """
    # This backend quantizes the shared activation once and therefore cannot
    # represent route-specific expert scales. Validate that the checkpoint's
    # per-expert scales really are uniform before collapsing them, then let
    # amax_for_moe_activation_quant do the EPLB-aware cross-rank reduction.
    num_experts = w13.shape[0]
    enable_eplb = layer.moe_config.moe_parallel_config.enable_eplb
    a13_scale = require_uniform_nvfp4_expert_scale(
        merge_nvfp4_gate_up_input_scales(a13_scale),
        num_local_experts=num_experts,
        name="a13_scale",
    )
    a2_scale = require_uniform_nvfp4_expert_scale(
        a2_scale,
        num_local_experts=num_experts,
        name="a2_scale",
    )
    a13_scale = amax_for_moe_activation_quant(a13_scale, enable_eplb).repeat(
        num_experts
    )
    a2_scale = amax_for_moe_activation_quant(a2_scale, enable_eplb).repeat(num_experts)

    if layer.activation.is_gated:
        w13, w13_scale = reorder_w13_to_w31_for_flashinfer_cutedsl(
            layer.activation, w13, w13_scale
        )

        # Interleave up/gate rows for w13 weights and scales.
        w13 = interleave_linear_and_gate(w13, group_size=64, dim=1)
        w13_scale = interleave_linear_and_gate(w13_scale, group_size=64, dim=1)

    w13_scale = swizzle_blockscale(w13_scale)
    w2_scale = swizzle_blockscale(w2_scale)

    return (
        w13,
        w13_scale,
        w13_scale_2,
        a13_scale,
        w2,
        w2_scale,
        w2_scale_2,
        a2_scale,
    )


def nvfp4_swizzled_scale_to_cutedsl_mma_view(scale: torch.Tensor) -> torch.Tensor:
    """View a swizzled (E, M_padded, K_sf_padded) block-scale tensor in the
    MMA layout expected by the CuteDSL MoE kernel.

    The returned tensor aliases `scale`'s storage, so in-place updates of the
    registered Parameter (weight reloads, EPLB rearrangement) are visible to
    the kernel with no extra bookkeeping.
    """
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    num_experts, m_padded, k_sf_padded = scale.shape
    mma_view = convert_sf_to_mma_layout(
        scale.reshape(num_experts * m_padded, k_sf_padded),
        m=m_padded,
        k=k_sf_padded * 16,
        num_groups=num_experts,
        sf_vec_size=16,
    )
    assert mma_view.data_ptr() == scale.data_ptr(), (
        "convert_sf_to_mma_layout no longer returns a view of its input; "
        "the quant config would go stale after weight updates."
    )
    return mma_view


def prepare_static_weights_for_trtllm_fp4_moe(
    # args_dequant,
    # args,
    gemm1_weights,
    gemm2_weights,
    gemm1_scales_linear_fp4_bytes,
    gemm2_scales_linear_fp4_bytes,
    hidden_size,
    intermediate_size,
    num_experts,
    is_gated_activation: bool,
):
    from flashinfer import nvfp4_block_scale_interleave
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    _cache_permute_indices: dict[torch.Size, torch.Tensor] = {}
    """Prepare quantized weights for kernel (done offline with weights)."""
    epilogue_tile_m = 128  # FIXME: this depends on the kernel internals
    gemm1_intermediate_size = (
        2 * intermediate_size if is_gated_activation else intermediate_size
    )

    # Convert quantized weights to proper formats
    gemm1_weights_fp4 = gemm1_weights.view(torch.float8_e4m3fn).reshape(
        num_experts, gemm1_intermediate_size, hidden_size // 2
    )  # packed fp4
    gemm1_scales_linear_fp4 = gemm1_scales_linear_fp4_bytes.view(
        torch.float8_e4m3fn
    ).reshape(
        num_experts, gemm1_intermediate_size, hidden_size // 16
    )  # fp8 scaling factors

    gemm2_weights_fp4 = gemm2_weights.view(torch.float8_e4m3fn).reshape(
        num_experts, hidden_size, intermediate_size // 2
    )  # packed fp4
    gemm2_scales_linear_fp4 = gemm2_scales_linear_fp4_bytes.view(
        torch.float8_e4m3fn
    ).reshape(num_experts, hidden_size, intermediate_size // 16)  # fp8 scaling factors

    gemm1_weights_fp4_shuffled = []
    gemm1_scales_fp4_shuffled = []
    gemm2_weights_fp4_shuffled = []
    gemm2_scales_fp4_shuffled = []
    for i in range(num_experts):
        # Calculate the permute indices for the following:
        # 1. Reorder rows of W1 and scales for fused gated activation
        # 2. Shuffle weights and scaling factors for transposed mma output
        # for both w3_w1 and w2 weights and scale factors
        permute_indices = _maybe_get_cached_w3_w1_permute_indices(
            _cache_permute_indices,
            gemm1_weights_fp4[i].view(torch.uint8),
            epilogue_tile_m,
            is_gated_act_gemm=is_gated_activation,
        )
        gemm1_weights_fp4_shuffled.append(
            gemm1_weights_fp4[i]
            .view(torch.uint8)[permute_indices.to(gemm1_weights_fp4.device)]
            .contiguous()
        )

        permute_sf_indices = _maybe_get_cached_w3_w1_permute_indices(
            _cache_permute_indices,
            gemm1_scales_linear_fp4[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
            is_gated_act_gemm=is_gated_activation,
        )
        gemm1_scales_fp4_shuffled.append(
            nvfp4_block_scale_interleave(
                gemm1_scales_linear_fp4[i]
                .view(torch.uint8)[
                    permute_sf_indices.to(gemm1_scales_linear_fp4.device)
                ]
                .contiguous()
            )
        )

        permute_indices = get_w2_permute_indices_with_cache(
            _cache_permute_indices,
            gemm2_weights_fp4[i].view(torch.uint8),
            epilogue_tile_m,
        )
        gemm2_weights_fp4_shuffled.append(
            gemm2_weights_fp4[i]
            .view(torch.uint8)[permute_indices.to(gemm2_weights_fp4.device)]
            .contiguous()
        )

        permute_sf_indices = get_w2_permute_indices_with_cache(
            _cache_permute_indices,
            gemm2_scales_linear_fp4[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
        )
        gemm2_scales_fp4_shuffled.append(
            nvfp4_block_scale_interleave(
                gemm2_scales_linear_fp4[i]
                .view(torch.uint8)[
                    permute_sf_indices.to(gemm2_scales_linear_fp4.device)
                ]
                .contiguous()
            )
        )

    # Stack weights for all experts
    gemm1_weights_fp4_shuffled = torch.stack(gemm1_weights_fp4_shuffled)
    gemm1_scales_fp4_shuffled = (
        torch.stack(gemm1_scales_fp4_shuffled)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, gemm1_intermediate_size, hidden_size // 16)
    )

    gemm2_weights_fp4_shuffled = torch.stack(gemm2_weights_fp4_shuffled)
    gemm2_scales_fp4_shuffled = (
        torch.stack(gemm2_scales_fp4_shuffled)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, hidden_size, intermediate_size // 16)
    )
    return (
        gemm1_weights_fp4_shuffled,
        gemm1_scales_fp4_shuffled,
        gemm2_weights_fp4_shuffled,
        gemm2_scales_fp4_shuffled,
    )


def prepare_nvfp4_moe_layer_for_fi_or_cutlass(
    backend: "NvFp4MoeBackend",
    layer: "RoutedExperts",
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_scale_2: torch.Tensor,
    a13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_scale_2: torch.Tensor,
    a2_scale: torch.Tensor,
    is_act_and_mul: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    # Delayed import for circular dependency avoidance.
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        NvFp4MoeBackend,
        is_global_sf_supported_for_nvfp4_backend,
    )

    assert backend in [
        NvFp4MoeBackend.VLLM_CUTLASS,
        NvFp4MoeBackend.FLASHINFER_CUTLASS,
        NvFp4MoeBackend.FLASHINFER_TRTLLM,
        NvFp4MoeBackend.FLASHINFER_CUTEDSL_BATCHED,
        NvFp4MoeBackend.FLASHINFER_B12X,
    ]

    # Reorder [w1, w3] to [w3, w1] for FI NVFP4 MoE kernels.
    is_gated = layer.activation.is_gated
    if (
        is_gated
        and is_act_and_mul
        and backend
        in [
            NvFp4MoeBackend.FLASHINFER_CUTLASS,
            NvFp4MoeBackend.FLASHINFER_TRTLLM,
            NvFp4MoeBackend.FLASHINFER_B12X,
        ]
    ):
        w13, w13_scale = reorder_w1w3_to_w3w1(w13, w13_scale)

    a13_scale = merge_nvfp4_gate_up_input_scales(a13_scale)

    # For some FI kernels, the input scales are shared by all experts.
    if is_global_sf_supported_for_nvfp4_backend(backend):
        num_experts = w13.shape[0]
        # Fail closed on genuinely per-expert scales, then reduce EPLB-aware.
        # The `else` branch upstream keeps here (a13_scale.max(dim=1)) is gone:
        # merge_nvfp4_gate_up_input_scales above already collapsed the gate/up
        # pair to one validated scale per expert, so a13_scale is 1-D by now.
        enable_eplb = layer.moe_config.moe_parallel_config.enable_eplb
        a13_scale = require_uniform_nvfp4_expert_scale(
            a13_scale,
            num_local_experts=num_experts,
            name="a13_scale",
        )
        a2_scale = require_uniform_nvfp4_expert_scale(
            a2_scale,
            num_local_experts=num_experts,
            name="a2_scale",
        )
        a13_scale = amax_for_moe_activation_quant(a13_scale, enable_eplb).repeat(
            num_experts
        )
        a2_scale = amax_for_moe_activation_quant(a2_scale, enable_eplb).repeat(
            num_experts
        )

    # Shuffle weights and scales for FI TRTLLM NVFP4 MoE kernels.
    if backend == NvFp4MoeBackend.FLASHINFER_TRTLLM:
        w13, w13_scale, w2, w2_scale, padded_hidden = (
            align_trtllm_fp4_moe_hidden_dim_for_fi(w13, w13_scale, w2, w2_scale)
        )
        if layer.moe_config.hidden_dim_unpadded is None:
            layer.moe_config.hidden_dim_unpadded = layer.moe_config.hidden_dim
        layer.moe_config.hidden_dim = padded_hidden

        # Align weights for FI NVFP4 MoE kernels.
        # FlashInfer's TRT-LLM block-scale shuffle asserts the gate/up row dim
        # (= up_mult * padded_intermediate, up_mult=2 when gated) is a multiple of
        # 128. So gated needs padded_intermediate % 64 (2*64=128); the old value 16
        # left 2*intermediate a multiple of only 32, so an NVFP4 MoE whose rank-local
        # intermediate is not 128-aligned at TP>1 (e.g. Gemma-4-26B-A4B at tp4) hit
        # `assert M % 128 == 0`. Padded rows are zero -> outputs unchanged.
        min_alignment = 64 if is_gated else 128
        w13, w13_scale, w2, w2_scale, padded_intermediate = (
            align_fp4_moe_weights_for_fi(
                w13, w13_scale, w2, w2_scale, is_act_and_mul, min_alignment
            )
        )
        layer.moe_config.intermediate_size_per_partition = padded_intermediate

        w13, w13_scale, w2, w2_scale = prepare_static_weights_for_trtllm_fp4_moe(
            w13,
            w2,
            w13_scale,
            w2_scale,
            hidden_size=w2.size(-2),
            intermediate_size=w13.size(-2) // 2 if is_gated else w13.size(-2),
            num_experts=w13.size(0),
            is_gated_activation=is_gated,
        )
    else:
        # Swizzle the block scales for other FI NVFP4 MoE kernels.
        w13_scale = swizzle_blockscale(w13_scale)

        # Apply padding if needed.
        pad_size = w13_scale.size(1) - w13.size(1)
        if pad_size > 0:
            if is_act_and_mul:
                raise NotImplementedError(
                    "Intermediate size padding for w1 and w3, for %s "
                    "NvFp4 backend, but this is not currently supported",
                    backend.value,
                )
            w13 = torch.nn.functional.pad(w13, (0, 0, 0, pad_size))
            w2 = torch.nn.functional.pad(w2, (0, pad_size // 2, 0, 0))
            w2_scale = torch.nn.functional.pad(w2_scale, (0, pad_size // 16))

        w2_scale = swizzle_blockscale(w2_scale)

    return w13, w13_scale, w13_scale_2, a13_scale, w2, w2_scale, w2_scale_2, a2_scale
