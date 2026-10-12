# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch
import torch.nn as nn

from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
)
from vllm.model_executor.layers.rotary_embedding.common import (
    rotate_gptj,
    rotate_neox,
)
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_einsum, get_tma_aligned_size
from vllm.utils.torch_utils import direct_register_custom_op

# Device capability is constant per process. Compute it once at import so the
# o_proj forward references a Python constant instead of calling the (C-level,
# Dynamo-untraceable) current_platform.get_device_capability() inside the graph.
_DSV4_DEVICE_CAP = current_platform.get_device_capability()


def deepseek_v4_inv_rope_woa(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a_weight: torch.Tensor,
    out: torch.Tensor,
    rope_head_dim: int,
    n_local_groups: int,
    o_lora_rank: int,
    is_neox_style: bool,
) -> None:
    """Inverse-RoPE + BF16 wo_a einsum for the dsv4_int (INT8 wo_a) path.

    The dsv4_int checkpoint stores wo_a as INT8 and dequantizes it once to
    BF16 at load time (`_dsv4_int_dequanted`), so it must not go through the
    FP8 einsum. Computes out = einsum("tgd,grd->tgr", inv_rope(o), wo_a).
    """
    head_size = o.shape[-1]
    nope_dim = head_size - rope_head_dim
    o_pass = o[..., :nope_dim] if nope_dim > 0 else None
    o_rot = o[..., nope_dim:]

    safe_positions = positions.clamp(0, cos_sin_cache.shape[0] - 1)
    cos_sin = cos_sin_cache[safe_positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    if is_neox_style:
        cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2)
        sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2)
        rotate_fn = rotate_neox
    else:
        cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
        sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)
        rotate_fn = rotate_gptj
    o_rot = (o_rot.float() * cos - rotate_fn(o_rot.float()) * sin).to(o.dtype)
    o_ref = torch.cat((o_pass, o_rot), dim=-1) if o_pass is not None else o_rot
    o_ref = o_ref.view(o.shape[0], n_local_groups, -1).to(torch.bfloat16)
    wo_a = wo_a_weight.view(n_local_groups, o_lora_rank, o_ref.shape[-1]).to(
        torch.bfloat16
    )
    out.copy_(torch.einsum("tgd,grd->tgr", o_ref, wo_a))


def _deepseek_v4_inv_rope_woa_fake(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a_weight: torch.Tensor,
    out: torch.Tensor,
    rope_head_dim: int,
    n_local_groups: int,
    o_lora_rank: int,
    is_neox_style: bool,
) -> None:
    return None


direct_register_custom_op(
    op_name="deepseek_v4_inv_rope_woa",
    op_func=deepseek_v4_inv_rope_woa,
    mutates_args=["out"],
    fake_impl=_deepseek_v4_inv_rope_woa_fake,
)


def compute_fp8_einsum_recipe(
    block_size: int = 128,
) -> tuple[tuple[int, int, int], bool]:
    """fp8_einsum recipe + scale layout for the current GPU arch.

    SM90 keeps block-row FP32 scales. SM100 uses packed per-row E8M0 scales.

    Returns ``(einsum_recipe, tma_aligned_scales)`` for ``deep_gemm_fp8_o_proj``.
    """
    cap = _DSV4_DEVICE_CAP
    assert cap is not None, "DeepseekV4 attention requires a CUDA device"
    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, block_size)
    tma_aligned_scales = cap.major >= 10
    return einsum_recipe, tma_aligned_scales


def deep_gemm_fp8_o_proj(
    o: torch.Tensor | QuantizedActivation,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: Callable[[torch.Tensor], torch.Tensor],
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
    tma_aligned_scales: bool,
    is_neox_style: bool = False,
) -> torch.Tensor:
    """O projection: inverse RoPE + grouped wo_a + wo_b.

    Shared by the FlashMLA and FlashInfer CUDA backends. The attention
    layer selects the recipe at initialization. ``wo_b`` is any callable over
    the flattened ``z``: the projection module itself, or a wrapper that also
    reduce-scatters its output (DeepSeek-V4.1 GEMM-RS).

    A QuantizedActivation ``o`` was already rotated and cast by the attention
    kernel, one ``wo_a`` group per slot with the live groups first, so only
    the padding groups are sliced off.
    """
    # dsv4_int stores wo_a as INT8 and dequantizes it to BF16 once at load
    # (`_dsv4_int_dequanted`); that path uses inverse-RoPE + a BF16 einsum, not
    # the FP8 einsum (which asserts fp8 weights).
    if getattr(wo_a, "_dsv4_int_dequanted", False):
        assert isinstance(o, torch.Tensor), "INT8 wo_a requires BF16 attention output"
        z = torch.empty(
            (o.shape[0], n_groups, o_lora_rank),
            dtype=torch.bfloat16,
            device=o.device,
        )
        torch.ops.vllm.deepseek_v4_inv_rope_woa(
            o,
            positions,
            cos_sin_cache,
            wo_a.weight,
            z,
            rope_dim,
            n_groups,
            o_lora_rank,
            is_neox_style,
        )
        return wo_b(z.flatten(1))
    use_fp8 = wo_a.weight.dtype == torch.float8_e4m3fn
    if isinstance(o, QuantizedActivation):
        assert use_fp8
        o_proj_input, o_scale = o.data[:, :n_groups], o.scale[:, :n_groups]
    else:
        o_proj_input, o_scale = fused_inv_rope_fp8_quant(
            o,
            positions,
            cos_sin_cache,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            quant_group_size=einsum_recipe[2],
            tma_aligned_scales=tma_aligned_scales,
            quantize=use_fp8,
        )
    z = torch.empty(
        (o_proj_input.shape[0], n_groups, o_lora_rank),
        device=o_proj_input.device,
        dtype=torch.bfloat16,
    )
    if use_fp8:
        weight_scale = (
            wo_a.weight_scale
            if hasattr(wo_a, "weight_scale")
            else wo_a.weight_scale_inv
        )
        # DeepGEMM fp8_einsum is Hopper/sm_100 only. On Ampere (sm_8x) and
        # consumer Blackwell (sm_12x) use the software fp8 einsum (triton on
        # sm_89+, torch fallback on sm_86), the same "bhr,hdr->bhd" contraction.
        cap = _DSV4_DEVICE_CAP
        if cap is not None and cap.major in (8, 12):
            from vllm.models.deepseek_v4.common.ops.fp8_einsum import (
                deepseek_v4_sm12x_fp8_einsum,
            )
            from vllm.models.deepseek_v4.nvidia_imma.triton_kernels import (
                _normalize_deepseek_v4_fp8_einsum_inputs,
            )

            a, a_scale, b, b_scale = _normalize_deepseek_v4_fp8_einsum_inputs(
                o_proj_input, o_scale, wo_a.weight, weight_scale, z
            )
            deepseek_v4_sm12x_fp8_einsum(a, a_scale, b, b_scale, z)
        else:
            fp8_einsum(
                "bhr,hdr->bhd",
                (o_proj_input, o_scale),
                (wo_a.weight, weight_scale),
                z,
                recipe=einsum_recipe,
            )
    else:
        grouped_weight = wo_a.weight.view(n_groups, o_lora_rank, -1)
        torch.bmm(
            o_proj_input.transpose(0, 1),
            grouped_weight.transpose(1, 2),
            out=z.transpose(0, 1),
        )
    return wo_b(z.flatten(1))


# TRTLLM-gen's DSv4 RopeQuant epilogue (flashinfer-ai/flashinfer#4918) has one
# fixed schedule: 128 query heads in groups of 8 512-wide heads, and FP8 with
# one packed UE8M0 scale per 128 elements.
_ROPE_QUANT_NUM_HEADS = 128
_ROPE_QUANT_HEADS_PER_GROUP = 8
_ROPE_QUANT_HEAD_DIM = 512
_ROPE_QUANT_BLOCK = 128


def rope_quant_unsupported_reason(layer: torch.nn.Module) -> str | None:
    """Why ``layer`` cannot take FlashInfer's fused inverse RoPE + FP8 output.

    DSv4 always has 8 heads per ``wo_a`` group at any TP size, and the rest of
    the kernel's fixed shape (head/RoPE dims, per-128 einsum scales, FP32 RoPE
    cache) is what the SM100 layer uses anyway. What varies is the local head
    count and the configs the layer also serves unquantized.
    """
    if layer.n_local_heads != _ROPE_QUANT_NUM_HEADS:
        # Padding fewer heads up to the kernel's fixed 128 costs more
        # attention than the fused epilogue saves.
        return f"RopeQuant needs {_ROPE_QUANT_NUM_HEADS} local query heads"
    if layer.kv_cache_torch_dtype != torch.float8_e4m3fn:
        return "RopeQuant needs the per-tensor FP8 KV cache"
    if layer.wo_a.weight.dtype != torch.float8_e4m3fn:
        return "RopeQuant needs an FP8 wo_a"
    return None


def rope_quant_attn_out(num_tokens: int, device: torch.device) -> QuantizedActivation:
    """Allocate the output pair the RopeQuant kernel writes.

    Values are group-major, ``[num_tokens, 16, 8 * 512]`` over physical
    ``[group, token, K]``. Scales are MN-major packed UE8M0, one int32 per
    head over its four 128-wide blocks. Both are what ``fp8_einsum`` consumes.
    """
    n_groups = _ROPE_QUANT_NUM_HEADS // _ROPE_QUANT_HEADS_PER_GROUP
    group_width = _ROPE_QUANT_HEADS_PER_GROUP * _ROPE_QUANT_HEAD_DIM
    data = torch.empty(
        (n_groups, num_tokens, group_width), dtype=torch.float8_e4m3fn, device=device
    ).transpose(0, 1)
    aligned = get_tma_aligned_size(num_tokens, torch.int32.itemsize)
    scale = torch.empty(
        (n_groups, group_width // (_ROPE_QUANT_BLOCK * 4), aligned),
        dtype=torch.int32,
        device=device,
    ).permute(2, 0, 1)[:num_tokens]
    return QuantizedActivation(
        data=data,
        scale=scale,
        orig_dtype=torch.bfloat16,
        orig_shape=data.shape,
        quant_key=kFp8Dynamic128Sym,
    )
