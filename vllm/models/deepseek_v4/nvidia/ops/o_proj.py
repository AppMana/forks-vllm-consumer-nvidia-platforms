# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.model_executor.layers.rotary_embedding.common import (
    rotate_gptj,
    rotate_neox,
)
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_einsum
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
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
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
    layer selects the recipe at initialization.
    """
    # dsv4_int stores wo_a as INT8 and dequantizes it to BF16 once at load
    # (`_dsv4_int_dequanted`); that path uses inverse-RoPE + a BF16 einsum, not
    # the FP8 einsum (which asserts fp8 weights).
    if getattr(wo_a, "_dsv4_int_dequanted", False):
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
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
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
