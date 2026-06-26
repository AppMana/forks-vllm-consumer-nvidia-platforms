# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.model_executor.layers.rotary_embedding.common import (
    rotate_gptj,
    rotate_neox,
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


def compute_fp8_einsum_recipe() -> tuple[tuple[int, int, int], bool]:
    """fp8_einsum recipe + scale layout for the current GPU arch.

    SM90: FP32 block scales stay [g, r/128, d/128] → sfb_gran_mn=128.
    SM100: INT32 packed scales become [g, r, ...] → sfb_gran_mn=1.

    Returns ``(einsum_recipe, tma_aligned_scales)`` for ``deep_gemm_fp8_o_proj``.
    """
    cap = _DSV4_DEVICE_CAP
    assert cap is not None, "DeepseekV4 attention requires a CUDA device"
    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)
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
    """O projection: inverse RoPE + FP8 quant + einsum + wo_b.

    Shared by the FlashMLA and FlashInfer CUDA backends. ``einsum_recipe`` /
    ``tma_aligned_scales`` come from ``compute_fp8_einsum_recipe``.
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
    o_fp8, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        tma_aligned_scales=tma_aligned_scales,
    )
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    # DeepGEMM fp8_einsum is Hopper/sm_100 only. On Ampere (sm_8x) and consumer
    # Blackwell (sm_12x) use the software fp8 einsum (triton on sm_89+, torch
    # fallback on sm_86), which computes the same "bhr,hdr->bhd" contraction.
    cap = _DSV4_DEVICE_CAP
    if cap is not None and cap.major in (8, 12):
        from vllm.models.deepseek_v4.common.ops.fp8_einsum import (
            deepseek_v4_sm12x_fp8_einsum,
        )
        from vllm.models.deepseek_v4.nvidia_sm86.triton_kernels import (
            _normalize_deepseek_v4_fp8_einsum_inputs,
        )

        # Reshape wo_a (2D -> [groups, out_rank, hidden]) + unpack scales, then
        # run the software einsum (torch on sm_86, triton on sm_12x).
        a, a_scale, b, b_scale = _normalize_deepseek_v4_fp8_einsum_inputs(
            o_fp8, o_scale, wo_a.weight, wo_a.weight_scale_inv, z
        )
        deepseek_v4_sm12x_fp8_einsum(a, a_scale, b, b_scale, z)
    else:
        fp8_einsum(
            "bhr,hdr->bhd",
            (o_fp8, o_scale),
            (wo_a.weight, wo_a.weight_scale_inv),
            z,
            recipe=einsum_recipe,
        )
    return wo_b(z.flatten(1))
