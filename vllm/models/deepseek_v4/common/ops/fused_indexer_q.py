# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.import_utils import has_cutedsl

from .fp8e4m3_arith import fp8e4m3_encode_from_fp32

# MXFP4: 32 elements per block, packed 2 nibbles per byte, ue8m0 block scale.
MXFP4_BLOCK_SIZE = 32


@triton.jit
def _get_cos_sin(
    cos_sin_cache_ptr,
    cos_sin_cache_stride,
    pos,
    HALF_ROT_DIM: tl.constexpr,
):
    block = tl.arange(0, HALF_ROT_DIM)
    cos = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block)
    cos = cos.to(tl.float32)
    sin = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block + HALF_ROT_DIM)
    sin = sin.to(tl.float32)
    return cos, sin


@triton.jit
def _fp32x2_to_fp4x2(x_lo, x_hi):
    # NOTE: $1 is high nibble, $2 is low nibble
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b8 tmp;
            cvt.rn.satfinite.e2m1x2.f32 tmp, $1, $2;
            cvt.u32.u8 $0, tmp;
        }
        """,
        constraints="=r,f,f",
        args=[x_hi, x_lo],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)


@triton.jit
def _quantize_mxfp4_pair(x_lo, x_hi):
    """Quantize a block of MXFP4_BLOCK_SIZE fp32 values given as two
    interleaved halves (x_lo = values at even positions in the block,
    x_hi = values at odd positions). Returns:
        - packed : uint8[BLOCK/2]  (low nibble = quant(x_lo), high = quant(x_hi))
        - ue8m0  : scalar uint8    (block scale = 2^(ue8m0 - 127))
    """
    amax = tl.maximum(tl.max(tl.abs(x_lo)), tl.max(tl.abs(x_hi)))
    # 6 * 2^-126 is from https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/kernel.py#L163
    amax = tl.maximum(amax, 6.0 * (2**-126))
    # ue8m0 block scale: 2^ceil(log2(amax/6.0)).
    log2_ratio = tl.math.ceil(tl.math.log2(amax * (1.0 / 6.0)))
    log2_ratio = tl.minimum(tl.maximum(log2_ratio, -127.0), 127.0)
    scale = tl.math.exp2(log2_ratio)
    ue8m0 = (log2_ratio + 127.0).to(tl.uint8)

    inv_scale = 1.0 / scale
    packed = _fp32x2_to_fp4x2(x_lo * inv_scale, x_hi * inv_scale)
    return packed, ue8m0


@triton.jit
def _round_clamp_int8(v):
    """Round-half-away-from-zero (portable, no libdevice), clamp to symmetric
    INT8 [-127, 127], return int8. Matches torch.round/rintf closely enough for
    the indexer's quantization recall."""
    r = tl.where(v >= 0, tl.math.floor(v + 0.5), -tl.math.floor(-v + 0.5))
    r = tl.minimum(tl.maximum(r, -127.0), 127.0)
    return r.to(tl.int8)


@triton.jit
def _fused_indexer_q_rope_quant_kernel(
    pos_ptr,
    # Index Q RoPE
    index_q_ptr,
    index_q_stride0,
    index_q_stride1,
    index_q_cos_sin_ptr,
    index_q_cos_sin_stride,
    INDEX_Q_HALF_ROT_DIM: tl.constexpr,
    # Index Q Quantize
    index_q_fp8_ptr,  # uint8 view of the float8 output tensor on sm_8x
    index_q_fp8_stride0,
    index_q_fp8_stride1,
    INDEX_Q_HEAD_DIM: tl.constexpr,
    # Index weights
    index_weights_ptr,
    index_weights_stride,
    index_weights_softmax_scale,
    index_weights_head_scale,
    index_weights_out_ptr,
    index_weights_out_stride,
    QK_INT8: tl.constexpr = False,
):
    # Layout matches the unfused reference (DeepseekV4ScalingRotaryEmbedding
    # + per_token_group_quant_fp8): GPT-J interleaved RoPE applied to the
    # LAST rope_dim dims of each head; the leading [0, NOPE_DIM) is passed
    # through unchanged.
    INDEX_Q_ROT_DIM: tl.constexpr = 2 * INDEX_Q_HALF_ROT_DIM
    INDEX_Q_NOPE_DIM: tl.constexpr = INDEX_Q_HEAD_DIM - INDEX_Q_ROT_DIM
    tl.static_assert(INDEX_Q_NOPE_DIM >= 0)

    tok_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    pos = tl.load(pos_ptr + tok_idx)
    cos, sin = _get_cos_sin(
        index_q_cos_sin_ptr,
        index_q_cos_sin_stride,
        pos,
        INDEX_Q_HALF_ROT_DIM,
    )
    half_offset = tl.arange(0, INDEX_Q_HALF_ROT_DIM)
    base_ptr = index_q_ptr + tok_idx * index_q_stride0 + head_idx * index_q_stride1

    # Interleaved (GPT-J) RoPE on dims [NOPE_DIM, HEAD_DIM):
    #   even = q[NOPE_DIM + 2*i],  odd = q[NOPE_DIM + 2*i + 1]
    rot_base = base_ptr + INDEX_Q_NOPE_DIM
    x_even = tl.load(rot_base + half_offset * 2).to(tl.float32)
    x_odd = tl.load(rot_base + half_offset * 2 + 1).to(tl.float32)
    r_even = x_even * cos - x_odd * sin
    r_odd = x_odd * cos + x_even * sin

    # Match reference numerics: fp32 → bf16 → fp32 before the ue8m0 absmax.
    # Same pattern as the K-side compressor kernel (fused_compress_quant_cache.py).
    r_even = r_even.to(tl.bfloat16).to(tl.float32)
    r_odd = r_odd.to(tl.bfloat16).to(tl.float32)

    amax = tl.maximum(tl.max(tl.abs(r_even)), tl.max(tl.abs(r_odd)))
    if INDEX_Q_NOPE_DIM > 0:
        nope_offset = tl.arange(0, INDEX_Q_NOPE_DIM)
        x_nope = tl.load(base_ptr + nope_offset).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x_nope)))
    if QK_INT8:
        # Symmetric INT8 query for the s8 x s8 integer-MMA decode indexer.
        # Plain absmax/127 (matches the prefill use_imma path in deep_gemm and the
        # int8 K-cache writer's absmax scale); stored as int8 into an int8 tensor.
        index_q_scale = tl.maximum(amax, 1e-4) / 127.0
    else:
        index_q_scale = tl.div_rn(tl.maximum(amax, 1e-4), 448.0)
        index_q_scale = tl.math.exp2(tl.math.ceil(tl.math.log2(index_q_scale)))

    # Store quantized values to index_q (fp8 e4m3 bytes, or int8 when QK_INT8).
    fp8_base_ptr = (
        index_q_fp8_ptr + tok_idx * index_q_fp8_stride0 + head_idx * index_q_fp8_stride1
    )
    if INDEX_Q_NOPE_DIM > 0:
        qn = tl.div_rn(x_nope, index_q_scale)
        if QK_INT8:
            tl.store(fp8_base_ptr + nope_offset, _round_clamp_int8(qn))
        else:
            tl.store(fp8_base_ptr + nope_offset, fp8e4m3_encode_from_fp32(qn))
    fp8_rot_base = fp8_base_ptr + INDEX_Q_NOPE_DIM
    qe = tl.div_rn(r_even, index_q_scale)
    qo = tl.div_rn(r_odd, index_q_scale)
    if QK_INT8:
        tl.store(fp8_rot_base + half_offset * 2, _round_clamp_int8(qe))
        tl.store(fp8_rot_base + half_offset * 2 + 1, _round_clamp_int8(qo))
    else:
        tl.store(fp8_rot_base + half_offset * 2, fp8e4m3_encode_from_fp32(qe))
        tl.store(fp8_rot_base + half_offset * 2 + 1, fp8e4m3_encode_from_fp32(qo))

    # FP8 weight-fold contract:
    #   index_weights_out = index_weights * q_scale * softmax_scale * head_scale
    # The per-token-per-head q_scale (fp32) IS folded into the output weights
    # here because FP8 Q is stored WITHOUT a companion scale tensor — the
    # downstream fp8_fp4_mqa_logits/fp8_fp4_paged_mqa_logits kernels use `weights` to
    # apply per-token Q scale inline. See the MXFP4 kernel below for the
    # contrasting convention (scales live with the Q values, weights are NOT
    # q-scaled).
    index_weights = tl.load(
        index_weights_ptr + tok_idx * index_weights_stride + head_idx
    )
    index_weights = index_weights.to(tl.float32)
    index_weights *= index_q_scale
    index_weights *= index_weights_softmax_scale
    index_weights *= index_weights_head_scale
    tl.store(
        index_weights_out_ptr + tok_idx * index_weights_out_stride + head_idx,
        index_weights,
    )


@triton.jit
def _fused_indexer_q_rope_mxfp4_kernel(
    pos_ptr,
    # Index Q RoPE input (fp/bf16)
    index_q_ptr,
    index_q_stride0,
    index_q_stride1,
    index_q_cos_sin_ptr,
    index_q_cos_sin_stride,
    INDEX_Q_HALF_ROT_DIM: tl.constexpr,
    # MXFP4 Q outputs
    index_q_mxfp4_ptr,  # uint8, (T, H, HEAD_DIM // 2)
    index_q_mxfp4_stride0,
    index_q_mxfp4_stride1,
    index_q_scale_ptr,  # uint8 ue8m0, (T, H, HEAD_DIM // BLOCK)
    index_q_scale_stride0,
    index_q_scale_stride1,
    INDEX_Q_HEAD_DIM: tl.constexpr,
    MXFP4_BLOCK: tl.constexpr,
    # Weights (NO per-token q_scale fold for MXFP4; per-block scales stay
    # with the Q values in the output scale tensor).
    index_weights_ptr,
    index_weights_stride,
    index_weights_softmax_scale,
    index_weights_head_scale,
    index_weights_out_ptr,
    index_weights_out_stride,
):
    INDEX_Q_ROT_DIM: tl.constexpr = 2 * INDEX_Q_HALF_ROT_DIM
    INDEX_Q_NOPE_DIM: tl.constexpr = INDEX_Q_HEAD_DIM - INDEX_Q_ROT_DIM
    NUM_NOPE_BLOCKS: tl.constexpr = INDEX_Q_NOPE_DIM // MXFP4_BLOCK
    NUM_ROPE_BLOCKS: tl.constexpr = INDEX_Q_ROT_DIM // MXFP4_BLOCK
    HALF_BLOCK: tl.constexpr = MXFP4_BLOCK // 2
    tl.static_assert(INDEX_Q_NOPE_DIM >= 0)
    tl.static_assert(INDEX_Q_NOPE_DIM % MXFP4_BLOCK == 0)
    tl.static_assert(INDEX_Q_ROT_DIM % MXFP4_BLOCK == 0)
    tl.static_assert(MXFP4_BLOCK % 2 == 0)

    tok_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    pos = tl.load(pos_ptr + tok_idx)

    q_base = index_q_ptr + tok_idx * index_q_stride0 + head_idx * index_q_stride1
    out_base = (
        index_q_mxfp4_ptr
        + tok_idx * index_q_mxfp4_stride0
        + head_idx * index_q_mxfp4_stride1
    )
    scale_base = (
        index_q_scale_ptr
        + tok_idx * index_q_scale_stride0
        + head_idx * index_q_scale_stride1
    )

    half_off = tl.arange(0, HALF_BLOCK)

    # ---- NoPE blocks: direct load, pair as (even-index, odd-index) values ----
    for b in tl.static_range(NUM_NOPE_BLOCKS):
        base = b * MXFP4_BLOCK
        x_lo = tl.load(q_base + base + half_off * 2).to(tl.float32)
        x_hi = tl.load(q_base + base + half_off * 2 + 1).to(tl.float32)
        packed, ue8m0 = _quantize_mxfp4_pair(x_lo, x_hi)
        tl.store(out_base + base // 2 + half_off, packed)
        tl.store(scale_base + b, ue8m0)

    # ---- RoPE blocks: apply GPT-J interleaved RoPE to the block's 16 pairs,
    # then quantize. Each block covers HALF_BLOCK (=16) cos/sin pairs. ----
    rot_q_base = q_base + INDEX_Q_NOPE_DIM
    for b in tl.static_range(NUM_ROPE_BLOCKS):
        pair_off = b * HALF_BLOCK + half_off  # indices in [0, HALF_ROT_DIM)
        cos_b = tl.load(
            index_q_cos_sin_ptr + pos * index_q_cos_sin_stride + pair_off
        ).to(tl.float32)
        sin_b = tl.load(
            index_q_cos_sin_ptr
            + pos * index_q_cos_sin_stride
            + pair_off
            + INDEX_Q_HALF_ROT_DIM
        ).to(tl.float32)
        x_even = tl.load(rot_q_base + pair_off * 2).to(tl.float32)
        x_odd = tl.load(rot_q_base + pair_off * 2 + 1).to(tl.float32)
        r_even = x_even * cos_b - x_odd * sin_b
        r_odd = x_odd * cos_b + x_even * sin_b
        # bf16 roundtrip for parity with the FP8 kernel / reference numerics.
        r_even = r_even.to(tl.bfloat16).to(tl.float32)
        r_odd = r_odd.to(tl.bfloat16).to(tl.float32)
        packed, ue8m0 = _quantize_mxfp4_pair(r_even, r_odd)
        rope_byte_off = (INDEX_Q_NOPE_DIM + b * MXFP4_BLOCK) // 2
        tl.store(out_base + rope_byte_off + half_off, packed)
        tl.store(scale_base + NUM_NOPE_BLOCKS + b, ue8m0)

    # MXFP4 weight-fold contract:
    #   index_weights_out = index_weights * softmax_scale * head_scale
    # NOTE: q_scale is NOT folded here (contrast with the FP8 kernel above).
    # MXFP4 Q emits a separate ue8m0 scale tensor of shape
    # (T, H, HEAD_DIM // MXFP4_BLOCK) alongside the packed values, so each
    # per-block scale is applied by the downstream MXFP4 logits kernel when
    # dequantizing Q — there is no per-token scalar to fold into `weights`.
    index_weights = tl.load(
        index_weights_ptr + tok_idx * index_weights_stride + head_idx
    ).to(tl.float32)
    index_weights *= index_weights_softmax_scale
    index_weights *= index_weights_head_scale
    tl.store(
        index_weights_out_ptr + tok_idx * index_weights_out_stride + head_idx,
        index_weights,
    )


def _supports_fp8e4nv_in_triton() -> bool:
    """Same gate as fused_inv_rope_fp8_quant — Triton's tl.float8e4nv requires
    sm_89+. Fall back to torch on sm_8x (Ampere)."""
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        return True
    cap = current_platform.get_device_capability()
    if cap is None:
        return True
    return cap.major != 8


def _fused_indexer_q_rope_fp8_torch(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch fallback mirroring `_fused_indexer_q_rope_quant_kernel`.

    Steps:
      1. Apply forward RoPE (interleaved GPT-J style) to the trailing
         rope_dim elements of each head's query.
      2. Compute per-token-per-head q_scale = ceil(log2(amax/448))-rounded
         power-of-2 over the entire head_dim.
      3. Cast (q / q_scale) to torch.float8_e4m3fn (software cast on sm_8x).
      4. Fold q_scale * softmax_scale * head_scale into index_weights_out.

    Returns (index_q_fp8, index_weights_out) matching the kernel contract.
    """
    num_tokens, num_index_q_heads, index_q_head_dim = index_q.shape
    rope_half = index_q_cos_sin_cache.shape[-1] // 2
    rope_dim = rope_half * 2
    nope_dim = index_q_head_dim - rope_dim

    safe_positions = positions.clamp(0, index_q_cos_sin_cache.shape[0] - 1)
    cos_sin = index_q_cos_sin_cache[safe_positions].to(torch.float32)
    cos = cos_sin[:, :rope_half]  # (T, rope_half)
    sin = cos_sin[:, rope_half:]

    q_f32 = index_q.to(torch.float32)
    q_nope = q_f32[..., :nope_dim]
    q_rope = q_f32[..., nope_dim:]
    # The Triton kernel uses fwd RoPE (rope_q): pair (even, odd) -> (cos*even - sin*odd, cos*odd + sin*even).
    # Match the fp32 → bf16 → fp32 round-trip used by the Triton path so the
    # ue8m0 absmax matches bit-for-bit.
    x_even = q_rope[..., 0::2].to(torch.bfloat16).to(torch.float32)
    x_odd = q_rope[..., 1::2].to(torch.bfloat16).to(torch.float32)
    cos_b = cos[:, None, :]
    sin_b = sin[:, None, :]
    r_even = x_even * cos_b - x_odd * sin_b
    r_odd = x_odd * cos_b + x_even * sin_b
    r_even_bf16 = r_even.to(torch.bfloat16).to(torch.float32)
    r_odd_bf16 = r_odd.to(torch.bfloat16).to(torch.float32)
    rope_rotated = torch.empty_like(q_rope)
    rope_rotated[..., 0::2] = r_even_bf16
    rope_rotated[..., 1::2] = r_odd_bf16
    q_full = (
        torch.cat([q_nope, rope_rotated], dim=-1) if nope_dim > 0 else rope_rotated
    )

    # Per-token-per-head amax over full head_dim.
    fp8_max = 448.0
    amax = q_full.abs().amax(dim=-1).clamp_min(1e-4)  # (T, H)
    scale_raw = amax / fp8_max
    log2_scale = torch.log2(scale_raw).ceil()
    q_scale = torch.pow(2.0, log2_scale).to(torch.float32)  # (T, H)

    q_scaled = q_full / q_scale.unsqueeze(-1)
    q_clamped = q_scaled.clamp(-fp8_max, fp8_max)
    index_q_fp8 = q_clamped.to(torch.float8_e4m3fn)

    weights_f32 = index_weights.to(torch.float32)
    index_weights_out = (
        weights_f32 * q_scale * index_weights_softmax_scale * index_weights_head_scale
    )
    return index_q_fp8, index_weights_out


def fused_indexer_q_rope_quant(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    # Index weights
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
    use_fp4: bool = False,
    q_is_int8: bool = False,
) -> tuple[
    torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
]:
    """Fused RoPE + quantize Q for the sparse indexer.

    Weight-fold semantics (important — the two paths differ):

    FP8 path (use_fp4=False, default):
        q_fp8      : (T, H, HEAD_DIM) float8_e4m3fn, per-token-per-head
                     scalar scale (NOT stored — folded into weights below)
        weights_out = weights * q_scale * softmax_scale * head_scale
        Rationale: a single per-token q_scale is a scalar the downstream FP8
        logits kernel would otherwise multiply in. Folding it into `weights`
        avoids emitting a separate tensor and is free for the logits kernel.

    MXFP4 path (use_fp4=True):
        q_packed   : (T, H, HEAD_DIM // 2) uint8 (2 E2M1 nibbles per byte)
        q_scale    : (T, H, HEAD_DIM // MXFP4_BLOCK_SIZE) uint8 ue8m0 bytes
        weights_out = weights * softmax_scale * head_scale
        Rationale: MXFP4 has PER-BLOCK (32-element) scales that live with
        the Q values — they cannot be folded into a per-token weight
        scalar, so `weights` carries only the softmax and head scales.

    Returns (q_quant, weights_out) where q_quant is either a Tensor (FP8) or
    a (values, scales) tuple (MXFP4). This matches the union type accepted
    by `SparseAttnIndexer.forward_*`.
    """
    assert positions.ndim == 1
    assert index_q.ndim == 3
    assert index_q_cos_sin_cache.ndim == 2

    num_tokens = positions.shape[0]
    num_index_q_heads = index_q.shape[1]
    index_q_head_dim = index_q.shape[2]

    index_weights_out = torch.empty_like(index_weights, dtype=torch.float32)

    if use_fp4:
        assert index_q_head_dim % MXFP4_BLOCK_SIZE == 0, (
            f"head_dim={index_q_head_dim} must be a multiple of MXFP4 block "
            f"size {MXFP4_BLOCK_SIZE}"
        )
        num_scale_blocks = index_q_head_dim // MXFP4_BLOCK_SIZE
        index_q_packed = torch.empty(
            (num_tokens, num_index_q_heads, index_q_head_dim // 2),
            dtype=torch.uint8,
            device=index_q.device,
        )
        index_q_scale = torch.empty(
            (num_tokens, num_index_q_heads, num_scale_blocks),
            dtype=torch.uint8,
            device=index_q.device,
        )
        if has_cutedsl():
            # lazily import, otherwise some tests fail due to CUDA driver init failure.
            from vllm.models.deepseek_v4.nvidia.ops.fused_indexer_q_cutedsl import (
                fused_indexer_q_rope_quant_mxfp4_cutedsl,
            )

            fused_indexer_q_rope_quant_mxfp4_cutedsl(
                positions,
                index_q,
                index_q_cos_sin_cache,
                index_weights,
                index_weights_softmax_scale,
                index_weights_head_scale,
                index_q_packed,
                index_q_scale,
                index_weights_out,
            )
        else:
            _fused_indexer_q_rope_mxfp4_kernel[(num_tokens, num_index_q_heads)](
                positions,
                index_q,
                index_q.stride(0),
                index_q.stride(1),
                index_q_cos_sin_cache,
                index_q_cos_sin_cache.stride(0),
                index_q_cos_sin_cache.shape[-1] // 2,
                index_q_packed,
                index_q_packed.stride(0),
                index_q_packed.stride(1),
                index_q_scale,
                index_q_scale.stride(0),
                index_q_scale.stride(1),
                index_q_head_dim,
                MXFP4_BLOCK_SIZE,
                index_weights,
                index_weights.stride(0),
                index_weights_softmax_scale,
                index_weights_head_scale,
                index_weights_out,
                index_weights_out.stride(0),
                num_warps=1,  # TODO: Tune this
            )

        # Values stay uint8 (2 E2M1 nibbles per byte). Scales are 4 ue8m0
        # bytes per (token, head) reinterpreted as one int32, then squeezed
        # from (T, H, 1) to (T, H) to match DeepGEMM's expected q_sf rank
        # (prefill wants 2-D (seq_len, num_heads); decode reshapes this to
        # 3-D (batch, next_n, num_heads)).
        return (
            index_q_packed,
            index_q_scale.view(torch.int32).squeeze(-1),
        ), index_weights_out

    if q_is_int8:
        # INT8 integer-MMA query for the s8 x s8 indexer on Ampere. Output is a
        # symmetric INT8 tensor (NOT fp8); the per-(token, head) q scale is folded
        # into index_weights_out identically to the fp8 path. cuTeDSL/fp8 fast
        # paths are fp8-only, so always use the Triton kernel here.
        index_q_int8 = torch.empty_like(index_q, dtype=torch.int8)
        _fused_indexer_q_rope_quant_kernel[(num_tokens, num_index_q_heads)](
            positions,
            index_q,
            index_q.stride(0),
            index_q.stride(1),
            index_q_cos_sin_cache,
            index_q_cos_sin_cache.stride(0),
            index_q_cos_sin_cache.shape[-1] // 2,
            index_q_int8,
            index_q_int8.stride(0),
            index_q_int8.stride(1),
            index_q_head_dim,
            index_weights,
            index_weights.stride(0),
            index_weights_softmax_scale,
            index_weights_head_scale,
            index_weights_out,
            index_weights_out.stride(0),
            QK_INT8=True,
            num_warps=1,
        )
        return index_q_int8, index_weights_out

    index_q_fp8 = torch.empty_like(index_q, dtype=torch.float8_e4m3fn)

    if _supports_fp8e4nv_in_triton() and has_cutedsl():
        # lazily import, otherwise some tests fail due to CUDA driver init failure.
        from vllm.models.deepseek_v4.nvidia.ops.fused_indexer_q_cutedsl import (
            fused_indexer_q_rope_quant_fp8_cutedsl,
        )

        fused_indexer_q_rope_quant_fp8_cutedsl(
            positions,
            index_q,
            index_q_cos_sin_cache,
            index_weights,
            index_weights_softmax_scale,
            index_weights_head_scale,
            index_q_fp8,
            index_weights_out,
        )
        return index_q_fp8, index_weights_out

    # The Triton kernel stores E4M3 bytes using the arithmetic encoder from
    # fp8e4m3_arith, so it is usable on sm_8x where tl.float8e4nv casts fail.
    index_q_fp8_u8 = index_q_fp8.view(torch.uint8)
    _fused_indexer_q_rope_quant_kernel[(num_tokens, num_index_q_heads)](
        positions,
        index_q,
        index_q.stride(0),
        index_q.stride(1),
        index_q_cos_sin_cache,
        index_q_cos_sin_cache.stride(0),
        index_q_cos_sin_cache.shape[-1] // 2,
        index_q_fp8_u8,
        index_q_fp8_u8.stride(0),
        index_q_fp8_u8.stride(1),
        index_q_head_dim,
        index_weights,
        index_weights.stride(0),
        index_weights_softmax_scale,
        index_weights_head_scale,
        index_weights_out,
        index_weights_out.stride(0),
        num_warps=1,  # TODO: Tune this
    )
    return index_q_fp8, index_weights_out
