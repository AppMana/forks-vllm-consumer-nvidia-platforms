# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fallback kernels used by the local DeepSeek V4 path."""

import torch

from vllm.triton_utils import LOG2E, tl, triton
from vllm.models.deepseek_v4.common.ops.fp8e4m3_arith import (
    fp8e4m3_decode_to_fp32,
)

DEEPSEEK_V4_MLA_HEAD_DIM = 512
FP8_DS_MLA_FP8_DIM = 448
FP8_DS_MLA_SCALE_GROUP = 64
FP8_DS_MLA_SCALE_BYTES = 8
FP8_DS_MLA_TOKEN_BYTES = 576


def indexer_imma_enabled() -> bool:
    """True when the indexer logits run as int8 integer MMA (q quantized to
    symmetric INT8 with its scale folded into weights by the caller; requires
    the INT8 indexer cache). Enabled by the AppMana experimental checkpoint
    config field."""
    return _dsv4_int_indexer_auto_enabled() and indexer_cache_is_int8()


def indexer_cache_is_int8() -> bool:
    """True when the indexer K cache stores symmetric INT8 (scale_fmt "int8"
    in indexer_k_quant_and_cache) instead of FP8 e4m3. Enabled by the AppMana
    experimental checkpoint config field. Gate-1 recall study:
    tools/ampere/dsv4_indexer_int8_recall.py."""
    return _dsv4_int_indexer_auto_enabled()


def _dsv4_int_indexer_auto_enabled() -> bool:
    try:
        from vllm.model_executor.layers.quantization.dsv4_int import (
            dsv4_int4_experts_int8_dense_active,
        )
    except Exception:
        return False
    return dsv4_int4_experts_int8_dense_active()


def _view_packed_fp8_paged_mqa_kv_cache(
    kv_cache: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP8 values and fp32 scales from indexer cache block storage."""
    if kv_cache.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 kv_cache, got {kv_cache.dtype}")
    if kv_cache.dim() == 3:
        num_blocks, block_size, head_dim_with_scale = kv_cache.shape
        num_kv_heads = 1
    elif kv_cache.dim() == 4:
        num_blocks, block_size, num_kv_heads, head_dim_with_scale = kv_cache.shape
    else:
        raise ValueError(
            f"Expected 3D or 4D kv_cache, got {kv_cache.dim()} dimensions"
        )
    if num_kv_heads != 1:
        raise ValueError(f"Expected one KV head, got {num_kv_heads}")

    scale_bytes = head_dim_with_scale - head_dim
    if scale_bytes <= 0 or scale_bytes % torch.float32.itemsize != 0:
        raise ValueError(
            "Expected kv_cache last dimension to contain FP8 values followed "
            f"by fp32 scale bytes; got head_dim={head_dim}, "
            f"last_dim={head_dim_with_scale}"
        )

    block_stride = kv_cache.stride(0)
    base_storage_offset = kv_cache.storage_offset()
    scale_elems = scale_bytes // torch.float32.itemsize
    kv_values = torch.as_strided(
        kv_cache,
        size=(num_blocks, block_size, 1, head_dim),
        stride=(block_stride, head_dim, head_dim, 1),
        storage_offset=base_storage_offset,
    ).view(torch.float8_e4m3fn)
    kv_scale = torch.as_strided(
        kv_cache,
        size=(num_blocks, block_size, 1, scale_bytes),
        stride=(block_stride, scale_bytes, scale_bytes, 1),
        storage_offset=base_storage_offset + block_size * head_dim,
    ).view(torch.float32)
    return kv_values, kv_scale[..., :scale_elems]


# seq_kv tracks the context length; pin it off divisibility-by-16 specialization
# (same recompile-wedge class as _mqa_logits_workspace_kernel).
@triton.jit(do_not_specialize=["num_tokens", "seq_kv"])
def _sparse_attention_bf16_kernel(
    q_ptr,
    kv_ptr,
    indices_ptr,
    lengths_ptr,
    sink_ptr,
    out_ptr,
    num_tokens,
    num_heads: tl.constexpr,
    seq_kv: tl.int32,
    index_topk: tl.constexpr,
    sm_scale_log2: tl.constexpr,
    stride_qt: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kv_t: tl.constexpr,
    stride_kv_d: tl.constexpr,
    stride_indices_t: tl.constexpr,
    stride_indices_k: tl.constexpr,
    stride_out_t: tl.constexpr,
    stride_out_h: tl.constexpr,
    stride_out_d: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_SINK: tl.constexpr,
    LOG2E_CONST: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_block = tl.program_id(1)
    heads = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    mask_h = heads < num_heads

    q = tl.load(
        q_ptr
        + token_id * stride_qt
        + heads[:, None] * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=mask_h[:, None],
        other=0.0,
    )

    if HAS_SINK:
        sink = tl.load(sink_ptr + heads, mask=mask_h, other=-float("inf"))
        e_max = sink * LOG2E_CONST
        e_sum = tl.where(mask_h, 1.0, 0.0)
    else:
        e_max = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
        e_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    length = tl.load(lengths_ptr + token_id)
    for start in range(0, index_topk, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        idx = tl.load(
            indices_ptr + token_id * stride_indices_t + offs_n * stride_indices_k,
            mask=offs_n < index_topk,
            other=-1,
        )
        mask_kv = (offs_n < length) & (idx >= 0) & (idx < seq_kv)
        k = tl.load(
            kv_ptr + idx[None, :] * stride_kv_t + offs_d[:, None] * stride_kv_d,
            mask=mask_kv[None, :],
            other=0.0,
        )
        qk = tl.dot(q, k.to(q.dtype)) * sm_scale_log2
        qk = tl.where(
            mask_h[:, None] & mask_kv[None, :],
            qk,
            -3.4028234663852886e38,
        )

        v = tl.load(
            kv_ptr + idx[:, None] * stride_kv_t + offs_d[None, :] * stride_kv_d,
            mask=mask_kv[:, None],
            other=0.0,
        )

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        p = tl.where(mask_h[:, None] & mask_kv[None, :], p, 0.0)
        acc = acc * re_scale[:, None] + tl.dot(p.to(v.dtype), v)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    acc = acc / tl.maximum(e_sum, 1.0e-20)[:, None]
    tl.store(
        out_ptr
        + token_id * stride_out_t
        + heads[:, None] * stride_out_h
        + offs_d[None, :] * stride_out_d,
        acc.to(tl.bfloat16),
        mask=mask_h[:, None],
    )


def sparse_attention_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    out: torch.Tensor,
) -> None:
    if indices.ndim == 3:
        indices = indices.squeeze(1)
    if kv.ndim == 3:
        kv = kv.squeeze(1)

    num_tokens, num_heads, head_dim = q.shape
    if num_tokens == 0:
        return
    if head_dim != DEEPSEEK_V4_MLA_HEAD_DIM:
        raise ValueError(
            "DeepSeek V4 sparse Triton fallback expects "
            f"D={DEEPSEEK_V4_MLA_HEAD_DIM}, got {head_dim}"
        )
    assert kv.shape[-1] == head_dim
    assert out.shape[-1] == head_dim

    grid = (num_tokens, triton.cdiv(num_heads, 8))
    _sparse_attention_bf16_kernel[grid](
        q,
        kv,
        indices,
        lengths,
        attn_sink if attn_sink is not None else q,
        out,
        num_tokens,
        num_heads,
        kv.shape[0],
        indices.shape[-1],
        scale * LOG2E,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        indices.stride(0),
        indices.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_H=8,
        BLOCK_N=16,
        BLOCK_D=DEEPSEEK_V4_MLA_HEAD_DIM,
        HAS_SINK=attn_sink is not None,
        LOG2E_CONST=LOG2E,
        num_warps=8,
    )


@triton.jit(do_not_specialize=["num_tokens"])
def _decode_sparse_attention_fp8_kernel(
    q_ptr,
    swa_cache_fp8_ptr,
    swa_cache_bf16_ptr,
    swa_cache_u8_ptr,
    swa_indices_ptr,
    swa_lens_ptr,
    extra_cache_fp8_ptr,
    extra_cache_bf16_ptr,
    extra_cache_u8_ptr,
    extra_indices_ptr,
    extra_lens_ptr,
    sink_ptr,
    out_ptr,
    num_tokens,
    num_heads: tl.constexpr,
    swa_index_topk: tl.constexpr,
    extra_index_topk: tl.constexpr,
    swa_num_blocks: tl.constexpr,
    extra_num_blocks: tl.constexpr,
    swa_block_size: tl.constexpr,
    extra_block_size: tl.constexpr,
    swa_stride_block_bytes: tl.constexpr,
    extra_stride_block_bytes: tl.constexpr,
    sm_scale_log2: tl.constexpr,
    stride_qt: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_swa_indices_t: tl.constexpr,
    stride_swa_indices_k: tl.constexpr,
    stride_extra_indices_t: tl.constexpr,
    stride_extra_indices_k: tl.constexpr,
    stride_out_t: tl.constexpr,
    stride_out_h: tl.constexpr,
    stride_out_d: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FP8_DIM: tl.constexpr,
    SCALE_GROUP: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    TOKEN_BYTES: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    HAS_SINK: tl.constexpr,
    LOG2E_CONST: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_block = tl.program_id(1)
    heads = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    mask_h = heads < num_heads

    q = tl.load(
        q_ptr
        + token_id * stride_qt
        + heads[:, None] * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=mask_h[:, None],
        other=0.0,
    )

    if HAS_SINK:
        sink = tl.load(sink_ptr + heads, mask=mask_h, other=-float("inf"))
        e_max = sink * LOG2E_CONST
        e_sum = tl.where(mask_h, 1.0, 0.0)
    else:
        e_max = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
        e_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    swa_len = tl.load(swa_lens_ptr + token_id)
    extra_len = tl.load(extra_lens_ptr + token_id) if HAS_EXTRA else 0
    total_len = extra_len + swa_len

    for start in range(0, extra_index_topk + swa_index_topk, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        use_extra = HAS_EXTRA & (offs_n < extra_len)
        use_swa = (offs_n >= extra_len) & (offs_n < total_len)

        extra_cols = offs_n
        swa_cols = offs_n - extra_len
        extra_idx = tl.load(
            extra_indices_ptr
            + token_id * stride_extra_indices_t
            + extra_cols * stride_extra_indices_k,
            mask=HAS_EXTRA & (extra_cols < extra_index_topk),
            other=-1,
        )
        swa_idx = tl.load(
            swa_indices_ptr
            + token_id * stride_swa_indices_t
            + swa_cols * stride_swa_indices_k,
            mask=(swa_cols >= 0) & (swa_cols < swa_index_topk),
            other=-1,
        )
        idx = tl.where(use_extra, extra_idx, swa_idx)

        extra_block = idx // extra_block_size
        extra_pos = idx - extra_block * extra_block_size
        swa_block = idx // swa_block_size
        swa_pos = idx - swa_block * swa_block_size
        valid_extra = use_extra & (idx >= 0) & (extra_block < extra_num_blocks)
        valid_swa = use_swa & (idx >= 0) & (swa_block < swa_num_blocks)
        valid = valid_extra | valid_swa

        extra_token_base = extra_block * extra_stride_block_bytes
        extra_token_base += extra_pos * TOKEN_BYTES
        swa_token_base = swa_block * swa_stride_block_bytes
        swa_token_base += swa_pos * TOKEN_BYTES
        token_base = tl.where(use_extra, extra_token_base, swa_token_base)
        block_size = tl.where(use_extra, extra_block_size, swa_block_size)
        stride_block_bytes = tl.where(
            use_extra, extra_stride_block_bytes, swa_stride_block_bytes
        )
        pos = tl.where(use_extra, extra_pos, swa_pos)

        is_fp8 = offs_d < FP8_DIM
        scale_offsets = (
            tl.where(use_extra, extra_block, swa_block)[:, None]
            * stride_block_bytes[:, None]
            + block_size[:, None] * TOKEN_BYTES
            + pos[:, None] * SCALE_BYTES
            + (offs_d[None, :] // SCALE_GROUP)
        )
        encoded_scale = tl.load(
            tl.where(use_extra[:, None], extra_cache_u8_ptr, swa_cache_u8_ptr)
            + scale_offsets,
            mask=valid[:, None] & is_fp8[None, :],
            other=127,
        ).to(tl.float32)
        fp8_scale = tl.exp2(encoded_scale - 127.0)

        fp8_offsets = token_base[:, None] + offs_d[None, :]
        fp8_vals = (
            fp8e4m3_decode_to_fp32(
                tl.load(
                    tl.where(use_extra[:, None], extra_cache_fp8_ptr, swa_cache_fp8_ptr)
                    + fp8_offsets,
                    mask=valid[:, None] & is_fp8[None, :],
                    other=0,
                )
            )
            * fp8_scale
        )

        bf16_offsets = (token_base[:, None] + FP8_DIM) // 2
        bf16_offsets += offs_d[None, :] - FP8_DIM
        bf16_vals = tl.load(
            tl.where(use_extra[:, None], extra_cache_bf16_ptr, swa_cache_bf16_ptr)
            + bf16_offsets,
            mask=valid[:, None] & (~is_fp8[None, :]),
            other=0.0,
        ).to(tl.float32)
        k = tl.where(is_fp8[None, :], fp8_vals, bf16_vals)

        qk = tl.dot(q, tl.trans(k.to(q.dtype))) * sm_scale_log2
        qk = tl.where(
            mask_h[:, None] & valid[None, :],
            qk,
            -3.4028234663852886e38,
        )

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        p = tl.where(mask_h[:, None] & valid[None, :], p, 0.0)
        acc = acc * re_scale[:, None] + tl.dot(p.to(k.dtype), k)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    acc = acc / tl.maximum(e_sum, 1.0e-20)[:, None]
    tl.store(
        out_ptr
        + token_id * stride_out_t
        + heads[:, None] * stride_out_h
        + offs_d[None, :] * stride_out_d,
        acc.to(tl.bfloat16),
        mask=mask_h[:, None],
    )


def decode_sparse_attention_triton(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    out: torch.Tensor,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_lens: torch.Tensor | None = None,
) -> None:
    if swa_indices.ndim == 3:
        swa_indices = swa_indices.squeeze(1)
    if extra_indices is not None and extra_indices.ndim == 3:
        extra_indices = extra_indices.squeeze(1)

    num_tokens, num_heads, head_dim = q.shape
    if num_tokens == 0:
        return
    if head_dim != DEEPSEEK_V4_MLA_HEAD_DIM:
        raise ValueError(
            "DeepSeek V4 decode Triton fallback expects "
            f"D={DEEPSEEK_V4_MLA_HEAD_DIM}, got {head_dim}"
        )
    has_extra = (
        extra_cache is not None and extra_indices is not None and extra_lens is not None
    )
    if not has_extra:
        extra_cache = swa_cache
        extra_indices = swa_indices[:, :1]
        extra_lens = swa_lens

    assert extra_cache is not None
    assert extra_indices is not None
    assert extra_lens is not None
    grid = (num_tokens, triton.cdiv(num_heads, 8))
    _decode_sparse_attention_fp8_kernel[grid](
        q,
        swa_cache,
        swa_cache.view(torch.bfloat16),
        swa_cache,
        swa_indices,
        swa_lens,
        extra_cache,
        extra_cache.view(torch.bfloat16),
        extra_cache,
        extra_indices,
        extra_lens,
        attn_sink if attn_sink is not None else q,
        out,
        num_tokens,
        num_heads,
        swa_indices.shape[-1],
        extra_indices.shape[-1] if has_extra else 0,
        swa_cache.shape[0],
        extra_cache.shape[0],
        swa_cache.shape[1],
        extra_cache.shape[1],
        swa_cache.stride(0),
        extra_cache.stride(0),
        scale * LOG2E,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        swa_indices.stride(0),
        swa_indices.stride(1),
        extra_indices.stride(0),
        extra_indices.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_H=8,
        BLOCK_N=16,
        BLOCK_D=DEEPSEEK_V4_MLA_HEAD_DIM,
        FP8_DIM=FP8_DS_MLA_FP8_DIM,
        SCALE_GROUP=FP8_DS_MLA_SCALE_GROUP,
        SCALE_BYTES=FP8_DS_MLA_SCALE_BYTES,
        TOKEN_BYTES=FP8_DS_MLA_TOKEN_BYTES,
        HAS_EXTRA=has_extra,
        HAS_SINK=attn_sink is not None,
        LOG2E_CONST=LOG2E,
        num_warps=8,
    )


@triton.jit(do_not_specialize=["B"])
def _deepseek_v4_fp8_einsum_triton_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    out_ptr,
    B,
    G: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    a_stride_b: tl.constexpr,
    a_stride_g: tl.constexpr,
    a_stride_k: tl.constexpr,
    as_stride_b: tl.constexpr,
    as_stride_g: tl.constexpr,
    as_stride_kb: tl.constexpr,
    b_stride_g: tl.constexpr,
    b_stride_n: tl.constexpr,
    b_stride_k: tl.constexpr,
    bs_stride_g: tl.constexpr,
    bs_stride_nb: tl.constexpr,
    bs_stride_kb: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_g: tl.constexpr,
    out_stride_n: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        kb = k0 // BLOCK_K

        a = tl.load(
            a_ptr
            + offs_b[:, None] * a_stride_b
            + pid_g * a_stride_g
            + k[None, :] * a_stride_k,
            mask=(offs_b[:, None] < B) & (k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr
            + pid_g * b_stride_g
            + offs_n[:, None] * b_stride_n
            + k[None, :] * b_stride_k,
            mask=(offs_n[:, None] < N) & (k[None, :] < K),
            other=0.0,
        )
        a_s = tl.load(
            a_scale_ptr
            + offs_b * as_stride_b
            + pid_g * as_stride_g
            + kb * as_stride_kb,
            mask=offs_b < B,
            other=0.0,
        ).to(tl.float32)
        b_s = tl.load(
            b_scale_ptr
            + pid_g * bs_stride_g
            + (offs_n // BLOCK_K) * bs_stride_nb
            + kb * bs_stride_kb,
            mask=offs_n < N,
            other=0.0,
        ).to(tl.float32)
        acc += (
            tl.dot(a, tl.trans(b), out_dtype=tl.float32) * a_s[:, None] * b_s[None, :]
        )

    tl.store(
        out_ptr
        + offs_b[:, None] * out_stride_b
        + pid_g * out_stride_g
        + offs_n[None, :] * out_stride_n,
        acc,
        mask=(offs_b[:, None] < B) & (offs_n[None, :] < N),
    )


def _e8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    return (scale.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)


def _unpack_int32_e8m0_scales(
    packed_scale: torch.Tensor,
    num_blocks: int,
) -> torch.Tensor:
    shifts = torch.arange(4, device=packed_scale.device, dtype=torch.int32) * 8
    unpacked = (packed_scale.to(torch.int32).unsqueeze(-1) >> shifts) & 0xFF
    unpacked = unpacked.reshape(*packed_scale.shape[:-1], -1)[..., :num_blocks]
    return (unpacked << 23).view(torch.float32)


def _normalize_deepseek_v4_fp8_einsum_inputs(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, G, K = a.shape
    _, out_g, N = out.shape
    assert out_g == G
    k_blocks = triton.cdiv(K, 128)
    n_blocks = triton.cdiv(N, 128)

    if b.ndim == 2:
        b = b.view(G, N, K)
    if b_scale.ndim == 2:
        b_scale = b_scale.view(G, n_blocks, k_blocks)

    if a_scale.dtype == torch.int32:
        a_scale = _unpack_int32_e8m0_scales(a_scale, k_blocks)
    if b_scale.dtype == torch.int32:
        b_scale = _unpack_int32_e8m0_scales(b_scale, k_blocks)

    if a_scale.dtype == torch.float8_e8m0fnu:
        a_scale = _e8m0_to_fp32(a_scale)
    if b_scale.dtype == torch.float8_e8m0fnu:
        b_scale = _e8m0_to_fp32(b_scale)

    return a, a_scale.contiguous(), b, b_scale.contiguous()


def deepseek_v4_fp8_einsum_triton(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    a, a_scale, b, b_scale = _normalize_deepseek_v4_fp8_einsum_inputs(
        a, a_scale, b, b_scale, out
    )
    B, G, K = a.shape
    N = out.shape[-1]
    grid = (triton.cdiv(B, 16), G, triton.cdiv(N, 32))
    _deepseek_v4_fp8_einsum_triton_kernel[grid](
        a,
        a_scale,
        b,
        b_scale,
        out,
        B,
        G,
        N,
        K,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        a_scale.stride(0),
        a_scale.stride(1),
        a_scale.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        b_scale.stride(0),
        b_scale.stride(1),
        b_scale.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_B=16,
        BLOCK_N=32,
        BLOCK_K=128,
        num_warps=4,
    )


# stride_lm is the context-dependent output row-stride; pin it off divisibility
# specialization too (num_q/seq_len_kv already pinned).
@triton.jit(do_not_specialize=["num_q", "seq_len_kv", "stride_lm"])
def _fp8_mqa_logits_kernel(
    q_ptr,
    k_ptr,
    scale_ptr,
    weights_ptr,
    cu_seqlen_ks_ptr,
    cu_seqlen_ke_ptr,
    logits_ptr,
    num_q,
    seq_len_kv,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wh: tl.constexpr,
    stride_lm: tl.int64,
    stride_ln: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    valid_m = offs_m < num_q
    valid_n = offs_n < seq_len_kv
    seq_start = tl.load(cu_seqlen_ks_ptr + offs_m, mask=valid_m, other=0)
    seq_end = tl.load(cu_seqlen_ke_ptr + offs_m, mask=valid_m, other=0)
    seq_mask = (offs_n[None, :] >= seq_start[:, None]) & (
        offs_n[None, :] < seq_end[:, None]
    )

    logits = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for h in tl.range(0, num_heads):
        scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for d0 in tl.range(0, head_dim, BLOCK_D):
            d = d0 + offs_d
            q = tl.load(
                q_ptr
                + offs_m[:, None] * stride_qm
                + h * stride_qh
                + d[None, :] * stride_qd,
                mask=valid_m[:, None] & (d[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            k = tl.load(
                k_ptr + offs_n[:, None] * stride_kn + d[None, :] * stride_kd,
                mask=valid_n[:, None] & (d[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            scores += tl.dot(q, tl.trans(k), input_precision="tf32")
        scale = tl.load(scale_ptr + offs_n, mask=valid_n, other=0.0)
        weighted = tl.maximum(scores * scale[None, :], 0.0)
        weight = tl.load(
            weights_ptr + offs_m * stride_wm + h * stride_wh,
            mask=valid_m,
            other=0.0,
        )
        logits += weighted * weight[:, None]

    store_mask = valid_m[:, None] & valid_n[None, :]
    logits = tl.where(seq_mask & store_mask, logits, float("-inf"))
    tl.store(
        logits_ptr + offs_m[:, None] * stride_lm + offs_n[None, :] * stride_ln,
        logits,
        mask=store_mask,
    )


def fp8_mqa_logits_triton(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    k_fp8, scale = kv
    num_q, num_heads, head_dim = q.shape
    seq_len_kv = k_fp8.shape[0]
    logits = torch.empty(
        (num_q, seq_len_kv),
        device=q.device,
        dtype=torch.float32,
    )
    if num_q == 0 or seq_len_kv == 0:
        return logits

    grid = (triton.cdiv(num_q, 8), triton.cdiv(seq_len_kv, 64))
    _fp8_mqa_logits_kernel[grid](
        q,
        k_fp8,
        scale,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        logits,
        num_q,
        seq_len_kv,
        num_heads,
        head_dim,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_fp8.stride(0),
        k_fp8.stride(1),
        weights.stride(0),
        weights.stride(1),
        logits.stride(0),
        logits.stride(1),
        BLOCK_M=8,
        BLOCK_N=64,
        BLOCK_D=64,
        num_warps=4,
    )
    return logits


@triton.jit(do_not_specialize=["logits_width", "num_rows", "token_start", "stride_lm"])
def _fp8_paged_mqa_logits_kernel(
    q_ptr,
    kv_ptr,
    scale_ptr,
    weights_ptr,
    context_lens_ptr,
    block_tables_ptr,
    logits_ptr,
    token_start,
    num_rows,
    logits_width,
    next_n: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    block_table_width: tl.constexpr,
    kv_num_blocks: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kvb: tl.constexpr,
    stride_kvs: tl.constexpr,
    stride_kvd: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_ss: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wh: tl.constexpr,
    stride_clb: tl.constexpr,
    stride_cln: tl.constexpr,
    stride_btb: tl.constexpr,
    stride_btk: tl.constexpr,
    stride_lm: tl.int64,
    stride_ln: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    K_IS_INT8: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_local_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = token_start + offs_local_n
    offs_d = tl.arange(0, BLOCK_D)

    valid_m = offs_m < num_rows
    valid_n = offs_local_n < logits_width
    batch = offs_m // next_n
    q_pos = offs_m - batch * next_n
    context_len = tl.load(
        context_lens_ptr + batch * stride_clb + q_pos * stride_cln,
        mask=valid_m,
        other=0,
    )
    context_mask = valid_n[None, :] & (offs_n[None, :] < context_len[:, None])

    block_rank = offs_n // block_size
    block_offset = offs_n - block_rank * block_size
    valid_block_rank = context_mask & (block_rank[None, :] < block_table_width)
    block_idx = tl.load(
        block_tables_ptr
        + batch[:, None] * stride_btb
        + block_rank[None, :] * stride_btk,
        mask=valid_m[:, None] & valid_block_rank,
        other=0,
    )
    valid_block = valid_block_rank & (block_idx >= 0) & (block_idx < kv_num_blocks)

    logits = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    scale = tl.load(
        scale_ptr + block_idx * stride_sb + block_offset[None, :] * stride_ss,
        mask=valid_block,
        other=0.0,
    )
    for h in tl.range(0, num_heads):
        scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for d0 in tl.range(0, head_dim, BLOCK_D):
            d = d0 + offs_d
            q_u8 = tl.load(
                q_ptr
                + batch[:, None] * stride_qb
                + q_pos[:, None] * stride_qn
                + h * stride_qh
                + d[None, :] * stride_qd,
                mask=valid_m[:, None] & (d[None, :] < head_dim),
                other=0,
            )
            q = fp8e4m3_decode_to_fp32(q_u8)
            k_u8 = tl.load(
                kv_ptr
                + block_idx[:, :, None] * stride_kvb
                + block_offset[None, :, None] * stride_kvs
                + d[None, None, :] * stride_kvd,
                mask=valid_block[:, :, None] & (d[None, None, :] < head_dim),
                other=0,
            )
            if K_IS_INT8:
                k = k_u8.to(tl.int8, bitcast=True).to(tl.float32)
            else:
                k = fp8e4m3_decode_to_fp32(k_u8)
            scores += tl.sum(q[:, None, :] * k, axis=2)
        weighted = tl.maximum(scores * scale, 0.0)
        weight = tl.load(
            weights_ptr + offs_m * stride_wm + h * stride_wh,
            mask=valid_m,
            other=0.0,
        )
        logits += weighted * weight[:, None]

    store_mask = valid_m[:, None] & valid_n[None, :]
    logits = tl.where(valid_block & store_mask, logits, float("-inf"))
    tl.store(
        logits_ptr + offs_m[:, None] * stride_lm + offs_local_n[None, :] * stride_ln,
        logits,
        mask=store_mask,
    )


def fp8_paged_mqa_logits_triton(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    token_start: int = 0,
    token_count: int | None = None,
) -> torch.Tensor:
    batch_size, next_n, num_heads, head_dim = q.size()
    # The indexer query is symmetric INT8 (s8 x s8 integer-MMA) when the int8
    # cache + checkpoint-enabled IMMA are on; fused_indexer_q emits an int8 q
    # tensor and folds its scale into `weights`, so detect it by dtype here.
    q_is_int8 = q.dtype == torch.int8
    if head_dim % 64 == 0 and num_heads % 4 == 0:
        return fp8_paged_mqa_logits_rowwise_triton(
            q,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            max_model_len,
            token_start=token_start,
            token_count=token_count,
            q_is_int8=q_is_int8,
        )

    kv_values, kv_scale = _view_packed_fp8_paged_mqa_kv_cache(kv_cache, head_dim)
    _, block_size, _, _ = kv_values.size()
    num_rows = batch_size * next_n
    if token_count is None:
        token_count = max_model_len - token_start
    assert token_start >= 0
    assert token_count >= 0
    assert token_start + token_count <= max_model_len
    block_n = 64
    logits_width = triton.cdiv(token_count, block_n) * block_n
    logits = torch.empty(
        (num_rows, logits_width),
        device=q.device,
        dtype=torch.float32,
    )
    if num_rows == 0 or token_count == 0:
        return logits[:, :token_count]

    context_lens_2d = context_lens.reshape(batch_size, -1)
    if context_lens_2d.shape[1] == 1 and next_n != 1:
        context_lens_2d = context_lens_2d.expand(batch_size, next_n).contiguous()
    grid = (triton.cdiv(num_rows, 4), triton.cdiv(logits_width, block_n))
    # sm_8x: pass q/kv as uint8 so the kernel decodes arithmetically.
    _fp8_paged_mqa_logits_kernel[grid](
        q.view(torch.uint8),
        kv_values.view(torch.uint8),
        kv_scale,
        weights,
        context_lens_2d,
        block_tables,
        logits,
        token_start,
        num_rows,
        logits_width,
        next_n,
        num_heads,
        head_dim,
        block_size,
        block_tables.shape[1],
        kv_values.shape[0],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        kv_values.stride(0),
        kv_values.stride(1),
        kv_values.stride(3),
        kv_scale.stride(0),
        kv_scale.stride(1),
        weights.stride(0),
        weights.stride(1),
        context_lens_2d.stride(0),
        context_lens_2d.stride(1),
        block_tables.stride(0),
        block_tables.stride(1),
        logits.stride(0),
        logits.stride(1),
        BLOCK_M=4,
        BLOCK_N=block_n,
        BLOCK_D=64,
        K_IS_INT8=indexer_cache_is_int8(),
        num_warps=4,
    )
    return logits[:, :token_count]


@triton.jit(do_not_specialize=["logits_width", "num_rows", "token_start", "stride_lm"])
def _fp8_paged_mqa_logits_rowwise_kernel(
    q_ptr,
    kv_ptr,
    scale_ptr,
    weights_ptr,
    context_lens_ptr,
    block_tables_ptr,
    logits_ptr,
    token_start,
    num_rows,
    logits_width,
    next_n: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    block_table_width: tl.constexpr,
    kv_num_blocks: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kvb: tl.constexpr,
    stride_kvs: tl.constexpr,
    stride_kvd: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_ss: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wh: tl.constexpr,
    stride_clb: tl.constexpr,
    stride_cln: tl.constexpr,
    stride_btb: tl.constexpr,
    stride_btk: tl.constexpr,
    stride_lm: tl.int64,
    stride_ln: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    K_IS_INT8: tl.constexpr,
    QK_INT8: tl.constexpr,
):
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_local_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = token_start + offs_local_n
    offs_d = tl.arange(0, BLOCK_D)

    valid_row = row < num_rows
    valid_n = offs_local_n < logits_width
    batch = row // next_n
    q_pos = row - batch * next_n
    context_len = tl.load(
        context_lens_ptr + batch * stride_clb + q_pos * stride_cln,
        mask=valid_row,
        other=0,
    )
    if token_start + pid_n * BLOCK_N >= context_len:
        logits = tl.full((BLOCK_N,), float("-inf"), dtype=tl.float32)
        tl.store(
            logits_ptr + row * stride_lm + offs_local_n * stride_ln,
            logits,
            mask=valid_row & valid_n,
        )
        return
    context_mask = valid_n & (offs_n < context_len)

    block_rank = offs_n // block_size
    block_offset = offs_n - block_rank * block_size
    valid_block_rank = context_mask & (block_rank < block_table_width)
    block_idx = tl.load(
        block_tables_ptr + batch * stride_btb + block_rank * stride_btk,
        mask=valid_row & valid_block_rank,
        other=0,
    )
    valid_block = valid_block_rank & (block_idx >= 0) & (block_idx < kv_num_blocks)

    scale = tl.load(
        scale_ptr + block_idx * stride_sb + block_offset * stride_ss,
        mask=valid_block,
        other=0.0,
    )
    logits = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for h0 in tl.range(0, num_heads, BLOCK_H):
        heads = h0 + tl.arange(0, BLOCK_H)
        valid_h = heads < num_heads
        if QK_INT8:
            # Integer-MMA path: q and the cache both hold symmetric INT8.
            # The per-(row, head) q scale is folded into `weights` by the
            # caller and the per-candidate k scale is applied below through
            # `scale`, so the s32 dot needs no extra scale handling.
            scores_i32 = tl.zeros((BLOCK_H, BLOCK_N), dtype=tl.int32)
            for d0 in tl.range(0, head_dim, BLOCK_D):
                d = d0 + offs_d
                q_u8 = tl.load(
                    q_ptr
                    + batch * stride_qb
                    + q_pos * stride_qn
                    + heads[:, None] * stride_qh
                    + d[None, :] * stride_qd,
                    mask=valid_row & valid_h[:, None] & (d[None, :] < head_dim),
                    other=0,
                )
                k_u8 = tl.load(
                    kv_ptr
                    + block_idx[None, :] * stride_kvb
                    + block_offset[None, :] * stride_kvs
                    + d[:, None] * stride_kvd,
                    mask=valid_block[None, :] & (d[:, None] < head_dim),
                    other=0,
                )
                scores_i32 += tl.dot(
                    q_u8.to(tl.int8, bitcast=True),
                    k_u8.to(tl.int8, bitcast=True),
                    out_dtype=tl.int32,
                )
            scores = scores_i32.to(tl.float32)
        else:
            scores = tl.zeros((BLOCK_H, BLOCK_N), dtype=tl.float32)
            for d0 in tl.range(0, head_dim, BLOCK_D):
                d = d0 + offs_d
                q_u8 = tl.load(
                    q_ptr
                    + batch * stride_qb
                    + q_pos * stride_qn
                    + heads[:, None] * stride_qh
                    + d[None, :] * stride_qd,
                    mask=valid_row & valid_h[:, None] & (d[None, :] < head_dim),
                    other=0,
                )
                q = fp8e4m3_decode_to_fp32(q_u8)
                k_u8 = tl.load(
                    kv_ptr
                    + block_idx[None, :] * stride_kvb
                    + block_offset[None, :] * stride_kvs
                    + d[:, None] * stride_kvd,
                    mask=valid_block[None, :] & (d[:, None] < head_dim),
                    other=0,
                )
                if K_IS_INT8:
                    k = k_u8.to(tl.int8, bitcast=True).to(tl.float32)
                else:
                    k = fp8e4m3_decode_to_fp32(k_u8)
                scores += tl.dot(q, k, input_precision="tf32")

        weighted = tl.maximum(scores * scale[None, :], 0.0)
        weight = tl.load(
            weights_ptr + row * stride_wm + heads * stride_wh,
            mask=valid_row & valid_h,
            other=0.0,
        )
        logits += tl.sum(weighted * weight[:, None], axis=0)

    logits = tl.where(valid_block & valid_row, logits, float("-inf"))
    tl.store(
        logits_ptr + row * stride_lm + offs_local_n * stride_ln,
        logits,
        mask=valid_row & valid_n,
    )


def fp8_paged_mqa_logits_rowwise_triton(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    token_start: int = 0,
    token_count: int | None = None,
    q_is_int8: bool = False,
) -> torch.Tensor:
    batch_size, next_n, num_heads, head_dim = q.size()
    kv_values, kv_scale = _view_packed_fp8_paged_mqa_kv_cache(kv_cache, head_dim)
    _, block_size, _, _ = kv_values.size()
    num_rows = batch_size * next_n
    if token_count is None:
        token_count = max_model_len - token_start
    assert token_start >= 0
    assert token_count >= 0
    assert token_start + token_count <= max_model_len
    block_n = 128
    logits_width = triton.cdiv(token_count, block_n) * block_n
    logits = torch.empty(
        (num_rows, logits_width),
        device=q.device,
        dtype=torch.float32,
    )
    if num_rows == 0 or token_count == 0:
        return logits[:, :token_count]

    context_lens_2d = context_lens.reshape(batch_size, -1)
    if context_lens_2d.shape[1] == 1 and next_n != 1:
        context_lens_2d = context_lens_2d.expand(batch_size, next_n).contiguous()
    grid = (num_rows, triton.cdiv(logits_width, block_n))
    # sm_8x: pass q/kv as uint8 so the kernel can decode arithmetically
    # without invoking the unsupported fp8e4nv path. Strides are unchanged
    # since fp8 and uint8 share a 1-byte element width.
    _fp8_paged_mqa_logits_rowwise_kernel[grid](
        q.view(torch.uint8),
        kv_values.view(torch.uint8),
        kv_scale,
        weights,
        context_lens_2d,
        block_tables,
        logits,
        token_start,
        num_rows,
        logits_width,
        next_n,
        num_heads,
        head_dim,
        block_size,
        block_tables.shape[1],
        kv_values.shape[0],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        kv_values.stride(0),
        kv_values.stride(1),
        kv_values.stride(3),
        kv_scale.stride(0),
        kv_scale.stride(1),
        weights.stride(0),
        weights.stride(1),
        context_lens_2d.stride(0),
        context_lens_2d.stride(1),
        block_tables.stride(0),
        block_tables.stride(1),
        logits.stride(0),
        logits.stride(1),
        BLOCK_N=block_n,
        BLOCK_D=64,
        BLOCK_H=8,
        K_IS_INT8=indexer_cache_is_int8(),
        QK_INT8=q_is_int8,
        num_warps=4,
    )
    return logits[:, :token_count]


@triton.jit(do_not_specialize=["M", "stride_outs", "stride_sqs"])
def _tf32_hc_prenorm_gemm_kernel(
    x_ptr,
    fn_ptr,
    out_ptr,
    sqrsum_ptr,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_fnn: tl.constexpr,
    stride_fnk: tl.constexpr,
    stride_outs,
    stride_outm: tl.constexpr,
    stride_outn: tl.constexpr,
    stride_sqs,
    stride_sqm: tl.constexpr,
    NUM_SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    split_k = tl.cdiv(K, NUM_SPLIT)
    split_begin = pid_s * split_k
    split_end = tl.minimum(split_begin + split_k, K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    sq = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k0 in tl.range(0, split_k, BLOCK_K):
        k = split_begin + k0 + offs_k
        k_mask = k < split_end
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        fn = tl.load(
            fn_ptr + offs_n[None, :] * stride_fnn + k[:, None] * stride_fnk,
            mask=(offs_n[None, :] < N) & k_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        acc += tl.dot(x, fn, input_precision="tf32", out_dtype=tl.float32)
        sq += tl.sum(x * x, axis=1)

    tl.store(
        out_ptr
        + pid_s * stride_outs
        + offs_m[:, None] * stride_outm
        + offs_n[None, :] * stride_outn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )

    if pid_n == 0:
        tl.store(
            sqrsum_ptr + pid_s * stride_sqs + offs_m * stride_sqm,
            sq,
            mask=offs_m < M,
        )


def tf32_hc_prenorm_gemm_triton(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    num_split: int,
) -> None:
    assert x.dim() == 2
    assert fn.dim() == 2
    assert out.dim() == 3
    assert sqrsum.dim() == 2

    m, k = x.shape
    n = fn.shape[0]
    assert fn.shape[1] == k
    assert out.shape == (num_split, m, n)
    assert sqrsum.shape == (num_split, m)

    if m == 0:
        return

    block_m = 16
    block_n = triton.next_power_of_2(n)
    block_n = min(max(block_n, 16), 32)
    block_k = 64
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n), num_split)
    _tf32_hc_prenorm_gemm_kernel[grid](
        x,
        fn,
        out,
        sqrsum,
        m,
        k,
        n,
        x.stride(0),
        x.stride(1),
        fn.stride(0),
        fn.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        sqrsum.stride(0),
        sqrsum.stride(1),
        num_split,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )


# do_not_specialize the context-varying runtime ints. Triton otherwise compiles a
# separate kernel per divisibility-by-16 class of num_rows / seq_len_kv / the output
# row-stride stride_lm; during decode the context grows by 1 token per step and keeps
# crossing the ÷16 boundary, recompiling ~every 16 tokens. Stacked over a long
# generation these ~165ms compiles wedge the whole PP chain (one rank pinned in
# Triton launch, the rest blocked on the collective). Pinning them off compiles once.
@triton.jit(do_not_specialize=["num_rows", "seq_len_kv", "stride_lm"])
def _mqa_logits_workspace_kernel(
    q_ptr,
    k_ptr,
    k_scale_ptr,
    weights_ptr,
    ks_ptr,
    ke_ptr,
    logits_ptr,
    num_rows,
    seq_len_kv,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wh: tl.constexpr,
    stride_lm: tl.int64,
    stride_ln: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    QK_INT8: tl.constexpr,
):
    """Fused MQA indexer logits over a dense [N, D] K workspace.

    logits[m, n] = sum_h relu((q[m,h,:] . k[n,:]) * k_scale[n]) * w[m,h]
    with n outside [ks[m], ke[m]) forced to -inf. QK_INT8 runs the dot as
    s8 x s8 -> s32 (q scale pre-folded into weights by the caller; k scale
    applied post-dot, order-safe because both scales are positive).
    """
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    valid_row = row < num_rows
    valid_n = offs_n < seq_len_kv
    ks = tl.load(ks_ptr + row, mask=valid_row, other=0)
    ke = tl.load(ke_ptr + row, mask=valid_row, other=0)
    in_range = (offs_n >= ks) & (offs_n < ke)

    k_post_scale = tl.load(k_scale_ptr + offs_n, mask=valid_n, other=0.0).to(
        tl.float32
    )
    logits = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for h0 in tl.range(0, num_heads, BLOCK_H):
        heads = h0 + tl.arange(0, BLOCK_H)
        valid_h = heads < num_heads
        if QK_INT8:
            scores_i32 = tl.zeros((BLOCK_H, BLOCK_N), dtype=tl.int32)
            for d0 in tl.range(0, head_dim, BLOCK_D):
                d = d0 + offs_d
                q_u8 = tl.load(
                    q_ptr
                    + row * stride_qm
                    + heads[:, None] * stride_qh
                    + d[None, :] * stride_qd,
                    mask=valid_row & valid_h[:, None] & (d[None, :] < head_dim),
                    other=0,
                )
                k_u8 = tl.load(
                    k_ptr + offs_n[None, :] * stride_kn + d[:, None] * stride_kd,
                    mask=valid_n[None, :] & (d[:, None] < head_dim),
                    other=0,
                )
                scores_i32 += tl.dot(
                    q_u8.to(tl.int8, bitcast=True),
                    k_u8.to(tl.int8, bitcast=True),
                    out_dtype=tl.int32,
                )
            scores = scores_i32.to(tl.float32)
        else:
            scores = tl.zeros((BLOCK_H, BLOCK_N), dtype=tl.float32)
            for d0 in tl.range(0, head_dim, BLOCK_D):
                d = d0 + offs_d
                q_u8 = tl.load(
                    q_ptr
                    + row * stride_qm
                    + heads[:, None] * stride_qh
                    + d[None, :] * stride_qd,
                    mask=valid_row & valid_h[:, None] & (d[None, :] < head_dim),
                    other=0,
                )
                q = fp8e4m3_decode_to_fp32(q_u8)
                k_u8 = tl.load(
                    k_ptr + offs_n[None, :] * stride_kn + d[:, None] * stride_kd,
                    mask=valid_n[None, :] & (d[:, None] < head_dim),
                    other=0,
                )
                k = fp8e4m3_decode_to_fp32(k_u8)
                scores += tl.dot(q, k, input_precision="tf32")

        weighted = tl.maximum(scores * k_post_scale[None, :], 0.0)
        weight = tl.load(
            weights_ptr + row * stride_wm + heads * stride_wh,
            mask=valid_row & valid_h,
            other=0.0,
        )
        logits += tl.sum(weighted * weight[:, None], axis=0)

    logits = tl.where(in_range & valid_n, logits, float("-inf"))
    tl.store(
        logits_ptr + row * stride_lm + offs_n * stride_ln,
        logits,
        mask=valid_row & valid_n,
    )


def mqa_logits_workspace_triton(
    q: torch.Tensor,  # [M, H, D] fp8 bytes, or int8 when qk_int8
    kv: tuple[torch.Tensor, torch.Tensor],  # ([N, D] fp8/int8, [N] fp32 scale)
    weights: torch.Tensor,  # [M, H] fp32 (q scale pre-folded when qk_int8)
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    qk_int8: bool = False,
) -> torch.Tensor:
    """Fused replacement for the chunked torch MQA logits on sm_8x."""
    k_values, k_scales = kv
    M, H, D = q.shape
    N = k_values.shape[0]
    logits = torch.empty((M, N), dtype=torch.float32, device=q.device)
    BLOCK_N = 128
    grid = (M, triton.cdiv(N, BLOCK_N))
    _mqa_logits_workspace_kernel[grid](
        q.view(torch.uint8),
        k_values.view(torch.uint8),
        k_scales.reshape(-1).float(),
        weights,
        cu_seqlen_ks.to(torch.int32),
        cu_seqlen_ke.to(torch.int32),
        logits,
        M,
        N,
        num_heads=H,
        head_dim=D,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_qd=q.stride(2),
        stride_kn=k_values.stride(0),
        stride_kd=k_values.stride(1),
        stride_wm=weights.stride(0),
        stride_wh=weights.stride(1),
        stride_lm=logits.stride(0),
        stride_ln=logits.stride(1),
        BLOCK_N=BLOCK_N,
        BLOCK_D=min(128, triton.next_power_of_2(D)),
        BLOCK_H=16,
        QK_INT8=qk_int8,
        num_warps=4,
    )
    return logits
