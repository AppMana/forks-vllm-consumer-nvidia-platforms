# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F
from torch import Tensor

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


_MHC_PRE_NUM_SPLIT_BUCKETS = (1, 2, 4, 8, 16, 32)


def _bucket_mhc_pre_num_split(split_k: int) -> int:
    for bucket in reversed(_MHC_PRE_NUM_SPLIT_BUCKETS):
        if split_k >= bucket:
            return bucket
    return 1


@triton.jit
def _rmsnorm_nw_kernel(
    x_ptr,
    out_ptr,
    stride_row,
    D,
    eps,
    RBLOCK: tl.constexpr,
):
    """Weight-free RMSNorm Triton kernel: out = x * rsqrt(mean(x², -1) + eps)."""
    row = tl.program_id(0)
    cols = tl.arange(0, RBLOCK)
    mask = cols < D

    x = tl.load(
        x_ptr + row * stride_row + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    var = tl.sum(x * x, 0) / D
    rstd = tl.rsqrt(var + eps)

    out = (x * rstd).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + row * D + cols, out, mask=mask, eviction_policy="evict_first")


def rmsnorm_nw(x: Tensor, eps: float) -> Tensor:
    """Weight-free RMSNorm over the last dimension.

    Treats *x* as ``[num_rows, D]`` where ``num_rows = product(shape[:-1])``.
    Returns a contiguous tensor with the same shape and dtype as *x*.
    """
    orig_shape = x.shape
    D = orig_shape[-1]
    x_2d = x.reshape(-1, D)
    num_rows = x_2d.shape[0]

    out = torch.empty_like(x_2d)
    RBLOCK = triton.next_power_of_2(D)

    _rmsnorm_nw_kernel[(num_rows,)](
        x_2d,
        out,
        x_2d.stride(0),
        D,
        eps,
        RBLOCK=RBLOCK,
        num_warps=1 if RBLOCK <= 512 else (4 if RBLOCK <= 4096 else 8),
    )
    return out.view(orig_shape)


@triton.jit
def _hc_head_reduce_store_kernel(
    pre_ptr,
    x_ptr,
    out_ptr,
    hidden_size: tl.constexpr,
    hc_mult: tl.constexpr,
    pre_stride_t: tl.constexpr,
    pre_stride_m: tl.constexpr,
    x_stride_t: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_stride_h: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offsets = block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offsets < hidden_size

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for mix_idx in tl.static_range(0, hc_mult):
        pre = tl.load(pre_ptr + token_idx * pre_stride_t + mix_idx * pre_stride_m).to(
            tl.float32
        )
        x = tl.load(
            x_ptr
            + token_idx * x_stride_t
            + mix_idx * x_stride_m
            + offsets * x_stride_h,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += pre * x

    tl.store(
        out_ptr + token_idx * out_stride_t + offsets * out_stride_h,
        acc,
        mask=mask,
    )


def hc_head_reduce_triton_kernel(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
) -> None:
    x_flat = x.flatten(-2)
    x_normed = rmsnorm_nw(x_flat, norm_eps)
    mixes = F.linear(x_normed.float(), hc_fn)
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps

    hidden_size = x.shape[-1]
    hc_mult = x.shape[-2]
    block_h = 1024
    _hc_head_reduce_store_kernel[(x.shape[0], (hidden_size + block_h - 1) // block_h)](
        pre,
        x,
        out,
        hidden_size,
        hc_mult,
        pre.stride(0),
        pre.stride(1),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_H=block_h,
        num_warps=4,
    )


def _hc_head_triton(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> None:
    """Fill pre-allocated `out` (T, H) in-place with the hc_head result."""
    if hs_flat.shape[0] == 0:
        return

    hc_head_reduce_triton_kernel(
        hs_flat,
        fn,
        hc_scale,
        hc_base,
        out,
        rms_eps,
        hc_eps,
    )
    return


direct_register_custom_op(
    op_name="hc_head_triton",
    op_func=_hc_head_triton,
    mutates_args=["out"],
)


def _compute_mhc_pre_num_split(
    num_tokens: int,
    hc_hidden_size: int,
    device: torch.device,
) -> int:
    if num_tokens <= 0 or device.type != "cuda":
        return 1
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    num_sms = torch.cuda.get_device_properties(index).multi_processor_count
    block_k = 64
    block_m = 64
    grid_size = triton.cdiv(num_tokens, block_m)
    split_k = num_sms // grid_size
    num_block_k = triton.cdiv(hc_hidden_size, block_k)
    split_k = min(split_k, num_block_k // 4)
    return _bucket_mhc_pre_num_split(max(split_k, 1))


@triton.jit(
    do_not_specialize=["num_tokens", "gemm_stride_s", "sq_stride_s"]
)
def _mhc_pre_fuse_triton_kernel(
    gemm_out_ptr,
    sqrsum_ptr,
    scale_ptr,
    base_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    layer_input_ptr,
    num_tokens,
    hidden: tl.constexpr,
    hc: tl.constexpr,
    hc_mult3: tl.constexpr,
    hc_hidden_size: tl.constexpr,
    sinkhorn_repeat: tl.constexpr,
    gemm_stride_s,
    gemm_stride_t: tl.constexpr,
    gemm_stride_n: tl.constexpr,
    sq_stride_s,
    sq_stride_t: tl.constexpr,
    residual_stride_t: tl.constexpr,
    residual_stride_i: tl.constexpr,
    residual_stride_h: tl.constexpr,
    post_stride_t: tl.constexpr,
    post_stride_i: tl.constexpr,
    comb_stride_t: tl.constexpr,
    comb_stride_i: tl.constexpr,
    comb_stride_j: tl.constexpr,
    layer_stride_t: tl.constexpr,
    layer_stride_h: tl.constexpr,
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    NUM_SPLIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    hidden_block = tl.program_id(1)
    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < hc_mult3

    mixes = tl.zeros((BLOCK_N,), dtype=tl.float32)
    sqsum = tl.full((), 0.0, dtype=tl.float32)
    for split in tl.static_range(0, NUM_SPLIT):
        mixes += tl.load(
            gemm_out_ptr
            + split * gemm_stride_s
            + token * gemm_stride_t
            + offs_n * gemm_stride_n,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        sqsum += tl.load(
            sqrsum_ptr + split * sq_stride_s + token * sq_stride_t
        ).to(tl.float32)

    mixes *= tl.rsqrt(sqsum / hc_hidden_size + rms_eps)

    if hidden_block == 0:
        for i in tl.static_range(0, hc):
            post_logit = (
                tl.sum(tl.where(offs_n == hc + i, mixes, 0.0), axis=0)
                * tl.load(scale_ptr + 1)
                + tl.load(base_ptr + hc + i)
            )
            tl.store(
                post_ptr + token * post_stride_t + i * post_stride_i,
                tl.sigmoid(post_logit) * hc_post_mult_value,
            )

        for row in tl.static_range(0, hc):
            row_vals = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for col in tl.static_range(0, hc):
                idx = 2 * hc + row * hc + col
                mix_idx = tl.sum(tl.where(offs_n == idx, mixes, 0.0), axis=0)
                val = mix_idx * tl.load(scale_ptr + 2) + tl.load(base_ptr + idx)
                row_vals = tl.where(offs_n == col, val, row_vals)
            row_vals = row_vals - tl.max(
                tl.where(offs_n < hc, row_vals, -float("inf")), axis=0
            )
            row_vals = tl.exp(row_vals)
            row_vals = row_vals / tl.sum(
                tl.where(offs_n < hc, row_vals, 0.0), axis=0
            )
            row_vals = row_vals + hc_sinkhorn_eps
            for col in tl.static_range(0, hc):
                row_col = tl.sum(tl.where(offs_n == col, row_vals, 0.0), axis=0)
                tl.store(
                    comb_ptr
                    + token * comb_stride_t
                    + row * comb_stride_i
                    + col * comb_stride_j,
                    row_col,
                )

        # First column normalization after softmax, then alternating row/column
        # normalizations for the remaining Sinkhorn iterations.
        for col in tl.static_range(0, hc):
            denom = tl.full((), 0.0, dtype=tl.float32)
            for row in tl.static_range(0, hc):
                denom += tl.load(
                    comb_ptr
                    + token * comb_stride_t
                    + row * comb_stride_i
                    + col * comb_stride_j,
                )
            denom += hc_sinkhorn_eps
            for row in tl.static_range(0, hc):
                ptr = (
                    comb_ptr
                    + token * comb_stride_t
                    + row * comb_stride_i
                    + col * comb_stride_j
                )
                val = tl.load(ptr) / denom
                tl.store(ptr, val)

        for _ in tl.static_range(1, sinkhorn_repeat):
            for row in tl.static_range(0, hc):
                denom = tl.full((), 0.0, dtype=tl.float32)
                for col in tl.static_range(0, hc):
                    denom += tl.load(
                        comb_ptr
                        + token * comb_stride_t
                        + row * comb_stride_i
                        + col * comb_stride_j,
                    )
                denom += hc_sinkhorn_eps
                for col in tl.static_range(0, hc):
                    ptr = (
                        comb_ptr
                        + token * comb_stride_t
                        + row * comb_stride_i
                        + col * comb_stride_j
                    )
                    val = tl.load(ptr) / denom
                    tl.store(ptr, val)

            for col in tl.static_range(0, hc):
                denom = tl.full((), 0.0, dtype=tl.float32)
                for row in tl.static_range(0, hc):
                    denom += tl.load(
                        comb_ptr
                        + token * comb_stride_t
                        + row * comb_stride_i
                        + col * comb_stride_j,
                    )
                denom += hc_sinkhorn_eps
                for row in tl.static_range(0, hc):
                    ptr = (
                        comb_ptr
                        + token * comb_stride_t
                        + row * comb_stride_i
                        + col * comb_stride_j
                    )
                    val = tl.load(ptr) / denom
                    tl.store(ptr, val)

    offs_h = hidden_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < hidden
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for i in tl.static_range(0, hc):
        mix_i = tl.sum(tl.where(offs_n == i, mixes, 0.0), axis=0)
        pre_i = (
            tl.sigmoid(mix_i * tl.load(scale_ptr + 0) + tl.load(base_ptr + i))
            + hc_pre_eps
        )
        res = tl.load(
            residual_ptr
            + token * residual_stride_t
            + i * residual_stride_i
            + offs_h * residual_stride_h,
            mask=mask_h,
            other=0.0,
        ).to(tl.float32)
        acc += pre_i * res
    tl.store(
        layer_input_ptr + token * layer_stride_t + offs_h * layer_stride_h,
        acc,
        mask=mask_h,
    )


def _mhc_pre_fuse_triton(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    residual_flat: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_splits, num_tokens, hc_mult3 = gemm_out_mul.shape
    hc = residual_flat.shape[1]
    hidden = residual_flat.shape[2]
    post_mix = torch.empty(num_tokens, hc, dtype=torch.float32, device=residual_flat.device)
    comb_mix = torch.empty(num_tokens, hc, hc, dtype=torch.float32, device=residual_flat.device)
    layer_input = torch.empty(
        num_tokens, hidden, dtype=torch.bfloat16, device=residual_flat.device
    )
    if num_tokens == 0:
        return post_mix, comb_mix, layer_input

    block_h = 1024
    block_n = triton.next_power_of_2(hc_mult3)
    _mhc_pre_fuse_triton_kernel[(num_tokens, triton.cdiv(hidden, block_h))](
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
        num_tokens,
        hidden,
        hc,
        hc_mult3,
        hc * hidden,
        sinkhorn_repeat,
        gemm_out_mul.stride(0),
        gemm_out_mul.stride(1),
        gemm_out_mul.stride(2),
        gemm_out_sqrsum.stride(0),
        gemm_out_sqrsum.stride(1),
        residual_flat.stride(0),
        residual_flat.stride(1),
        residual_flat.stride(2),
        post_mix.stride(0),
        post_mix.stride(1),
        comb_mix.stride(0),
        comb_mix.stride(1),
        comb_mix.stride(2),
        layer_input.stride(0),
        layer_input.stride(1),
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        NUM_SPLIT=num_splits,
        BLOCK_N=block_n,
        BLOCK_H=block_h,
        num_warps=4,
    )
    return post_mix, comb_mix, layer_input


def _mhc_pre_fuse_from_gemm_torch(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    residual_flat: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_splits, num_tokens, hc_mult3 = gemm_out_mul.shape
    del num_splits
    hc = residual_flat.shape[1]
    hidden = residual_flat.shape[2]
    assert hc_mult3 == hc * (hc + 2)

    mixes = gemm_out_mul.sum(dim=0)
    sqrsum = gemm_out_sqrsum.sum(dim=0)
    mixes = mixes * torch.rsqrt(sqrsum.unsqueeze(-1) / (hc * hidden) + rms_eps)

    pre_mix = torch.sigmoid(mixes[:, :hc] * hc_scale[0] + hc_base[:hc])
    pre_mix = pre_mix + hc_pre_eps

    post_mix = torch.sigmoid(
        mixes[:, hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc]
    )
    post_mix = post_mix * hc_post_mult_value

    comb_mix = mixes[:, 2 * hc :].view(num_tokens, hc, hc)
    comb_mix = comb_mix * hc_scale[2] + hc_base[2 * hc :].view(1, hc, hc)
    comb_mix = torch.softmax(comb_mix, dim=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = (
        pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32)
    ).sum(dim=1).to(torch.bfloat16)
    return post_mix, comb_mix, layer_input


def mhc_pre_triton(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from vllm.models.deepseek_v4.nvidia_sm86.triton_kernels import (
        tf32_hc_prenorm_gemm_triton,
    )

    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    hc_hidden_size = hc_mult * hidden_size
    outer_shape = residual.shape[:-2]

    assert fn.shape == (hc_mult3, hc_hidden_size)
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    residual_flat = residual.reshape(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    if num_tokens == 0:
        return (
            torch.empty(*outer_shape, hc_mult, 1, dtype=torch.float32, device=residual.device),
            torch.empty(*outer_shape, hc_mult, hc_mult, dtype=torch.float32, device=residual.device),
            torch.empty(*outer_shape, hidden_size, dtype=torch.bfloat16, device=residual.device),
        )

    requested_splits = max(1, int(n_splits))
    n_splits = (
        _compute_mhc_pre_num_split(num_tokens, hc_hidden_size, residual.device)
        if requested_splits == 1
        else requested_splits
    )
    gemm_out_mul = torch.empty(
        n_splits,
        num_tokens,
        hc_mult3,
        dtype=torch.float32,
        device=residual.device,
    )
    gemm_out_sqrsum = torch.empty(
        n_splits,
        num_tokens,
        dtype=torch.float32,
        device=residual.device,
    )
    tf32_hc_prenorm_gemm_triton(
        residual_flat.reshape(num_tokens, hc_hidden_size),
        fn,
        gemm_out_mul,
        gemm_out_sqrsum,
        n_splits,
    )

    post_mix, comb_mix, layer_input = _mhc_pre_fuse_triton(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_flat,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )
    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


def _mhc_pre_triton_op(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return mhc_pre_triton(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
    )


def _mhc_pre_triton_fake(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del fn, hc_scale, hc_base, rms_eps, hc_pre_eps
    del hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits
    outer_shape = residual.shape[:-2]
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    return (
        torch.empty(
            *outer_shape,
            hc_mult,
            1,
            dtype=torch.float32,
            device=residual.device,
        ),
        torch.empty(
            *outer_shape,
            hc_mult,
            hc_mult,
            dtype=torch.float32,
            device=residual.device,
        ),
        torch.empty(
            *outer_shape,
            hidden_size,
            dtype=residual.dtype,
            device=residual.device,
        ),
    )


direct_register_custom_op(
    op_name="mhc_pre_triton",
    op_func=_mhc_pre_triton_op,
    fake_impl=_mhc_pre_triton_fake,
)


@triton.jit(do_not_specialize=["num_tokens"])
def _mhc_post_triton_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    out_ptr,
    num_tokens,
    hc: tl.constexpr,
    hidden: tl.constexpr,
    x_stride_t: tl.constexpr,
    x_stride_h: tl.constexpr,
    residual_stride_t: tl.constexpr,
    residual_stride_i: tl.constexpr,
    residual_stride_h: tl.constexpr,
    post_stride_t: tl.constexpr,
    post_stride_j: tl.constexpr,
    comb_stride_t: tl.constexpr,
    comb_stride_i: tl.constexpr,
    comb_stride_j: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_j: tl.constexpr,
    out_stride_h: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    j = tl.program_id(1)
    hidden_block = tl.program_id(2)
    offs_h = hidden_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < hidden

    x = tl.load(
        x_ptr + token * x_stride_t + offs_h * x_stride_h,
        mask=mask_h,
        other=0.0,
    ).to(tl.float32)
    post = tl.load(post_ptr + token * post_stride_t + j * post_stride_j).to(
        tl.float32
    )
    acc = post * x

    for i in tl.static_range(0, hc):
        comb = tl.load(
            comb_ptr + token * comb_stride_t + i * comb_stride_i + j * comb_stride_j
        ).to(tl.float32)
        res = tl.load(
            residual_ptr
            + token * residual_stride_t
            + i * residual_stride_i
            + offs_h * residual_stride_h,
            mask=mask_h,
            other=0.0,
        ).to(tl.float32)
        acc += comb * res

    tl.store(
        out_ptr + token * out_stride_t + j * out_stride_j + offs_h * out_stride_h,
        acc,
        mask=mask_h,
    )


def mhc_post_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    num_tokens = residual.numel() // (residual.shape[-2] * residual.shape[-1])
    hc = residual.shape[-2]
    hidden = residual.shape[-1]
    x_flat = x.reshape(num_tokens, hidden)
    residual_flat = residual.reshape(num_tokens, hc, hidden)
    post_flat = post_layer_mix.reshape(num_tokens, hc, 1)
    comb_flat = comb_res_mix.reshape(num_tokens, hc, hc)
    out = torch.empty_like(residual_flat)
    if num_tokens == 0:
        return out.view_as(residual)

    block_h = 1024
    _mhc_post_triton_kernel[(num_tokens, hc, triton.cdiv(hidden, block_h))](
        x_flat,
        residual_flat,
        post_flat,
        comb_flat,
        out,
        num_tokens,
        hc,
        hidden,
        x_flat.stride(0),
        x_flat.stride(1),
        residual_flat.stride(0),
        residual_flat.stride(1),
        residual_flat.stride(2),
        post_flat.stride(0),
        post_flat.stride(1),
        comb_flat.stride(0),
        comb_flat.stride(1),
        comb_flat.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_H=block_h,
        num_warps=8,
    )
    return out.view_as(residual)


def _mhc_post_triton_op(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    return mhc_post_triton(x, residual, post_layer_mix, comb_res_mix)


def _mhc_post_triton_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    del x, post_layer_mix, comb_res_mix
    return torch.empty_like(residual)


direct_register_custom_op(
    op_name="mhc_post_triton",
    op_func=_mhc_post_triton_op,
    fake_impl=_mhc_post_triton_fake,
)


@triton.jit(
    do_not_specialize=["num_tokens", "gemm_stride_s", "sq_stride_s"]
)
def _mhc_fused_post_prenorm_gemm_triton_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    fn_ptr,
    gemm_out_ptr,
    sqrsum_ptr,
    residual_out_ptr,
    num_tokens,
    hc: tl.constexpr,
    hidden: tl.constexpr,
    n_out: tl.constexpr,
    x_stride_t: tl.constexpr,
    x_stride_h: tl.constexpr,
    residual_stride_t: tl.constexpr,
    residual_stride_i: tl.constexpr,
    residual_stride_h: tl.constexpr,
    post_stride_t: tl.constexpr,
    post_stride_j: tl.constexpr,
    comb_stride_t: tl.constexpr,
    comb_stride_i: tl.constexpr,
    comb_stride_j: tl.constexpr,
    fn_stride_n: tl.constexpr,
    fn_stride_i: tl.constexpr,
    fn_stride_h: tl.constexpr,
    gemm_stride_s,
    gemm_stride_t: tl.constexpr,
    gemm_stride_n: tl.constexpr,
    sq_stride_s,
    sq_stride_t: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_i: tl.constexpr,
    out_stride_h: tl.constexpr,
    NUM_SPLIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    n_block = tl.program_id(1)
    split = tl.program_id(2)

    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, BLOCK_H)
    h_per_split = tl.cdiv(hidden, NUM_SPLIT)
    h_start = split * h_per_split
    h_end = tl.minimum(h_start + h_per_split, hidden)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    sq = tl.full((), 0.0, dtype=tl.float32)
    for h0 in tl.range(0, h_per_split, BLOCK_H):
        h = h_start + h0 + offs_h
        mask_h = h < h_end
        x = tl.load(
            x_ptr + token * x_stride_t + h * x_stride_h,
            mask=mask_h,
            other=0.0,
        ).to(tl.float32)

        for j in tl.static_range(0, hc):
            post = tl.load(post_ptr + token * post_stride_t + j * post_stride_j).to(
                tl.float32
            )
            new_r = post * x
            for i in tl.static_range(0, hc):
                comb = tl.load(
                    comb_ptr
                    + token * comb_stride_t
                    + i * comb_stride_i
                    + j * comb_stride_j
                ).to(tl.float32)
                res = tl.load(
                    residual_ptr
                    + token * residual_stride_t
                    + i * residual_stride_i
                    + h * residual_stride_h,
                    mask=mask_h,
                    other=0.0,
                ).to(tl.float32)
                new_r += comb * res

            if n_block == 0:
                tl.store(
                    residual_out_ptr
                    + token * out_stride_t
                    + j * out_stride_i
                    + h * out_stride_h,
                    new_r,
                    mask=mask_h,
                )
                sq += tl.sum(tl.where(mask_h, new_r * new_r, 0.0), axis=0)

            fn = tl.load(
                fn_ptr
                + offs_n[:, None] * fn_stride_n
                + j * fn_stride_i
                + h[None, :] * fn_stride_h,
                mask=(offs_n[:, None] < n_out) & mask_h[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(fn * new_r[None, :], axis=1)

    tl.store(
        gemm_out_ptr
        + split * gemm_stride_s
        + token * gemm_stride_t
        + offs_n * gemm_stride_n,
        acc,
        mask=offs_n < n_out,
    )
    if n_block == 0:
        tl.store(
            sqrsum_ptr + split * sq_stride_s + token * sq_stride_t,
            sq,
        )


def _mhc_fused_post_prenorm_gemm_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    n_splits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = residual.numel() // (residual.shape[-2] * residual.shape[-1])
    hc = residual.shape[-2]
    hidden = residual.shape[-1]
    n_out = fn.shape[0]
    x_flat = x.reshape(num_tokens, hidden)
    residual_flat = residual.reshape(num_tokens, hc, hidden)
    post_flat = post_layer_mix.reshape(num_tokens, hc)
    comb_flat = comb_res_mix.reshape(num_tokens, hc, hc)
    fn_view = fn.view(n_out, hc, hidden)
    residual_out = torch.empty_like(residual_flat)
    gemm_out_mul = torch.empty(
        n_splits,
        num_tokens,
        n_out,
        dtype=torch.float32,
        device=residual.device,
    )
    gemm_out_sqrsum = torch.empty(
        n_splits,
        num_tokens,
        dtype=torch.float32,
        device=residual.device,
    )
    if num_tokens == 0:
        return residual_out, gemm_out_mul, gemm_out_sqrsum

    block_n = 4
    block_h = 256
    grid = (num_tokens, triton.cdiv(n_out, block_n), n_splits)
    _mhc_fused_post_prenorm_gemm_triton_kernel[grid](
        x_flat,
        residual_flat,
        post_flat,
        comb_flat,
        fn_view,
        gemm_out_mul,
        gemm_out_sqrsum,
        residual_out,
        num_tokens,
        hc,
        hidden,
        n_out,
        x_flat.stride(0),
        x_flat.stride(1),
        residual_flat.stride(0),
        residual_flat.stride(1),
        residual_flat.stride(2),
        post_flat.stride(0),
        post_flat.stride(1),
        comb_flat.stride(0),
        comb_flat.stride(1),
        comb_flat.stride(2),
        fn_view.stride(0),
        fn_view.stride(1),
        fn_view.stride(2),
        gemm_out_mul.stride(0),
        gemm_out_mul.stride(1),
        gemm_out_mul.stride(2),
        gemm_out_sqrsum.stride(0),
        gemm_out_sqrsum.stride(1),
        residual_out.stride(0),
        residual_out.stride(1),
        residual_out.stride(2),
        NUM_SPLIT=n_splits,
        BLOCK_N=block_n,
        BLOCK_H=block_h,
        num_warps=8,
    )
    return residual_out, gemm_out_mul, gemm_out_sqrsum


def mhc_fused_post_pre_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = residual.numel() // (residual.shape[-2] * residual.shape[-1])
    if num_tokens <= 16:
        requested_splits = max(1, int(n_splits))
        hidden_size = residual.shape[-1]
        n_splits = 4 if requested_splits == 1 else requested_splits
        if num_tokens < 8 and hidden_size <= 4096 and requested_splits == 1:
            n_splits = 8
        residual_cur, gemm_out_mul, gemm_out_sqrsum = (
            _mhc_fused_post_prenorm_gemm_triton(
                x,
                residual,
                post_layer_mix,
                comb_res_mix,
                fn,
                n_splits,
            )
        )
        post_mix_cur, comb_mix_cur, layer_input_cur = _mhc_pre_fuse_triton(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_cur,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
        outer_shape = residual.shape[:-2]
        hc_mult = residual.shape[-2]
        return (
            residual_cur.view_as(residual),
            post_mix_cur.view(*outer_shape, hc_mult, 1),
            comb_mix_cur.view(*outer_shape, hc_mult, hc_mult),
            layer_input_cur.view(*outer_shape, hidden_size),
        )

    residual_cur = mhc_post_triton(x, residual, post_layer_mix, comb_res_mix)
    post_mix_cur, comb_mix_cur, layer_input_cur = mhc_pre_triton(
        residual_cur,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
    )
    return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur


def _mhc_fused_post_pre_triton_op(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return mhc_fused_post_pre_triton(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
    )


def _mhc_fused_post_pre_triton_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del x, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base
    del rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value
    del sinkhorn_repeat, n_splits
    outer_shape = residual.shape[:-2]
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    return (
        torch.empty_like(residual),
        torch.empty(
            *outer_shape,
            hc_mult,
            1,
            dtype=torch.float32,
            device=residual.device,
        ),
        torch.empty(
            *outer_shape,
            hc_mult,
            hc_mult,
            dtype=torch.float32,
            device=residual.device,
        ),
        torch.empty(
            *outer_shape,
            hidden_size,
            dtype=residual.dtype,
            device=residual.device,
        ),
    )


direct_register_custom_op(
    op_name="mhc_fused_post_pre_triton",
    op_func=_mhc_fused_post_pre_triton_op,
    fake_impl=_mhc_fused_post_pre_triton_fake,
)
