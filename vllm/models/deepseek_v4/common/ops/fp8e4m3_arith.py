# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Arithmetic-only FP8 E4M3 encode/decode for sm_8x.

Triton's NVIDIA backend refuses to lower the `tl.float8e4nv` cast on
sm_8x — even though FP8 hardware MMA isn't strictly required for plain
cast/bitcast. The kernels here let sm_8x kernels keep their structure by
swapping the unsupported cast for an arithmetic decode/encode that
operates on `uint8` and `fp32` only.

Decoder is bit-exact with PyTorch's `torch.float8_e4m3fn.to(fp32)` for
all 256 bytes (excluding NaN, encoded as 0x7F/0xFF).

Encoder is round-to-nearest-even to match PyTorch's
`torch.float8_e4m3fn` conversion.
"""
from __future__ import annotations

from vllm.triton_utils import tl, triton


@triton.jit
def fp8e4m3_decode_to_fp32(x_uint8):
    """Decode an E4M3FN byte to fp32 using arithmetic only.

    Layout: sign[7] exp[6:3] mant[2:0]; bias=7.

    - Subnormal (exp_bits == 0): value = sign * (mant/8) * 2^(-6)
    - Normal              : value = sign * (1 + mant/8) * 2^(exp_bits - 7)
    - NaN (0x7F, 0xFF)    : returns 448.0 / -448.0 (E4M3FN max finite);
      kv-cache values cannot be NaN, so this is benign.
    """
    b = x_uint8.to(tl.int32)
    sign_bit = (b >> 7) & 1
    exp_bits = (b >> 3) & 0xF
    mant_bits = b & 0x7
    mant_f = mant_bits.to(tl.float32) / 8.0
    is_subnormal = exp_bits == 0
    leading = tl.where(is_subnormal, 0.0, 1.0)
    exp_val = tl.where(
        is_subnormal, -6.0, (exp_bits - 7).to(tl.float32)
    )
    mag = (leading + mant_f) * tl.exp2(exp_val)
    sign_factor = tl.where(sign_bit == 1, -1.0, 1.0)
    return sign_factor * mag


@triton.jit
def _round_to_nearest_even(x):
    """Round positive fp32 values to nearest integer, ties to even."""
    floor_x = tl.floor(x)
    frac = x - floor_x
    floor_i = floor_x.to(tl.int32)
    tie_to_odd = (frac == 0.5) & ((floor_i & 1) == 1)
    round_up = (frac > 0.5) | tie_to_odd
    return (floor_i + round_up.to(tl.int32)).to(tl.int32)


@triton.jit
def fp8e4m3_encode_from_fp32(x):
    """Encode fp32 to E4M3FN byte using arithmetic only.

    Round-to-nearest-even, saturating at ±448. Output is the byte that
    PyTorch's `torch.float8_e4m3fn` stores for finite values.
    """
    fp8_max: tl.constexpr = 448.0
    x_clamped = tl.clamp(x, -fp8_max, fp8_max)
    sign = (x_clamped < 0).to(tl.int32)
    abs_x = tl.abs(x_clamped)
    is_zero = abs_x == 0.0
    log2x = tl.log2(tl.where(is_zero, 1.0, abs_x))
    exp_unbiased = tl.floor(log2x).to(tl.int32)
    exp_bits = exp_unbiased + 7
    is_subnormal = exp_bits <= 0
    norm_mant_int = _round_to_nearest_even(
        (abs_x / tl.exp2(exp_unbiased.to(tl.float32)) - 1.0) * 8.0
    )
    overflow = norm_mant_int >= 8
    final_exp_normal = tl.where(overflow, exp_bits + 1, exp_bits)
    final_mant_normal = tl.where(overflow, 0, norm_mant_int)
    sub_mant = _round_to_nearest_even(abs_x * 512.0)
    sub_overflow = sub_mant >= 8
    final_exp_sub = tl.where(sub_overflow, 1, 0)
    final_mant_sub = tl.where(sub_overflow, sub_mant - 8, sub_mant)
    final_exp = tl.where(is_subnormal, final_exp_sub, final_exp_normal)
    final_mant = tl.where(is_subnormal, final_mant_sub, final_mant_normal)
    is_max = (final_exp > 15) | ((final_exp == 15) & (final_mant > 6))
    final_exp = tl.where(is_max, 15, final_exp)
    final_mant = tl.where(is_max, 6, final_mant)  # E4M3FN max-finite = 0x7E
    final_exp = tl.where(is_zero, 0, final_exp)
    final_mant = tl.where(is_zero, 0, final_mant)
    byte = (sign << 7) | (final_exp << 3) | final_mant
    return byte.to(tl.uint8)
