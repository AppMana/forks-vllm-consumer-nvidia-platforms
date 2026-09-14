# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80+ integer tensor-core kernels for DeepSeek V4 vision."""

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
)
from vllm.triton_utils import tl, triton

VISION_GROUP_SIZE = 32


def quantize_vision_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize [output, input] weights using independent 32-input channel groups."""
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("Vision weights must be a floating-point matrix")
    w = weight.float()
    if not torch.isfinite(w).all():
        raise ValueError("Vision weights contain non-finite values")
    n, k = w.shape
    grouped = F.pad(w, (0, (-k) % VISION_GROUP_SIZE)).reshape(n, -1, VISION_GROUP_SIZE)
    scale = grouped.abs().amax(dim=2) / 127
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = (grouped / scale[..., None]).round().clamp(-127, 127).to(torch.int8)
    return quantized.reshape(n, -1)[:, :k].contiguous(), scale


@register_weight_loader_v2_supported_method
class VisionInt8LinearMethod(LinearMethodBase):
    """W8A8 IMMA with groupwise activation scales to contain vision outliers."""

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        loader = extra_weight_attrs.get("weight_loader")
        n = sum(output_partition_sizes)
        layer.weight_block_size = (1, VISION_GROUP_SIZE)
        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(n, input_size_per_partition, dtype=torch.int8),
                input_dim=1,
                output_dim=0,
                weight_loader=loader,
            ),
        )
        layer.register_parameter(
            "weight_scale",
            BlockQuantScaleParameter(
                data=torch.empty(
                    n,
                    triton.cdiv(input_size_per_partition, VISION_GROUP_SIZE),
                    dtype=torch.float32,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=loader,
            ),
        )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if (
            layer.weight.dtype != torch.int8
            or layer.weight_scale.dtype != torch.float32
        ):
            raise ValueError("Vision IMMA requires signed INT8 weights and FP32 scales")
        layer.weight = nn.Parameter(layer.weight.data.contiguous(), requires_grad=False)
        layer.weight_scale = nn.Parameter(
            layer.weight_scale.data.contiguous(), requires_grad=False
        )

    def apply(
        self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1]).contiguous()
        m, k = x.shape
        n = layer.weight.shape[0]
        out = torch.empty((m, n), device=x.device, dtype=x.dtype)
        _vision_linear_int8[(triton.cdiv(m, 32), triton.cdiv(n, 64))](
            x,
            layer.weight,
            layer.weight_scale,
            bias,
            out,
            m,
            n,
            k,
            32,
            64,
            VISION_GROUP_SIZE,
            bias is not None,
            num_warps=4,
        )
        return out.reshape(*shape, n)


@triton.jit
def _round_int8(x):
    return tl.minimum(tl.maximum(tl.floor(x + 0.5), -127.0), 127.0).to(tl.int8)


@triton.jit
def _vision_linear_int8(
    X,
    W,
    S,
    Bias,
    Out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    dims = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for group in range(tl.cdiv(K, BK)):
        kk = group * BK + dims
        x = tl.load(
            X + rows[:, None] * K + kk[None, :],
            (rows[:, None] < M) & (kk[None, :] < K),
            0,
        ).to(tl.float32)
        scale = tl.maximum(tl.max(tl.abs(x), 1) / 127.0, 1.0e-20)
        xq = _round_int8(x / scale[:, None])
        w = tl.load(
            W + cols[None, :] * K + kk[:, None],
            (cols[None, :] < N) & (kk[:, None] < K),
            0,
        )
        ws = tl.load(S + cols * tl.cdiv(K, BK) + group, cols < N, 0)
        product = tl.dot(xq, w).to(tl.float32)
        acc += product * scale[:, None] * ws[None, :]
    if HAS_BIAS:
        bias = tl.load(Bias + cols, cols < N, 0).to(tl.float32)
        acc += bias[None, :]
    tl.store(
        Out + rows[:, None] * N + cols[None, :],
        acc,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _vision_attention_int8(
    Q,
    K,
    V,
    Out,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    head = tl.program_id(1)
    dims = tl.arange(0, D)
    q = tl.load(
        Q + (rows[:, None] * H + head) * D + dims[None, :], rows[:, None] < N, 0
    ).to(tl.float32)
    qs = tl.maximum(tl.max(tl.abs(q), 1) / 127.0, 1.0e-20)
    qi = _round_int8(q / qs[:, None])
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    for start in range(tl.cdiv(N, BN)):
        cols = start * BN + tl.arange(0, BN)
        offsets = (cols[:, None] * H + head) * D + dims[None, :]
        k = tl.load(K + offsets, cols[:, None] < N, 0).to(tl.float32)
        ks = tl.maximum(tl.max(tl.abs(k), 1) / 127.0, 1.0e-20)
        ki = _round_int8(k / ks[:, None])
        scores = tl.dot(qi, tl.trans(ki)).to(tl.float32)
        scores = scores * qs[:, None] * ks[None, :] * (D**-0.5)
        scores = tl.where(cols[None, :] < N, scores, -float("inf"))
        new_maximum = tl.maximum(maximum, tl.max(scores, 1))
        alpha = tl.exp(maximum - new_maximum)
        p = tl.exp(scores - new_maximum[:, None])
        denominator = denominator * alpha + tl.sum(p, 1)
        ps = tl.maximum(tl.max(p, 1) / 127.0, 1.0e-20)
        pi = _round_int8(p / ps[:, None])
        v = tl.load(V + offsets, cols[:, None] < N, 0).to(tl.float32)
        vs = tl.maximum(tl.max(tl.abs(v), 0) / 127.0, 1.0e-20)
        vi = _round_int8(v / vs[None, :])
        product = tl.dot(pi, vi).to(tl.float32)
        acc = acc * alpha[:, None] + product * ps[:, None] * vs[None, :]
        maximum = new_maximum
    tl.store(
        Out + (rows[:, None] * H + head) * D + dims[None, :],
        acc / denominator[:, None],
        rows[:, None] < N,
    )


def vision_attention_int8(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Noncausal [1, tokens, heads, dim] attention with INT8 QK and PV MMA."""
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("Vision IMMA expects matching [1, tokens, heads, dim] Q/K/V")
    if not q.is_cuda or torch.cuda.get_device_capability(q.device)[0] < 8:
        raise ValueError("Vision IMMA requires an SM80+ CUDA device")
    _, n, h, d = q.shape
    if d not in (32, 64, 128) or n == 0:
        raise ValueError("Vision IMMA requires nonempty tokens and head dim 32/64/128")
    q, k, v = (x.contiguous() for x in (q, k, v))
    out = torch.empty_like(q)
    _vision_attention_int8[(triton.cdiv(n, 32), h)](
        q,
        k,
        v,
        out,
        n,
        h,
        d,
        32,
        64,
        num_warps=4,
    )
    return out
