# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vision kernels must execute integer MMA and preserve dense-attention outputs."""

import pytest
import torch
from torch import nn

from vllm.models.deepseek_v4.common.vision_int8 import (
    VisionInt8LinearMethod,
    _vision_attention_int8,
    _vision_linear_int8,
    quantize_vision_weight,
    vision_attention_int8,
)


@pytest.mark.parametrize("n", [1, 33, 65, 576, 3456])
def test_vision_attention_int8_matches_reference(n):
    torch.manual_seed(42)
    q, k, v = [
        torch.randn(1, n, 16, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    ]
    actual = vision_attention_int8(q, k, v)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)
    relative_error = (
        actual.float() - expected.float()
    ).norm() / expected.float().norm()
    assert torch.isfinite(actual).all()
    assert relative_error < 0.04
    # Both dot products must compile to integer MMA, not a floating-point substitute.
    compiled = _vision_attention_int8.warmup(
        q,
        k,
        v,
        torch.empty_like(q),
        n,
        16,
        64,
        32,
        64,
        num_warps=4,
        grid=((n + 31) // 32, 16),
    )
    ptx = compiled.asm["ptx"]
    assert ".s32.s8.s8.s32" in ptx


@pytest.mark.parametrize(
    "shape",
    [
        (1024, 588),
        (3072, 1024),
        (1024, 1024),
        (5632, 1024),
        (1024, 2816),
        (4096, 9216),
        (4096, 4096),
    ],
)
def test_vision_linear_int8_loads_and_handles_real_shapes(shape):
    torch.manual_seed(17)
    n, k = shape
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    q, scale = quantize_vision_weight(w)
    layer = nn.Module()
    layer.weight = nn.Parameter(q, requires_grad=False)
    layer.weight_scale = nn.Parameter(scale, requires_grad=False)
    method = VisionInt8LinearMethod()
    method.process_weights_after_loading(layer)
    x = torch.randn(1, k, 37, device="cuda", dtype=torch.bfloat16).transpose(-1, -2)
    actual = method.apply(layer, x, bias)
    expected = torch.nn.functional.linear(x, w, bias)
    assert actual.shape == expected.shape
    assert (actual.float() - expected.float()).norm() / expected.float().norm() < 0.025


def test_vision_linear_compiles_integer_mma():
    x = torch.ones(37, 588, device="cuda", dtype=torch.bfloat16)
    w, scales = quantize_vision_weight(
        torch.ones(1024, 588, device="cuda", dtype=torch.bfloat16)
    )
    out = torch.empty(37, 1024, device="cuda", dtype=torch.bfloat16)
    compiled = _vision_linear_int8.warmup(
        x,
        w,
        scales,
        None,
        out,
        37,
        1024,
        588,
        32,
        64,
        32,
        False,
        num_warps=4,
        grid=(2, 16),
    )
    assert ".s32.s8.s8.s32" in compiled.asm["ptx"]


def test_vision_attention_zero_and_large_inputs_remain_finite():
    q = torch.zeros(1, 65, 16, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.ones_like(q)
    torch.testing.assert_close(vision_attention_int8(q, q, v), v)
    torch.manual_seed(2)
    q = torch.randn_like(q) * 100
    assert torch.isfinite(vision_attention_int8(q, q, v)).all()


def test_vision_int8_cuda_graph_replay_reads_new_inputs():
    """Captured vision work must use current image data rather than capture inputs."""
    q, k, v = [
        torch.randn(1, 65, 16, 64, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    for _ in range(3):
        vision_attention_int8(q, k, v)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = vision_attention_int8(q, k, v)
    q.copy_(torch.randn_like(q))
    v.copy_(torch.randn_like(v))
    graph.replay()
    expected = vision_attention_int8(q, k, v)
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)


def _compiled_variants(kernel) -> int:
    """Number of distinct specializations Triton has compiled for a kernel."""
    return sum(len(entry[0]) for entry in kernel.device_caches.values())


def test_vision_kernels_do_not_specialize_on_image_token_count():
    """Image resolution varies per request; it must not recompile the kernels.

    Triton specializes on every constexpr value and on divisibility by 16 of
    every plain integer argument, so token counts that are constexpr (or
    merely runtime) cost a fresh compile per novel image size: about 200 ms
    per linear shape and 500 ms for attention on an A5000, which is the
    whole of the vision latency gap against BF16. Sizes below mix multiples
    of 16 with odd counts to cover both specialization keys.
    """
    torch.manual_seed(5)
    w, scales = quantize_vision_weight(
        torch.randn(1024, 588, device="cuda", dtype=torch.bfloat16)
    )
    layer = nn.Module()
    layer.weight = nn.Parameter(w, requires_grad=False)
    layer.weight_scale = nn.Parameter(scales, requires_grad=False)
    method = VisionInt8LinearMethod()
    method.process_weights_after_loading(layer)
    before = _compiled_variants(_vision_linear_int8)
    for tokens in (97, 331, 512, 529, 33):
        x = torch.randn(1, tokens, 588, device="cuda", dtype=torch.bfloat16)
        method.apply(layer, x, None)
    torch.accelerator.synchronize()
    assert _compiled_variants(_vision_linear_int8) - before <= 1

    before = _compiled_variants(_vision_attention_int8)
    for tokens in (97, 331, 512, 529, 33):
        q, k, v = [
            torch.randn(1, tokens, 16, 64, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        ]
        vision_attention_int8(q, k, v)
    torch.accelerator.synchronize()
    assert _compiled_variants(_vision_attention_int8) - before <= 1
