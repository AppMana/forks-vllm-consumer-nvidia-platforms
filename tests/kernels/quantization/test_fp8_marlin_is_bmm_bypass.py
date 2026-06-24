# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Marlin FP8 must NOT repack `is_bmm=True` linears.

DeepSeek V4-Flash builds `wo_a` as a fused per-group `ColumnParallelLinear`
with `is_bmm=True` and consumes it directly via ``layer.weight`` /
``layer.weight_scale_inv`` from a custom FP8 einsum kernel — its ``apply()``
is never called.

On sm_8x (Ampere) we lack native FP8 hardware, so the linear-method selector
picks the Marlin FP8 kernel. Marlin's ``process_weights_after_loading``
normally repacks the weight via ``gptq_marlin_repack`` into a
``(size_k // 16, size_n * 16 // pack_factor)`` layout — which mangles the
``(out, in)`` shape the V4 fp8 einsum reads. The bypass under test keeps
``is_bmm=True`` layers in canonical FP8 block-quant ``(out, in)`` form.
"""
from __future__ import annotations

import pytest
import torch

from vllm.model_executor.kernels.linear.scaled_mm.marlin import (
    MarlinFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for Marlin FP8 process_weights_after_loading",
)


class _FakeBmmLinear(torch.nn.Module):
    """Mimics a vLLM ColumnParallelLinear that DeepSeek V4 marks `is_bmm`.

    Only the attributes Marlin's process_weights_after_loading reads are set.
    """

    def __init__(self, out_features: int, in_features: int, *, is_bmm: bool) -> None:
        super().__init__()
        # Block-quant FP8: `(out, in)` packed bytes + `(out/128, in/128)` fp32 scales.
        weight = torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn)
        scale = torch.empty(
            out_features // 128, in_features // 128, dtype=torch.float32
        )
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale_inv = torch.nn.Parameter(scale, requires_grad=False)
        self.input_size_per_partition = in_features
        self.output_size_per_partition = out_features
        self.weight_block_size = (128, 128)
        self.orig_dtype = torch.bfloat16
        self.logical_widths = [out_features]
        if is_bmm:
            self.is_bmm = True
            # n_groups for V4-Flash o_groups=8 (stored on the layer; not read
            # by Marlin's process_weights_after_loading but kept realistic).
            self.bmm_batch_size = 8


def _make_kernel(out_features: int, in_features: int) -> MarlinFP8ScaledMMLinearKernel:
    cfg = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8Static128BlockSym,
        activation_quant_key=kFp8Dynamic128Sym,
        weight_shape=(out_features, in_features),
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )
    return MarlinFP8ScaledMMLinearKernel(
        c=cfg,
        layer_param_names=("weight", "weight_scale_inv"),
    )


def test_is_bmm_layer_keeps_canonical_shape(default_vllm_config) -> None:
    """V4-Flash wo_a: 4096→8192 fp8 block-quant must NOT be Marlin-repacked."""
    out_features, in_features = 8192, 4096  # n_groups*o_lora_rank, hidden/n_groups
    layer = _FakeBmmLinear(out_features, in_features, is_bmm=True).cuda()
    kernel = _make_kernel(out_features, in_features)

    kernel.process_weights_after_loading(layer)

    # Weight stays `(out, in)` so the V4 fp8 einsum's `b.view(g, r, d)` works.
    assert layer.weight.shape == (out_features, in_features), (
        f"is_bmm layer weight got mangled: {layer.weight.shape}, "
        f"expected ({out_features}, {in_features})"
    )
    assert layer.weight.dtype == torch.float8_e4m3fn
    # Scale stays `(out/128, in/128)`.
    assert layer.weight_scale_inv.shape == (
        out_features // 128,
        in_features // 128,
    )
    # No marlin workspace allocated — bypass took the early return path.
    assert not hasattr(layer, "workspace") or layer.workspace is None


def test_non_bmm_layer_still_repacks(default_vllm_config) -> None:
    """Regular FP8 block-quant linears must still Marlin-repack on sm_8x."""
    out_features, in_features = 8192, 4096
    layer = _FakeBmmLinear(out_features, in_features, is_bmm=False).cuda()
    kernel = _make_kernel(out_features, in_features)

    # The non-bmm path needs a working Marlin repack. On sm_89+ the Marlin
    # kernel is normally bypassed (use VLLM_TEST_FORCE_FP8_MARLIN to force).
    cap = torch.cuda.get_device_capability()
    if cap[0] >= 9:
        pytest.skip("Marlin repack only exercised on sm_8x in production")

    kernel.process_weights_after_loading(layer)

    # Marlin repack changes shape: (size_k // 16, size_n * 16 // pack_factor)
    # = (in/16, out*16/4) = (256, 32768) for 4096→8192.
    assert layer.weight.shape != (out_features, in_features), (
        "non-bmm layer should have been Marlin-repacked"
    )
    assert layer.weight.shape == (in_features // 16, out_features * 16 // 4)
