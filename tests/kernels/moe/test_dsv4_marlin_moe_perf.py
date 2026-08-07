# SPDX-License-Identifier: Apache-2.0
"""Direct W4A16/W4A8 Marlin comparison at DSV4 TP=2 prefill shapes."""

from __future__ import annotations

import os

import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.dsv4_int import Dsv4Int4MoEMethod
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_act_int8_process_scales,
    marlin_make_workspace_new,
    marlin_moe_permute_scales,
)
from vllm.scalar_type import scalar_types


HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 2048
GROUP_SIZE = 32
TOPK = 8

pytestmark = pytest.mark.skipif(
    os.environ.get("DSV4_RUN_PERF_TESTS") != "1",
    reason="set DSV4_RUN_PERF_TESTS=1 for the allocation-heavy performance probe",
)


def _elapsed_ms(call, iterations: int = 3) -> float:
    call()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / iterations


def _make_variant(
    w13_source: torch.Tensor,
    w2_source: torch.Tensor,
    w13_scale_source: torch.Tensor,
    w2_scale_source: torch.Tensor,
    *,
    input_dtype: torch.dtype | None,
):
    int8_activations = input_dtype is torch.int8
    w13 = Dsv4Int4MoEMethod._repack_int4_for_marlin(
        w13_source.clone(),
        size_n=2 * INTERMEDIATE_SIZE,
        size_k=HIDDEN_SIZE,
        is_a_8bit=int8_activations,
    )
    w2 = Dsv4Int4MoEMethod._repack_int4_for_marlin(
        w2_source.clone(),
        size_n=HIDDEN_SIZE,
        size_k=INTERMEDIATE_SIZE,
        is_a_8bit=int8_activations,
    )
    w13_scale = marlin_moe_permute_scales(
        w13_scale_source.transpose(1, 2).contiguous(),
        size_k=HIDDEN_SIZE,
        size_n=2 * INTERMEDIATE_SIZE,
        group_size=GROUP_SIZE,
        is_a_8bit=int8_activations,
    )
    w2_scale = marlin_moe_permute_scales(
        w2_scale_source.transpose(1, 2).contiguous(),
        size_k=INTERMEDIATE_SIZE,
        size_n=HIDDEN_SIZE,
        group_size=GROUP_SIZE,
        is_a_8bit=int8_activations,
    )
    global1 = global2 = None
    if int8_activations:
        w13_scale, global1 = marlin_act_int8_process_scales(w13_scale)
        w2_scale, global2 = marlin_act_int8_process_scales(w2_scale)
    return w13, w2, w13_scale, w2_scale, global1, global2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", [1024, 8192])
def test_dsv4_marlin_prefill_w4a16_vs_w4a8(tokens: int) -> None:
    torch.manual_seed(121 + tokens)
    device = torch.device("cuda")
    experts = 16
    w13_source = torch.randint(
        -128,
        128,
        (experts, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE // 2),
        dtype=torch.int8,
        device=device,
    )
    w2_source = torch.randint(
        -128,
        128,
        (experts, HIDDEN_SIZE, INTERMEDIATE_SIZE // 2),
        dtype=torch.int8,
        device=device,
    )
    w13_scales = (
        torch.rand(
            experts,
            2 * INTERMEDIATE_SIZE,
            HIDDEN_SIZE // GROUP_SIZE,
            dtype=torch.float32,
            device=device,
        )
        * 0.02
        + 0.001
    ).to(torch.bfloat16)
    w2_scales = (
        torch.rand(
            experts,
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE // GROUP_SIZE,
            dtype=torch.float32,
            device=device,
        )
        * 0.02
        + 0.001
    ).to(torch.bfloat16)
    variants = {
        "w4a16": _make_variant(
            w13_source,
            w2_source,
            w13_scales,
            w2_scales,
            input_dtype=None,
        ),
        "w4a8": _make_variant(
            w13_source,
            w2_source,
            w13_scales,
            w2_scales,
            input_dtype=torch.int8,
        ),
    }
    del w13_source, w2_source, w13_scales, w2_scales
    x = torch.randn(tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    scores = torch.rand(tokens, TOPK, dtype=torch.float32, device=device)
    topk_weights = scores / scores.sum(dim=-1, keepdim=True)
    topk_ids = torch.randint(
        0, experts, (tokens, TOPK), dtype=torch.int32, device=device
    )
    empty = torch.empty(experts, 0, dtype=torch.int32, device=device)
    workspace = marlin_make_workspace_new(device, 4)

    def call(name: str):
        w13, w2, s13, s2, global1, global2 = variants[name]
        return fused_marlin_moe(
            x,
            w13,
            w2,
            None,
            None,
            s13,
            s2,
            topk_weights,
            topk_ids,
            quant_type_id=scalar_types.uint4b8.id,
            global_num_experts=experts,
            g_idx1=empty,
            g_idx2=empty,
            sort_indices1=empty,
            sort_indices2=empty,
            workspace=workspace,
            is_k_full=True,
            input_dtype=torch.int8 if name == "w4a8" else None,
            input_global_scale1=global1,
            input_global_scale2=global2,
        )

    outputs = {name: call(name) for name in variants}
    for name, output in outputs.items():
        assert output.shape == x.shape, name
        assert torch.isfinite(output).all(), name
    w4a16_ms = _elapsed_ms(lambda: call("w4a16"), iterations=1)
    w4a8_ms = _elapsed_ms(lambda: call("w4a8"), iterations=1)
    print(
        f"tokens={tokens} hidden={HIDDEN_SIZE} intermediate={INTERMEDIATE_SIZE} "
        f"topk={TOPK} w4a16_ms={w4a16_ms:.3f} w4a8_ms={w4a8_ms:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    for token_count in (1024, 8192):
        test_dsv4_marlin_prefill_w4a16_vs_w4a8(token_count)
