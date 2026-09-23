#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time Humming fused MoE at DeepSeek V4.1 Flash expert shapes.

Builds the experts exactly the way tests/kernels/moe/test_moe.py does
(convert_to_humming_moe_kernel_format + HummingIndexedExperts.apply) and
times one MoE layer for several weight/activation formats, so W3A8 can be
compared with W4A8, W8A8 and W4A16 on the same GPU. Weights are random
codes with unit group scales: timing only, not numerics.

Run from the repository root:
    python tools/ampere/bench_humming_moe.py --formats uint3:int8 uint4:int8 \
        uint8:int8 uint4:bfloat16 --tokens 1 8 64 512 2048 8192
"""

import argparse
import json

import torch
from torch.nn import Parameter

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
    HummingIndexedExperts,
)
from vllm.model_executor.layers.quantization.utils import humming_utils
from vllm.platforms import current_platform
from vllm.utils import humming


def build(
    weight_dtype: str,
    input_dtype: str,
    num_experts: int,
    top_k: int,
    hidden: int,
    inter: int,
    group_size: int,
):
    activation = MoEActivation.SILU
    moe_config = make_dummy_moe_config(
        num_experts=num_experts,
        experts_per_token=top_k,
        hidden_dim=hidden,
        intermediate_size=inter,
        activation=activation,
        max_num_tokens=8192,
    )
    layer = torch.nn.Module()
    layer.moe_config = moe_config
    layer.params_dtype = torch.bfloat16
    weight_schema = humming.HummingWeightSchema(
        b_dtype=weight_dtype, weight_scale_group_size=group_size
    )
    for name, shape_n, shape_k, stacks in (
        ("w13", inter * 2, hidden, 2),
        ("w2", hidden, inter, 1),
    ):
        attrs = weight_schema.get_tensors_attrs(
            shape_n=shape_n,
            shape_k=shape_k,
            param_dtype=layer.params_dtype,
            num_experts=num_experts,
            stack_size=stacks,
        )
        for tensor_name, a in attrs.items():
            if a["dtype"] == torch.int32:
                t = torch.randint(
                    -(2**31), 2**31 - 1, a["shape"], dtype=torch.int32, device="cuda"
                )
            else:
                t = torch.ones(a["shape"], dtype=a["dtype"], device="cuda")
            layer.register_parameter(
                f"{name}_{tensor_name}", Parameter(t, requires_grad=False)
            )
    humming_utils.convert_to_humming_moe_kernel_format(
        layer,
        weight_schema=weight_schema,
        input_schema=humming.HummingInputSchema(
            a_dtype=humming.dtypes.DataType.from_str(input_dtype)
        ),
    )
    layer.local_num_experts = layer.global_num_experts = num_experts
    layer.hidden_size = hidden
    layer.intermediate_size_per_partition = inter
    quant_config = humming_utils.get_humming_moe_quant_config(layer)
    experts = HummingIndexedExperts(moe_config=moe_config, quant_config=quant_config)
    return experts, layer


def time_one(experts, layer, vllm_config, tokens: int, iters: int) -> float:
    cfg = experts.moe_config
    top_k, hidden, num_experts = cfg.experts_per_token, cfg.hidden_dim, cfg.num_experts
    ws13, ws2, _ = experts.workspace_shapes(
        M=tokens,
        N=cfg.intermediate_size,
        K=hidden,
        topk=top_k,
        global_num_experts=num_experts,
        local_num_experts=num_experts,
        expert_tokens_meta=None,
        activation=cfg.activation,
    )
    dev, dt = torch.device("cuda"), layer.params_dtype
    workspace13 = torch.empty(ws13, dtype=dt, device=dev)
    workspace2 = torch.empty(ws2, dtype=dt, device=dev)
    hidden_states = torch.randn((tokens, hidden), dtype=dt, device=dev)
    output = torch.empty_like(hidden_states)
    topk_weights = torch.full((tokens, top_k), 1 / top_k, dtype=dt, device=dev)
    # Uniform routing: every expert sees tokens * top_k / num_experts rows.
    topk_ids = (
        torch.randperm(num_experts, device=dev)[:top_k].repeat(tokens, 1)
        + torch.arange(tokens, device=dev).unsqueeze(1) * top_k
    ) % num_experts
    topk_ids = topk_ids.to(torch.int32)

    def run():
        experts.apply(
            output=output,
            hidden_states=hidden_states,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=cfg.activation,
            global_num_experts=num_experts,
            expert_map=None,
            a1q_scale=None,
            a2_scale=None,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )

    with set_forward_context(None, vllm_config, num_tokens=tokens):
        for _ in range(3):
            run()
        torch.accelerator.synchronize()
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            run()
        end.record()
        torch.accelerator.synchronize()
    return start.elapsed_time(end) / iters * 1000.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--formats",
        nargs="+",
        default=["uint3:int8", "uint4:int8", "uint8:int8", "uint4:bfloat16"],
        help="weight_dtype:input_dtype pairs",
    )
    p.add_argument("--tokens", nargs="+", type=int, default=[1, 8, 64, 512, 2048])
    p.add_argument("--num-experts", type=int, default=384)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--hidden", type=int, default=5120)
    p.add_argument("--inter", type=int, default=2304)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        for fmt in args.formats:
            weight_dtype, input_dtype = fmt.split(":")
            experts, layer = build(
                weight_dtype,
                input_dtype,
                args.num_experts,
                args.top_k,
                args.hidden,
                args.inter,
                args.group_size,
            )
            for tokens in args.tokens:
                us = time_one(experts, layer, vllm_config, tokens, args.iters)
                flops = 2 * tokens * args.top_k * 3 * args.hidden * args.inter
                print(
                    json.dumps(
                        {
                            "device": current_platform.get_device_name(),
                            "format": fmt,
                            "tokens": tokens,
                            "us": round(us, 1),
                            "tflops": round(flops / us / 1e6, 2),
                        }
                    ),
                    flush=True,
                )
            del experts, layer
            torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
