# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical parity for vLLM's SparkInfer NVFP4 MoE integration.

The oracle is SparkInfer's pure-Torch NVFP4 implementation.  The test starts
from ModelOpt checkpoint layout, runs vLLM's load-time conversion and scale
adaptation, and compares both decode and prefill token counts.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.sparkinfer_moe import (
    SparkInferExperts,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    convert_to_nvfp4_moe_kernel_format,
    make_nvfp4_moe_quant_config,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

try:
    import pytest
except ImportError:
    pytest = None

if pytest is not None:
    pytest.importorskip("sparkinfer.moe")
    pytestmark = pytest.mark.skipif(
        not (
            torch.cuda.is_available()
            and current_platform.is_device_capability_family(120)
        ),
        reason="Requires SparkInfer NVFP4 on SM120/SM121",
    )


def _linearize_scales(
    scales: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """Undo scaled_fp4_quant's 128x4 swizzle to mimic checkpoint storage."""
    block_size = 16
    row_tiles = (rows + 127) // 128
    col_tiles = (cols + block_size * 4 - 1) // (block_size * 4)
    linear = []
    for scale in scales:
        tiled = scale.reshape(1, row_tiles, col_tiles, 32, 4, 4)
        tiled = tiled.permute(0, 1, 4, 3, 2, 5)
        linear.append(
            tiled.reshape(row_tiles * 128, col_tiles * 4)[:rows, : cols // 16]
        )
    return torch.stack(linear)


def _quantize_weights(
    experts: int,
    rows: int,
    cols: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fp4_max = 6.0
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    packed, scales, inverse_global_scales = [], [], []
    for _ in range(experts):
        weight = (
            torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16) / 15
        )
        quant_global_scale = (
            fp8_max * fp4_max / weight.abs().max().to(torch.float32)
        )
        weight_fp4, scale = ops.scaled_fp4_quant(weight, quant_global_scale)
        packed.append(weight_fp4)
        scales.append(scale)
        inverse_global_scales.append(torch.reciprocal(quant_global_scale))
    return (
        torch.stack(packed),
        _linearize_scales(torch.stack(scales), rows, cols),
        torch.stack(inverse_global_scales),
    )


def _load_checkpoint_weights(
    snapshot: Path,
    experts: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    from safetensors import safe_open

    shard = snapshot / "model-00002-of-00046.safetensors"
    values: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("w1", "w2", "w3")
    }
    scales: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("w1", "w2", "w3")
    }
    global_scales: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("w1", "w2", "w3")
    }
    input_scales: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("w1", "w2", "w3")
    }
    with safe_open(shard, framework="pt", device="cpu") as handle:
        for expert in range(experts):
            prefix = f"layers.0.ffn.experts.{expert}"
            for projection in ("w1", "w2", "w3"):
                values[projection].append(
                    handle.get_tensor(f"{prefix}.{projection}.weight")
                )
                scales[projection].append(
                    handle.get_tensor(f"{prefix}.{projection}.weight_scale")
                )
                global_scales[projection].append(
                    handle.get_tensor(f"{prefix}.{projection}.weight_scale_2")
                )
                input_scales[projection].append(
                    handle.get_tensor(f"{prefix}.{projection}.input_scale")
                )

    w1_global = torch.stack(global_scales["w1"])
    w3_global = torch.stack(global_scales["w3"])
    torch.testing.assert_close(w1_global, w3_global)
    return (
        torch.cat(
            [torch.stack(values["w1"]), torch.stack(values["w3"])], dim=1
        ).cuda(),
        torch.cat(
            [torch.stack(scales["w1"]), torch.stack(scales["w3"])], dim=1
        ).cuda(),
        w1_global.cuda(),
        torch.stack(
            [
                torch.stack(input_scales["w1"]),
                torch.stack(input_scales["w3"]),
            ],
            dim=1,
        ).cuda(),
        torch.stack(values["w2"]).cuda(),
        torch.stack(scales["w2"]).cuda(),
        torch.stack(global_scales["w2"]).cuda(),
        torch.stack(input_scales["w2"]).cuda(),
    )


@torch.inference_mode()
def _run_sparkinfer_nvfp4_torch_eager_parity(
    num_tokens: int,
    *,
    assert_metrics: bool = True,
):
    from sparkinfer.moe._shared.kernels.reference import (
        compare_to_reference,
        moe_reference_nvfp4,
    )

    set_random_seed(7 + num_tokens)
    # DSV4 TP=1 expert dimensions.  A reduced expert count keeps the fixture
    # compact without changing the GEMM geometry; top-k remains the model's 6.
    experts = int(os.getenv("NVFP4_TEST_EXPERTS", "8"))
    hidden, intermediate, topk = 4096, 2048, 6
    dtype = torch.bfloat16

    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        binding_path = os.getenv("NVFP4_TEST_BINDING")
        binding = (
            torch.load(binding_path, map_location="cpu", weights_only=True)
            if binding_path
            else None
        )
        hidden_states = (
            binding["hidden_states"].to(device="cuda", dtype=dtype)
            if binding
            else torch.randn(
                num_tokens, hidden, device="cuda", dtype=dtype
            )
            / 10
        )
        if binding and tuple(hidden_states.shape) != (num_tokens, hidden):
            raise ValueError(
                "binding hidden shape does not match requested test geometry: "
                f"{tuple(hidden_states.shape)} != {(num_tokens, hidden)}"
            )
        checkpoint_snapshot = os.getenv("NVFP4_CHECKPOINT_SNAPSHOT")
        if checkpoint_snapshot:
            (
                w13,
                w13_scale,
                w13_scale_2,
                w13_input_scale,
                w2,
                w2_scale,
                w2_scale_2,
                w2_input_scale,
            ) = _load_checkpoint_weights(Path(checkpoint_snapshot), experts)
        else:
            w13, w13_scale, w13_scale_2 = _quantize_weights(
                experts, 2 * intermediate, hidden
            )
            w2, w2_scale, w2_scale_2 = _quantize_weights(
                experts, hidden, intermediate
            )
            w13_input_scale = (
                torch.rand(experts, 2, device="cuda", dtype=torch.float32) * 0.5
                + 0.75
            )
            w2_input_scale = (
                torch.rand(experts, device="cuda", dtype=torch.float32) * 0.5
                + 0.75
            )

        converted = convert_to_nvfp4_moe_kernel_format(
            nvfp4_backend=NvFp4MoeBackend.SPARKINFER,
            layer=None,
            w13=w13,
            w13_scale=w13_scale,
            w13_scale_2=w13_scale_2,
            a13_scale=w13_input_scale,
            w2=w2,
            w2_scale=w2_scale,
            w2_scale_2=w2_scale_2,
            a2_scale=w2_input_scale,
            is_act_and_mul=True,
        )
        (
            w13_converted,
            w13_scale_converted,
            w13_scale_2,
            w13_input_scale_converted,
            w2_converted,
            w2_scale_converted,
            w2_scale_2,
            w2_input_scale_converted,
        ) = converted
        quant_config = make_nvfp4_moe_quant_config(
            backend=NvFp4MoeBackend.SPARKINFER,
            w13_scale=w13_scale_converted,
            w2_scale=w2_scale_converted,
            w13_scale_2=w13_scale_2,
            w2_scale_2=w2_scale_2,
            a13_scale=w13_input_scale_converted,
            a2_scale=w2_input_scale_converted,
        )
        moe_config = FusedMoEConfig(
            num_experts=experts,
            experts_per_token=topk,
            hidden_dim=hidden,
            intermediate_size=intermediate,
            num_local_experts=experts,
            num_logical_experts=experts,
            activation=MoEActivation.SILU,
            device="cuda",
            routing_method=RoutingMethodType.TopK,
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            in_dtype=dtype,
            max_num_tokens=max(32, num_tokens),
            swiglu_limit=10.0 if checkpoint_snapshot else None,
        )
        kernel = SparkInferExperts(moe_config, quant_config)
        layer = torch.nn.Module()
        layer.w13_weight = w13_converted
        layer.w13_weight_scale = w13_scale_converted
        layer.w13_weight_scale_2 = w13_scale_2
        layer.w13_input_scale = w13_input_scale_converted
        layer.w2_weight = w2_converted
        layer.w2_weight_scale = w2_scale_converted
        layer.w2_weight_scale_2 = w2_scale_2
        layer.w2_input_scale = w2_input_scale_converted
        kernel.process_weights_after_loading(layer)

        routing_mode = os.getenv("NVFP4_TEST_ROUTING", "random")
        if binding:
            routing_mode = "captured"
            topk_ids = binding["topk_ids"].to(device="cuda", dtype=torch.int64)
            topk_weights = binding["topk_weights"].to(
                device="cuda", dtype=torch.float32
            )
        elif routing_mode == "concentrated":
            topk_ids = torch.arange(
                topk,
                device="cuda",
                dtype=torch.int64,
            ).expand(num_tokens, -1)
            topk_logits = torch.randn(
                num_tokens,
                topk,
                device="cuda",
                dtype=torch.float32,
            )
        elif routing_mode == "random":
            routing = torch.randn(
                num_tokens,
                experts,
                device="cuda",
                dtype=torch.float32,
            )
            topk_logits, topk_ids = torch.topk(routing, topk, dim=-1)
        elif routing_mode == "padded":
            topk_ids = torch.full(
                (num_tokens, topk),
                -1,
                device="cuda",
                dtype=torch.int64,
            )
            topk_weights = torch.zeros(
                num_tokens,
                topk,
                device="cuda",
                dtype=torch.float32,
            )
        else:
            raise ValueError(f"unknown NVFP4_TEST_ROUTING={routing_mode!r}")
        if not binding and routing_mode != "padded":
            topk_weights = torch.softmax(topk_logits, dim=-1)
        output = torch.empty_like(hidden_states)
        kernel.apply(
            output=output,
            hidden_states=hidden_states,
            w1=w13_converted,
            w2=w2_converted,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=MoEActivation.SILU,
            global_num_experts=experts,
            expert_map=None,
            a1q_scale=None,
            a2_scale=None,
            workspace13=None,
            workspace2=None,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )
        assert kernel._experts is not None
        reference_topk_ids = topk_ids.clamp_min(0).to(torch.int32)
        reference = moe_reference_nvfp4(
            hidden_states,
            w13_converted,
            w13_scale_converted,
            kernel._experts.w1_alphas,
            w2_converted,
            w2_scale_converted,
            kernel._experts.w2_alphas,
            kernel._experts.a1_gscale,
            kernel._experts.a2_gscale,
            reference_topk_ids,
            topk_weights,
            experts,
            hidden,
            intermediate,
            quant_scale_math=(
                "reciprocal_multiply" if num_tokens <= 8 else "direct_division"
            ),
        )
        squared_router_reference = moe_reference_nvfp4(
            hidden_states,
            w13_converted,
            w13_scale_converted,
            kernel._experts.w1_alphas,
            w2_converted,
            w2_scale_converted,
            kernel._experts.w2_alphas,
            kernel._experts.a1_gscale,
            kernel._experts.a2_gscale,
            reference_topk_ids,
            topk_weights.square(),
            experts,
            hidden,
            intermediate,
            quant_scale_math=(
                "reciprocal_multiply" if num_tokens <= 8 else "direct_division"
            ),
        )
        torch.cuda.synchronize()

        metrics = compare_to_reference(output, reference)
        squared_router_metrics = compare_to_reference(
            output, squared_router_reference
        )
        print(
            f"num_tokens={num_tokens} "
            f"routing={routing_mode} "
            f"output_finite={output.isfinite().all().item()} "
            f"reference_finite={reference.isfinite().all().item()} "
            f"output_absmax={output.float().abs().max().item()} "
            f"reference_absmax={reference.float().abs().max().item()} "
            f"w1_alpha=[{kernel._experts.w1_alphas.min().item()}, "
            f"{kernel._experts.w1_alphas.max().item()}] "
            f"a1_gscale=[{kernel._experts.a1_gscale.min().item()}, "
            f"{kernel._experts.a1_gscale.max().item()}] "
            f"squared_router={squared_router_metrics}"
        )
        if assert_metrics:
            assert metrics.max_abs <= 8e-4, metrics
            assert metrics.rmse <= 5e-5, metrics
            assert metrics.cos > 0.9999, metrics
        print(f"num_tokens={num_tokens}: {metrics}")


if pytest is not None:

    @pytest.mark.parametrize("num_tokens", [1, 9])
    def test_sparkinfer_nvfp4_matches_torch_eager_after_vllm_conversion(
        num_tokens: int,
    ):
        _run_sparkinfer_nvfp4_torch_eager_parity(num_tokens)


if __name__ == "__main__":
    if not (
        torch.cuda.is_available()
        and current_platform.is_device_capability_family(120)
    ):
        raise SystemExit("Requires SparkInfer NVFP4 on SM120/SM121")
    token_counts = [
        int(value)
        for value in os.getenv("NVFP4_TEST_TOKENS", "1,9").split(",")
        if value
    ]
    for token_count in token_counts:
        _run_sparkinfer_nvfp4_torch_eager_parity(
            token_count,
            assert_metrics=False,
        )
