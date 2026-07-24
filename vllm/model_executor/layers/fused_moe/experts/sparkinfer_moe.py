# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sparkinfer fused-MoE experts for GB10 (SM120/SM121).

Native NVFP4 MoE via ``sparkinfer.moe.fused_moe`` (the CuTe-DSL kernel family
that consumes modelopt's on-disk NVFP4 layout directly): route -> FC1 ->
SwiGLU -> FC2 -> scatter in one launch. This replaces the Marlin weight-only
FP4 fallback ("GPU does not have native support for FP4") that the NVFP4
oracle otherwise selects on sm12x — Marlin dequantizes to bf16 and runs a
bf16 GEMM, forfeiting the FP4 tensor cores.

Weight lifecycle mirrors ``FlashInferB12xExperts`` (which binds FlashInfer PR
#3080's b12x wrapper we do not ship): the modelopt tensors vLLM already
stores map 1:1 onto sparkinfer's ``prepare_weights`` inputs —

    vLLM w13_weight (U8-packed E2M1)      -> w1_fp4
    vLLM w13_weight_scale (E4M3, /16 grp) -> w1_blockscale
    vLLM w13_weight_scale_2 (F32/expert)  -> w1_global_scale
    activation input global scale         -> a1_gscale   (and w2/a2 likewise)

Runtime is plan(Caps) -> bind(scratch, a, experts, topk) -> run, all
allocation-free / CUDA-graph-capture safe once the per-device scratch is
grown at plan time (vLLM's eager warmup precedes graph capture).
"""

from __future__ import annotations

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform


def _has_sparkinfer_moe() -> bool:
    try:
        from sparkinfer.moe import fused_moe
    except ImportError:
        return False
    return bool(fused_moe.is_supported())


class SparkInferExperts(mk.FusedMoEExpertsModular):
    """Native NVFP4 fused MoE on GB10 via sparkinfer.moe.fused_moe."""

    _ACTIVATION_MAP: dict[MoEActivation, str] = {
        MoEActivation.SILU: "silu",
        MoEActivation.RELU2_NO_MUL: "relu2",
    }

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        assert quant_config.quant_dtype == "nvfp4", (
            "SparkInferExperts only supports nvfp4 quantization."
        )
        self.out_dtype = moe_config.in_dtype
        self.num_local_experts = moe_config.num_local_experts
        self.global_num_experts = moe_config.num_experts
        self.topk = moe_config.experts_per_token
        self.hidden_dim = moe_config.hidden_dim
        self.intermediate_size_per_partition = (
            moe_config.intermediate_size_per_partition
        )
        self.max_num_tokens = moe_config.max_num_tokens

        activation = moe_config.activation
        if activation not in self._ACTIVATION_MAP:
            raise ValueError(
                f"SparkInferExperts does not support activation {activation!r}; "
                f"supported: {list(self._ACTIVATION_MAP.keys())}"
            )
        self._activation_str = self._ACTIVATION_MAP[activation]

        # Built lazily in process_weights_after_loading / first apply.
        self._weight_plan = None
        self._experts = None
        self._plan = None  # runtime scratch plan
        self._scratch: torch.Tensor | None = None
        self._planned_max_tokens = 0

    # -- capability gates (mirror FlashInferB12xExperts) --------------------

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return (
            p.is_cuda()
            and p.is_device_capability_family(120)
            and _has_sparkinfer_moe()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # sparkinfer performs in-kernel BF16->FP4 activation quant, so both
        # statically-calibrated (kNvfp4Dynamic activation) and W4A16
        # (activation_key=None) modelopt checkpoints are runtime-compatible.
        return (weight_key, activation_key) in (
            (kNvfp4Static, kNvfp4Dynamic),
            (kNvfp4Static, None),
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (MoEActivation.SILU, MoEActivation.RELU2_NO_MUL)

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # No expert parallelism yet: local expert count must equal global.
        return not moe_parallel_config.use_ep

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # sparkinfer applies the topk weights internally.
        return TopKWeightAndReduceNoOP()

    @property
    def expects_unquantized_inputs(self) -> bool:
        # sparkinfer takes BF16 hidden states and quantizes to FP4 in-kernel.
        return True

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # sparkinfer manages its own scratch (see _ensure_plan).
        return (1,), (0,), (M, K)

    # -- weight preparation -------------------------------------------------

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from sparkinfer.moe import fused_moe

        self._weight_plan = fused_moe.plan_weights(
            quant_modes="nvfp4",
            source_format="modelopt_nvfp4",
            activation=self._activation_str,
            params_dtype=self.out_dtype,
            num_experts=self.num_local_experts,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_size_per_partition,
        )
        # a1/a2 global activation scales: the modelopt input_scale companions,
        # or a synthesized 1.0 per expert for W4A16 checkpoints (sparkinfer
        # then uses its own in-kernel dynamic per-block activation scale, the
        # same treatment FlashInferB12xExperts applies).
        device = layer.w13_weight.device
        a1_gscale = self.a1_gscale
        a2_gscale = self.a2_gscale
        if a1_gscale is None:
            a1_gscale = torch.ones(
                self.num_local_experts, device=device, dtype=torch.float32
            )
        if a2_gscale is None:
            a2_gscale = torch.ones(
                self.num_local_experts, device=device, dtype=torch.float32
            )

        self._experts = fused_moe.prepare_weights(
            plan=self._weight_plan,
            w1_global_scale=layer.w13_weight_scale_2,
            w2_global_scale=layer.w2_weight_scale_2,
            w1_fp4=layer.w13_weight,
            w1_blockscale=layer.w13_weight_scale,
            w2_fp4=layer.w2_weight,
            w2_blockscale=layer.w2_weight_scale,
            a1_gscale=a1_gscale,
            a2_gscale=a2_gscale,
            params_dtype=self.out_dtype,
        )

    def _ensure_plan(self, num_tokens: int, device: torch.device) -> None:
        """Grow the runtime scratch plan to cover ``num_tokens``. Growth only
        happens when a larger batch than any seen appears; vLLM's eager
        warmup drives the max-token plan before graph capture, so captured
        launches reuse a stable scratch address."""
        from sparkinfer.moe import fused_moe

        cap_tokens = max(int(self.max_num_tokens), int(num_tokens))
        if self._plan is not None and cap_tokens <= self._planned_max_tokens:
            return
        self._plan = fused_moe.plan(
            fused_moe.Caps(
                max_tokens=cap_tokens,
                num_topk=self.topk,
                device=device,
                weight_plan=self._weight_plan,
                quant_mode="nvfp4",
            )
        )
        (spec,) = self._plan.scratch_specs()
        self._scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        self._planned_max_tokens = cap_tokens

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ):
        assert self._experts is not None and self._weight_plan is not None, (
            "process_weights_after_loading must run before SparkInferExperts.apply"
        )
        from sparkinfer.moe import fused_moe

        num_tokens = int(hidden_states.shape[0])
        self._ensure_plan(num_tokens, hidden_states.device)

        binding = fused_moe.bind(
            self._plan,
            scratch=self._scratch,
            a=hidden_states,
            experts=self._experts,
            topk_weights=topk_weights,
            topk_ids=topk_ids.to(torch.int32),
            output=output,
        )
        result = fused_moe.run(binding=binding)
        if result.data_ptr() != output.data_ptr():
            output.copy_(result)
