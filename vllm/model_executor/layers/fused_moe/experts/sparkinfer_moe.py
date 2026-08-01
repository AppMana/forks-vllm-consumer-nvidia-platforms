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
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    merge_nvfp4_gate_up_input_scales,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


def _modelopt_activation_gscales(
    w13_input_scale: torch.Tensor,
    w2_input_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert checkpoint input scales to SparkInfer reciprocal scales.

    ModelOpt stores ``*.input_scale`` in dequant form. SparkInfer's
    ``modelopt_nvfp4`` API instead takes its reciprocal as ``a*_gscale`` and
    derives the W4A4 runtime alpha as::

        weight_scale / a_gscale == weight_scale * checkpoint_input_scale

    Gate and up projections share one fused FC1 activation quantizer. ModelOpt
    emits the same input scale for both projections, so reject a malformed
    checkpoint instead of silently replacing either value with their maximum.
    """
    if w2_input_scale.ndim != 1 or w2_input_scale.shape[0] != w13_input_scale.shape[0]:
        raise ValueError(
            "w2_input_scale must have shape (num_experts,), got "
            f"{tuple(w2_input_scale.shape)}"
        )
    w13_scale = merge_nvfp4_gate_up_input_scales(w13_input_scale)
    w2_scale = w2_input_scale.to(torch.float32)
    if not torch.all(torch.isfinite(w13_scale) & (w13_scale > 0)):
        raise ValueError("w13_input_scale must be finite and positive")
    if not torch.all(torch.isfinite(w2_scale) & (w2_scale > 0)):
        raise ValueError("w2_input_scale must be finite and positive")
    return torch.reciprocal(w13_scale).contiguous(), torch.reciprocal(
        w2_scale
    ).contiguous()


def _sparkinfer_moe_unavailable_reason() -> str | None:
    """None when the sparkinfer NVFP4 MoE can run; else why it cannot."""
    try:
        from sparkinfer.moe import fused_moe
    except ImportError as exc:
        return f"sparkinfer.moe is not importable ({exc})"
    if not fused_moe.is_supported():
        return "sparkinfer.moe.fused_moe reports unsupported on this device"
    return None


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
        # SwiGLU clamp params (DSV4-Flash sets swiglu_limit). sparkinfer applies
        # the clamp natively via its Caps, so — unlike the Marlin fallback path
        # — we do NOT need to drop out of the clamp-capable backend set.
        self.swiglu_limit = moe_config.swiglu_limit
        self.swiglu_alpha = moe_config.swiglu_alpha
        self.swiglu_beta = moe_config.swiglu_beta

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
        # Per-token-count (plan, scratch) cache. sparkinfer selects a kernel
        # REGIME (micro / dynamic / tiny-decode) from the token count and the
        # scratch layout is regime-specific — one plan cannot serve both a
        # decode batch (micro) and a prefill batch (dynamic). Keying by exact
        # token count gives each CUDA-graph-captured decode size its own stable
        # scratch address, which capture requires.
        self._plan_cache: dict[int, tuple] = {}
        # Captured decode sizes must keep their scratch address for the life of
        # the graph, so they are never evicted. Every other size — prefill and
        # chunked-prefill token counts — is unbounded, so it is served by a
        # bounded LRU. Without the bound the cache grew a plan plus a scratch
        # tensor per distinct prefill length and never released one, which is a
        # steady GPU-memory climb over a long-lived server rather than a leak
        # that shows up in a short run.
        try:
            from vllm.config import get_current_vllm_config

            capture_sizes = (
                get_current_vllm_config().compilation_config.cudagraph_capture_sizes
            )
            self._pinned_sizes = frozenset(capture_sizes or ())
        except Exception:
            # Could not read the capture list. Evicting blind risks freeing the
            # scratch a captured graph points at, so leave the cache unbounded
            # -- the old behaviour -- rather than guess.
            self._pinned_sizes = frozenset()
            self._evict_enabled = False
        else:
            self._evict_enabled = True
        self._max_unpinned_plans = 8

    # -- capability gates (mirror FlashInferB12xExperts) --------------------

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        if not (p.is_cuda() and p.is_device_capability_family(120)):
            return False
        reason = _sparkinfer_moe_unavailable_reason()
        if reason is not None:
            # sm_12x is exactly where this expert is the intended path, so
            # losing it here is a silent drop to the Marlin bf16 fallback.
            logger.warning_once(
                "sparkinfer NVFP4 fused MoE unavailable on this sm_12x "
                "device (%s); the MoE oracle will fall back to a "
                "dequantize-to-bf16 path and forfeit the FP4 tensor cores.",
                reason,
            )
            return False
        return True

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
        # Preserve the checkpoint's calibrated W4A4 activation scaling.
        # Dynamic per-block quantization does not make this global factor 1:
        # SparkInfer's ModelOpt API takes its reciprocal and folds it into the
        # runtime alpha as weight_scale / reciprocal.
        a1_gscale, a2_gscale = _modelopt_activation_gscales(
            layer.w13_input_scale,
            layer.w2_input_scale,
        )
        logger.info_once(
            "sparkinfer ModelOpt NVFP4 tensors: weight=%s block_scale=%s "
            "weight_global=%s activation_global=%s",
            layer.w13_weight.dtype,
            layer.w13_weight_scale.dtype,
            layer.w13_weight_scale_2.dtype,
            a1_gscale.dtype,
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

    def _get_plan_scratch(self, num_tokens: int, device: torch.device) -> tuple:
        """Return the (plan, scratch) for exactly ``num_tokens``, building and
        caching on first sight. Sizing the plan at the actual token count keeps
        it in the kernel regime that ``run`` will re-derive for that count, so
        the bound workspace metadata matches (a single max-token plan is always
        'dynamic' and mismatches decode-sized 'micro' launches)."""
        from sparkinfer.moe import fused_moe

        cached = self._plan_cache.get(num_tokens)
        if cached is not None:
            return cached
        plan = fused_moe.plan(
            fused_moe.Caps(
                max_tokens=num_tokens,
                num_topk=self.topk,
                device=device,
                weight_plan=self._weight_plan,
                quant_mode="nvfp4",
                swiglu_limit=self.swiglu_limit,
                swiglu_alpha=self.swiglu_alpha,
                swiglu_beta=self.swiglu_beta,
            )
        )
        (spec,) = plan.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        self._plan_cache[num_tokens] = (plan, scratch)
        self._evict_unpinned()
        return plan, scratch

    def _evict_unpinned(self) -> None:
        """Drop the oldest non-captured plans once past the bound.

        dict preserves insertion order, so the first non-pinned key is the
        least recently inserted. Pinned (cudagraph-captured) sizes are skipped:
        their scratch address is baked into a captured graph.
        """
        if not self._evict_enabled:
            return
        unpinned = [k for k in self._plan_cache if k not in self._pinned_sizes]
        for key in unpinned[: max(0, len(unpinned) - self._max_unpinned_plans)]:
            del self._plan_cache[key]

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
        plan, scratch = self._get_plan_scratch(num_tokens, hidden_states.device)

        binding = fused_moe.bind(
            plan,
            scratch=scratch,
            a=hidden_states,
            experts=self._experts,
            topk_weights=topk_weights,
            topk_ids=topk_ids.to(torch.int32),
            output=output,
            input_scales_static=True,
        )
        result = fused_moe.run(binding=binding)
        if result.data_ptr() != output.data_ptr():
            output.copy_(result)
