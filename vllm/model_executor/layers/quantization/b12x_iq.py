# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native GGUF IQ expert method using the canonical planned b12x API.

The GGUF loader supplies indexed source descriptors; raw expert Parameters are
never allocated. The runner owns routing, shared experts and cross-rank sums.
"""

from bisect import bisect_left
from typing import Any

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.utils.b12x import B12xWarmupUnit, b12x_warmup_token_counts


class B12xIQMoEMethod(FusedMoEMethodBase):
    def __init__(self, moe, *, gate, up, down):
        super().__init__(moe)
        parallel = moe.moe_parallel_config
        if (
            parallel.tp_size != 1
            or parallel.dp_size != 1
            or parallel.is_sequence_parallel
            or parallel.use_all2all_kernels
            or parallel.enable_eplb
        ):
            raise ValueError(
                "Native IQ requires full experts, replicated tokens, DP1, no SP/EPLB"
            )
        if moe.activation != MoEActivation.SILU or moe.in_dtype != torch.bfloat16:
            raise ValueError("Native IQ requires BF16 SiLU")
        if moe.has_bias or moe.is_lora_enabled:
            raise ValueError("Native IQ does not support expert bias or LoRA")
        self.sources = (gate, up, down)
        self.experts: Any = None
        self._plans: dict[int, tuple[Any, tuple[torch.Tensor, ...]]] = {}
        self._capacities: tuple[int, ...] = ()
        self._output: torch.Tensor | None = None

    @property
    def topk_indices_dtype(self):
        return torch.int32

    def get_fused_moe_quant_config(self, layer):
        # Encodings belong to the prepared source, independently for FC1/FC2.
        return None

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **kwargs,
    ):
        gate, _, down = self.sources
        if (hidden_size, intermediate_size_per_partition, params_dtype) != (
            gate.columns,
            gate.rows,
            torch.bfloat16,
        ):
            raise ValueError("Native IQ source geometry differs from the layer")
        if (
            down.experts != self.moe.num_experts
            or num_experts != self.moe.num_local_experts
        ):
            raise ValueError("Native IQ expert counts differ from the layer")
        # Carries the module device without allocating checkpoint-sized weights.
        layer.register_buffer(
            "_b12x_iq_device", torch.empty(0, device=self.moe.device), persistent=False
        )

    def process_weights_after_loading(self, layer):
        from b12x.moe import fused_moe as moe

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Native IQ weights must be loaded before capture")
        if layer.apply_router_weight_on_input:
            raise ValueError("Native IQ expects router weights on expert output")
        if self.experts is not None:
            raise RuntimeError(
                "Reload requires rebuilding native IQ plans and captures"
            )
        device = layer._b12x_iq_device.device
        mapping = layer.expert_map
        if mapping is None:
            ids = tuple(range(self.moe.num_experts))
        else:
            host_map = mapping.cpu().tolist()
            ordered = sorted(
                (local, global_id)
                for global_id, local in enumerate(host_map)
                if local >= 0
            )
            if [local for local, _ in ordered] != list(
                range(self.moe.num_local_experts)
            ):
                raise ValueError("Expert map must assign each local slot exactly once")
            ids = tuple(global_id for _, global_id in ordered)
        gate, up, down = self.sources
        fc2 = (
            None
            if down.quant_type == gate.quant_type
            else (
                "fp4_e8m0_k32"
                if down.quant_type == "MXFP4"
                else down.quant_type.lower()
            )
        )
        plan = moe.plan_weights(
            source=moe.PackedSource(
                format=gate.quant_type.lower(), fc2_format=fc2, w13_layout="w31"
            ),
            activation=moe.ActivationSpec(
                mode="a16",
                nonlinearity="silu",
                io_dtype=torch.bfloat16,
                swiglu_limit=self.moe.swiglu_limit,
                swiglu_alpha=self.moe.swiglu_alpha,
                swiglu_beta=self.moe.swiglu_beta,
            ),
            geometry=moe.MoEGeometry(
                num_experts=len(ids),
                hidden_size=gate.columns,
                intermediate_size=gate.rows,
            ),
        )
        self.experts = moe.prepare_weights(
            plan=plan, weights=moe.GGUFExpertWeights(gate, up, down, ids, device)
        )
        layer.b12x_warmup_provider = self
        # Memory profiling runs before kernel_warmup. Allocate and prepare the
        # serving capacity now so profiling accounts for resident workspaces.
        config = get_current_vllm_config_or_none()
        sizes = []
        if config is not None:
            sizes.extend(config.compilation_config.cudagraph_capture_sizes or [])
            sizes.extend(
                size
                for size in config.compilation_config.compile_sizes or []
                if isinstance(size, int)
            )
        counts = b12x_warmup_token_counts(
            max_tokens=self.moe.max_num_tokens, cudagraph_capture_sizes=sizes
        )
        self.warmup_launches(layer, token_counts=counts)

    def get_b12x_warmup_unit(self, layer, token_counts, output_dtype):
        if output_dtype != torch.bfloat16:
            raise ValueError("Native IQ warmup requires BF16")
        # Every owner needs prepared state and buffers, even when compiled
        # callables are shared across identical geometries by b12x's cache.
        return B12xWarmupUnit(
            name="native IQ MoE",
            key=(type(self), id(self)),
            compile=lambda: self.warmup_launches(layer, token_counts=token_counts),
        )

    @torch.inference_mode()
    def warmup_launches(self, layer, *, token_counts):
        from b12x.moe import fused_moe as moe
        from b12x.preparation import PreparationSession, PreparedCall, require_prepared

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Native IQ planning must precede capture")
        if self.experts is None:
            raise RuntimeError("Load native IQ weights before warmup")
        counts = tuple(sorted({int(n) for n in token_counts if int(n) > 0}))
        if not counts:
            raise ValueError("Native IQ warmup requires positive capacities")
        if counts[-1] > self.moe.max_num_tokens:
            raise ValueError("Warmup count exceeds the configured native IQ capacity")
        missing = tuple(n for n in counts if n not in self._plans)
        if not missing:
            return
        device, h, topk = (
            self.experts.device,
            self.experts.hidden_size,
            self.moe.experts_per_token,
        )
        x = torch.zeros((counts[-1], h), dtype=torch.bfloat16, device=device)
        if self._output is None:
            self._output = torch.empty(
                self.moe.max_num_tokens, h, dtype=torch.bfloat16, device=device
            )
        output = self._output
        ids = torch.zeros((counts[-1], topk), dtype=torch.int32, device=device)
        probs = torch.full(
            (counts[-1], topk), 1 / topk, dtype=torch.float32, device=device
        )
        with PreparationSession(
            device=device, autotune=False, compile_workers=1
        ) as session:
            for capacity in missing:
                declaration = moe.plan_execution(
                    experts=self.experts,
                    capacity=moe.ExecutionCapacity(
                        max_tokens=capacity,
                        top_k=topk,
                        route_num_experts=self.moe.num_experts,
                    ),
                )

                def factory(state, capacity=capacity):
                    scratch = tuple(
                        torch.empty(s.shape, dtype=s.dtype, device=device)
                        for s in state.scratch.scratch_specs()
                    )
                    binding = state.bind(
                        scratch=scratch,
                        a=x[:capacity],
                        topk_ids=ids[:capacity],
                        topk_weights=probs[:capacity],
                        output=output[:capacity],
                        route_expert_map=layer.expert_map,
                    )
                    return PreparedCall(
                        run=binding.run,
                        output=output[:capacity],
                        owners=(binding, scratch),
                    )

                # Scalar capacity declaration: live M is never a compile key.
                session.prepare(
                    (
                        declaration.request(
                            name=f"native-iq-{capacity}",
                            prepare_call=factory,
                            benchmark_call=factory,
                        ),
                    )
                )
                state = require_prepared(declaration, "moe.decode")
                scratch = tuple(
                    torch.empty(s.shape, dtype=s.dtype, device=device)
                    for s in state.scratch.scratch_specs()
                )
                self._plans[capacity] = (state, scratch)
        self._capacities = tuple(sorted(self._plans))

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        if not self._capacities or self._output is None:
            raise RuntimeError("Native IQ execution requires warmup before forward")
        m = x.shape[0]
        index = bisect_left(self._capacities, m)
        if index == len(self._capacities):
            raise ValueError("Token count exceeds prepared native IQ capacity")
        if topk_ids.dtype != torch.int32 or topk_weights.dtype != torch.float32:
            raise ValueError("Native IQ requires int32 IDs and float32 router weights")
        state, scratch = self._plans[self._capacities[index]]
        output = self._output[:m]
        binding = state.bind(
            scratch=scratch,
            a=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            output=output,
            route_expert_map=layer.expert_map,
        )
        binding.run()
        # No shared-expert execution or all-reduce here. MoERunner owns both.
        return output
