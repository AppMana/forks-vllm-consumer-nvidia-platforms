# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in AOT CUDA GGUF expert execution with shared worker scratch."""

from bisect import bisect_left

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.quantization.b12x_iq import B12xIQMoEMethod
from vllm.utils.b12x import b12x_warmup_token_counts
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)


def _consumer_scratch(plan):
    if is_workspace_manager_initialized():
        return tuple(
            current_workspace_manager().get_simultaneous(*plan.scratch_specs())
        )
    return plan.allocate_scratch()


class ConsumerIQMoEMethod(B12xIQMoEMethod):
    def __init__(self, moe, *, gate, up, down):
        super().__init__(moe, gate=gate, up=up, down=down)
        if moe.swiglu_alpha not in (None, 1.0) or moe.swiglu_beta not in (None, 0.0):
            raise ValueError("Consumer IQ requires ordinary SiLU alpha/beta")
        if moe.swiglu_limit is None:
            raise ValueError("Consumer IQ requires an explicit GGUF clamp")

    def process_weights_after_loading(self, layer):
        from consumer_int8.gguf_moe import prepare_experts

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Consumer IQ loading must precede capture")
        if layer.apply_router_weight_on_input:
            raise ValueError("Consumer IQ expects router weights on expert output")
        if self.experts is not None:
            raise RuntimeError("Reload requires rebuilding consumer IQ captures")
        mapping = layer.expert_map
        if mapping is None:
            ids = tuple(range(self.moe.num_experts))
        else:
            ordered = sorted(
                (local, global_id)
                for global_id, local in enumerate(mapping.cpu().tolist())
                if local >= 0
            )
            if [local for local, _ in ordered] != list(
                range(self.moe.num_local_experts)
            ):
                raise ValueError("Expert map must assign each local slot exactly once")
            ids = tuple(global_id for _, global_id in ordered)
        self.experts = prepare_experts(
            *self.sources, ids, device=layer._b12x_iq_device.device
        )
        layer.b12x_warmup_provider = self
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

    @torch.inference_mode()
    def warmup_launches(self, layer, *, token_counts):
        from consumer_int8.gguf_moe import ExecutionPlan

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Consumer IQ warmup must precede capture")
        if self.experts is None:
            raise RuntimeError("Load consumer IQ weights before warmup")
        counts = tuple(sorted({int(n) for n in token_counts if int(n) > 0}))
        if not counts or counts[-1] > self.moe.max_num_tokens:
            raise ValueError("Invalid consumer IQ warmup capacity")
        missing = tuple(n for n in counts if n not in self._plans)
        if not missing:
            return
        device, h, topk = (
            self.experts.device,
            self.experts.hidden_size,
            self.moe.experts_per_token,
        )
        x = torch.zeros(counts[-1], h, dtype=torch.bfloat16, device=device)
        ids = torch.full(
            (counts[-1], topk),
            self.experts.expert_ids[0],
            dtype=torch.int32,
            device=device,
        )
        probs = torch.full((counts[-1], topk), 1 / topk, device=device)
        if self._output is None:
            self._output = torch.empty(
                self.moe.max_num_tokens, h, dtype=torch.bfloat16, device=device
            )
        for capacity in missing:
            plan = ExecutionPlan(
                self.experts, capacity, topk, swiglu_limit=self.moe.swiglu_limit
            )
            scratch = _consumer_scratch(plan)
            plan.run(
                x[:capacity],
                ids[:capacity],
                probs[:capacity],
                self._output[:capacity],
                scratch,
            )
            self._plans[capacity] = (
                plan,
                None if is_workspace_manager_initialized() else scratch,
            )
        self._capacities = tuple(sorted(self._plans))

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        if not self._capacities or self._output is None:
            raise RuntimeError("Consumer IQ requires warmup before forward")
        index = bisect_left(self._capacities, x.shape[0])
        if index == len(self._capacities):
            raise ValueError("Token count exceeds consumer IQ capacity")
        if topk_ids.dtype != torch.int32 or topk_weights.dtype != torch.float32:
            raise ValueError("Consumer IQ requires int32 IDs and float32 weights")
        plan, scratch = self._plans[self._capacities[index]]
        if scratch is None:
            scratch = _consumer_scratch(plan)
        output = self._output[: x.shape[0]]
        plan.run(x, topk_ids, topk_weights, output, scratch)
        return output
