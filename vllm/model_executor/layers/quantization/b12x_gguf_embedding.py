# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact Q8_0 token rows through the planned b12x embedding operation."""

import torch

from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.utils.b12x import B12xWarmupUnit


class B12xGGUFEmbeddingMethod(QuantizeMethodBase):
    def __init__(self, source, *, max_tokens):
        self.source = source
        self.max_tokens = int(max_tokens)
        if source.quant_type != "Q8_0" or self.max_tokens <= 0:
            raise ValueError(
                "Native token embedding requires Q8_0 and positive capacity"
            )
        self.weight = self.state = self.output = None

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **kwargs,
    ):
        if (
            params_dtype != torch.bfloat16
            or input_size_per_partition != self.source.columns
            or layer.org_vocab_size != self.source.rows
            or layer.num_added_embeddings
        ):
            raise ValueError(
                "Native Q8 embedding differs from checkpoint vocabulary/width"
            )
        if len(output_partition_sizes) != 1:
            raise ValueError("Native Q8 embedding expects one vocabulary partition")
        self.rows = output_partition_sizes[0]
        self.row_start = layer.shard_indices.org_vocab_start_index
        layer.register_buffer("_b12x_gguf_device", torch.empty(0), persistent=False)

    def process_weights_after_loading(self, layer):
        from b12x.loader.gguf_dense import prepare_dense_partition

        if self.weight is not None:
            raise RuntimeError("Reload requires rebuilding native embedding plans")
        self.weight = prepare_dense_partition(
            self.source,
            device=layer._b12x_gguf_device.device,
            row_start=self.row_start,
            rows=self.rows,
        )
        layer.b12x_warmup_provider = self
        self.warmup_launches(layer, token_counts=(self.max_tokens,))

    def get_b12x_warmup_unit(self, layer, token_counts, output_dtype):
        if output_dtype != torch.bfloat16:
            raise ValueError("Native Q8 embedding output must be BF16")
        return B12xWarmupUnit(
            name="native GGUF embedding",
            key=(type(self), id(self)),
            compile=lambda: self.warmup_launches(layer, token_counts=token_counts),
        )

    @torch.inference_mode()
    def warmup_launches(self, layer, *, token_counts):
        from b12x.preparation import PreparationSession, PreparedCall, require_prepared
        from b12x.sequence import embedding

        counts = tuple(int(n) for n in token_counts)
        if not counts or min(counts) <= 0 or max(counts) > self.max_tokens:
            raise ValueError("Native Q8 embedding warmup exceeds prepared capacity")
        if self.state is not None:
            return
        if self.weight is None or torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Load and prepare native embedding before capture")
        device = self.weight.values.device
        ids = torch.zeros(self.max_tokens, dtype=torch.int64, device=device)
        self.output = torch.empty(
            self.max_tokens, self.source.columns, dtype=torch.bfloat16, device=device
        )
        plan = embedding.plan(
            embedding.query_from_call(self.weight, ids, out=self.output), device=device
        )
        with PreparationSession(
            device=device, autotune=False, compile_workers=1
        ) as session:
            session.prepare(
                (
                    plan.request(
                        name="native Q8 embedding",
                        prepare_call=lambda state: PreparedCall(
                            run=lambda: state.run(self.weight, ids, out=self.output)
                        ),
                    ),
                )
            )
            self.state = require_prepared(plan, "sequence.embedding")
            session.freeze()

    def embedding(self, layer, input_):
        if (
            self.state is None
            or self.output is None
            or input_.numel() > self.max_tokens
        ):
            raise ValueError("Native Q8 embedding is unprepared or exceeds capacity")
        out = self.output[: input_.numel()].view(*input_.shape, self.source.columns)
        self.state.run(self.weight, input_, out=out)
        return out

    def apply(self, layer, x, bias=None):
        raise NotImplementedError("Token embedding cannot be used as an LM head")
