# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint-derived native IQ dispatch for DeepSeek V4.1 GGUF shards."""

from pathlib import Path
from typing import Any

import regex as re
import torch

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.b12x_iq import B12xIQMoEMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

_EXPERT_PREFIX = re.compile(r"(?:^|\.)layers\.(\d+)\.ffn\.experts$")


class Dsv41GGUFConfig(QuantizationConfig):
    """Index exact source tensors before the model allocates expert weights.

    Q6_K/Q8_0 projections use compact native linear methods, including grouped
    WO-A, token embeddings and the LM head. Engram reads original Q5_K rows
    through bounded disk staging. Unsupported layers fail explicitly.
    """

    # Shared-expert intermediate padding preserves Q6_K blocks under TP.
    weight_block_size = [256, 256]

    def __init__(self, files, max_tokens=None):
        super().__init__()
        from b12x.loader.gguf_experts import index_experts
        from gguf import GGUFReader

        self.files = tuple(Path(path).resolve() for path in files)
        if not self.files or len(set(self.files)) != len(self.files):
            raise ValueError("GGUF shard paths must be nonempty and unique")
        self.max_tokens = max_tokens
        self.tensor_types: dict[str, str] = {}
        self.engram_source = None
        metadata: dict[str, Any] = {}
        keys = (
            "general.architecture",
            "deepseek41.block_count",
            "deepseek41.swiglu_clamp_exp",
            "deepseek41.swiglu_clamp_shexp",
        )
        for path in self.files:
            reader = GGUFReader(path, mode="r")
            for key in keys:
                field = reader.fields.get(key)
                if field is None:
                    continue
                value = field.contents()
                if key in metadata and metadata[key] != value:
                    raise ValueError(f"Conflicting GGUF metadata: {key}")
                metadata[key] = value
            for tensor in reader.tensors:
                if tensor.name in self.tensor_types:
                    raise ValueError(f"Duplicate GGUF tensor: {tensor.name}")
                self.tensor_types[tensor.name] = tensor.tensor_type.name
                if tensor.name == "per_layer_token_embd.weight":
                    from .b12x_gguf_engram import GGUFEngramSource

                    if tensor.tensor_type.name != "Q5_K" or len(tensor.shape) != 2:
                        raise ValueError("Native GGUF Engram requires a Q5_K matrix")
                    self.engram_source = GGUFEngramSource(
                        path,
                        tensor.name,
                        int(tensor.data_offset),
                        int(tensor.shape[1]),
                        int(tensor.shape[0]),
                    )
            del reader
        if metadata.get("general.architecture") != "deepseek41":
            raise ValueError("Native DSV4.1 requires deepseek41 GGUF architecture")
        self.num_layers = int(metadata["deepseek41.block_count"])
        self.routed_clamps = tuple(metadata["deepseek41.swiglu_clamp_exp"])
        self.shared_clamps = tuple(metadata["deepseek41.swiglu_clamp_shexp"])
        if self.num_layers <= 0 or any(
            len(values) != self.num_layers
            for values in (self.routed_clamps, self.shared_clamps)
        ):
            raise ValueError("GGUF clamp arrays must cover every layer")
        import math

        if any(
            not math.isfinite(value) or value <= 0
            for value in self.routed_clamps + self.shared_clamps
        ):
            raise ValueError("GGUF clamps must be finite and positive")
        from b12x.loader.gguf_dense import index_dense

        self.dense_sources = index_dense(self.files)
        index = index_experts(self.files)
        self.expert_sources = {}
        for layer in range(self.num_layers):
            names = [
                f"blk.{layer}.ffn_{part}_exps.weight" for part in ("gate", "up", "down")
            ]
            missing = [name for name in names if name not in index]
            if missing:
                raise ValueError(f"Incomplete GGUF expert layer: {missing}")
            gate, up, down = (index.pop(name) for name in names)
            if (gate.columns, gate.rows, gate.experts, gate.quant_type) != (
                up.columns,
                up.rows,
                up.experts,
                up.quant_type,
            ) or (down.columns, down.rows, down.experts) != (
                gate.rows,
                gate.columns,
                gate.experts,
            ):
                raise ValueError(f"Inconsistent GGUF expert geometry in layer {layer}")
            supported = {"IQ2_XS", "IQ2_XXS", "IQ2_S", "IQ3_XXS", "IQ3_S"}
            pair = (gate.quant_type, down.quant_type)
            if gate.quant_type not in supported or (
                down.quant_type not in {gate.quant_type, "MXFP4"}
                and pair
                not in {
                    ("IQ2_XXS", "IQ3_XXS"),
                    ("IQ2_S", "IQ3_S"),
                    ("IQ3_XXS", "IQ3_S"),
                }
            ):
                raise ValueError(f"Unsupported native GGUF expert pair: {pair}")
            self.expert_sources[layer] = (gate, up, down)
        if index:
            raise ValueError(f"Unexpected GGUF expert tensors: {sorted(index)}")

    def create_engram_embedding(self, layout, layer_hash_index):
        from vllm.config import get_current_vllm_config
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        from vllm.models.deepseek_v41.common.engram import get_engram_dp_size

        from .b12x_gguf_engram import B12xGGUFEngramEmbedding

        if self.engram_source is None or get_engram_dp_size() != 1:
            raise ValueError("Native GGUF Engram requires its Q5_K table and DP=1")
        config = get_current_vllm_config()
        return B12xGGUFEngramEmbedding(
            self.engram_source,
            layout,
            layer_hash_index,
            max_tokens=self.max_tokens
            or config.scheduler_config.max_num_batched_tokens,
            tp_rank=get_tensor_model_parallel_rank(),
            tp_size=get_tensor_model_parallel_world_size(),
            device=torch.device("cuda", torch.accelerator.current_device_index()),
        )

    @classmethod
    def get_name(cls):
        return "dsv41_gguf"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 120

    @staticmethod
    def get_config_filenames():
        return ["dsv41_gguf_config.json"]

    @classmethod
    def from_config(cls, config):
        return cls(config["gguf_files"], max_tokens=config.get("max_tokens"))

    def get_quant_method(self, layer, prefix):
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
            VocabParallelEmbedding,
        )

        if isinstance(layer, VocabParallelEmbedding):
            from vllm.config import get_current_vllm_config
            from vllm.model_executor.layers.quantization.b12x_gguf_embedding import (
                B12xGGUFEmbeddingMethod,
            )
            from vllm.model_executor.layers.quantization.b12x_gguf_linear import (
                B12xGGUFLinearMethod,
                GGUFLinearSlice,
            )

            capacity = self.max_tokens
            if capacity is None:
                capacity = (
                    get_current_vllm_config().scheduler_config.max_num_batched_tokens
                )
            if layer.num_added_embeddings:
                raise ValueError(
                    "Native GGUF vocabulary does not support added embedding rows"
                )
            name = (
                "output.weight"
                if isinstance(layer, ParallelLMHead)
                else "token_embd.weight"
            )
            if ".engram." in prefix or name not in self.dense_sources:
                raise NotImplementedError(
                    f"Unsupported native GGUF embedding: {prefix}"
                )
            source = self.dense_sources[name]
            if (
                source.rows != layer.org_vocab_size
                or source.columns != layer.embedding_dim
            ):
                raise ValueError("Native GGUF vocabulary geometry differs from model")
            if isinstance(layer, ParallelLMHead):
                return B12xGGUFLinearMethod(
                    (
                        GGUFLinearSlice(
                            source,
                            layer.num_embeddings_per_partition,
                            source.columns,
                            row_start=layer.shard_indices.org_vocab_start_index,
                        ),
                    ),
                    max_tokens=capacity,
                )
            return B12xGGUFEmbeddingMethod(source, max_tokens=capacity)

        if isinstance(layer, RoutedExperts):
            match = _EXPERT_PREFIX.search(prefix)
            if match is None:
                raise ValueError(f"Unrecognized DSV4.1 expert prefix: {prefix}")
            number = int(match[1])
            if number not in self.expert_sources:
                raise ValueError(
                    f"GGUF has no routed layer {number}; drafter needs its own config"
                )
            if layer.moe_config.swiglu_limit != self.routed_clamps[number]:
                raise ValueError("Model routed clamp disagrees with GGUF metadata")
            gate, up, down = self.expert_sources[number]
            return B12xIQMoEMethod(layer.moe_config, gate=gate, up=up, down=down)
        if isinstance(layer, LinearBase):
            return self._linear_method(layer, prefix)
        return None

    def _linear_method(self, layer, prefix):
        from vllm.config import get_current_vllm_config
        from vllm.model_executor.layers.quantization.b12x_gguf_linear import (
            B12xGGUFGroupedLinearMethod,
            B12xGGUFLinearMethod,
        )

        match = re.search(r"(?:^|\.)layers\.(\d+)\.(.*)$", prefix)
        if match is None:
            raise NotImplementedError(f"Unmapped GGUF linear: {prefix}")
        number, suffix = int(match[1]), match[2]
        mapping = {
            "attn.fused_wqa_wkv": ("attn_q_a", "attn_kv"),
            "attn.wq_b": ("attn_q_b",),
            "attn.wo_a": ("attn_output_a",),
            "attn.wo_b": ("attn_output_b",),
            "attn.indexer.wq_b": ("indexer.attn_q_b",),
            "engram.wkv": ("engram_kv",),
            "ffn.shared_experts.gate_up_proj": ("ffn_gate_shexp", "ffn_up_shexp"),
            "ffn.shared_experts.down_proj": ("ffn_down_shexp",),
        }
        if suffix not in mapping:
            raise NotImplementedError(f"Unmapped GGUF linear: {prefix}")
        names = tuple(f"blk.{number}.{name}.weight" for name in mapping[suffix])
        if any(name not in self.tensor_types for name in names):
            raise ValueError(f"Missing GGUF linear sources: {names}")
        if all(self.tensor_types[name] in ("BF16", "F32") for name in names):
            return UnquantizedLinearMethod()
        if any(name not in self.dense_sources for name in names):
            raise ValueError(f"Unsupported GGUF linear formats: {names}")
        # TP identity is assigned after get_quant_method; create_weights derives
        # slices using the completed layer, including disable_tp replication.
        if isinstance(layer, RowParallelLinear):
            partition = "row"
        elif isinstance(layer, ColumnParallelLinear):
            partition = "column"
        elif isinstance(layer, ReplicatedLinear):
            partition = "replicated"
        else:
            raise NotImplementedError(f"Unsupported GGUF linear class: {type(layer)}")
        capacity = self.max_tokens
        if capacity is None:
            capacity = get_current_vllm_config().scheduler_config.max_num_batched_tokens
        method = (
            B12xGGUFGroupedLinearMethod
            if suffix == "attn.wo_a"
            else B12xGGUFLinearMethod
        )
        return method(
            sources=tuple(self.dense_sources[name] for name in names),
            partition=partition,
            max_tokens=capacity,
            pad_to_block=suffix.startswith("ffn.shared_experts."),
        )
