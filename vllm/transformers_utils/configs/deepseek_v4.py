# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers import PretrainedConfig

from vllm.transformers_utils.configs.deepseek_v4_appmana import (
    SPARSE_MLA_DECODE_FP8_FLASH,
    SPARSE_MLA_DECODE_FP8_TRITON,
    SPARSE_MLA_DECODE_INT8_TRITON,
    SPARSE_MLA_PREFILL_TRITON,
)

# Deprecated aliases for the role-keyed HF override strings. The canonical
# symbol constants (and the "appmana" config block that replaces these
# per-role keys) live in deepseek_v4_appmana.py.
DEEPSEEK_V4_SM86_SPARSE_MLA_DECODE_FP8 = SPARSE_MLA_DECODE_FP8_FLASH
DEEPSEEK_V4_SM86_SPARSE_MLA_DECODE_FP8_TRITON = SPARSE_MLA_DECODE_FP8_TRITON
DEEPSEEK_V4_SM86_SPARSE_MLA_DECODE_INT8 = SPARSE_MLA_DECODE_INT8_TRITON
DEEPSEEK_V4_SM86_SPARSE_MLA_PREFILL = SPARSE_MLA_PREFILL_TRITON


class DeepseekV4Config(PretrainedConfig):
    model_type = "deepseek_v4"

    def __init__(
        self,
        max_position_embeddings: int = 1048576,
        rope_scaling: dict[str, Any] | None = None,
        rope_parameters: dict[str, Any] | None = None,
        rope_theta: float = 10000.0,
        deepseek_v4_sm86_sparse_mla_decode_fp8: str = (
            DEEPSEEK_V4_SM86_SPARSE_MLA_DECODE_FP8
        ),
        deepseek_v4_sm86_sparse_mla_decode_int8: str = (
            DEEPSEEK_V4_SM86_SPARSE_MLA_DECODE_INT8
        ),
        deepseek_v4_sm86_sparse_mla_prefill: str = (
            DEEPSEEK_V4_SM86_SPARSE_MLA_PREFILL
        ),
        appmana: dict[str, Any] | None = None,
        **kwargs,
    ):
        self.max_position_embeddings = max_position_embeddings
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        self.rope_parameters = rope_scaling or rope_parameters
        self.deepseek_v4_sm86_sparse_mla_decode_fp8 = (
            deepseek_v4_sm86_sparse_mla_decode_fp8
        )
        self.deepseek_v4_sm86_sparse_mla_decode_int8 = (
            deepseek_v4_sm86_sparse_mla_decode_int8
        )
        self.deepseek_v4_sm86_sparse_mla_prefill = (
            deepseek_v4_sm86_sparse_mla_prefill
        )
        # Unified checkpoint-config-driven kernel selection block; see
        # deepseek_v4_appmana.py for the schema and registry.
        self.appmana = appmana
        super().__init__(**kwargs)
