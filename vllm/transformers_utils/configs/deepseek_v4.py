# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers import PretrainedConfig

DEEPSEEK_V4_SM86_SPARSE_MLA_DECODE_FP8 = "flash_mla.flash_sparse_mla_decode"
DEEPSEEK_V4_SM86_SPARSE_MLA_DECODE_FP8_TRITON = (
    "vllm.models.deepseek_v4.nvidia_sm86.triton_kernels."
    "decode_sparse_attention_triton"
)
DEEPSEEK_V4_SM86_SPARSE_MLA_DECODE_INT8 = (
    "flash_mla.triton_sparse_int8_mla_decode"
)
DEEPSEEK_V4_SM86_SPARSE_MLA_PREFILL = (
    "vllm.models.deepseek_v4.nvidia_sm86.triton_kernels.sparse_attention_triton"
)


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
        super().__init__(**kwargs)
