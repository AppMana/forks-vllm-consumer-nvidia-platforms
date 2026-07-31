# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    UniformTypeKVCacheSpecs,
)


def _attention_spec(dtype: torch.dtype) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=dtype,
    )


def test_packed_mixed_precision_cache_requires_block_zeroing():
    fp8_spec = _attention_spec(torch.float8_e4m3fn)
    bf16_spec = _attention_spec(torch.bfloat16)
    config = KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["fp8_layer"],
                UniformTypeKVCacheSpecs(
                    block_size=16,
                    kv_cache_specs={"fp8_layer": fp8_spec},
                ),
            ),
            KVCacheGroupSpec(
                ["bf16_layer"],
                UniformTypeKVCacheSpecs(
                    block_size=16,
                    kv_cache_specs={"bf16_layer": bf16_spec},
                ),
            ),
        ],
    )

    assert config.has_mixed_precision_kv_cache
    assert config.needs_kv_cache_zeroing
