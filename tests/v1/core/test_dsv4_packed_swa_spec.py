# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.core.kv_cache_utils import _promote_local_kv_cache_specs
from vllm.v1.kv_cache_interface import (
    KVQuantMode,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)


def _packed_dsv4_swa_spec(
    cache_dtype: str,
) -> SlidingWindowMLASpec:
    is_fp8 = cache_dtype == "fp8_ds_mla"
    return SlidingWindowMLASpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        kv_quant_mode=(
            KVQuantMode.FP8_PER_TENSOR if is_fp8 else KVQuantMode.NONE
        ),
        sliding_window=128,
        cache_dtype_str=cache_dtype,
        alignment=576 if is_fp8 else 528,
        compress_ratio=1,
        model_version="deepseek_v4",
    )


@pytest.mark.parametrize(
    ("cache_dtype", "quant_mode", "bytes_per_token"),
    [
        ("fp8_ds_mla", KVQuantMode.FP8_PER_TENSOR, 584),
        ("int8_ds_mla", KVQuantMode.NONE, 528),
    ],
)
def test_dsv4_packed_swa_merge_preserves_quantized_shape_selection(
    cache_dtype: str,
    quant_mode: KVQuantMode,
    bytes_per_token: int,
):
    spec = _packed_dsv4_swa_spec(cache_dtype)

    merged = SlidingWindowMLASpec.merge([spec])

    assert merged.kv_quant_mode == quant_mode
    assert merged.cache_dtype_str == cache_dtype
    assert merged.real_page_size_bytes == 64 * bytes_per_token


@pytest.mark.parametrize(
    ("cache_dtype", "quant_mode", "alignment", "bytes_per_token"),
    [
        ("fp8_ds_mla", KVQuantMode.FP8_PER_TENSOR, 576, 584),
        ("int8_ds_mla", KVQuantMode.NONE, 528, 528),
    ],
)
def test_dsv4_packed_swa_promotion_preserves_quantized_shape_selection(
    cache_dtype: str,
    quant_mode: KVQuantMode,
    alignment: int,
    bytes_per_token: int,
):
    swa = _packed_dsv4_swa_spec(cache_dtype)
    full = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        kv_quant_mode=quant_mode,
        cache_dtype_str=cache_dtype,
        alignment=alignment,
        compress_ratio=4,
        model_version="deepseek_v4",
    )

    promoted = _promote_local_kv_cache_specs({"swa": swa, "full": full})["swa"]

    assert isinstance(promoted, MLAAttentionSpec)
    assert promoted.kv_quant_mode == quant_mode
    assert promoted.cache_dtype_str == cache_dtype
    assert promoted.alignment == alignment
    assert promoted.real_page_size_bytes == 256 * bytes_per_token
