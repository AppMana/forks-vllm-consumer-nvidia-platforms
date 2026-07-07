# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 int8_ds_mla cache layout tests."""

import inspect

import torch

from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    compress_norm_rope_store_triton,
)
from vllm.models.deepseek_v4.common.ops.cache_utils import (
    dequantize_and_gather_int8_ds_mla_cache,
    dequantize_global_slots_int8_ds_mla_cache,
    get_int8_ds_mla_cache_views,
    quantize_and_insert_int8_ds_mla_cache,
)
from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLABackend
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _expected_int8_dequant(k: torch.Tensor) -> torch.Tensor:
    scale = (k.float().abs().amax(dim=-1) / 127.0).clamp_min(1.0e-12)
    q = torch.round(k.float() / scale.unsqueeze(-1)).clamp(-127, 127)
    return (q * scale.unsqueeze(-1)).to(torch.bfloat16)


def test_int8_ds_mla_compressor_launcher_accepts_runtime_selector() -> None:
    signature = inspect.signature(compress_norm_rope_store_triton)
    assert "int8_ds_mla" in signature.parameters


def test_int8_ds_mla_cache_shapes_and_page_sizes() -> None:
    assert DeepseekV4FlashMLABackend.get_kv_cache_shape(
        3, 64, 1, 512, "int8_ds_mla"
    ) == (3, 64, 516)
    assert DeepseekSparseSWABackend.get_kv_cache_shape(
        3, 64, 1, 512, "int8_ds_mla"
    ) == (3, 64, 516)

    main_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        compress_ratio=4,
        cache_dtype_str="int8_ds_mla",
        alignment=516,
        model_version="deepseek_v4",
    )
    assert main_spec.real_page_size_bytes == 64 * 516

    swa_spec = SlidingWindowMLASpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        sliding_window=128,
        cache_dtype_str="int8_ds_mla",
        alignment=516,
        model_version="deepseek_v4",
    )
    assert swa_spec.real_page_size_bytes == 64 * 516


def test_int8_ds_mla_insert_global_dequant_and_views() -> None:
    device = _device()
    block_size = 4
    k = torch.randn(6, 512, dtype=torch.bfloat16, device=device)
    slot_mapping = torch.tensor([0, 1, 5, -1, 8, 9], dtype=torch.int64, device=device)
    k_cache = torch.zeros(3, block_size, 516, dtype=torch.uint8, device=device)

    quantize_and_insert_int8_ds_mla_cache(k, k_cache, slot_mapping, block_size)

    rows_i8, scales = get_int8_ds_mla_cache_views(k_cache, block_size)
    assert rows_i8.shape == (3, block_size, 512)
    assert rows_i8.dtype == torch.int8
    assert scales.shape == (3, block_size)
    assert scales.dtype == torch.float32
    assert rows_i8.stride()[1] == 516

    slot_ids = torch.tensor([[0, 5, -1], [8, 9, 1]], dtype=torch.int64, device=device)
    out = torch.empty(2, 3, 512, dtype=torch.bfloat16, device=device)
    dequantize_global_slots_int8_ds_mla_cache(out, k_cache, slot_ids, block_size)

    expected = _expected_int8_dequant(k)
    torch.testing.assert_close(out[0, 0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(out[0, 1], expected[2], rtol=0, atol=0)
    torch.testing.assert_close(out[1, 0], expected[4], rtol=0, atol=0)
    torch.testing.assert_close(out[1, 1], expected[5], rtol=0, atol=0)
    torch.testing.assert_close(out[1, 2], expected[1], rtol=0, atol=0)
    assert torch.equal(out[0, 2], torch.zeros_like(out[0, 2]))


def test_int8_ds_mla_gather_dequant() -> None:
    device = _device()
    block_size = 4
    k = torch.randn(8, 512, dtype=torch.bfloat16, device=device)
    k_cache = torch.zeros(2, block_size, 516, dtype=torch.uint8, device=device)
    slot_mapping = torch.arange(8, dtype=torch.int64, device=device)
    quantize_and_insert_int8_ds_mla_cache(k, k_cache, slot_mapping, block_size)

    out = torch.empty(1, 5, 512, dtype=torch.bfloat16, device=device)
    seq_lens = torch.tensor([8], dtype=torch.int32, device=device)
    gather_lens = torch.tensor([3], dtype=torch.int32, device=device)
    block_table = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    dequantize_and_gather_int8_ds_mla_cache(
        out,
        k_cache,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        block_table=block_table,
        block_size=block_size,
        offset=2,
    )

    expected = _expected_int8_dequant(k)
    torch.testing.assert_close(out[0, 2:5], expected[5:8], rtol=0, atol=0)
