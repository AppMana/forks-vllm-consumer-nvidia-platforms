# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for CUDA kernels in cache_kernels.cu."""

import pytest
import torch

try:
    from vllm import _custom_ops as ops
except ImportError:
    pytest.skip(
        "Could not import vllm._custom_ops. (pip install -e .)", allow_module_level=True
    )


@pytest.mark.skipif(torch.accelerator.device_count() < 1, reason="Need CUDA device")
def test_gather_cache_oob():
    """
    Tests for OOB read in gather_and_maybe_dequant_cache (Issue #27909).
    This test constructs a boundary case identified in the issue where
    seq_starts causes the block_table offset to read out of bounds.
    """

    batch_size = 1
    block_size = 64
    entry_size = 128

    block_table = torch.tensor([[1, 2]], dtype=torch.int32, device="cuda")

    # This will result in offset = 128 / block_size = 128 / 64 = 2
    # This will cause the kernel to try to read from
    # block_table[0, 2], but its size is only 2.
    seq_starts = torch.tensor([128], dtype=torch.int32, device="cuda")

    seq_len = 65
    cu_seq_lens = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")

    # src_cache: [num_blocks, block_size, entry_size]
    num_blocks = 5
    src_cache = torch.randn(
        (num_blocks, block_size, entry_size), dtype=torch.float16, device="cuda"
    )

    dst = torch.empty((seq_len, entry_size), dtype=torch.float16, device="cuda")

    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    # Calling the C++ function gather_and_maybe_dequant_cache
    ops.gather_and_maybe_dequant_cache(
        src_cache,
        dst,
        block_table,
        cu_seq_lens,
        batch_size,
        "auto",  # kv_cache_dtype
        scale,
        seq_starts,
    )

    torch.accelerator.synchronize()
    assert True


@pytest.mark.skipif(torch.accelerator.device_count() < 1, reason="Need CUDA device")
def test_int8_ds_mla_cache_insert_is_cudagraph_safe():
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        _INT8_DS_MLA_DIM,
        _INT8_DS_MLA_TOKEN_BYTES,
        get_int8_ds_mla_cache_views,
        quantize_and_insert_int8_ds_mla_cache,
    )

    torch.manual_seed(0)
    block_size = 64
    num_blocks = 4
    num_tokens = 6
    k = torch.randn(
        (num_tokens, _INT8_DS_MLA_DIM), dtype=torch.bfloat16, device="cuda"
    )
    slot_mapping = torch.tensor([0, 63, -1, 64, 130, -1], device="cuda")
    cache = torch.full(
        (num_blocks, block_size, _INT8_DS_MLA_TOKEN_BYTES),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    )

    quantize_and_insert_int8_ds_mla_cache(k, cache, slot_mapping, block_size)
    torch.cuda.synchronize()

    data, scales = get_int8_ds_mla_cache_views(cache, block_size)
    valid = slot_mapping >= 0
    k_valid = k[valid].to(torch.float32)
    ref_scales = (k_valid.abs().amax(dim=-1) / 127.0).clamp_min(1.0e-12)
    ref_q = torch.round(k_valid / ref_scales.unsqueeze(-1)).clamp(-127, 127).to(
        torch.int8
    )
    slots = slot_mapping[valid].to(torch.int64)
    block_idx = slots // block_size
    pos = slots % block_size
    assert torch.equal(data[block_idx, pos], ref_q)
    torch.testing.assert_close(scales[block_idx, pos], ref_scales)

    graph_cache = torch.empty_like(cache)
    quantize_and_insert_int8_ds_mla_cache(k, graph_cache, slot_mapping, block_size)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        quantize_and_insert_int8_ds_mla_cache(k, graph_cache, slot_mapping, block_size)
    graph.replay()
    torch.cuda.synchronize()

    small_block_cache = torch.empty(
        (num_blocks, 2, _INT8_DS_MLA_TOKEN_BYTES),
        dtype=torch.uint8,
        device="cuda",
    )
    small_block_slots = torch.tensor([0, 1, -1, 2, 3, -1], device="cuda")
    quantize_and_insert_int8_ds_mla_cache(
        k, small_block_cache, small_block_slots, block_size=2
    )
    torch.cuda.synchronize()


if __name__ == "__main__":
    pytest.main([__file__])
