# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sm_86 correctness gate: the Triton sparse-MLA decode primitive vs the
pure-torch reference oracle.

This is decoupled from the fork's monolithic ``deepseek_v4/attention.py``: it
imports only the kernel primitives (``sparse_mla_kernels``) used by
``nvidia_sm86``'s ``decode_sparse_attention_triton`` and the reference oracle
(``sparse_mla_reference``), and asserts they match within tolerance. It is the
red-first gate that must be green before the nvidia_sm86 attention wiring is
trusted on Ampere.
"""

import pytest
import torch

from vllm.v1.attention.backends.mla.sparse_mla_kernels import (
    accumulate_fp8ds_global_slots_sparse_mla_attention_chunk,
    accumulate_indexed_sparse_mla_attention_chunk,
    finish_gathered_sparse_mla_attention,
    merge_sparse_mla_subset_with_sink,
)
from vllm.v1.attention.backends.mla.sparse_mla_reference import (
    reference_attention_no_sink,
    reference_sparse_mla_prefill,
)

_FP8_DIM = 448
_ROPE_DIM = 64
_SCALE_DIM = 8
_TOKEN_DATA_SIZE = _FP8_DIM + _ROPE_DIM * 2


def _write_fp8_ds_mla_token(
    k_cache: torch.Tensor,
    slot: int,
    block_size: int,
) -> torch.Tensor:
    block_idx = slot // block_size
    block_offset = slot % block_size

    values = (
        (torch.arange(_FP8_DIM, device=k_cache.device, dtype=torch.float32) % 17) - 8
    ) / 16.0
    values = values + float(slot) / 32.0
    scale_exponents = torch.tensor(
        [-2, -1, 0, 1, 2, -2, 1],
        device=k_cache.device,
        dtype=torch.float32,
    )
    scales = torch.exp2(scale_exponents)
    scale_per_dim = scales.repeat_interleave(64)

    fp8_values = (values / scale_per_dim).to(torch.float8_e4m3fn)
    expected_nope = fp8_values.float() * scale_per_dim
    rope = (
        torch.linspace(-1.0, 1.0, _ROPE_DIM, device=k_cache.device) + float(slot) / 16.0
    ).to(torch.bfloat16)

    flat_block = k_cache[block_idx].view(-1)
    token_data_start = block_offset * _TOKEN_DATA_SIZE
    token_scale_start = block_size * _TOKEN_DATA_SIZE + block_offset * _SCALE_DIM
    flat_block[token_data_start : token_data_start + _FP8_DIM] = fp8_values.view(
        torch.uint8
    )
    flat_block[token_data_start + _FP8_DIM : token_data_start + _TOKEN_DATA_SIZE] = (
        rope.view(torch.uint8)
    )

    encoded_scales = (scale_exponents.to(torch.int32) + 127).to(torch.uint8)
    flat_block[token_scale_start : token_scale_start + encoded_scales.numel()] = (
        encoded_scales
    )
    flat_block[
        token_scale_start + encoded_scales.numel() : token_scale_start + _SCALE_DIM
    ] = 127

    return torch.cat([expected_nope, rope.float()]).to(torch.bfloat16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_triton_fp8ds_global_slots_attention_chunk_matches_reference() -> None:
    torch.manual_seed(10)
    block_size = 4
    num_blocks = 3
    k_cache = torch.zeros(
        num_blocks,
        block_size,
        _TOKEN_DATA_SIZE + _SCALE_DIM,
        dtype=torch.uint8,
        device="cuda",
    )
    expected_by_slot = {
        slot: _write_fp8_ds_mla_token(k_cache, slot, block_size)
        for slot in (0, 1, 3, 4, 7, 8)
    }
    slot_ids = torch.tensor(
        [
            [0, 3, -1, 8, 1],
            [7, -1, 4, 0, 8],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    lens = torch.tensor([4, 5], dtype=torch.int32, device="cuda")
    q = torch.randn(2, 1, 3, 512, device="cuda", dtype=torch.bfloat16)
    scale = 0.0625

    max_score = torch.full((2, 3), float("-inf"), device="cuda")
    denom = torch.zeros((2, 3), device="cuda")
    acc = torch.zeros((2, 3, 512), device="cuda")
    accumulate_fp8ds_global_slots_sparse_mla_attention_chunk(
        q=q,
        k_cache=k_cache,
        slot_ids=slot_ids[:, :2],
        lens=lens,
        block_size=block_size,
        candidate_offset=0,
        scale=scale,
        max_score=max_score,
        denom=denom,
        acc=acc,
    )
    accumulate_fp8ds_global_slots_sparse_mla_attention_chunk(
        q=q,
        k_cache=k_cache,
        slot_ids=slot_ids[:, 2:],
        lens=lens,
        block_size=block_size,
        candidate_offset=2,
        scale=scale,
        max_score=max_score,
        denom=denom,
        acc=acc,
    )

    output = torch.empty_like(acc)
    lse = torch.empty_like(max_score)
    finish_gathered_sparse_mla_attention(
        max_score=max_score,
        denom=denom,
        acc=acc,
        output=output,
        lse=lse,
    )

    gathered = torch.zeros(2, 5, 512, device="cuda", dtype=torch.bfloat16)
    for token_idx in range(slot_ids.shape[0]):
        for topk_idx in range(slot_ids.shape[1]):
            slot = int(slot_ids[token_idx, topk_idx].item())
            if slot >= 0:
                gathered[token_idx, topk_idx] = expected_by_slot[slot]
    offsets = torch.arange(slot_ids.shape[1], device="cuda")
    valid_tokens = (offsets[None, :] < lens[:, None]) & (slot_ids >= 0)
    expected_output, expected_lse = reference_attention_no_sink(
        q,
        gathered,
        valid_tokens,
        scale,
    )

    torch.testing.assert_close(output, expected_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse, expected_lse, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_triton_indexed_bf16_prefill_chunks_match_reference() -> None:
    torch.manual_seed(17)
    q = torch.randn(5, 5, 16, device="cuda", dtype=torch.bfloat16)
    q_active = q[:, :3]
    kv = torch.randn(2, 7, 16, device="cuda", dtype=torch.bfloat16)
    kv_flat = kv.reshape(-1, q.shape[-1])
    combined_indices = torch.tensor(
        [
            [0, 3, -1, 5, 3, 1],
            [4, -1, 2, 2, 1, 8],
            [-1, -1, -1, -1, -1, -1],
            [8, 0, 9, -1, 7, 4],
            [13, 12, 0, 12, -1, 3],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    combined_lens = torch.tensor([5, 4, 0, 6, 5], dtype=torch.int32, device="cuda")
    sink = torch.tensor([-0.5, 1.0, 0.25], dtype=torch.float32, device="cuda")
    scale = 0.375
    output = torch.empty_like(q_active)

    for token_start in (0, 2, 4):
        token_end = min(token_start + 2, q.shape[0])
        q_chunk = q[token_start:token_end]
        indices_chunk = combined_indices[token_start:token_end]
        lens_chunk = combined_lens[token_start:token_end]
        max_score = torch.full(
            (q_chunk.shape[0], q_active.shape[1]),
            float("-inf"),
            device="cuda",
        )
        denom = torch.zeros_like(max_score)
        acc = torch.zeros(
            q_chunk.shape[0],
            q_active.shape[1],
            q_chunk.shape[-1],
            device="cuda",
            dtype=torch.float32,
        )
        for index_start in (0, 3):
            index_end = min(index_start + 3, combined_indices.shape[-1])
            accumulate_indexed_sparse_mla_attention_chunk(
                q=q_chunk,
                kv_flat=kv_flat,
                indices=indices_chunk[:, index_start:index_end],
                lens=lens_chunk,
                candidate_offset=index_start,
                scale=scale,
                max_score=max_score,
                denom=denom,
                acc=acc,
            )
        subset_output = torch.empty_like(acc)
        subset_lse = torch.empty_like(max_score)
        finish_gathered_sparse_mla_attention(
            max_score=max_score,
            denom=denom,
            acc=acc,
            output=subset_output,
            lse=subset_lse,
        )
        merge_sparse_mla_subset_with_sink(
            subset_output=subset_output,
            subset_lse=subset_lse,
            attn_sink=sink,
            output=output[token_start:token_end],
        )

    expected = torch.empty_like(q_active)
    reference_sparse_mla_prefill(
        q=q_active,
        kv=kv,
        combined_indices=combined_indices,
        combined_lens=combined_lens,
        scale=scale,
        attn_sink=sink,
        output=expected,
        topk_chunk_size=3,
        query_chunk_size=2,
    )

    torch.testing.assert_close(output.float(), expected.float(), rtol=2e-2, atol=2e-2)
