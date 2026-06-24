# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.models.deepseek_v4.attention as deepseek_v4_attention
from vllm.v1.attention.backends.mla.sparse_mla_kernels import (
    accumulate_indexed_sparse_mla_attention_chunk,
    accumulate_indexed_sparse_mla_attention_chunk_multihead,
    finish_sparse_mla_attention_with_sink,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_multihead_indexed_sparse_mla_accumulate_matches_scalar_path() -> None:
    torch.manual_seed(0)
    num_tokens = 5
    num_heads = 8
    head_dim = 512
    num_kv_tokens = 257
    num_candidates = 128
    q = torch.randn(
        num_tokens,
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    kv_flat = torch.randn(
        num_kv_tokens,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    indices = torch.randint(
        -1,
        num_kv_tokens,
        (num_tokens, num_candidates),
        device="cuda",
        dtype=torch.int32,
    )
    lens = torch.tensor([128, 79, 0, 13, 96], device="cuda", dtype=torch.int32)
    sink = torch.randn(num_heads, device="cuda", dtype=torch.float32) * 0.01
    scale = head_dim**-0.5

    def run_accumulate(use_multihead: bool) -> torch.Tensor:
        max_score = torch.full(
            (num_tokens, num_heads),
            float("-inf"),
            device="cuda",
            dtype=torch.float32,
        )
        denom = torch.zeros_like(max_score)
        acc = torch.zeros(
            num_tokens,
            num_heads,
            head_dim,
            device="cuda",
            dtype=torch.float32,
        )
        if use_multihead:
            accumulate_indexed_sparse_mla_attention_chunk_multihead(
                q=q,
                kv_flat=kv_flat,
                indices=indices,
                lens=lens,
                candidate_offset=0,
                scale=scale,
                max_score=max_score,
                denom=denom,
                acc=acc,
                head_block_size=4,
            )
        else:
            accumulate_indexed_sparse_mla_attention_chunk(
                q=q,
                kv_flat=kv_flat,
                indices=indices,
                lens=lens,
                candidate_offset=0,
                scale=scale,
                max_score=max_score,
                denom=denom,
                acc=acc,
            )
        output = torch.empty_like(q)
        finish_sparse_mla_attention_with_sink(max_score, denom, acc, sink, output)
        return output.float()

    expected = run_accumulate(use_multihead=False)
    actual = run_accumulate(use_multihead=True)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_deepseek_v4_sparse_prefill_matmul_path_matches_default(
    monkeypatch,
) -> None:
    torch.manual_seed(1)
    num_tokens = 7
    num_heads = 8
    head_dim = 512
    num_kv_tokens = 333
    num_candidates = 65
    q = torch.randn(
        num_tokens,
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    kv = torch.randn(
        1,
        num_kv_tokens,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    combined_indices = torch.randint(
        -1,
        num_kv_tokens,
        (num_tokens, num_candidates),
        device="cuda",
        dtype=torch.int32,
    )
    combined_lens = torch.tensor(
        [65, 11, 0, 64, 23, 49, 4],
        device="cuda",
        dtype=torch.int32,
    )
    attn = SimpleNamespace(
        prefix="test",
        num_heads=num_heads,
        scale=head_dim**-0.5,
        attn_sink=torch.randn(num_heads, device="cuda", dtype=torch.float32) * 0.01,
    )

    def run_prefill(use_matmul: bool) -> torch.Tensor:
        monkeypatch.setenv(
            "VLLM_TRITON_MLA_SPARSE_MATMUL_PREFILL",
            "1" if use_matmul else "0",
        )
        monkeypatch.setenv("VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE", "17")
        monkeypatch.setenv("VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE", "3")
        max_score = torch.empty(3, num_heads, device="cuda", dtype=torch.float32)
        denom = torch.empty_like(max_score)
        acc = torch.empty(3, num_heads, head_dim, device="cuda", dtype=torch.float32)
        output = torch.empty_like(q)
        deepseek_v4_attention.DeepseekV4MLAAttention._forward_sparse_mla_prefill_triton(
            attn,
            q=q,
            kv=kv,
            combined_indices=combined_indices,
            combined_lens=combined_lens,
            output=output,
            state_buffers=(max_score, denom, acc),
        )
        torch.cuda.synchronize()
        return output.float()

    expected = run_prefill(use_matmul=False)
    actual = run_prefill(use_matmul=True)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
