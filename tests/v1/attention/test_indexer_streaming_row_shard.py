# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming indexer scoring must carry the weights of each TP-local row."""

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.sparse_attn_indexer as indexer
from vllm.config import CUDAGraphMode
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_streaming_scores_use_matching_query_and_weight_rows(monkeypatch, rank):
    sizes = [3, 2, 3]
    chunks = [
        SimpleNamespace(
            token_start=start,
            token_end=end,
            cu_seqlen_ks=torch.zeros(end - start, dtype=torch.int32),
            cu_seqlen_ke=torch.arange(start + 1, end + 1, dtype=torch.int32),
            local_cu_seq_lens=torch.tensor([0, 8], dtype=torch.int32),
            local_total_seq_lens=8,
            max_local_total_seq_lens=8,
            skip_kv_gather=True,
            pcp_deinterleave_idx=None,
        )
        for start, end in [(0, 5), (5, 8)]
    ]
    metadata = DeepseekV32IndexerMetadata(
        seq_lens=torch.tensor([8], dtype=torch.int32),
        max_seq_len=8,
        slot_mapping=torch.arange(8),
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
        num_prefill_tokens=8,
        prefill=SimpleNamespace(chunks=chunks, row_shard_sizes=sizes),
    )
    monkeypatch.setattr(
        indexer,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={"indexer": metadata},
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        ),
    )
    monkeypatch.setattr(indexer, "get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setattr(indexer, "_reserve_prefill_gather_workspace", lambda **kw: None)
    monkeypatch.setattr(indexer, "_merge_dcp_topk_global", lambda *args, **kw: None)
    monkeypatch.setattr(
        indexer, "should_stream_prefill_topk_for_context", lambda *a: True
    )
    monkeypatch.setattr(
        indexer,
        "current_workspace_manager",
        lambda: SimpleNamespace(
            get_simultaneous=lambda *specs: [
                torch.zeros(shape, dtype=dtype) for shape, dtype in specs
            ]
        ),
    )
    # The upstream exchange test covers gathering. Keep the local output here
    # so each rank's exact rows can be checked without a distributed runtime.
    monkeypatch.setattr(indexer.PrefillRowShard, "exchange_topk", lambda *a: None)
    calls = []

    def score(q, kv, weights, starts, ends, out, topk):
        row_ids = q[:, 0, 0].float()
        torch.testing.assert_close(weights[:, 0], row_ids + 100)
        torch.testing.assert_close(ends, row_ids.to(torch.int32) + 1)
        out.copy_(weights.to(torch.int32).expand(-1, topk))
        calls.extend(row_ids.tolist())

    monkeypatch.setattr(indexer, "streaming_prefill_topk", score)
    q = torch.arange(8, dtype=torch.float32).reshape(8, 1, 1)
    weights = torch.arange(100, 108, dtype=torch.float32).reshape(8, 1)
    out = torch.full((8, 2), -7, dtype=torch.int32)
    indexer.sparse_attn_indexer(
        hidden_states=torch.empty(8, 1),
        k_cache_prefix="indexer",
        kv_cache=torch.empty(0),
        q_quant=q,
        q_scale=None,
        k=None,
        weights=weights,
        quant_block_size=128,
        scale_fmt="ue8m0",
        topk_tokens=2,
        head_dim=128,
        max_model_len=8,
        total_seq_lens=8,
        topk_indices_buffer=out,
        skip_k_cache_insert=True,
        use_pcp=False,
        pcp_shard_decode_requests=False,
        dense_mha_metadata_layer_name="",
    )
    start, end = sum(sizes[:rank]), sum(sizes[: rank + 1])
    assert calls == list(range(start, end))
    torch.testing.assert_close(
        out[start:end], weights[start:end].to(torch.int32).expand(-1, 2)
    )
    assert torch.all(out[:start] == -1)
    assert torch.all(out[end:] == -1)
