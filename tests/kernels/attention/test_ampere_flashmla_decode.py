# SPDX-License-Identifier: Apache-2.0
"""Glue test: ampere_flashmla_sparse_decode vs the Triton matmul decode path.

Random ragged section lengths (including empty sections) over the
[T, C, 576] materialized decode workspace, sink-aware.
"""

import pytest
import torch

import vllm.models.deepseek_v4.attention  # noqa: F401  (break circular import)
from vllm.v1.attention.backends.mla.sparse_mla_kernels import (
    matmul_sparse_mla_attention_with_sink,
)

H, D = 128, 576


def _reference(q, kv, topk_lens, swa_lens, compressed_topk, scale, sink, num_heads):
    T, C = kv.shape[:2]
    valid = torch.zeros(T, C, dtype=torch.bool, device=kv.device)
    for t in range(T):
        valid[t, : topk_lens[t]] = True
        valid[t, compressed_topk : compressed_topk + swa_lens[t]] = True
    out = torch.empty(T, q.shape[1], D, dtype=torch.bfloat16, device=kv.device)
    matmul_sparse_mla_attention_with_sink(
        q=q, kv=kv, valid_tokens=valid, scale=scale, attn_sink=sink,
        output=out, num_heads=num_heads,
        value_block_size=512, candidate_block_size=128,
    )
    return out


@pytest.mark.parametrize("T", [1, 4, 12])
@pytest.mark.parametrize("compressed_topk,swa_cap", [(512, 576), (512, 0), (32, 64)])
def test_glue_matches_triton(T, compressed_topk, swa_cap):
    from vllm.v1.attention.backends.mla.ampere_flashmla_decode import (
        ampere_flashmla_sparse_decode,
    )

    torch.manual_seed(7 + T)
    device = "cuda"
    C = compressed_topk + swa_cap
    q = torch.randn(T, H, D, dtype=torch.bfloat16, device=device)
    kv = torch.randn(T, C, D, dtype=torch.bfloat16, device=device)
    sink = torch.randn(H, dtype=torch.float32, device=device)
    scale = D ** -0.5

    topk_lens = torch.randint(0, compressed_topk + 1, (T,), device=device)
    if swa_cap:
        swa_lens = torch.randint(1, swa_cap + 1, (T,), device=device)
    else:
        swa_lens = torch.zeros(T, dtype=torch.int64, device=device)
    # at least one valid candidate per token
    topk_lens[swa_lens == 0] = topk_lens[swa_lens == 0].clamp(min=1)

    ref = _reference(q, kv, topk_lens, swa_lens, compressed_topk, scale, sink, H)

    out = torch.empty(T, H, D, dtype=torch.bfloat16, device=device)
    ampere_flashmla_sparse_decode(
        q=q, combined_kv=kv, topk_lens=topk_lens, swa_lens=swa_lens,
        compressed_topk=compressed_topk, scale=scale, attn_sink=sink,
        output=out, num_heads=H,
    )
    torch.cuda.synchronize()

    cos = torch.nn.functional.cosine_similarity(
        ref.float().flatten(), out.float().flatten(), dim=0
    )
    assert cos > 0.9999, f"cos={cos}"
    err = (ref.float() - out.float()).abs().max()
    assert err < 0.05, f"max abs err {err}"


if __name__ == "__main__":
    for T in (1, 4, 12):
        for ct, sc in ((512, 576), (512, 0), (32, 64)):
            test_glue_matches_triton(T, ct, sc)
            print(f"T={T} topk={ct} swa={sc}: OK")
