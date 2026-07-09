# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.models.deepseek_v4 import attention as dsv4_attention


def test_attention_impl_skips_sparse_mla_without_attention_metadata(monkeypatch):
    monkeypatch.setattr(
        dsv4_attention,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )

    def fail_forward_mqa(*args, **kwargs):
        raise AssertionError("profile dummy run must not enter sparse MLA")

    fake_attention = SimpleNamespace(forward_mqa=fail_forward_mqa)
    out = torch.empty((2, 3, 4))

    dsv4_attention.DeepseekV4Attention.attention_impl(
        fake_attention,
        hidden_states=torch.empty((2, 8)),
        qr=torch.empty((2, 8)),
        kv=torch.empty((2, 4)),
        kv_score=None,
        indexer_kv_score=None,
        indexer_weights=None,
        positions=torch.arange(2),
        out=out,
    )

    assert torch.equal(out, torch.zeros_like(out))
