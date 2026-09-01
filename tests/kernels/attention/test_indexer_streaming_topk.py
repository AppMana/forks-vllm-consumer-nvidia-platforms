# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native activation chunking for the DSV4 sparse-attention indexer prefill.

During chunked prefill the indexer materializes the full score row
``logits[M, N]`` (fp32, N = compressed prior context) before top-k selection,
so peak memory is O(chunk x window). The streaming path tiles the context
dimension into slabs. Each slab and every running candidate merge use the
native CUDA prefill radix/histogram selector. Since top-k over a union equals
top-k over the union of each input's top-k, the result has the same score
multiset as one-shot selection. Boundary ties may choose different columns,
matching the existing native selector's documented semantics.
"""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

capability = current_platform.get_device_capability()
if not current_platform.is_cuda() or capability is None or capability.major < 8:
    pytest.skip(
        "indexer streaming top-k requires CUDA compute capability 8.0+",
        allow_module_level=True,
    )

from vllm.model_executor.layers.sparse_attn_indexer import (  # noqa: E402
    _fp32_sort_keys_into,
    _indexer_prefill_logits,
    oneshot_prefill_topk_reference,
    streaming_prefill_topk,
)

HEADS = 64
HEAD_DIM = 128


def _make_inputs(m, n, *, qk_int8, seed=0, ctx_start=None, device="cuda"):
    """Causal-suffix layout of one request: query token i sees columns
    [0, ctx_start + i + 1) of its gathered compressed context."""
    torch.manual_seed(seed)
    if qk_int8:
        q = torch.randint(
            -16, 16, (m, HEADS, HEAD_DIM), dtype=torch.int8, device=device
        )
        k = torch.randint(-16, 16, (n, HEAD_DIM), dtype=torch.int8, device=device)
    else:
        q = (torch.randn(m, HEADS, HEAD_DIM, device=device) * 0.3).to(
            torch.float8_e4m3fn
        )
        k = (torch.randn(n, HEAD_DIM, device=device) * 0.3).to(torch.float8_e4m3fn)
    k_scale = torch.rand(n, dtype=torch.float32, device=device) + 0.5
    weights = torch.randn(m, HEADS, dtype=torch.float32, device=device) * 0.05
    ks = torch.zeros(m, dtype=torch.int32, device=device)
    if ctx_start is None:
        ctx_start = n - m
    ke = (torch.arange(m, dtype=torch.int32, device=device) + ctx_start + 1).clamp_(
        max=n
    )
    return q, k, k_scale, weights, ks, ke


def _torch_int8_mqa_logits(q, k, k_scale, weights, ks, ke):
    """Known-correct eager form of the INT8 indexer logits equation."""
    scores = torch.einsum("mhd,nd->mhn", q.float(), k.float())
    logits = (torch.relu(scores * k_scale[None, None, :]) * weights[:, :, None]).sum(
        dim=1
    )
    columns = torch.arange(k.shape[0], device=q.device)
    valid = (columns[None, :] >= ks[:, None]) & (columns[None, :] < ke[:, None])
    return logits.masked_fill(~valid, -torch.inf)


def test_int8_logits_and_streaming_topk_match_torch_eager():
    """Prove both the modified logits path and streamed selection against eager."""
    m, n, topk = 4, 4097, 512
    q, k, k_scale, weights, ks, ke = _make_inputs(
        m, n, qk_int8=True, seed=17, ctx_start=n - m
    )

    eager_logits = _torch_int8_mqa_logits(q, k, k_scale, weights, ks, ke)
    native_logits = _indexer_prefill_logits(
        q, (k, k_scale), weights, ks, ke, qk_int8=True
    )
    torch.testing.assert_close(native_logits, eager_logits, rtol=2e-5, atol=2e-3)

    eager_keys = torch.empty((m, n), dtype=torch.int64, device="cuda")
    _fp32_sort_keys_into(eager_logits, 0, eager_keys)
    eager_top_keys = torch.topk(
        eager_keys, topk, dim=1, largest=True, sorted=True
    ).values
    eager_indices = ((1 << 32) - 1 - (eager_top_keys & 0xFFFFFFFF)).to(
        torch.int32
    ) - ks[:, None]

    actual = torch.empty(m, topk, dtype=torch.int32, device="cuda")
    streaming_prefill_topk(
        q,
        (k, k_scale),
        weights,
        ks,
        ke,
        actual,
        topk,
        slab_rows=2048,
        qk_int8=True,
    )
    _assert_sets_equal_mod_boundary_ties(actual, eager_indices, eager_logits, ke, topk)


@pytest.mark.parametrize("qk_int8", [True, False])
@pytest.mark.parametrize("slab_rows", [4096, 16384, 8191])
def test_streaming_matches_oneshot_scores(qk_int8, slab_rows):
    """Slab-tiled native selection matches one-shot scores."""
    m, n, topk = 512, 65536, 2048  # 256k-token window at CSA-4, chunk 512
    q, k, k_scale, weights, ks, ke = _make_inputs(m, n, qk_int8=qk_int8)

    ref = oneshot_prefill_topk_reference(
        q, (k, k_scale), weights, ks, ke, topk, qk_int8=qk_int8
    )
    out = torch.empty(m, topk, dtype=torch.int32, device="cuda")
    streaming_prefill_topk(
        q,
        (k, k_scale),
        weights,
        ks,
        ke,
        out,
        topk,
        slab_rows=slab_rows,
        qk_int8=qk_int8,
    )
    logits = _indexer_prefill_logits(q, (k, k_scale), weights, ks, ke, qk_int8=qk_int8)
    _assert_sets_equal_mod_boundary_ties(out, ref, logits, ke, topk)


def _assert_sets_equal_mod_boundary_ties(out, prod, logits, ke, topk):
    """Selected sets must be identical up to substitution among columns whose
    logit bit-equals the k-th boundary value (the production kernel's tie
    order at the boundary is unspecified -- even int8-IMMA random data
    produces genuine bit-equal fp32 ties there)."""
    m = out.shape[0]
    ke_cpu = ke.cpu()
    for r in range(m):
        o = set(out[r][out[r] >= 0].tolist())
        p = set(prod[r][prod[r] >= 0].tolist())
        if o == p:
            continue
        row = logits[r, : int(ke_cpu[r])]
        assert row.numel() > 0
        kth = torch.topk(row, min(topk, row.numel())).values[-1]
        # Same score multiset...
        lo = row[torch.tensor(sorted(o), device=row.device)].sort().values
        lp = row[torch.tensor(sorted(p), device=row.device)].sort().values
        assert torch.equal(lo, lp), f"row {r}: score multisets differ"
        # ...and every disagreement sits exactly on the boundary value.
        for c in o.symmetric_difference(p):
            assert row[c] == kth, f"row {r} col {c}: non-tie disagreement"


@pytest.mark.parametrize("qk_int8", [True])
def test_streaming_matches_production_kernel_set(qk_int8):
    """The selection must match ops.top_k_per_row_prefill up to boundary-tie
    permutation (the production kernel's tie order is unspecified)."""
    m, n, topk = 256, 32768, 2048
    q, k, k_scale, weights, ks, ke = _make_inputs(m, n, qk_int8=qk_int8, seed=1)

    logits = _indexer_prefill_logits(q, (k, k_scale), weights, ks, ke, qk_int8=qk_int8)
    prod = torch.full((m, topk), -1, dtype=torch.int32, device="cuda")
    ops.top_k_per_row_prefill(
        logits, ks, ke, prod, m, logits.stride(0), logits.stride(1), topk
    )

    out = torch.empty(m, topk, dtype=torch.int32, device="cuda")
    streaming_prefill_topk(
        q,
        (k, k_scale),
        weights,
        ks,
        ke,
        out,
        topk,
        slab_rows=8192,
        qk_int8=qk_int8,
    )
    _assert_sets_equal_mod_boundary_ties(out, prod, logits, ke, topk)


def test_streaming_ties_use_one_deterministic_total_order():
    """Equal scores must resolve identically across launches and slab sizes.

    INT8 indexer logits contain genuine bit-equal boundary scores. Selecting
    different physical columns for those ties changes the attention context,
    so equality of score multisets is insufficient for deterministic greedy
    generation.
    """
    m, n, topk = 16, 16384, 2048
    torch.manual_seed(2)
    q = torch.randint(-2, 3, (m, HEADS, HEAD_DIM), dtype=torch.int8, device="cuda")
    base = torch.randint(-2, 3, (256, HEAD_DIM), dtype=torch.int8, device="cuda")
    k = base.repeat(n // 256, 1)  # every score value repeats n/256 times
    k_scale = torch.ones(n, dtype=torch.float32, device="cuda")
    weights = torch.ones(m, HEADS, dtype=torch.float32, device="cuda")
    ks = torch.zeros(m, dtype=torch.int32, device="cuda")
    ke = torch.full((m,), n, dtype=torch.int32, device="cuda")

    ref = oneshot_prefill_topk_reference(
        q, (k, k_scale), weights, ks, ke, topk, qk_int8=True
    )
    for slab in (1024, 4096, 5000, 16384):
        for _ in range(2):
            out = torch.empty(m, topk, dtype=torch.int32, device="cuda")
            streaming_prefill_topk(
                q,
                (k, k_scale),
                weights,
                ks,
                ke,
                out,
                topk,
                slab_rows=slab,
                qk_int8=True,
            )
            assert torch.equal(out, ref), f"tie order changed at slab={slab}"


def test_short_rows_pad_minus_one():
    """Rows with fewer than top-k valid columns must select all of them and
    pad with -1, matching the production kernel's contract."""
    m, n, topk = 64, 8192, 2048
    q, k, k_scale, weights, ks, ke = _make_inputs(
        m, n, qk_int8=True, seed=3, ctx_start=64
    )
    # ke = 65..128: every row is shorter than topk; also one empty row.
    ke[0] = 0

    ref = oneshot_prefill_topk_reference(
        q, (k, k_scale), weights, ks, ke, topk, qk_int8=True
    )
    out = torch.empty(m, topk, dtype=torch.int32, device="cuda")
    streaming_prefill_topk(
        q, (k, k_scale), weights, ks, ke, out, topk, slab_rows=4096, qk_int8=True
    )
    logits = _indexer_prefill_logits(q, (k, k_scale), weights, ks, ke, qk_int8=True)
    _assert_sets_equal_mod_boundary_ties(out, ref, logits, ke, topk)
    assert (out[0] == -1).all()
    ke_cpu = ke.cpu()
    for i in range(m):
        row_len = int(ke_cpu[i])
        assert (out[i] >= 0).sum().item() == row_len
        got = out[i, :row_len].sort().values.cpu()
        assert torch.equal(got, torch.arange(row_len, dtype=torch.int32))


def test_multi_request_row_offsets():
    """Two requests concatenated in the gathered K buffer: indices must be
    LOCAL to each request (production kernel subtracts rowStart)."""
    topk = 512
    n1, n2, m = 8192, 12288, 128
    n = n1 + n2
    q, k, k_scale, weights, _, _ = _make_inputs(m, n, qk_int8=True, seed=4)
    ks = torch.empty(m, dtype=torch.int32, device="cuda")
    ke = torch.empty(m, dtype=torch.int32, device="cuda")
    half = m // 2
    ks[:half] = 0
    ke[:half] = n1 - half + torch.arange(half, dtype=torch.int32, device="cuda") + 1
    ks[half:] = n1
    ke[half:] = (
        n1 + n2 - half + torch.arange(half, dtype=torch.int32, device="cuda") + 1
    )

    out = torch.empty(m, topk, dtype=torch.int32, device="cuda")
    streaming_prefill_topk(
        q, (k, k_scale), weights, ks, ke, out, topk, slab_rows=4096, qk_int8=True
    )
    # Local index ranges: request 2's rows must all be < n2.
    assert out[half:].max().item() < n2

    # Set-compare against the production kernel too. Its indices are local;
    # re-localize both against ks=0 for the helper by adding row starts back.
    logits = _indexer_prefill_logits(q, (k, k_scale), weights, ks, ke, qk_int8=True)
    prod = torch.full((m, topk), -1, dtype=torch.int32, device="cuda")
    ops.top_k_per_row_prefill(
        logits, ks, ke, prod, m, logits.stride(0), logits.stride(1), topk
    )
    ks_col = ks.unsqueeze(1)
    out_glob = torch.where(out >= 0, out + ks_col, out)
    prod_glob = torch.where(prod >= 0, prod + ks_col, prod)
    _assert_sets_equal_mod_boundary_ties(out_glob, prod_glob, logits, ke, topk)


def test_sm120_int8_streaming_selects_triton_logits_per_slab(monkeypatch):
    from vllm.models.deepseek_v4.nvidia_imma import triton_kernels

    calls = []

    def fake_triton(q, kv, weights, k_start, k_end, *, qk_int8):
        del weights, k_start, k_end
        assert qk_int8
        calls.append(kv[0].shape[0])
        scores = torch.arange(kv[0].shape[0], dtype=torch.float32, device=q.device)
        return scores.expand(q.shape[0], -1).contiguous()

    monkeypatch.setattr(
        triton_kernels,
        "mqa_logits_workspace_triton",
        fake_triton,
    )
    q = torch.zeros((2, HEADS, HEAD_DIM), dtype=torch.int8, device="cuda")
    k = torch.zeros((8, HEAD_DIM), dtype=torch.int8, device="cuda")
    scales = torch.ones(8, dtype=torch.float32, device="cuda")
    weights = torch.ones((2, HEADS), dtype=torch.float32, device="cuda")
    k_start = torch.zeros(2, dtype=torch.int32, device="cuda")
    k_end = torch.full((2,), 8, dtype=torch.int32, device="cuda")
    out = torch.empty((2, 2), dtype=torch.int32, device="cuda")

    streaming_prefill_topk(
        q,
        (k, scales),
        weights,
        k_start,
        k_end,
        out,
        2,
        slab_rows=4,
        qk_int8=True,
    )

    assert calls == [4, 4]
    assert torch.equal(
        out, torch.tensor([[3, 7], [3, 7]], dtype=torch.int32, device="cuda")
    )


def test_vllm_block_gates_streaming_and_sets_slab_rows(monkeypatch):
    """The checkpoint "vllm" block turns the gate on and its
    indexer_prefill_topk_slab_rows key drives the default slab width, with the
    selection staying exact."""
    import vllm.model_executor.layers.sparse_attn_indexer as indexer_mod
    import vllm.transformers_utils.configs.dsv4.kernel_config as kernel_config

    monkeypatch.setattr(kernel_config, "_ACTIVE_CONFIG", None)
    assert not indexer_mod.should_use_prefill_streaming_topk(1, False)

    kernel_config.activate_kernel_config(
        kernel_config.resolve_kernel_config(
            {
                "kernels": [kernel_config.INDEXER_STREAMING_TOPK_PREFILL],
                "indexer_prefill_topk_slab_rows": 5000,
            }
        )
    )
    assert indexer_mod.should_use_prefill_streaming_topk(1, False)
    assert indexer_mod._resolved_prefill_topk_slab_rows() == 5000
    assert not indexer_mod.should_stream_prefill_topk_for_context(1, False, 5000)
    assert indexer_mod.should_stream_prefill_topk_for_context(1, False, 5001)

    m, n, topk = 128, 16384, 512
    q, k, k_scale, weights, ks, ke = _make_inputs(m, n, qk_int8=True, seed=5)
    ref = oneshot_prefill_topk_reference(
        q, (k, k_scale), weights, ks, ke, topk, qk_int8=True
    )
    # No explicit slab_rows: the config-block value (5000, unaligned) is used.
    out = torch.empty(m, topk, dtype=torch.int32, device="cuda")
    streaming_prefill_topk(q, (k, k_scale), weights, ks, ke, out, topk, qk_int8=True)
    logits = _indexer_prefill_logits(q, (k, k_scale), weights, ks, ke, qk_int8=True)
    _assert_sets_equal_mod_boundary_ties(out, ref, logits, ke, topk)
