# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-decode-step scratch must be sized by the batch's actual max context,
not max_model_len.

Two independent decode-step allocations were sized by max_model_len:

A. Indexer decode logits: the sm86 Triton paged-MQA-logits fallback was called
   without ``token_count``, allocating a [rows, max_model_len]-wide fp32 logits
   scratch, -inf-filling the full width, and making the top-k scan the full
   width -- every decode step, per compress-4 layer (up to 256x oversize at
   short contexts).

B. C128A decode gather width: the padded C128A topk index view
   (cdiv(max_model_len/128, 128)*128 == 8192 columns at 1M) was passed to the
   flash_mla decode kernel unnarrowed, so the kernel's per-call ``sel_kv``
   scratch was sized T x (swa_topk + 8192) x 512 bf16 regardless of the
   batch's actual compressed context.

Both bounds derive from ``max_seq_len`` on the built attention metadata; the
runner builds capture-time metadata with ``max_seq_len = max_model_len``
(gpu_model_runner._build_attention_metadata, for_cudagraph_capture=True), so
inside FULL cudagraphs the widths stay full and replays serve any context.
The scratch is masked/-inf'd, so narrowing must not change results: the
correctness tests below assert bit-identical outputs.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.utils import deep_gemm
from vllm.utils.math_utils import round_up
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec

requires_sm8x = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 8,
    reason="requires an Ampere (sm_8x) GPU",
)

_HEAD_DIM = 128
_NUM_HEADS = 64
_SCALE_BYTES = 4


def _make_paged_logits_inputs(
    batch_size: int,
    next_n: int,
    context_lens_values: list[int],
    max_model_len: int,
    block_size: int = 64,
    seed: int = 0,
):
    """fp8 indexer cache + query for fp8_fp4_paged_mqa_logits (sm86 path)."""
    torch.manual_seed(seed)
    device = torch.device("cuda")
    assert len(context_lens_values) == batch_size
    max_ctx = max(context_lens_values)
    num_blocks_per_req = (max_model_len + block_size - 1) // block_size
    num_blocks = batch_size * num_blocks_per_req
    token_bytes = _HEAD_DIM + _SCALE_BYTES

    q = (
        torch.randn(
            (batch_size, next_n, _NUM_HEADS, _HEAD_DIM),
            device=device,
            dtype=torch.float32,
        )
        .clamp(-4, 4)
        .to(torch.float8_e4m3fn)
    )

    kv_cache = torch.zeros(
        (num_blocks, block_size, token_bytes), device=device, dtype=torch.uint8
    )
    needed_blocks = (max_ctx + block_size - 1) // block_size
    kv_flat = kv_cache.view(num_blocks, -1)
    value_end = block_size * _HEAD_DIM
    kv_flat[:, :value_end] = torch.randint(
        0, 255, (num_blocks, value_end), device=device, dtype=torch.uint8
    )
    scales = (
        torch.rand((num_blocks, block_size, 1), device=device, dtype=torch.float32)
        * 0.05
        + 0.001
    )
    kv_flat[:, value_end:] = scales.view(torch.uint8).reshape(num_blocks, -1)

    weights = torch.rand(
        (batch_size * next_n, _NUM_HEADS), device=device, dtype=torch.float32
    )
    context_lens = torch.tensor(
        context_lens_values, device=device, dtype=torch.int32
    ).unsqueeze(-1)
    if next_n > 1:
        context_lens = context_lens.expand(batch_size, next_n).contiguous()
    block_tables = (
        torch.arange(num_blocks, device=device, dtype=torch.int32)
        .view(batch_size, num_blocks_per_req)
        .contiguous()
    )
    del needed_blocks
    return q, kv_cache, weights, context_lens, block_tables


def _call_paged_logits(q, kv_cache, weights, context_lens, block_tables,
                       max_model_len, token_count=None):
    kwargs = {}
    if token_count is not None:
        kwargs["token_count"] = token_count
    return deep_gemm.fp8_fp4_paged_mqa_logits(
        (q, None),
        kv_cache.unsqueeze(-2),
        weights,
        context_lens,
        block_tables,
        torch.empty(0, device=q.device, dtype=torch.int32),
        max_model_len,
        clean_logits=False,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# A. Indexer decode logits width
# ---------------------------------------------------------------------------


def test_decode_logits_token_count_helper_values():
    from vllm.model_executor.layers.sparse_attn_indexer import (
        DECODE_LOGITS_WIDTH_ALIGNMENT,
        _decode_logits_token_count,
    )

    assert DECODE_LOGITS_WIDTH_ALIGNMENT == 128
    max_model_len = 262144  # 1M / compress-4
    # 4k context / 4 = 1024 compressed -> already aligned.
    assert _decode_logits_token_count(1024, max_model_len) == 1024
    # 100k context / 4 = 25000 -> padded to the 128 alignment.
    assert _decode_logits_token_count(25000, max_model_len) == 25088
    # Alignment boundaries.
    assert _decode_logits_token_count(127, max_model_len) == 128
    assert _decode_logits_token_count(128, max_model_len) == 128
    assert _decode_logits_token_count(129, max_model_len) == 256
    # Zero/empty context still yields a valid non-empty width.
    assert _decode_logits_token_count(0, max_model_len) == 128
    # Capture-time metadata carries max_seq_len == max_model_len: full width.
    assert _decode_logits_token_count(max_model_len, max_model_len) == max_model_len
    # Never exceeds max_model_len even when the rounded width would.
    assert _decode_logits_token_count(90, 100) == 100


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_indexer_decode_metadata_carries_max_context_len():
    device = torch.device("cuda")
    kv_cache_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        compress_ratio=4,
    )
    vllm_config = create_vllm_config(
        model_name="deepseek-ai/DeepSeek-V2-Lite-Chat",
        max_model_len=4096,
        hf_config_override={
            "sliding_window": 128,
            "index_topk": 4,
            "compress_ratios": [4],
        },
    )
    builder = DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["dummy"],
        vllm_config=vllm_config,
        device=device,
    )

    # Two single-token decode requests.
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([280, 519], dtype=torch.int32, device=device)
    common = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens.cpu(),
        num_reqs=2,
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=519,
        block_table_tensor=torch.zeros(
            (2, 16), dtype=torch.int32, device=device
        ),
        slot_mapping=torch.zeros(2, dtype=torch.int64, device=device),
        causal=True,
    )
    md = builder.build(common_prefix_len=0, common_attn_metadata=common)
    assert md.decode is not None
    # Compressed units: 519 // 4.
    assert md.decode.max_context_len == 129


@requires_sm8x
def test_paged_logits_dispatch_accepts_token_count():
    max_model_len = 4096
    q, kv, w, ctx, bt = _make_paged_logits_inputs(2, 1, [100, 500], max_model_len)
    out = _call_paged_logits(q, kv, w, ctx, bt, max_model_len, token_count=512)
    torch.cuda.synchronize()
    assert out.shape == (2, 512)
    assert out.is_contiguous()


@requires_sm8x
@pytest.mark.parametrize(
    "context_lens_values",
    [
        [100, 127],       # below one 128 tile
        [128, 129],       # tile boundary
        [255, 256],
        [500, 1000],
        [1, 2047],
    ],
)
def test_paged_logits_bit_identical_narrowed_vs_full(context_lens_values):
    """Narrowing only drops -inf'd columns: the shared prefix of the logits and
    the resulting top-k selection are bit-identical."""
    max_model_len = 4096
    topk = 512
    batch_size = len(context_lens_values)
    q, kv, w, ctx, bt = _make_paged_logits_inputs(
        batch_size, 1, context_lens_values, max_model_len
    )
    token_count = min(round_up(max(max(context_lens_values), 1), 128), max_model_len)

    full = _call_paged_logits(q, kv, w, ctx, bt, max_model_len)
    narrowed = _call_paged_logits(
        q, kv, w, ctx, bt, max_model_len, token_count=token_count
    )
    torch.cuda.synchronize()
    assert narrowed.shape == (batch_size, token_count)
    assert torch.equal(full[:, :token_count], narrowed)
    # All full-width columns beyond the narrowed width are masked.
    assert torch.isneginf(full[:, token_count:]).all()

    workspace = torch.zeros(1024 * 1024, dtype=torch.uint8, device=q.device)
    idx_full = torch.full(
        (batch_size, topk), -1, dtype=torch.int32, device=q.device
    )
    idx_narrow = torch.full_like(idx_full, -1)
    torch.ops._C.persistent_topk(
        full, ctx, idx_full, workspace, topk, full.shape[1]
    )
    torch.ops._C.persistent_topk(
        narrowed, ctx, idx_narrow, workspace, topk, narrowed.shape[1]
    )
    torch.cuda.synchronize()
    assert torch.equal(
        idx_full.sort(dim=1).values, idx_narrow.sort(dim=1).values
    )


@requires_sm8x
def test_paged_logits_narrowed_peak_allocation():
    """The per-step logits scratch must scale with the batch's context, not
    max_model_len."""
    max_model_len = 262144  # production compress-4 width at 1M
    rows = 4
    q, kv, w, ctx, bt = _make_paged_logits_inputs(
        rows, 1, [1024] * rows, max_model_len, block_size=64
    )
    token_count = 1024

    # Warm up allocator/JIT.
    _call_paged_logits(q, kv, w, ctx, bt, max_model_len, token_count=token_count)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = _call_paged_logits(
        q, kv, w, ctx, bt, max_model_len, token_count=token_count
    )
    torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated() - base
    full_bytes = rows * max_model_len * 4
    narrow_bytes = rows * token_count * 4
    # Way below the full-width allocation; roomy upper bound for the narrow one.
    assert peak_delta < full_bytes // 4, (peak_delta, full_bytes)
    assert peak_delta <= narrow_bytes * 4 + (1 << 20), (peak_delta, narrow_bytes)
    del out


@requires_sm8x
def test_paged_logits_cudagraph_capture_replay():
    """Sizes are python ints per call: within one captured graph the width is
    frozen and replays with updated context_lens stay correct."""
    max_model_len = 4096
    token_count = 2048  # capture-time width bound (>= any replayed context)
    graphs = {}
    statics = {}
    for batch_size in (2, 4):
        q, kv, w, ctx, bt = _make_paged_logits_inputs(
            batch_size, 1, [700] * batch_size, max_model_len, seed=batch_size
        )
        fn = lambda q=q, kv=kv, w=w, ctx=ctx, bt=bt: _call_paged_logits(
            q, kv, w, ctx, bt, max_model_len, token_count=token_count
        )
        # Warm up (JIT + allocator) outside capture.
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn()
        graphs[batch_size] = g
        statics[batch_size] = (q, kv, w, ctx, bt, out)

    for batch_size in (2, 4):
        g = graphs[batch_size]
        q, kv, w, ctx, bt, out = statics[batch_size]
        for new_ctx in (123, 1024, 2000):
            ctx.fill_(new_ctx)
            g.replay()
            torch.cuda.synchronize()
            ref = _call_paged_logits(
                q, kv, w, ctx, bt, max_model_len, token_count=token_count
            )
            torch.cuda.synchronize()
            assert torch.equal(out, ref), (batch_size, new_ctx)


# ---------------------------------------------------------------------------
# B. C128A decode gather width
# ---------------------------------------------------------------------------


def test_c128a_decode_topk_width_helper_values():
    from vllm.models.deepseek_v4.sparse_mla import (
        _C128A_TOPK_ALIGNMENT,
        c128a_decode_topk_width,
    )

    full = 8192  # cdiv(1M/128, 128) * 128
    assert _C128A_TOPK_ALIGNMENT == 128
    # 100k context -> 781 compressed rows -> padded to 896.
    assert c128a_decode_topk_width(100_000, 128, full) == 896
    # 4k context -> 32 compressed rows -> one alignment tile.
    assert c128a_decode_topk_width(4096, 128, full) == 128
    # 500k context -> 3906 -> 3968.
    assert c128a_decode_topk_width(500_000, 128, full) == 3968
    # Capture-time metadata (max_seq_len == max_model_len) keeps full width.
    assert c128a_decode_topk_width(1_048_576, 128, full) == full
    # Never exceeds the persistent buffer width.
    assert c128a_decode_topk_width(10_000_000, 128, full) == full
    # Tiny/zero contexts keep a non-empty aligned width.
    assert c128a_decode_topk_width(0, 128, full) == 128


@requires_sm8x
def test_sm86_forward_decode_narrows_c128a_extra_indices(monkeypatch):
    """_forward_decode must pass the flash_mla kernel a contiguous C128A index
    view narrowed to the batch's max compressed context (not the 8192-wide
    persistent buffer)."""
    from vllm.models.deepseek_v4.nvidia_imma import attention as sm86_attention
    from vllm.transformers_utils.configs.dsv4.kernel_config import (
        SPARSE_MLA_DECODE_INT8_FLASH,
    )

    device = torch.device("cuda")
    num_decode_tokens = 4
    full_width = 8192
    max_seq_len = 100_000
    expected_width = 896  # round_up(100_000 // 128, 128)

    buffer = torch.full(
        (num_decode_tokens, full_width), -1, dtype=torch.int32, device=device
    )
    valid = max_seq_len // 128
    buffer[:, :valid] = torch.arange(valid, dtype=torch.int32, device=device)
    topk_lens = torch.full(
        (num_decode_tokens,), valid, dtype=torch.int32, device=device
    )

    captured = {}

    def fake_decode_int8(**kwargs):
        captured.update(kwargs)
        q = kwargs["q"]
        return torch.zeros(
            (q.shape[0], q.shape[1], q.shape[2]),
            dtype=torch.bfloat16,
            device=q.device,
        )

    monkeypatch.setattr(
        sm86_attention, "sparse_mla_decode_int8", fake_decode_int8
    )

    heads = 8
    swa_window = 64
    fake_self = SimpleNamespace(
        compress_ratio=128,
        kv_cache_dtype="int8_ds_mla",
        int8_decode_symbol=SPARSE_MLA_DECODE_INT8_FLASH,
        topk_indices_buffer=None,
        scale=0.1,
        attn_sink=torch.full((heads,), -float("inf"), device=device),
        n_local_heads=heads,
        swa_cache_layer=SimpleNamespace(
            kv_cache=torch.zeros((4, 16, 528), dtype=torch.uint8, device=device)
        ),
    )
    swa_metadata = SimpleNamespace(
        num_decodes=num_decode_tokens,
        num_decode_tokens=num_decode_tokens,
        is_valid_token=torch.ones(
            num_decode_tokens, dtype=torch.bool, device=device
        ),
        decode_swa_indices=torch.zeros(
            (num_decode_tokens, swa_window), dtype=torch.int32, device=device
        ),
        decode_swa_lens=torch.ones(
            num_decode_tokens, dtype=torch.int32, device=device
        ),
        block_size=16,
    )
    attn_metadata = SimpleNamespace(
        block_size=256 * 128,
        max_seq_len=max_seq_len,
        c128a_global_decode_topk_indices=buffer.view(
            num_decode_tokens, 1, full_width
        ),
        c128a_decode_topk_lens=topk_lens,
    )
    kv_cache = torch.zeros((8, 16, 528), dtype=torch.uint8, device=device)
    q = torch.zeros(
        (num_decode_tokens, heads, 512), dtype=torch.bfloat16, device=device
    )
    output = torch.zeros_like(q)

    sm86_attention.DeepseekV4TritonSM86Attention._forward_decode(
        fake_self, q, kv_cache, swa_metadata, attn_metadata, False, output
    )

    extra_indices = captured["extra_indices"]
    assert extra_indices.shape == (num_decode_tokens, expected_width)
    assert extra_indices.is_contiguous()
    torch.testing.assert_close(
        extra_indices, buffer[:, :expected_width], rtol=0, atol=0
    )


@requires_sm8x
def test_c128a_narrowed_indices_decode_bit_identical_and_smaller_scratch(
    monkeypatch,
):
    """flash_mla decode with the narrowed contiguous index view computes the
    same attention as the full 8192-wide padded view and allocates a much
    smaller sel_kv.

    Bitwise equality is asserted with the split heuristic pinned to a single
    split: the kernel's num_splits keys off swa_topk + extra_topk, so changing
    the padded width reorders the split-combine fp reduction (the same class
    of reordering that already happens when the batch size changes). With one
    split both runs reduce the identical selected slots in the identical
    order, so any difference would be a real indexing bug.
    """
    from flash_mla import sparse_mla_decode_int8

    torch.manual_seed(7)
    device = torch.device("cuda")
    T, H, D = 2, 64, 512
    block_size = 64
    full_width = 8192
    ctx_rows = 781  # 100k context / 128
    narrow_width = 896

    q = torch.randn((T, H, D), dtype=torch.bfloat16, device=device)

    def make_cache(num_blocks):
        rows = torch.randint(
            -100, 100, (num_blocks, block_size, D), dtype=torch.int8, device=device
        )
        scales = (
            torch.rand((num_blocks, block_size), dtype=torch.float32, device=device)
            * 0.01
            + 1e-4
        )
        return rows, scales

    swa_rows, swa_scales = make_cache(2)
    extra_blocks = (ctx_rows + block_size - 1) // block_size
    extra_rows, extra_scales = make_cache(extra_blocks)

    swa_topk = 64
    swa_indices = torch.arange(
        swa_topk, dtype=torch.int32, device=device
    ).expand(T, swa_topk).contiguous()
    swa_lens = torch.full((T,), swa_topk, dtype=torch.int32, device=device)

    extra_full = torch.full(
        (T, full_width), -1, dtype=torch.int32, device=device
    )
    extra_full[:, :ctx_rows] = torch.arange(
        ctx_rows, dtype=torch.int32, device=device
    )
    extra_lens = torch.full((T,), ctx_rows, dtype=torch.int32, device=device)
    extra_narrow = extra_full[:, :narrow_width].contiguous()

    def run(extra_indices):
        return sparse_mla_decode_int8(
            q=q,
            swa_cache=swa_rows,
            swa_scale=swa_scales,
            swa_indices=swa_indices,
            swa_lens=swa_lens,
            scale=D**-0.5,
            attn_sink=None,
            extra_cache=extra_rows,
            extra_scale=extra_scales,
            extra_indices=extra_indices,
            extra_lens=extra_lens,
        )

    # Default split heuristic: numerically equivalent (bf16 split-combine
    # reordering only; measured max abs diff ~2.4e-4).
    out_full = run(extra_full)
    torch.cuda.synchronize()
    out_narrow = run(extra_narrow)
    torch.cuda.synchronize()
    torch.testing.assert_close(out_full, out_narrow, rtol=1e-2, atol=2e-3)

    # Pinned single split: bit-identical.
    monkeypatch.setenv("FLASH_MLA_SLOTS_PER_SPLIT", "999999999")
    out_full_1s = run(extra_full)
    torch.cuda.synchronize()
    out_narrow_1s = run(extra_narrow)
    torch.cuda.synchronize()
    assert torch.equal(out_full_1s, out_narrow_1s)
    monkeypatch.delenv("FLASH_MLA_SLOTS_PER_SPLIT")

    def peak_bytes(extra_indices):
        run(extra_indices)  # warm
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        out = run(extra_indices)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        del out
        return peak

    peak_full = peak_bytes(extra_full)
    peak_narrow = peak_bytes(extra_narrow)
    # sel_kv is T x (swa_topk + extra_topk) x 512 bf16; narrowing extra_topk
    # from 8192 to 896 must cut the per-call scratch by several MiB.
    sel_kv_saving = T * (full_width - narrow_width) * D * 2
    assert peak_full - peak_narrow > sel_kv_saving // 2, (
        peak_full,
        peak_narrow,
        sel_kv_saving,
    )
