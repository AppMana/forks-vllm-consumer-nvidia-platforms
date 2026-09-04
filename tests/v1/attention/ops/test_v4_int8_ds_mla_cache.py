# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 int8_ds_mla cache layout tests."""

import inspect

import pytest
import torch

from vllm.models.deepseek_v4.attention import _resolve_dsv4_kv_cache_dtype
from vllm.models.deepseek_v4.common.ops.cache_utils import (
    dequantize_and_gather_int8_ds_mla_cache,
    dequantize_and_gather_k_cache,
    dequantize_global_slots_int8_ds_mla_cache,
    get_int8_ds_mla_cache_views,
    quantize_and_insert_int8_ds_mla_cache,
)
from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    compress_norm_rope_store_triton,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _expected_int8_dequant(k: torch.Tensor) -> torch.Tensor:
    scale = (k.float().abs().amax(dim=-1) / 127.0).clamp_min(1.0e-12)
    q = torch.round(k.float() / scale.unsqueeze(-1)).clamp(-127, 127)
    return (q * scale.unsqueeze(-1)).to(torch.bfloat16)


_INT8_DS_MLA_ATOL = 3.2e-2


def test_int8_ds_mla_compressor_launcher_accepts_runtime_selector() -> None:
    signature = inspect.signature(compress_norm_rope_store_triton)
    assert "int8_ds_mla" in signature.parameters


def test_int8_ds_mla_resolves_to_uint8_even_for_fp8_layout_backend() -> None:
    dtype_str, torch_dtype = _resolve_dsv4_kv_cache_dtype(
        use_fp8_ds_mla_layout=True,
        kv_cache_dtype="int8_ds_mla",
        cache_config=None,
    )
    assert dtype_str == "int8_ds_mla"
    assert torch_dtype is torch.uint8


def test_int8_ds_mla_cache_shapes_and_page_sizes() -> None:
    # The 528-byte row (512B signed-int8 + fp32 row scale + 12B pad) is carried
    # by the spec, not by a backend get_kv_cache_shape override: upstream drives
    # the page geometry from num_heads * num_states * state_content_size_bytes.
    main_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        tokens_per_state=4,
        state_content_bytes=528,
        cache_dtype_str="int8_ds_mla",
        alignment=528,
        model_version="deepseek_v4",
    )
    assert main_spec.num_states == 64
    assert main_spec.state_content_size_bytes == 528
    assert main_spec.real_page_size_bytes == 64 * 528
    assert main_spec.page_size_bytes == 64 * 528

    swa_spec = SlidingWindowMLASpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        sliding_window=128,
        state_content_bytes=528,
        cache_dtype_str="int8_ds_mla",
        alignment=528,
        model_version="deepseek_v4",
    )
    assert swa_spec.real_page_size_bytes == 64 * 528
    assert swa_spec.page_size_bytes == 64 * 528


def test_int8_ds_mla_insert_global_dequant_and_views() -> None:
    device = _device()
    block_size = 4
    k = torch.randn(6, 512, dtype=torch.bfloat16, device=device)
    slot_mapping = torch.tensor([0, 1, 5, -1, 8, 9], dtype=torch.int64, device=device)
    k_cache = torch.zeros(3, block_size, 528, dtype=torch.uint8, device=device)

    quantize_and_insert_int8_ds_mla_cache(k, k_cache, slot_mapping, block_size)

    rows_i8, scales = get_int8_ds_mla_cache_views(k_cache, block_size)
    assert rows_i8.shape == (3, block_size, 512)
    assert rows_i8.dtype == torch.int8
    assert scales.shape == (3, block_size)
    assert scales.dtype == torch.float32
    assert rows_i8.stride()[1] == 528

    slot_ids = torch.tensor([[0, 5, -1], [8, 9, 1]], dtype=torch.int64, device=device)
    out = torch.empty(2, 3, 512, dtype=torch.bfloat16, device=device)
    dequantize_global_slots_int8_ds_mla_cache(out, k_cache, slot_ids, block_size)

    expected = _expected_int8_dequant(k)
    torch.testing.assert_close(out[0, 0], expected[0], rtol=0, atol=_INT8_DS_MLA_ATOL)
    torch.testing.assert_close(out[0, 1], expected[2], rtol=0, atol=_INT8_DS_MLA_ATOL)
    torch.testing.assert_close(out[1, 0], expected[4], rtol=0, atol=_INT8_DS_MLA_ATOL)
    torch.testing.assert_close(out[1, 1], expected[5], rtol=0, atol=_INT8_DS_MLA_ATOL)
    torch.testing.assert_close(out[1, 2], expected[1], rtol=0, atol=_INT8_DS_MLA_ATOL)
    assert torch.equal(out[0, 2], torch.zeros_like(out[0, 2]))


def test_int8_ds_mla_gather_dequant() -> None:
    device = _device()
    block_size = 4
    k = torch.randn(8, 512, dtype=torch.bfloat16, device=device)
    k_cache = torch.zeros(2, block_size, 528, dtype=torch.uint8, device=device)
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
    torch.testing.assert_close(
        out[0, 2:5], expected[5:8], rtol=0, atol=_INT8_DS_MLA_ATOL
    )


def test_int8_ds_mla_gather_dequant_prefill_shape() -> None:
    device = _device()
    if device.type != "cuda":
        return

    block_size = 64
    num_blocks = 40
    num_tokens = num_blocks * block_size
    k = torch.randn(num_tokens, 512, dtype=torch.bfloat16, device=device)
    k_cache = torch.zeros(num_blocks, block_size, 528, dtype=torch.uint8, device=device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    quantize_and_insert_int8_ds_mla_cache(k, k_cache, slot_mapping, block_size)

    out = torch.empty(2, 1152, 512, dtype=torch.bfloat16, device=device)
    seq_lens = torch.tensor([1536, 2560], dtype=torch.int32, device=device)
    gather_lens = torch.tensor([512, 1024], dtype=torch.int32, device=device)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).view(1, -1)
    block_table = block_table.repeat(2, 1)
    dequantize_and_gather_int8_ds_mla_cache(
        out,
        k_cache,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        block_table=block_table,
        block_size=block_size,
        offset=128,
    )
    torch.cuda.synchronize()

    expected = _expected_int8_dequant(k)
    torch.testing.assert_close(
        out[0, 128:640], expected[1024:1536], rtol=0, atol=_INT8_DS_MLA_ATOL
    )
    torch.testing.assert_close(
        out[1, 128:1152], expected[1536:2560], rtol=0, atol=_INT8_DS_MLA_ATOL
    )


def test_generic_gather_dispatches_int8_ds_mla() -> None:
    device = _device()
    block_size = 4
    k = torch.randn(8, 512, dtype=torch.bfloat16, device=device)
    k_cache = torch.zeros(2, block_size, 528, dtype=torch.uint8, device=device)
    slot_mapping = torch.arange(8, dtype=torch.int64, device=device)
    quantize_and_insert_int8_ds_mla_cache(k, k_cache, slot_mapping, block_size)

    out = torch.empty(1, 3, 512, dtype=torch.bfloat16, device=device)
    seq_lens = torch.tensor([8], dtype=torch.int32, device=device)
    gather_lens = torch.tensor([3], dtype=torch.int32, device=device)
    block_table = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    dequantize_and_gather_k_cache(
        out,
        k_cache,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        block_table=block_table,
        block_size=block_size,
        offset=0,
        cache_dtype="int8_ds_mla",
    )

    expected = _expected_int8_dequant(k)
    torch.testing.assert_close(
        out[0], expected[5:8], rtol=0, atol=_INT8_DS_MLA_ATOL
    )


def test_int8_ds_mla_token_stride_is_16_byte_multiple() -> None:
    """516-byte token rows alternate 16B alignment across consecutive tokens;
    every packed-layout consumer (csrc uint4 stores, Triton vectorization)
    requires a 16-byte-multiple stride. 528 = 512 int8 + 4B fp32 scale + 12B pad.
    """
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        _INT8_DS_MLA_TOKEN_BYTES,
    )

    assert _INT8_DS_MLA_TOKEN_BYTES % 16 == 0
    assert _INT8_DS_MLA_TOKEN_BYTES == 528


def _reference_qnorm_rope(
    x: torch.Tensor,  # [..., 512] bf16
    positions: torch.Tensor,  # [N] int64
    cos_sin: torch.Tensor,  # [max_pos, 64] fp32
    eps: float | None,
) -> torch.Tensor:
    """Reference for the fused q-norm/rope (eps given) or kv rope (eps None).

    Matches the csrc fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
    semantics: fp32 math, RMSNorm over the whole 512-dim head (no weight),
    GPT-J RoPE on dims [448, 512) with interleaved (even, odd) pairs.
    """
    xf = x.to(torch.float32)
    if eps is not None:
        rms_rcp = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
        xf = xf * rms_rcp
    rope = xf[..., 448:].clone()
    even = rope[..., 0::2].clone()
    odd = rope[..., 1::2].clone()
    # cos_sin rows: 32 cos then 32 sin.
    cs = cos_sin[positions]  # [N, 64]
    shape = [positions.shape[0]] + [1] * (xf.dim() - 2) + [32]
    cos = cs[:, :32].view(shape)
    sin = cs[:, 32:].view(shape)
    rope[..., 0::2] = even * cos - odd * sin
    rope[..., 1::2] = odd * cos + even * sin
    out = torch.cat([xf[..., :448], rope], dim=-1)
    return out


def _reference_int8_quant_row(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-wise int8 quant matching the Triton writers (bf16 round-trip,
    absmax/127 scale, round-half-away-from-zero)."""
    xf = x.to(torch.bfloat16).to(torch.float32)
    absmax = xf.abs().amax(dim=-1).clamp_min(1.0e-12)
    scale = absmax / 127.0
    s = xf / scale.unsqueeze(-1)
    rounded = torch.trunc(s + torch.where(s >= 0, 0.5, -0.5))
    return rounded.clamp(-127, 127).to(torch.int8), scale


def _run_fused_insert_case(k_cache: torch.Tensor, block_size: int) -> None:
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        fused_qnorm_rope_kv_int8_ds_mla_insert,
    )

    device = k_cache.device
    torch.manual_seed(0)
    num_tokens = 6
    num_heads = 3
    padded_heads = 4
    eps = 1e-6
    q = torch.randn(num_tokens, num_heads, 512, dtype=torch.bfloat16, device=device)
    kv = torch.randn(num_tokens, 512, dtype=torch.bfloat16, device=device)
    positions = torch.tensor([0, 3, 7, 11, 2, 5], dtype=torch.int64, device=device)
    cos_sin = torch.randn(16, 64, dtype=torch.float32, device=device)
    slot_mapping = torch.tensor([0, 1, 5, -1, 8, 9], dtype=torch.int64, device=device)

    q_out = fused_qnorm_rope_kv_int8_ds_mla_insert(
        q,
        kv,
        k_cache,
        slot_mapping,
        positions,
        cos_sin,
        padded_heads,
        eps,
        block_size,
    )
    torch.cuda.synchronize()

    # ----- Q side: RMSNorm + RoPE + zero padding -----
    assert q_out.shape == (num_tokens, padded_heads, 512)
    q_ref = _reference_qnorm_rope(q, positions, cos_sin, eps).to(torch.bfloat16)
    torch.testing.assert_close(
        q_out[:, :num_heads], q_ref, rtol=1.6e-2, atol=1.6e-2
    )
    assert torch.equal(
        q_out[:, num_heads:], torch.zeros_like(q_out[:, num_heads:])
    )

    # ----- KV side: RoPE (no norm) + int8 row + fp32 scale -----
    kv_ref = _reference_qnorm_rope(kv.unsqueeze(1), positions, cos_sin, None)[:, 0]
    q_i8_ref, scale_ref = _reference_int8_quant_row(kv_ref)

    rows_i8, scales = get_int8_ds_mla_cache_views(k_cache, block_size)
    for token, slot in enumerate(slot_mapping.tolist()):
        if slot < 0:
            continue
        b, p = divmod(slot, block_size)
        got = rows_i8[b, p].to(torch.float32) * scales[b, p]
        want = q_i8_ref[token].to(torch.float32) * scale_ref[token]
        torch.testing.assert_close(got, want, rtol=0, atol=_INT8_DS_MLA_ATOL)


def test_fused_qnorm_rope_kv_int8_insert_contiguous() -> None:
    device = _device()
    if device.type != "cuda":
        return
    block_size = 4
    k_cache = torch.zeros(3, block_size, 528, dtype=torch.uint8, device=device)
    _run_fused_insert_case(k_cache, block_size)


def test_fused_qnorm_rope_kv_int8_insert_packed_crash_geometry() -> None:
    """Reconstruct the JobSet dsv4-m2-int4-int8kv-001 fault geometry: the SWA
    cache is a strided view of a packed multi-layer per-block buffer whose
    block stride is NOT a 16-byte multiple (observed 42696 ≡ 8 mod 16) and
    whose storage offset is only 4-byte aligned. The insert must neither fault
    (csrc uint4 stores did: CUDA misaligned address) nor corrupt bytes outside
    its own rows.
    """
    device = _device()
    if device.type != "cuda":
        return
    block_size = 4
    num_blocks = 3
    page_bytes = block_size * 528
    block_stride = page_bytes + 24  # ≡ 8 mod 16, like the packed layout
    storage_offset = 4  # 4-byte aligned only
    backing = torch.full(
        (storage_offset + num_blocks * block_stride,),
        0xAB,
        dtype=torch.uint8,
        device=device,
    )
    k_cache = backing.as_strided(
        (num_blocks, block_size, 528),
        (block_stride, 528, 1),
        storage_offset=storage_offset,
    )
    k_cache.zero_()
    _run_fused_insert_case(k_cache, block_size)
    # Bytes outside the cache view must be untouched.
    flat = backing.clone()
    flat_view = flat.as_strided(
        (num_blocks, block_size, 528),
        (block_stride, 528, 1),
        storage_offset=storage_offset,
    )
    flat_view.copy_(k_cache)
    mask = torch.ones_like(backing, dtype=torch.bool)
    mask_view = mask.as_strided(
        (num_blocks, block_size, 528),
        (block_stride, 528, 1),
        storage_offset=storage_offset,
    )
    mask_view.fill_(False)
    assert torch.all(backing[mask] == 0xAB)


def test_fused_qnorm_rope_kv_insert_dispatches_int8(monkeypatch) -> None:
    """attention._fused_qnorm_rope_kv_insert must route int8_ds_mla uint8
    caches to the int8 writer, NOT the csrc fp8_ds_mla writer (which writes a
    576/584-byte layout with 16B uint4 stores)."""
    import types

    import vllm.models.deepseek_v4.attention as dsv4_attention

    calls: list[str] = []

    def record_int8(*args, **kwargs):
        calls.append("int8")
        q = args[0]
        return q.new_zeros(q.shape[0], args[6], q.shape[-1])

    monkeypatch.setattr(
        dsv4_attention,
        "fused_qnorm_rope_kv_int8_ds_mla_insert",
        record_int8,
    )

    device = _device()
    layer = types.SimpleNamespace(
        swa_cache_layer=types.SimpleNamespace(
            prefix="swa",
            kv_cache=torch.zeros(2, 4, 528, dtype=torch.uint8, device=device),
        ),
        kv_cache_dtype="int8_ds_mla",
        rotary_emb=types.SimpleNamespace(
            cos_sin_cache=torch.zeros(8, 64, dtype=torch.float32, device=device)
        ),
        padded_heads=4,
        n_local_heads=4,
        eps=1e-6,
    )
    attn_metadata = {
        "swa": types.SimpleNamespace(
            slot_mapping=torch.zeros(2, dtype=torch.int64, device=device),
            block_size=4,
        )
    }
    q = torch.zeros(2, 4, 512, dtype=torch.bfloat16, device=device)
    kv = torch.zeros(2, 512, dtype=torch.bfloat16, device=device)
    positions = torch.zeros(2, dtype=torch.int64, device=device)

    dsv4_attention.DeepseekV4Attention._fused_qnorm_rope_kv_insert(
        layer, q, kv, positions, attn_metadata
    )
    assert calls == ["int8"]


@pytest.mark.parametrize("flat", [False, True])
def test_generic_gather_rejects_fp8_layout_for_explicit_int8_dtype(flat) -> None:
    block_size = 64
    shape = (2, block_size * 584) if flat else (2, block_size, 584)
    fp8_shaped_cache = torch.zeros(shape, dtype=torch.uint8)

    with pytest.raises(ValueError, match="528-byte INT8 cache layout"):
        dequantize_and_gather_k_cache(
            torch.empty(1, 1, 512, dtype=torch.bfloat16),
            fp8_shaped_cache,
            seq_lens=torch.tensor([1], dtype=torch.int32),
            gather_lens=torch.tensor([1], dtype=torch.int32),
            block_table=torch.tensor([[0]], dtype=torch.int32),
            block_size=block_size,
            offset=0,
            cache_dtype="int8_ds_mla",
        )


def test_generic_gather_rejects_wrong_block_count_for_int8_layout() -> None:
    block_size = 64
    wrong_block_count = torch.zeros(
        2, block_size - 1, 528, dtype=torch.uint8
    )
    with pytest.raises(ValueError, match="528-byte INT8 cache layout"):
        dequantize_and_gather_k_cache(
            torch.empty(1, 1, 512, dtype=torch.bfloat16),
            wrong_block_count,
            seq_lens=torch.tensor([1], dtype=torch.int32),
            gather_lens=torch.tensor([1], dtype=torch.int32),
            block_table=torch.tensor([[0]], dtype=torch.int32),
            block_size=block_size,
            offset=0,
            cache_dtype="int8_ds_mla",
        )


def test_fused_qnorm_rope_kv_int8_insert_guards_oob_slot_and_position() -> None:
    """A slot beyond the paged cache or a position beyond the cos/sin table is
    corrupt scheduler state; before the guards, the store/load through it was
    an illegal memory access that killed the engine (the production EOS-tail
    crash signature). The kernel must survive both, count each trip, and leave
    every in-bounds token's row correct."""
    device = _device()
    if device.type != "cuda":
        return
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        _insert_guard_counter,
        fused_qnorm_rope_kv_int8_ds_mla_insert,
    )

    block_size = 4
    k_cache = torch.zeros(3, block_size, 528, dtype=torch.uint8, device=device)
    num_slots = 3 * block_size

    torch.manual_seed(0)
    num_tokens = 4
    q = torch.randn(num_tokens, 2, 512, dtype=torch.bfloat16, device=device)
    kv = torch.randn(num_tokens, 512, dtype=torch.bfloat16, device=device)
    cos_sin = torch.randn(16, 64, dtype=torch.float32, device=device)
    # Token 1: slot far past the cache. Token 2: position far past the table.
    positions = torch.tensor([0, 3, 1 << 40, 5], dtype=torch.int64, device=device)
    slot_mapping = torch.tensor(
        [0, num_slots + (1 << 30), 2, 3], dtype=torch.int64, device=device
    )

    counter = _insert_guard_counter(device)
    before = counter.cpu().clone()
    q_out = fused_qnorm_rope_kv_int8_ds_mla_insert(
        q, kv, k_cache, slot_mapping, positions, cos_sin, 2, 1e-6, block_size
    )
    torch.cuda.synchronize()

    delta = counter.cpu() - before
    assert int(delta[0]) == 1, f"expected 1 oob-slot trip, got {int(delta[0])}"
    # The position guard runs in every (token, slot) program that reads the
    # position: padded_heads Q lanes plus the KV lane.
    assert int(delta[1]) == 3, f"expected 3 oob-position trips, got {int(delta[1])}"
    assert q_out.shape == (num_tokens, 2, 512)
    assert not q_out.isnan().any()

    # In-bounds tokens landed; the poisoned slot's write was dropped, so no
    # other row moved. Rows 0, 2, 3 written; row 1's target does not exist.
    rows_i8, scales = get_int8_ds_mla_cache_views(k_cache, block_size)
    written = {0, 2, 3}
    for slot in range(num_slots):
        b, p = divmod(slot, block_size)
        row_used = bool(rows_i8[b, p].any() or scales[b, p] != 0)
        assert row_used == (slot in written), f"slot {slot}: used={row_used}"
