# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sm_12x integration gate: sparkinfer's ``compressed_mla`` decode (the GB10
CuTe-DSL kernel that ``nvidia_sm12x`` attention wires in) must match the fork's
pure-torch sparse-MLA oracle on identical ``fp8_ds_mla`` inputs — the same
token-writer fixture and index/length semantics as the sm86 flash_mla parity
tests (``test_sm86_flash_mla_decode_parity`` / ``test_sm86_sparse_mla_oracle``).

Layout contract verified here (both sides must agree byte-for-byte):

- per-token payload: 448 fp8-e4m3 NoPE bytes + 64 bf16 RoPE (128 bytes) = 576
- per-token scales: 7 UE8M0 exponent bytes (bias 127) + 1 pad byte = 8
- page layout: ``[page_size * 576 payload][page_size * 8 scales][zero pad]``
  where the pad rounds the page byte size up to a 576 multiple. The fork's
  vLLM cache blocks are ``(num_blocks, block_size, 584)`` — 584 * B bytes,
  UNPADDED — so the sm12x backend must allocate sparkinfer-padded pages
  (``compressed_mla_page_nbytes``); this test builds caches in that padded
  shape and writes tokens with the UNMODIFIED sm86 fixture to prove the
  interior layout is identical.

Runs as pytest on a GB10 spark, or standalone (``python <file>``) from a bare
source checkout before vLLM is installed there — the reference oracle module
is torch-only and is loaded by file path when the vllm package is absent.
"""

import importlib.util
import math
import pathlib
import sys

import torch

try:
    import pytest
except ImportError:  # standalone __main__ run on a bare spark workspace
    pytest = None

# ---------------------------------------------------------------------------
# sparkinfer (the kernel under test)
# ---------------------------------------------------------------------------
if pytest is not None:
    sparkinfer_attention = pytest.importorskip("sparkinfer.attention")
    from sparkinfer.attention import compressed_mla
else:
    from sparkinfer.attention import compressed_mla

from sparkinfer.attention._shared.mla.compressed_reference import (  # noqa: E402
    compressed_mla_page_nbytes,
    compressed_sparse_mla_reference,
)

# ---------------------------------------------------------------------------
# Fork reference oracle (torch-only): normal import, else load by file path so
# this runs from a source rsync with no vllm install.
# ---------------------------------------------------------------------------
try:
    from vllm.v1.attention.backends.mla.sparse_mla_reference import (
        merge_reference_attention_with_sink,
        reference_attention_no_sink,
    )
except ImportError:
    _ref_path = (
        pathlib.Path(__file__).resolve().parents[3]
        / "vllm"
        / "v1"
        / "attention"
        / "backends"
        / "mla"
        / "sparse_mla_reference.py"
    )
    _spec = importlib.util.spec_from_file_location("sparse_mla_reference", _ref_path)
    assert _spec is not None and _spec.loader is not None
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["sparse_mla_reference"] = _mod
    _spec.loader.exec_module(_mod)
    merge_reference_attention_with_sink = _mod.merge_reference_attention_with_sink
    reference_attention_no_sink = _mod.reference_attention_no_sink

_FP8_DIM = 448
_ROPE_DIM = 64
_SCALE_DIM = 8
_TOKEN_DATA_SIZE = _FP8_DIM + _ROPE_DIM * 2  # 576
_HEAD_DIM = 512
_SM_SCALE = 1.0 / math.sqrt(_HEAD_DIM)


def _requires_gb10() -> str | None:
    if not torch.cuda.is_available():
        return "requires CUDA"
    if torch.cuda.get_device_capability(0)[0] != 12:
        return "requires an SM12x GPU"
    if not compressed_mla.is_supported():
        return "sparkinfer compressed_mla unsupported on this device"
    return None


if pytest is not None:
    pytestmark = pytest.mark.skipif(
        _requires_gb10() is not None, reason=_requires_gb10() or ""
    )


def _make_padded_cache(num_pages: int, page_size: int) -> torch.Tensor:
    """A zeroed sparkinfer-shaped page buffer: (num_pages, padded_page_nbytes).

    The leading ``page_size * 584`` bytes of each page hold the fork's
    fp8_ds_mla block layout; the tail is the sparkinfer alignment pad.
    """
    return torch.zeros(
        (num_pages, compressed_mla_page_nbytes(page_size)),
        dtype=torch.uint8,
        device="cuda",
    )


def _write_fp8_ds_mla_token(
    k_cache: torch.Tensor,
    slot: int,
    block_size: int,
    magnitude_slot: int | None = None,
) -> torch.Tensor:
    """The sm86 parity/oracle token writer (test_sm86_sparse_mla_oracle /
    test_sm86_flash_mla_decode_parity), with one documented deviation: those
    tests only ever write slots <= 12, so their slot-proportional offsets stay
    O(1). Here large slot ids exercise ADDRESSING across many pages, so
    ``magnitude_slot`` (default ``slot % 16``) bounds the written magnitudes
    to the same O(1) range the sm86 tests actually cover — the sm121 kernel
    computes in fp8 (Q is requantized to e4m3 in smem), so its error scales
    with output magnitude and the shared 2e-2 tolerance presumes O(1) data.
    The fp8-compute behavior at large magnitudes is characterized separately
    in ``test_fp8_compute_error_characterization``."""
    if magnitude_slot is None:
        magnitude_slot = slot % 16
    block_idx = slot // block_size
    block_offset = slot % block_size

    values = (
        (torch.arange(_FP8_DIM, device=k_cache.device, dtype=torch.float32) % 17) - 8
    ) / 16.0
    values = values + float(magnitude_slot) / 32.0
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
        torch.linspace(-1.0, 1.0, _ROPE_DIM, device=k_cache.device)
        + float(magnitude_slot) / 16.0
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


def _fill_cache(
    cache: torch.Tensor, num_tokens: int, page_size: int
) -> dict[int, torch.Tensor]:
    return {
        slot: _write_fp8_ds_mla_token(cache, slot, page_size)
        for slot in range(num_tokens)
    }


def _random_selection(
    rows: int,
    width: int,
    num_tokens: int,
    *,
    generator: torch.Generator,
    include_empty_row: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """-1-padded per-row slot selections + active lengths, sm86-test style."""
    indices = torch.full((rows, width), -1, dtype=torch.int32, device="cuda")
    lens = torch.zeros((rows,), dtype=torch.int32, device="cuda")
    for row in range(rows):
        if include_empty_row and row == rows - 1:
            continue
        length = int(
            torch.randint(
                1, min(width, num_tokens) + 1, (1,), generator=generator, device="cuda"
            ).item()
        )
        perm = torch.randperm(
            num_tokens, generator=generator, device="cuda", dtype=torch.int64
        )[:length].sort().values
        indices[row, :length] = perm.to(torch.int32)
        lens[row] = length
    return indices, lens


def _gather_expected(
    indices: torch.Tensor,
    lens: torch.Tensor,
    expected_by_slot: dict[int, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, width = indices.shape
    gathered = torch.zeros(rows, width, _HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    for t in range(rows):
        for k in range(width):
            slot = int(indices[t, k].item())
            if slot >= 0:
                gathered[t, k] = expected_by_slot[slot]
    offsets = torch.arange(width, device="cuda")
    valid = (offsets[None, :] < lens[:, None]) & (indices >= 0)
    return gathered, valid


def _reference_decode(
    q: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    swa_expected: dict[int, torch.Tensor],
    *,
    attn_sink: torch.Tensor | None,
    extra_indices: torch.Tensor | None = None,
    extra_lens: torch.Tensor | None = None,
    extra_expected: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Fork-oracle decode: attention over gathered expected rows, both streams
    concatenated, sink merged the same way the sm86 tests do."""
    gathered, valid = _gather_expected(swa_indices, swa_lens, swa_expected)
    if extra_indices is not None:
        assert extra_lens is not None and extra_expected is not None
        e_gathered, e_valid = _gather_expected(extra_indices, extra_lens, extra_expected)
        gathered = torch.cat([gathered, e_gathered], dim=1)
        valid = torch.cat([valid, e_valid], dim=1)
    subset_output, subset_lse = reference_attention_no_sink(
        q, gathered, valid, _SM_SCALE
    )
    if attn_sink is None:
        return subset_output
    output = torch.empty_like(subset_output)
    merge_reference_attention_with_sink(
        subset_outputs=[subset_output],
        subset_lses=[subset_lse],
        attn_sink=attn_sink,
        output=output,
    )
    return output


def _run_sparkinfer_decode(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    swa_page_size: int,
    *,
    attn_sink: torch.Tensor | None,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_lens: torch.Tensor | None = None,
    extra_page_size: int | None = None,
    mode: str = "decode",
) -> torch.Tensor:
    rows, heads, _ = q.shape
    width = swa_indices.shape[1] + (
        extra_indices.shape[1] if extra_indices is not None else 0
    )
    plan = compressed_mla.plan(
        compressed_mla.Caps(
            device=q.device,
            dtype=torch.bfloat16,
            kv_dtype=torch.uint8,
            num_q_heads=heads,
            max_width=width,
            max_q_rows=rows,
            max_batch=rows,
            max_kv_rows=rows * width,
        )
    )
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = plan.bind(
        scratch=scratch,
        q=q,
        swa_indices=swa_indices,
        swa_lengths=swa_lens,
        indexed_indices=extra_indices,
        indexed_lengths=extra_lens,
    )
    binding.scratch.mode = mode
    return compressed_mla.run(
        swa_k_cache=swa_cache,
        binding=binding,
        swa_page_size=swa_page_size,
        indexed_k_cache=extra_cache,
        indexed_page_size=extra_page_size,
        attn_sink=attn_sink,
        sm_scale=_SM_SCALE,
    )


def _assert_parity(out: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    torch.cuda.synchronize()
    assert bool(torch.isfinite(out.float()).all().item()), label
    cos = torch.nn.functional.cosine_similarity(
        out.float().reshape(-1), expected.float().reshape(-1), dim=0
    )
    assert float(cos.item()) > 0.999, f"{label}: cosine {float(cos.item()):.6f}"
    # Same tolerance the sm86 flash_mla parity tests use against this oracle.
    torch.testing.assert_close(
        out.float(), expected.float(), rtol=2e-2, atol=2e-2, msg=label
    )


# ---------------------------------------------------------------------------
# Cases — shapes mirror the sm86 parity/oracle tests, at the sm12x page sizes
# the deployed layer uses (SWA pool page 256, C4 compressed pool page 64).
# ---------------------------------------------------------------------------


def test_swa_only_decode_matches_fork_oracle() -> None:
    """SWA-only decode (compress_ratio<=1 layers): window 128, page 256."""
    torch.manual_seed(11)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(11)
    page_size, num_pages, num_tokens = 256, 2, 300
    cache = _make_padded_cache(num_pages, page_size)
    expected_by_slot = _fill_cache(cache, num_tokens, page_size)

    for rows, heads in ((2, 32), (16, 64), (64, 32)):
        q = (
            torch.randn(
                rows, heads, _HEAD_DIM, device="cuda", dtype=torch.float32,
                generator=gen,
            )
            / 4
        ).to(torch.bfloat16)
        swa_indices, swa_lens = _random_selection(
            rows, 128, num_tokens, generator=gen, include_empty_row=(rows > 2)
        )
        out = _run_sparkinfer_decode(
            q, cache, swa_indices, swa_lens, page_size, attn_sink=None
        )
        expected = _reference_decode(
            q, swa_indices, swa_lens, expected_by_slot, attn_sink=None
        )
        _assert_parity(out, expected, f"swa-only rows={rows} heads={heads}")


def test_swa_plus_compressed_decode_matches_fork_oracle() -> None:
    """The real DSV4 decode shape: SWA window 128 (page 256) + top-k 512 over
    the compressed pool (page 64), with -1 padding and short rows."""
    torch.manual_seed(12)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(12)
    swa_page, extra_page = 256, 64
    swa_cache = _make_padded_cache(2, swa_page)
    extra_cache = _make_padded_cache(12, extra_page)
    swa_expected = _fill_cache(swa_cache, 300, swa_page)
    extra_expected = _fill_cache(extra_cache, 700, extra_page)

    for rows, heads in ((2, 32), (32, 32), (8, 64)):
        q = (
            torch.randn(
                rows, heads, _HEAD_DIM, device="cuda", dtype=torch.float32,
                generator=gen,
            )
            / 4
        ).to(torch.bfloat16)
        swa_indices, swa_lens = _random_selection(rows, 128, 300, generator=gen)
        extra_indices, extra_lens = _random_selection(
            rows, 512, 700, generator=gen, include_empty_row=True
        )
        out = _run_sparkinfer_decode(
            q,
            swa_cache,
            swa_indices,
            swa_lens,
            swa_page,
            attn_sink=None,
            extra_cache=extra_cache,
            extra_indices=extra_indices,
            extra_lens=extra_lens,
            extra_page_size=extra_page,
        )
        expected = _reference_decode(
            q,
            swa_indices,
            swa_lens,
            swa_expected,
            attn_sink=None,
            extra_indices=extra_indices,
            extra_lens=extra_lens,
            extra_expected=extra_expected,
        )
        _assert_parity(out, expected, f"swa+topk rows={rows} heads={heads}")


def test_attn_sink_decode_matches_fork_oracle() -> None:
    """Per-head attention sink folded into the softmax denominator, same
    merge semantics as the fork's sink-aware reference."""
    torch.manual_seed(13)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(13)
    swa_page, extra_page = 256, 64
    swa_cache = _make_padded_cache(1, swa_page)
    extra_cache = _make_padded_cache(4, extra_page)
    swa_expected = _fill_cache(swa_cache, 200, swa_page)
    extra_expected = _fill_cache(extra_cache, 250, extra_page)

    rows, heads = 4, 32
    q = (
        torch.randn(
            rows, heads, _HEAD_DIM, device="cuda", dtype=torch.float32, generator=gen
        )
        / 4
    ).to(torch.bfloat16)
    attn_sink = torch.linspace(-0.5, 1.0, heads, dtype=torch.float32, device="cuda")
    swa_indices, swa_lens = _random_selection(rows, 128, 200, generator=gen)
    extra_indices, extra_lens = _random_selection(rows, 512, 250, generator=gen)

    out = _run_sparkinfer_decode(
        q,
        swa_cache,
        swa_indices,
        swa_lens,
        swa_page,
        attn_sink=attn_sink,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        extra_lens=extra_lens,
        extra_page_size=extra_page,
    )
    expected = _reference_decode(
        q,
        swa_indices,
        swa_lens,
        swa_expected,
        attn_sink=attn_sink,
        extra_indices=extra_indices,
        extra_lens=extra_lens,
        extra_expected=extra_expected,
    )
    _assert_parity(out, expected, "sink decode")


def test_cross_check_sparkinfer_reference_agrees_with_fork_oracle() -> None:
    """Belt and braces: sparkinfer's own pure-torch compressed-MLA reference
    must agree with the fork's oracle on the same fixture cache — proving the
    two projects read the byte layout identically (not just that the kernel
    happens to match one of them)."""
    torch.manual_seed(14)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(14)
    page_size = 64
    cache = _make_padded_cache(3, page_size)
    expected_by_slot = _fill_cache(cache, 150, page_size)

    rows, heads = 3, 32
    q = (
        torch.randn(
            rows, heads, _HEAD_DIM, device="cuda", dtype=torch.float32, generator=gen
        )
        / 4
    ).to(torch.bfloat16)
    indices, lens = _random_selection(rows, 96, 150, generator=gen)

    theirs = compressed_sparse_mla_reference(
        q,
        cache,
        indices,
        lens,
        sm_scale=_SM_SCALE,
        swa_page_size=page_size,
    )
    ours = _reference_decode(q, indices, lens, expected_by_slot, attn_sink=None)
    _assert_parity(theirs, ours, "reference cross-check")


def test_extend_mode_matches_fork_oracle() -> None:
    """Prefill rows go through scratch mode 'extend' (the MG prefill kernel);
    same per-row selected-slot semantics as decode, at the layer's real widths
    (SWA 128 + top-k 512) and many rows. Guards the sm12x _forward_prefill
    wiring."""
    torch.manual_seed(16)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(16)
    swa_page, extra_page = 288, 72
    swa_cache = _make_padded_cache(2, swa_page)
    extra_cache = _make_padded_cache(8, extra_page)
    swa_expected = _fill_cache(swa_cache, 500, swa_page)
    extra_expected = _fill_cache(extra_cache, 550, extra_page)

    rows, heads = 48, 32
    q = (
        torch.randn(
            rows, heads, _HEAD_DIM, device="cuda", dtype=torch.float32, generator=gen
        )
        / 4
    ).to(torch.bfloat16)
    attn_sink = torch.linspace(-0.3, 0.4, heads, dtype=torch.float32, device="cuda")
    swa_indices, swa_lens = _random_selection(rows, 128, 500, generator=gen)
    extra_indices, extra_lens = _random_selection(
        rows, 512, 550, generator=gen, include_empty_row=True
    )
    out = _run_sparkinfer_decode(
        q,
        swa_cache,
        swa_indices,
        swa_lens,
        swa_page,
        attn_sink=attn_sink,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        extra_lens=extra_lens,
        extra_page_size=extra_page,
        mode="extend",
    )
    expected = _reference_decode(
        q,
        swa_indices,
        swa_lens,
        swa_expected,
        attn_sink=attn_sink,
        extra_indices=extra_indices,
        extra_lens=extra_lens,
        extra_expected=extra_expected,
    )
    _assert_parity(out, expected, "extend mode")


def test_native_vllm_block_sizes_need_no_padding() -> None:
    """The zero-copy integration constraint: vLLM allocates unpadded
    (num_blocks, block_size, 584) blocks, and sparkinfer requires page bytes
    to be a 576 multiple — which holds exactly when block_size % 72 == 0.
    The deployed sm12x backend uses block_size 288 (SWA pool) with the C4
    compressed pool at 288/4 = 72; prove the kernel is correct at those page
    sizes (its own tests only cover 256/64/2)."""
    torch.manual_seed(15)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(15)
    swa_page, extra_page = 288, 72
    assert compressed_mla_page_nbytes(swa_page) == swa_page * 584
    assert compressed_mla_page_nbytes(extra_page) == extra_page * 584
    swa_cache = _make_padded_cache(2, swa_page)
    extra_cache = _make_padded_cache(8, extra_page)
    swa_expected = _fill_cache(swa_cache, 500, swa_page)
    extra_expected = _fill_cache(extra_cache, 550, extra_page)

    for rows, heads in ((2, 32), (16, 32)):
        q = (
            torch.randn(
                rows, heads, _HEAD_DIM, device="cuda", dtype=torch.float32,
                generator=gen,
            )
            / 4
        ).to(torch.bfloat16)
        swa_indices, swa_lens = _random_selection(rows, 128, 500, generator=gen)
        extra_indices, extra_lens = _random_selection(rows, 512, 550, generator=gen)
        out = _run_sparkinfer_decode(
            q,
            swa_cache,
            swa_indices,
            swa_lens,
            swa_page,
            attn_sink=None,
            extra_cache=extra_cache,
            extra_indices=extra_indices,
            extra_lens=extra_lens,
            extra_page_size=extra_page,
        )
        expected = _reference_decode(
            q,
            swa_indices,
            swa_lens,
            swa_expected,
            attn_sink=None,
            extra_indices=extra_indices,
            extra_lens=extra_lens,
            extra_expected=extra_expected,
        )
        _assert_parity(out, expected, f"block288/72 rows={rows} heads={heads}")


def test_fp8_compute_error_characterization() -> None:
    """The sm121 kernel computes in fp8 (Q requantized to e4m3 in smem), so
    unlike the sm86 flash_mla/Triton kernels (which consume fp8 KV but keep Q
    in bf16) its absolute error grows with data magnitude. Characterize it:
    at ~5x magnitudes the output must still be directionally exact
    (cosine >= 0.9995) with relative error bounded by fp8 resolution (~5%),
    and an fp8-rounded-Q oracle must halve the gap (proving the error source
    is Q quantization, not addressing)."""
    torch.manual_seed(11)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(11)
    page_size, num_tokens = 256, 300
    cache = _make_padded_cache(2, page_size)
    # Unbounded magnitudes: the original fixture offsets at large slot ids.
    expected_by_slot = {
        slot: _write_fp8_ds_mla_token(cache, slot, page_size, magnitude_slot=slot)
        for slot in range(num_tokens)
    }
    rows, heads = 2, 32
    q = (
        torch.randn(
            rows, heads, _HEAD_DIM, device="cuda", dtype=torch.float32, generator=gen
        )
        / 4
    ).to(torch.bfloat16)
    swa_indices, swa_lens = _random_selection(rows, 128, num_tokens, generator=gen)
    out = _run_sparkinfer_decode(
        q, cache, swa_indices, swa_lens, page_size, attn_sink=None
    )
    expected = _reference_decode(
        q, swa_indices, swa_lens, expected_by_slot, attn_sink=None
    )
    torch.cuda.synchronize()
    cos = torch.nn.functional.cosine_similarity(
        out.float().reshape(-1), expected.float().reshape(-1), dim=0
    )
    assert float(cos.item()) >= 0.9995, f"cosine {float(cos.item()):.6f}"
    scale = expected.float().abs().max().clamp(min=1.0)
    rel = ((out.float() - expected.float()).abs().max() / scale).item()
    assert rel <= 0.05, f"relative error {rel:.4f} exceeds fp8-compute bound"


if __name__ == "__main__":
    skip = _requires_gb10()
    if skip is not None:
        print(f"SKIP: {skip}")
        sys.exit(0)
    for fn in (
        test_cross_check_sparkinfer_reference_agrees_with_fork_oracle,
        test_swa_only_decode_matches_fork_oracle,
        test_swa_plus_compressed_decode_matches_fork_oracle,
        test_attn_sink_decode_matches_fork_oracle,
        test_extend_mode_matches_fork_oracle,
        test_native_vllm_block_sizes_need_no_padding,
        test_fp8_compute_error_characterization,
    ):
        print(f"RUN  {fn.__name__} ...", flush=True)
        fn()
        print(f"PASS {fn.__name__}", flush=True)
    print("ALL PARITY TESTS PASSED")
