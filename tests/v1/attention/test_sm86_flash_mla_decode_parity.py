# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sm_86 integration gate: the precompiled flash_mla CUDA sparse-MLA decode kernel
(now wired into ``nvidia_sm86`` attention) must match the Triton decode primitive it
replaces, on identical fp8_ds_mla inputs. Both target the same oracle, so they must
agree; this guards the ``_forward_decode`` flash_mla dispatch.

Skipped unless run on Ampere (sm_8x) with the flash_mla kernel importable.
"""

import inspect
import math

import pytest
import torch

flash_mla = pytest.importorskip("flash_mla")
from flash_mla import sparse_mla_decode_fp8, sparse_mla_prefill  # noqa: E402

from vllm.models.deepseek_v4.nvidia_sm86.triton_kernels import (  # noqa: E402
    decode_sparse_attention_triton,
)
from vllm.models.deepseek_v4.nvidia_sm86.attention import (  # noqa: E402
    DeepseekV4SM86Attention,
)
from vllm.transformers_utils.configs.dsv4.kernel_config import (  # noqa: E402
    SPARSE_MLA_DECODE_FP8_FLASH,
    SPARSE_MLA_DECODE_FP8_TRITON,
)

_FP8_DIM = 448
_ROPE_DIM = 64
_SCALE_DIM = 8
_TOKEN_DATA_SIZE = _FP8_DIM + _ROPE_DIM * 2  # 576
_HEAD_DIM = 512


def _write_fp8_ds_mla_token(k_cache: torch.Tensor, slot: int, block_size: int) -> None:
    block_idx = slot // block_size
    block_offset = slot % block_size
    values = ((torch.arange(_FP8_DIM, device=k_cache.device, dtype=torch.float32) % 17) - 8) / 16.0
    values = values + float(slot) / 32.0
    scale_exponents = torch.tensor([-2, -1, 0, 1, 2, -2, 1], device=k_cache.device, dtype=torch.float32)
    scale_per_dim = torch.exp2(scale_exponents).repeat_interleave(64)
    fp8_values = (values / scale_per_dim).to(torch.float8_e4m3fn)
    rope = (torch.linspace(-1.0, 1.0, _ROPE_DIM, device=k_cache.device) + float(slot) / 16.0).to(torch.bfloat16)
    flat = k_cache[block_idx].view(-1)
    ds = block_offset * _TOKEN_DATA_SIZE
    ss = block_size * _TOKEN_DATA_SIZE + block_offset * _SCALE_DIM
    flat[ds : ds + _FP8_DIM] = fp8_values.view(torch.uint8)
    flat[ds + _FP8_DIM : ds + _TOKEN_DATA_SIZE] = rope.view(torch.uint8)
    enc = (scale_exponents.to(torch.int32) + 127).to(torch.uint8)
    flat[ss : ss + enc.numel()] = enc
    flat[ss + enc.numel() : ss + _SCALE_DIM] = 127


def _cos_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    return 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 8,
    reason="flash_mla sparse-MLA decode requires Ampere (sm_8x)",
)
@pytest.mark.parametrize("topk", [256, 512])
@pytest.mark.parametrize("num_tokens", [1, 4])
def test_flash_mla_decode_matches_triton(topk: int, num_tokens: int) -> None:
    torch.manual_seed(0)
    dev = "cuda"
    H, block_size = 64, 32
    scale = 1.0 / math.sqrt(_HEAD_DIM)
    num_slots = topk + 64
    nb = (num_slots + block_size - 1) // block_size
    cache = torch.zeros(nb, block_size, _TOKEN_DATA_SIZE + _SCALE_DIM, dtype=torch.uint8, device=dev)
    for slot in range(num_slots):
        _write_fp8_ds_mla_token(cache, slot, block_size)

    q = torch.randn(num_tokens, H, _HEAD_DIM, device=dev, dtype=torch.bfloat16)
    lens = torch.randint(topk - 16, topk + 1, (num_tokens,), dtype=torch.int32, device=dev)
    idx = torch.stack(
        [torch.randperm(num_slots, device=dev)[:topk].to(torch.int32) for _ in range(num_tokens)]
    )
    sink = torch.randn(H, device=dev, dtype=torch.float32) * 0.1

    flash_out = sparse_mla_decode_fp8(
        q=q, swa_cache=cache, swa_indices=idx, swa_lens=lens, scale=scale, attn_sink=sink
    )
    tri_out = torch.empty_like(q)
    decode_sparse_attention_triton(
        q=q, swa_cache=cache, swa_indices=idx, swa_lens=lens, scale=scale, attn_sink=sink, out=tri_out
    )

    cd = _cos_diff(flash_out.float(), tri_out.float())
    assert cd < 8e-5, f"flash_mla vs Triton cos_diff={cd:.2e} (topk={topk} num_tokens={num_tokens})"


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 8,
    reason="flash_mla sparse-MLA decode requires Ampere (sm_8x)",
)
def test_flash_mla_decode_matches_triton_with_extra_cache() -> None:
    torch.manual_seed(13)
    dev = "cuda"
    num_tokens, H, block_size = 1, 64, 32
    swa_topk, extra_topk = 128, 1664
    scale = 1.0 / math.sqrt(_HEAD_DIM)

    swa_slots = swa_topk + 64
    extra_slots = extra_topk + 256
    swa_cache = torch.zeros(
        (swa_slots + block_size - 1) // block_size,
        block_size,
        _TOKEN_DATA_SIZE + _SCALE_DIM,
        dtype=torch.uint8,
        device=dev,
    )
    extra_cache = torch.zeros(
        (extra_slots + block_size - 1) // block_size,
        block_size,
        _TOKEN_DATA_SIZE + _SCALE_DIM,
        dtype=torch.uint8,
        device=dev,
    )
    for slot in range(swa_slots):
        _write_fp8_ds_mla_token(swa_cache, slot, block_size)
    for slot in range(extra_slots):
        _write_fp8_ds_mla_token(extra_cache, slot, block_size)

    q = torch.randn(num_tokens, H, _HEAD_DIM, device=dev, dtype=torch.bfloat16)
    swa_lens = torch.full((num_tokens,), swa_topk, dtype=torch.int32, device=dev)
    extra_lens = torch.full((num_tokens,), extra_topk, dtype=torch.int32, device=dev)
    swa_idx = torch.stack(
        [
            torch.randperm(swa_slots, device=dev)[:swa_topk].to(torch.int32)
            for _ in range(num_tokens)
        ]
    )
    extra_idx = torch.stack(
        [
            torch.randperm(extra_slots, device=dev)[:extra_topk].to(torch.int32)
            for _ in range(num_tokens)
        ]
    )
    sink = torch.randn(H, device=dev, dtype=torch.float32) * 0.1

    flash_out = sparse_mla_decode_fp8(
        q=q,
        swa_cache=swa_cache,
        swa_indices=swa_idx,
        swa_lens=swa_lens,
        scale=scale,
        attn_sink=sink,
        extra_cache=extra_cache,
        extra_indices=extra_idx,
        extra_lens=extra_lens,
    )
    tri_out = torch.empty_like(q)
    decode_sparse_attention_triton(
        q=q,
        swa_cache=swa_cache,
        swa_indices=swa_idx,
        swa_lens=swa_lens,
        scale=scale,
        attn_sink=sink,
        out=tri_out,
        extra_cache=extra_cache,
        extra_indices=extra_idx,
        extra_lens=extra_lens,
    )

    cd = _cos_diff(flash_out.float(), tri_out.float())
    assert cd < 8e-5, f"flash_mla vs Triton extra-cache cos_diff={cd:.2e}"


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 8,
    reason="flash_mla sparse-MLA prefill requires Ampere (sm_8x)",
)
def test_flash_mla_prefill_rejects_or_matches_real_swa_metadata_shape() -> None:
    """Real vLLM SWA metadata is [T, 1, window], not the 2-D fake used above."""
    torch.manual_seed(11)
    dev = "cuda"
    T, H, topk = 4, 64, 16
    block_size = 32
    scale = 1.0 / math.sqrt(_HEAD_DIM)
    num_slots = topk + 64
    nb = (num_slots + block_size - 1) // block_size
    cache = torch.zeros(nb, block_size, _TOKEN_DATA_SIZE + _SCALE_DIM, dtype=torch.uint8, device=dev)
    for slot in range(num_slots):
        _write_fp8_ds_mla_token(cache, slot, block_size)

    q = torch.randn(T, H, _HEAD_DIM, device=dev, dtype=torch.bfloat16)
    idx_2d = torch.stack(
        [torch.randperm(num_slots, device=dev)[:topk].to(torch.int32) for _ in range(T)]
    )
    idx_real = idx_2d.unsqueeze(1)
    lens = torch.full((T,), topk, dtype=torch.int32, device=dev)
    sink = torch.randn(H, device=dev, dtype=torch.float32) * 0.1

    expected = sparse_mla_prefill(
        q=q,
        swa_cache=cache,
        swa_indices=idx_2d,
        swa_lens=lens,
        scale=scale,
        attn_sink=sink,
    )
    actual = sparse_mla_prefill(
        q=q,
        swa_cache=cache,
        swa_indices=idx_real,
        swa_lens=lens,
        scale=scale,
        attn_sink=sink,
    )

    cd = _cos_diff(actual.float(), expected.float())
    assert cd < 8e-5, f"real vLLM [T,1,W] SWA metadata shape changed FlashMLA prefill output: cos_diff={cd:.2e}"


def test_sm86_prefill_dispatches_triton_and_native() -> None:
    """The prefill role selects between the Triton staging path and the
    native flash_mla prefill via the vllm kernel config."""
    source = inspect.getsource(DeepseekV4SM86Attention._forward_prefill)
    assert "sparse_attention_triton" in source
    assert "_forward_prefill_flash" in source
    native = inspect.getsource(DeepseekV4SM86Attention._forward_prefill_flash)
    assert "sparse_mla_prefill(" in native


def test_sm86_int8_dispatch_supports_native_and_triton_fqns() -> None:
    """int8_ds_mla: the decode role selects between the Triton int8 decode and
    the native flash_mla int8 decode; the FLASH prefill symbol dispatches to
    the fused int8 prefill on int8 caches."""
    from vllm.transformers_utils.configs.dsv4.kernel_config import (
        SPARSE_MLA_DECODE_INT8_FLASH,
        SPARSE_MLA_DECODE_INT8_TRITON,
    )

    assert SPARSE_MLA_DECODE_INT8_FLASH == "flash_mla.sparse_mla_decode_int8"
    assert SPARSE_MLA_DECODE_INT8_TRITON == "flash_mla.sparse_mla_decode_int8_triton"
    source = inspect.getsource(DeepseekV4SM86Attention._forward_decode)
    assert "SPARSE_MLA_DECODE_INT8_FLASH" in source
    assert "SPARSE_MLA_DECODE_INT8_TRITON" in source
    assert "sparse_mla_decode_int8(" in source
    assert "sparse_mla_decode_int8_triton(" in source
    native = inspect.getsource(DeepseekV4SM86Attention._forward_prefill_flash)
    assert "sparse_mla_prefill_int8(" in native
    assert "get_int8_ds_mla_cache_views(" in native


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 8,
    reason="flash_mla sparse-MLA kernels require Ampere (sm_8x)",
)
def test_flash_mla_int8_native_matches_triton_on_vllm_528B_views() -> None:
    """Numeric parity on the EXACT artifacts vLLM hands the kernels: a paged
    528-byte-token int8_ds_mla byte cache read through
    ``get_int8_ds_mla_cache_views`` strided views. Native int8 decode and
    native fused int8 prefill vs the incumbent Triton int8 decode."""
    from flash_mla import (
        sparse_mla_decode_int8,
        sparse_mla_prefill_int8,
        sparse_mla_decode_int8_triton,
    )

    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        get_int8_ds_mla_cache_views,
    )

    torch.manual_seed(29)
    dev = "cuda"
    T, H, block_size = 4, 64, 64
    swa_topk, extra_topk = 128, 384
    scale = 1.0 / math.sqrt(_HEAD_DIM)
    token_bytes = 528  # 512 int8 + 4B fp32 scale + 12B pad

    def build(num_slots: int) -> tuple[torch.Tensor, torch.Tensor]:
        nb = (num_slots + block_size - 1) // block_size
        rows = (torch.randn(nb * block_size, _HEAD_DIM, device=dev) * 2.0).to(
            torch.bfloat16
        )
        s = (rows.float().abs().amax(dim=-1) / 127.0).clamp_min(1e-12)
        q_i8 = (
            torch.round(rows.float() / s.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
        )
        cache = torch.zeros(
            nb, block_size, token_bytes, dtype=torch.uint8, device=dev
        )
        cache[:, :, :512] = q_i8.view(nb, block_size, 512).view(torch.uint8)
        cache[:, :, 512:516] = (
            s.to(torch.float32).view(nb, block_size, 1).view(torch.uint8)
        )
        # the vLLM paged cache arrives flattened per block; the oracle's K is
        # the dequantized rows ROUNDED TO BF16 (the kernels stage K as bf16),
        # matching the flash_mla fork's own oracle convention.
        dequant = (q_i8.float() * s.unsqueeze(-1)).to(torch.bfloat16).float()
        return cache.view(nb, block_size * token_bytes), dequant

    swa_cache, swa_K = build(swa_topk + 32)
    extra_cache, extra_K = build(extra_topk + 32)
    swa_rows, swa_scales = get_int8_ds_mla_cache_views(swa_cache, block_size)
    extra_rows, extra_scales = get_int8_ds_mla_cache_views(extra_cache, block_size)
    assert swa_rows.stride(1) == token_bytes  # strided, not contiguous

    q = torch.randn(T, H, _HEAD_DIM, device=dev, dtype=torch.bfloat16)
    swa_lens = torch.randint(1, swa_topk + 1, (T,), dtype=torch.int32, device=dev)
    extra_lens = torch.randint(1, extra_topk + 1, (T,), dtype=torch.int32, device=dev)
    swa_idx = torch.randint(
        0, swa_rows.shape[0] * block_size, (T, swa_topk), dtype=torch.int32, device=dev
    )
    extra_idx = torch.randint(
        0,
        extra_rows.shape[0] * block_size,
        (T, extra_topk),
        dtype=torch.int32,
        device=dev,
    )
    sink = torch.randn(H, device=dev, dtype=torch.float32) * 0.1

    # fp32 oracle over the dequantized rows (concatenated swa + extra streams).
    ref = torch.zeros(T, H, _HEAD_DIM, device=dev, dtype=torch.float32)
    for t in range(T):
        K = torch.cat(
            [
                swa_K[swa_idx[t, : swa_lens[t]].long()],
                extra_K[extra_idx[t, : extra_lens[t]].long()],
            ]
        )
        scores = (q[t].float() @ K.t()) * scale
        m = torch.maximum(
            scores.max(dim=-1, keepdim=True).values, sink[:, None].float()
        )
        p = torch.exp(scores - m)
        ref[t] = (p @ K) / (
            p.sum(-1, keepdim=True) + torch.exp(sink[:, None].float() - m)
        )

    kwargs = dict(
        scale=scale,
        attn_sink=sink,
        extra_cache=extra_rows,
        extra_scale=extra_scales,
        extra_indices=extra_idx,
        extra_lens=extra_lens,
    )
    tri = sparse_mla_decode_int8_triton(
        q, swa_rows, swa_scales, swa_idx, swa_lens, **kwargs
    )
    native_decode = sparse_mla_decode_int8(
        q, swa_rows, swa_scales, swa_idx, swa_lens, **kwargs
    )
    native_prefill = sparse_mla_prefill_int8(
        q, swa_rows, swa_scales, swa_idx, swa_lens, **kwargs
    )

    for name, out in (
        ("triton decode", tri),
        ("native decode", native_decode),
        ("native prefill", native_prefill),
    ):
        cd = _cos_diff(out.float(), ref)
        assert cd < 8e-5, f"int8 {name} vs fp32 oracle on 528B views cos_diff={cd:.2e}"
    # Native kernels hold the tight elementwise bound vs the oracle (the
    # Triton decode additionally quantizes Q to int8, so it is exempt here
    # and covered by its cos bound above).
    torch.testing.assert_close(native_decode.float(), ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(native_prefill.float(), ref, rtol=2e-2, atol=2e-2)


def test_sm86_fp8_decode_dispatch_supports_native_and_triton_fqns() -> None:
    assert SPARSE_MLA_DECODE_FP8_FLASH == (
        "flash_mla.sparse_mla_decode_fp8"
    )
    assert SPARSE_MLA_DECODE_FP8_TRITON == (
        "vllm.models.deepseek_v4.nvidia_sm86.triton_kernels."
        "decode_sparse_attention_triton"
    )
    source = inspect.getsource(DeepseekV4SM86Attention._forward_decode)
    assert "SPARSE_MLA_DECODE_FP8_FLASH" in source
    assert "SPARSE_MLA_DECODE_FP8_TRITON" in source
    assert "sparse_mla_decode_fp8(" in source
    assert "decode_sparse_attention_triton(" in source


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 8,
    reason="flash_mla sparse-MLA prefill requires Ampere (sm_8x)",
)
def test_flash_mla_prefill_matches_triton_reference_with_extra_cache() -> None:
    """Numeric parity: native flash_mla prefill vs the Triton reference on
    identical synthetic fp8_ds_mla inputs, including short rows (-1 padded
    SWA windows) and a second compressed cache stream."""
    torch.manual_seed(7)
    dev = "cuda"
    T, H, block_size = 6, 64, 32
    swa_topk, extra_topk = 128, 512
    scale = 1.0 / math.sqrt(_HEAD_DIM)

    swa_slots = swa_topk + 64
    extra_slots = extra_topk + 128
    swa_cache = torch.zeros(
        (swa_slots + block_size - 1) // block_size,
        block_size,
        _TOKEN_DATA_SIZE + _SCALE_DIM,
        dtype=torch.uint8,
        device=dev,
    )
    extra_cache = torch.zeros(
        (extra_slots + block_size - 1) // block_size,
        block_size,
        _TOKEN_DATA_SIZE + _SCALE_DIM,
        dtype=torch.uint8,
        device=dev,
    )
    for slot in range(swa_slots):
        _write_fp8_ds_mla_token(swa_cache, slot, block_size)
    for slot in range(extra_slots):
        _write_fp8_ds_mla_token(extra_cache, slot, block_size)

    q = torch.randn(T, H, _HEAD_DIM, device=dev, dtype=torch.bfloat16)
    # Short rows: valid entries compact-left, -1 padding after (the layout
    # produced by build_flashinfer_mixed_sparse_indices for prefill tokens).
    swa_lens = torch.randint(
        swa_topk // 2, swa_topk + 1, (T,), dtype=torch.int32, device=dev
    )
    extra_lens = torch.randint(
        extra_topk // 2, extra_topk + 1, (T,), dtype=torch.int32, device=dev
    )
    swa_idx = torch.full((T, swa_topk), -1, dtype=torch.int32, device=dev)
    extra_idx = torch.full((T, extra_topk), -1, dtype=torch.int32, device=dev)
    for t in range(T):
        swa_idx[t, : swa_lens[t]] = torch.randperm(swa_slots, device=dev)[
            : swa_lens[t]
        ].to(torch.int32)
        extra_idx[t, : extra_lens[t]] = torch.randperm(extra_slots, device=dev)[
            : extra_lens[t]
        ].to(torch.int32)
    sink = torch.randn(H, device=dev, dtype=torch.float32) * 0.1

    flash_out = sparse_mla_prefill(
        q=q,
        swa_cache=swa_cache,
        swa_indices=swa_idx,
        swa_lens=swa_lens,
        scale=scale,
        attn_sink=sink,
        extra_cache=extra_cache,
        extra_indices=extra_idx,
        extra_lens=extra_lens,
    )
    # Triton reference: the decode kernel computes the same absorbed sparse
    # attention over the same (swa, extra) global-slot selection.
    tri_out = torch.empty_like(q)
    decode_sparse_attention_triton(
        q=q,
        swa_cache=swa_cache,
        swa_indices=swa_idx,
        swa_lens=swa_lens,
        scale=scale,
        attn_sink=sink,
        out=tri_out,
        extra_cache=extra_cache,
        extra_indices=extra_idx,
        extra_lens=extra_lens,
    )

    torch.testing.assert_close(
        flash_out.float(), tri_out.float(), rtol=2e-2, atol=2e-2
    )
    cd = _cos_diff(flash_out.float(), tri_out.float())
    assert cd < 8e-5, f"flash_mla prefill vs Triton cos_diff={cd:.2e}"
