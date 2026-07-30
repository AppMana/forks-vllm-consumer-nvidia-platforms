"""Reproduce the live DSV4 paged-INT8 address overflow on GB10."""

import pytest
import torch

from vllm.models.deepseek_v4.nvidia_imma import triton_kernels

HEAD_DIM = 128
HEADS = 64
TOKEN_BYTES = HEAD_DIM + 4
BLOCK_SIZE = 64
KV_NUM_BLOCKS = 6_650
KV_PAGE_STRIDE = 438_784
LIVE_MAX_PAGE = 5_712
MAX_MODEL_LEN = 16_384
CONTEXT_LEN = 10_769

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1),
    reason="SM121 CUDA GPU required",
)


def make_inputs():
    device = torch.device("cuda")
    q = (
        torch.arange(HEADS * HEAD_DIM, dtype=torch.int32, device=device)
        .remainder_(31)
        .sub_(15)
        .to(torch.int8)
        .reshape(1, 1, HEADS, HEAD_DIM)
    )
    raw = torch.empty_strided(
        (KV_NUM_BLOCKS, BLOCK_SIZE, 1, TOKEN_BYTES),
        (KV_PAGE_STRIDE, TOKEN_BYTES, TOKEN_BYTES, 1),
        dtype=torch.uint8,
        device=device,
    )
    page = raw[LIVE_MAX_PAGE, :, 0]
    page[:, :HEAD_DIM].copy_(
        torch.arange(BLOCK_SIZE * HEAD_DIM, dtype=torch.int32, device=device)
        .remainder_(29)
        .sub_(14)
        .to(torch.int8)
        .view(torch.uint8)
        .reshape(BLOCK_SIZE, HEAD_DIM)
    )
    page[:, HEAD_DIM:].view(torch.float32).fill_(0.02)
    weights = torch.linspace(
        0.25, 1.25, HEADS, dtype=torch.float32, device=device
    ).reshape(1, HEADS)
    context_lens = torch.tensor([[CONTEXT_LEN]], dtype=torch.int32, device=device)
    block_table = torch.full(
        (1, MAX_MODEL_LEN // BLOCK_SIZE),
        LIVE_MAX_PAGE,
        dtype=torch.int32,
        device=device,
    )
    return q, raw, weights, context_lens, block_table


def test_int8_paged_logits_widens_physical_page_addressing(monkeypatch) -> None:
    """The exact live page/stride contract remains addressable above 2**31."""
    assert LIVE_MAX_PAGE * KV_PAGE_STRIDE > 2**31 - 1
    q, cache, weights, lens, table = make_inputs()
    monkeypatch.setattr(triton_kernels, "indexer_cache_is_int8", lambda: True)
    logits = triton_kernels.fp8_paged_mqa_logits_triton(
        q,
        cache,
        weights,
        lens,
        table,
        max_model_len=MAX_MODEL_LEN,
        token_count=MAX_MODEL_LEN,
    )
    torch.cuda.synchronize()
    assert logits.shape == (1, MAX_MODEL_LEN)
    assert torch.isfinite(logits[0, :CONTEXT_LEN]).all()
    assert torch.isneginf(logits[0, CONTEXT_LEN:]).all()
