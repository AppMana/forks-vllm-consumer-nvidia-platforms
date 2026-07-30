#!/usr/bin/env python3
"""Exercise the DSV4 C4 decode-indexer boundary on GB10.

At 32,768 uncompressed tokens the compressed C4 context crosses 8,192 rows,
and vLLM expands the decode logits scratch from 8,192 to 8,320 columns.
"""

import torch

from vllm.utils import deep_gemm


HEAD_DIM = 128
HEADS = 64
TOKEN_BYTES = 132
BLOCK_SIZE = 64
MAX_MODEL_LEN = 16_384
TOPK = 512


def make_inputs(context_len: int):
    torch.manual_seed(context_len)
    device = torch.device("cuda")
    num_blocks = (MAX_MODEL_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
    q = (
        torch.randn((1, 1, HEADS, HEAD_DIM), device=device)
        .clamp(-4, 4)
        .to(torch.float8_e4m3fn)
    )
    raw = torch.zeros(
        (num_blocks, BLOCK_SIZE, TOKEN_BYTES),
        dtype=torch.uint8,
        device=device,
    )
    values = raw[..., :HEAD_DIM].view(torch.float8_e4m3fn)
    values.copy_(
        torch.randn(values.shape, device=device).clamp(-4, 4).to(values.dtype)
    )
    raw[..., HEAD_DIM : HEAD_DIM + 4].view(torch.float32).fill_(0.02)
    weights = torch.rand((1, HEADS), device=device, dtype=torch.float32)
    context_lens = torch.tensor([[context_len]], device=device, dtype=torch.int32)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32)[None]
    schedule = torch.empty(0, device=device, dtype=torch.int32)
    return q, raw.unsqueeze(-2), weights, context_lens, block_table, schedule


def exercise(context_len: int, token_count: int) -> None:
    q, cache, weights, lens, table, schedule = make_inputs(context_len)
    logits = deep_gemm.fp8_fp4_paged_mqa_logits(
        (q, None),
        cache,
        weights,
        lens,
        table,
        schedule,
        max_model_len=MAX_MODEL_LEN,
        clean_logits=False,
        token_count=token_count,
    )
    indices = torch.full((1, TOPK), -1, dtype=torch.int32, device="cuda")
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
    torch.ops._C.persistent_topk(
        logits, lens, indices, workspace, TOPK, logits.shape[1]
    )
    torch.cuda.synchronize()
    assert logits.shape == (1, token_count)
    assert torch.isfinite(logits[0, :context_len]).all()
    assert (indices >= 0).all() and (indices < context_len).all()
    print(f"PASS context_len={context_len} token_count={token_count}")


def main() -> None:
    assert torch.cuda.get_device_capability() == (12, 1)
    exercise(8_192, 8_192)
    exercise(8_211, 8_320)


if __name__ == "__main__":
    main()
