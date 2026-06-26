# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sm_86 integration gate: the precompiled flash_mla CUDA sparse-MLA decode kernel
(now wired into ``nvidia_sm86`` attention) must match the Triton decode primitive it
replaces, on identical fp8_ds_mla inputs. Both target the same oracle, so they must
agree; this guards the ``_forward_decode`` flash_mla dispatch.

Skipped unless run on Ampere (sm_8x) with the flash_mla kernel importable.
"""

import math

import pytest
import torch

flash_mla = pytest.importorskip("flash_mla")
from flash_mla import flash_sparse_mla_decode  # noqa: E402

from vllm.models.deepseek_v4.nvidia_sm86.triton_kernels import (  # noqa: E402
    decode_sparse_attention_triton,
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

    flash_out = flash_sparse_mla_decode(
        q=q, swa_cache=cache, swa_indices=idx, swa_lens=lens, scale=scale, attn_sink=sink
    )
    tri_out = torch.empty_like(q)
    decode_sparse_attention_triton(
        q=q, swa_cache=cache, swa_indices=idx, swa_lens=lens, scale=scale, attn_sink=sink, out=tri_out
    )

    cd = _cos_diff(flash_out.float(), tri_out.float())
    assert cd < 8e-5, f"flash_mla vs Triton cos_diff={cd:.2e} (topk={topk} num_tokens={num_tokens})"
