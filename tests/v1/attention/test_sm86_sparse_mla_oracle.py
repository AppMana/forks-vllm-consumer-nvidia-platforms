# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sm_86 correctness gate: the fused Triton sparse-MLA decode kernel that
``nvidia_sm86`` calls, against the pure-torch reference oracle.

Imports the kernel directly rather than through ``deepseek_v4/attention.py``,
so a decode-kernel regression is attributable without standing up a model.
"""

import pytest
import torch

from vllm.v1.attention.backends.mla.sparse_mla_reference import (
    reference_attention_no_sink,
)

_FP8_DIM = 448
_ROPE_DIM = 64
_SCALE_DIM = 8
_TOKEN_DATA_SIZE = _FP8_DIM + _ROPE_DIM * 2


def _write_fp8_ds_mla_token(
    k_cache: torch.Tensor,
    slot: int,
    block_size: int,
) -> torch.Tensor:
    block_idx = slot // block_size
    block_offset = slot % block_size

    values = (
        (torch.arange(_FP8_DIM, device=k_cache.device, dtype=torch.float32) % 17) - 8
    ) / 16.0
    values = values + float(slot) / 32.0
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
        torch.linspace(-1.0, 1.0, _ROPE_DIM, device=k_cache.device) + float(slot) / 16.0
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_decode_sparse_attention_triton_fused_kernel_matches_reference() -> None:
    """The fused fp8 decode attention kernel (what nvidia_sm86 actually calls)
    vs the pure-torch oracle. Swa-only (no compressed extra cache)."""
    from vllm.models.deepseek_v4.nvidia_sm86.triton_kernels import (
        decode_sparse_attention_triton,
    )

    torch.manual_seed(11)
    block_size = 4
    num_blocks = 3
    num_heads = 3
    k_cache = torch.zeros(
        num_blocks,
        block_size,
        _TOKEN_DATA_SIZE + _SCALE_DIM,
        dtype=torch.uint8,
        device="cuda",
    )
    expected_by_slot = {
        slot: _write_fp8_ds_mla_token(k_cache, slot, block_size)
        for slot in (0, 1, 3, 4, 7, 8)
    }
    swa_indices = torch.tensor(
        [[0, 3, 8, 1, -1], [7, 4, 0, 8, 3]],
        dtype=torch.int32,
        device="cuda",
    )
    swa_lens = torch.tensor([4, 5], dtype=torch.int32, device="cuda")
    q = torch.randn(2, num_heads, 512, device="cuda", dtype=torch.bfloat16)
    scale = 0.0625
    out = torch.empty(2, num_heads, 512, device="cuda", dtype=torch.bfloat16)

    decode_sparse_attention_triton(
        q=q,
        swa_cache=k_cache,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        scale=scale,
        attn_sink=None,
        out=out,
    )

    topk = swa_indices.shape[1]
    gathered = torch.zeros(2, topk, 512, device="cuda", dtype=torch.bfloat16)
    for t in range(2):
        for k in range(topk):
            slot = int(swa_indices[t, k].item())
            if slot >= 0:
                gathered[t, k] = expected_by_slot[slot]
    offsets = torch.arange(topk, device="cuda")
    valid = (offsets[None, :] < swa_lens[:, None]) & (swa_indices >= 0)
    expected_output, _ = reference_attention_no_sink(
        q.unsqueeze(1), gathered, valid, scale
    )

    torch.testing.assert_close(
        out.float(), expected_output.float(), rtol=2e-2, atol=2e-2
    )
