# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ampere (sm_86) DeepSeek-V4 sparse-MLA attention.

Subclasses ``DeepseekV4FlashMLAAttention`` to reuse all projection / metadata /
indexer / o_proj machinery, and overrides only the two backend-specific
sparse-attention kernel calls:

- decode: the precompiled ``flash_mla.flash_sparse_mla_decode`` CUDA kernel — one
  launch for the whole decode batch, ~4.4x faster than the per-row Triton path it
  replaced, and a ``.so`` (no Triton JIT / recompile-wedge / warmup). It is a HARD
  dependency (imported at module top, no fallback): a missing kernel fails loudly
  at startup rather than silently degrading. Matches the Triton reference to ~1e-6
  (``test_sm86_flash_mla_decode_parity``).
- prefill: the precompiled ``flash_mla.flash_sparse_mla_prefill`` CUDA kernel
  over paged fp8_ds_mla KV caches and sparse slot ids.

``_o_proj`` is inherited from the FlashMLA layer; on sm_86 its fp8 einsum /
inv-rope ops fall back to the torch software casts (WS6), so no override is
needed. INT8 FP8 tensor cores are absent on Ampere; indexer kernels upcast FP8
inputs to bf16 internally unless the checkpoint enables the INT8 IMMA indexer.
"""

import torch

# HARD dependency: the Ampere sm_86 sparse-MLA decode runs the precompiled flash_mla
# CUDA kernel. No try/except, no env gate, no Triton fallback — if the kernel is not
# present the import fails loudly at startup (we never want a silent degrade to the
# slower per-row Triton path).
from flash_mla import flash_sparse_mla_decode, flash_sparse_mla_prefill

from vllm.models.deepseek_v4.common.ops.cache_utils import (
    compute_global_topk_indices_and_lens,
)
from vllm.models.deepseek_v4.nvidia.flashmla import DeepseekV4FlashMLAAttention
from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLAMetadata


class DeepseekV4TritonSM86Attention(DeepseekV4FlashMLAAttention):
    """DeepSeek-V4 sparse-MLA on Ampere via the sm86 FlashMLA CUDA kernels."""

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        # The Triton sparse-MLA kernels tile heads with cdiv(num_heads, 8) and
        # support any head count, so no padding to {64, 128} is needed (unlike
        # the FlashMLA FP8 decode kernel).
        return num_heads

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        # ----- prep (identical to the FlashMLA layer) -----
        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    self.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices[:num_decode_tokens]
        swa_lens = swa_metadata.decode_swa_lens[:num_decode_tokens]
        swa_k_cache = self.swa_cache_layer.kv_cache

        # q arrives padded to self.padded_heads as (num_decode_tokens, H, D);
        # the FP8 caches are consumed directly (dequantized inside the kernel).
        q_rows = q[:, 0] if q.dim() == 4 else q

        # Precompiled Ampere CUDA sparse-MLA decode: ONE launch for all decode tokens
        # (the old Triton path looped per row), ~4.4x faster, no Triton JIT/recompile.
        # Matches the Triton reference to ~1e-6 (test_sm86_flash_mla_decode_parity).
        extra_idx = None
        if topk_indices is not None:
            extra_idx = topk_indices.reshape(num_decode_tokens, -1)
        out = flash_sparse_mla_decode(
            q=q_rows,
            swa_cache=swa_k_cache,
            swa_indices=swa_indices,
            swa_lens=swa_lens,
            scale=self.scale,
            attn_sink=self.attn_sink,
            extra_cache=None if swa_only else kv_cache,
            extra_indices=extra_idx,
            extra_lens=None if topk_lens is None else topk_lens,
        )
        output[:num_decode_tokens].copy_(out)
        if output.shape[1] > self.n_local_heads:
            output[:, self.n_local_heads :].zero_()

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata,
    ) -> None:
        swa_only = attn_metadata is None

        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        assert query_start_loc_cpu is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
        else:
            topk_indices = None

        extra_sparse_indices = None
        extra_sparse_lens = None
        if not swa_only:
            assert attn_metadata is not None
            if compressed_k_cache is None:
                raise RuntimeError(
                    "Compressed sparse MLA prefill requires compressed KV cache."
                )
            block_size = attn_metadata.block_size // self.compress_ratio
            prefill_token_slice = slice(
                num_decode_tokens, num_decode_tokens + num_prefill_tokens
            )
            assert topk_indices is not None
            extra_sparse_indices, extra_sparse_lens = (
                compute_global_topk_indices_and_lens(
                    topk_indices,
                    swa_metadata.token_to_req_indices[prefill_token_slice],
                    attn_metadata.block_table,
                    block_size,
                    swa_metadata.is_valid_token[prefill_token_slice],
                )
            )

        assert swa_metadata.prefill_swa_indices is not None
        assert swa_metadata.prefill_swa_lens is not None

        num_chunks = (
            swa_metadata.num_prefills + self.PREFILL_CHUNK_SIZE - 1
        ) // self.PREFILL_CHUNK_SIZE
        assert num_chunks > 0, "prefill chunk plan must be non-empty"
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.PREFILL_CHUNK_SIZE
            chunk_end = min(
                chunk_start + self.PREFILL_CHUNK_SIZE, swa_metadata.num_prefills
            )
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            out = flash_sparse_mla_prefill(
                q=q[query_start:query_end],
                swa_cache=swa_k_cache,
                swa_indices=swa_metadata.prefill_swa_indices[query_start:query_end],
                swa_lens=swa_metadata.prefill_swa_lens[query_start:query_end],
                scale=self.scale,
                attn_sink=self.attn_sink,
                extra_cache=None if swa_only else compressed_k_cache,
                extra_indices=(
                    None
                    if extra_sparse_indices is None
                    else extra_sparse_indices[query_start:query_end]
                ),
                extra_lens=(
                    None
                    if extra_sparse_lens is None
                    else extra_sparse_lens[query_start:query_end]
                ),
            )
            output[query_start:query_end].copy_(out)
            if output.shape[1] > self.n_local_heads:
                output[query_start:query_end, self.n_local_heads :].zero_()
