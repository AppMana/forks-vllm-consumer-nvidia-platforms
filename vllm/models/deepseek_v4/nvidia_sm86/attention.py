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
- prefill: ``sparse_attention_triton`` over the gathered bf16 KV.

We deliberately do NOT use the native ``ampere_flashmla`` decode tail: that
kernel is sized for sm_80 (A100, 164 KB smem) and overflows sm_86's 100 KB
per-SM shared-memory cap (see pzhao-eng/FlashMLA#9). The Triton path is the
smem-frugal route that fits sm_86. ``_o_proj`` is inherited from the FlashMLA
layer; on sm_86 its fp8 einsum / inv-rope ops fall back to the torch software
casts (WS6), so no override is needed. INT8 FP8 tensor cores are absent on
Ampere; the Triton kernels upcast FP8 inputs to bf16 internally.
"""

import torch

# HARD dependency: the Ampere sm_86 sparse-MLA decode runs the precompiled flash_mla
# CUDA kernel. No try/except, no env gate, no Triton fallback — if the kernel is not
# present the import fails loudly at startup (we never want a silent degrade to the
# slower per-row Triton path).
from flash_mla import flash_sparse_mla_decode

from vllm.models.deepseek_v4.common.ops.cache_utils import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
)
from vllm.models.deepseek_v4.nvidia.flashmla import DeepseekV4FlashMLAAttention
from vllm.models.deepseek_v4.nvidia_sm86.triton_kernels import (
    sparse_attention_triton,
)
from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLAMetadata
from vllm.v1.worker.workspace import current_workspace_manager


class DeepseekV4TritonSM86Attention(DeepseekV4FlashMLAAttention):
    """DeepSeek-V4 sparse-MLA on Ampere via portable Triton kernels."""

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

        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            top_k = topk_indices.shape[-1]
        else:
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
        chunk_plan = swa_metadata.get_prefill_chunk_plan(
            compress_ratio=self.compress_ratio,
            prefill_chunk_size=self.PREFILL_CHUNK_SIZE,
        )
        assert chunk_plan, "prefill chunk plan must be non-empty when num_prefills > 0"
        workspace_manager = current_workspace_manager()
        for chunk_start, chunk_end, chunk_N, chunk_M in chunk_plan:
            chunk_size = chunk_end - chunk_start
            kv = workspace_manager.get_simultaneous(
                ((chunk_size, chunk_M, q.shape[-1]), torch.bfloat16),
            )[0]
            if not swa_only:
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )

            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=chunk_N,
            )

            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                chunk_M,
                chunk_N,
            )
            sparse_attention_triton(
                q=q[query_start:query_end],
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices.unsqueeze(1),
                lengths=combined_lens,
                scale=self.scale,
                attn_sink=self.attn_sink,
                out=output[query_start:query_end],
            )
