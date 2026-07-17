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

import os

import torch

# HARD dependencies: Ampere sm_86 sparse-MLA decode/prefill run through
# explicit flash_mla symbols. No try/except, no env gate, no silent fallback.
from flash_mla import (
    flash_sparse_mla_decode,
    flash_sparse_mla_prefill,
    sparse_int8_mla_decode,
    sparse_int8_mla_prefill,
    triton_sparse_int8_mla_decode,
)

from vllm.logger import init_logger
from vllm.models.deepseek_v4.common.ops.cache_utils import (
    build_flashinfer_mixed_sparse_indices,
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
    get_int8_ds_mla_cache_views,
)
from vllm.models.deepseek_v4.nvidia.flashmla import DeepseekV4FlashMLAAttention
from vllm.models.deepseek_v4.nvidia_sm86.triton_kernels import (
    decode_sparse_attention_triton,
    sparse_attention_triton,
)
from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLAMetadata
from vllm.transformers_utils.configs.deepseek_v4_appmana import (
    ROLE_SPARSE_MLA_DECODE_FP8,
    ROLE_SPARSE_MLA_DECODE_INT8,
    ROLE_SPARSE_MLA_PREFILL,
    SPARSE_MLA_DECODE_FP8_FLASH,
    SPARSE_MLA_DECODE_FP8_TRITON,
    SPARSE_MLA_DECODE_INT8_FLASH,
    SPARSE_MLA_DECODE_INT8_TRITON,
    SPARSE_MLA_PREFILL_FLASH,
    ResolvedAppmanaKernelConfig,
    resolve_appmana_kernel_config_from_hf_config,
    resolved_proof_line,
)
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)


def validate_sm86_kernel_selection(
    resolved: ResolvedAppmanaKernelConfig, *, kv_cache_dtype: str
) -> None:
    """SM86-specific cross-checks between kernel selection and cache dtype."""
    if kv_cache_dtype not in ("fp8_ds_mla", "int8_ds_mla"):
        raise ValueError(
            "DeepSeek V4 SM86 sparse MLA requires fp8_ds_mla or "
            f"int8_ds_mla KV cache, got {kv_cache_dtype}"
        )
    # The fused native prefill consumes both fp8_ds_mla and int8_ds_mla paged
    # caches (flash_mla 93bbf4e: whole-cache int8 dequant pass, runtime row
    # stride); on int8 caches _forward_prefill_flash dispatches to the int8
    # variant of the same kernel, so no cross-check is needed here.


class DeepseekV4TritonSM86Attention(DeepseekV4FlashMLAAttention):
    """DeepSeek-V4 sparse-MLA on Ampere via portable Triton kernels."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Unified kernel selection: the "appmana" checkpoint config block
        # (with the deprecated role-keyed HF override strings as fallback).
        # Resolution fails closed on unknown symbols / duplicate roles.
        resolved = resolve_appmana_kernel_config_from_hf_config(self.config)
        validate_sm86_kernel_selection(resolved, kv_cache_dtype=self.kv_cache_dtype)
        self.fp8_decode_symbol = resolved.roles[ROLE_SPARSE_MLA_DECODE_FP8]
        self.int8_decode_symbol = resolved.roles[ROLE_SPARSE_MLA_DECODE_INT8]
        self.prefill_symbol = resolved.roles[ROLE_SPARSE_MLA_PREFILL]
        if self.kv_cache_dtype == "int8_ds_mla":
            decode_symbol = self.int8_decode_symbol
        else:
            decode_symbol = self.fp8_decode_symbol
        logger.info_once(
            "DeepSeek V4 SM86 sparse MLA kernels: kv_cache_dtype=%s, "
            "decode=%s, prefill=%s",
            self.kv_cache_dtype,
            decode_symbol,
            self.prefill_symbol,
        )
        # Startup proof line: every active role -> symbol plus the resolved
        # cache dtype, single line, stable format (benchmark validity check).
        logger.info_once(
            "%s", resolved_proof_line(resolved, kv_cache_dtype=self.kv_cache_dtype)
        )

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
        import os as _os

        if _os.environ.get("APPMANA_DSPARK_SYNC_DEBUG") == "1" and num_decode_tokens > 1:
            from vllm.logger import init_logger as _il

            qr = q_rows[:num_decode_tokens].float()
            swa_i = swa_indices[:num_decode_tokens]
            _il(__name__).warning(
                "dspark-decode-debug prefix=%s cratio=%s q.shape=%s ndt=%d "
                "q_row_var=%.3e q_rows_identical=%s swa_idx.shape=%s "
                "swa_rows_identical=%s swa_lens=%s",
                getattr(self, "prefix", "?"),
                getattr(self, "compress_ratio", "?"),
                tuple(q.shape),
                num_decode_tokens,
                float(qr.var(dim=0).mean()),
                bool((q_rows[:num_decode_tokens] == q_rows[0]).all()),
                tuple(swa_i.shape),
                bool((swa_i == swa_i[0]).all()),
                swa_lens[:num_decode_tokens].tolist(),
            )

        # Precompiled Ampere CUDA sparse-MLA decode: ONE launch for all decode tokens
        # (the old Triton path looped per row), ~4.4x faster, no Triton JIT/recompile.
        # Matches the Triton reference to ~1e-6 (test_sm86_flash_mla_decode_parity).
        extra_idx = None
        if topk_indices is not None:
            extra_idx = topk_indices.reshape(num_decode_tokens, -1)
        if self.kv_cache_dtype == "int8_ds_mla":
            swa_rows, swa_scales = get_int8_ds_mla_cache_views(
                swa_k_cache, swa_metadata.block_size
            )
            extra_rows = None
            extra_scales = None
            if not swa_only:
                assert kv_cache is not None
                assert attn_metadata is not None
                extra_rows, extra_scales = get_int8_ds_mla_cache_views(
                    kv_cache, attn_metadata.block_size // self.compress_ratio
                )
            if self.int8_decode_symbol == SPARSE_MLA_DECODE_INT8_FLASH:
                out = sparse_int8_mla_decode(
                    q=q_rows,
                    swa_cache=swa_rows,
                    swa_scale=swa_scales,
                    swa_indices=swa_indices,
                    swa_lens=swa_lens,
                    scale=self.scale,
                    attn_sink=self.attn_sink,
                    extra_cache=extra_rows,
                    extra_scale=extra_scales,
                    extra_indices=extra_idx,
                    extra_lens=None if topk_lens is None else topk_lens,
                )
            elif self.int8_decode_symbol == SPARSE_MLA_DECODE_INT8_TRITON:
                out = triton_sparse_int8_mla_decode(
                    q=q_rows,
                    swa_cache=swa_rows,
                    swa_scale=swa_scales,
                    swa_indices=swa_indices,
                    swa_lens=swa_lens,
                    scale=self.scale,
                    attn_sink=self.attn_sink,
                    extra_cache=extra_rows,
                    extra_scale=extra_scales,
                    extra_indices=extra_idx,
                    extra_lens=None if topk_lens is None else topk_lens,
                )
            else:
                raise ValueError(
                    "Unsupported DeepSeek V4 SM86 int8 decode kernel "
                    f"{self.int8_decode_symbol!r}"
                )
        elif self.kv_cache_dtype == "fp8_ds_mla":
            if self.fp8_decode_symbol == SPARSE_MLA_DECODE_FP8_FLASH:
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
            elif self.fp8_decode_symbol == SPARSE_MLA_DECODE_FP8_TRITON:
                decode_sparse_attention_triton(
                    q=q_rows,
                    swa_cache=swa_k_cache,
                    swa_indices=swa_indices,
                    swa_lens=swa_lens,
                    scale=self.scale,
                    attn_sink=self.attn_sink,
                    out=output[:num_decode_tokens],
                    extra_cache=None if swa_only else kv_cache,
                    extra_indices=extra_idx,
                    extra_lens=None if topk_lens is None else topk_lens,
                )
                out = None
            else:
                raise ValueError(
                    "Unsupported DeepSeek V4 SM86 fp8 decode kernel "
                    f"{self.fp8_decode_symbol!r}"
                )
        else:
            raise ValueError(f"Unsupported SM86 KV cache dtype {self.kv_cache_dtype}")
        if out is not None:
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
        if self.prefill_symbol == SPARSE_MLA_PREFILL_FLASH:
            self._forward_prefill_flash(
                q=q,
                positions=positions,
                compressed_k_cache=compressed_k_cache,
                swa_k_cache=swa_k_cache,
                output=output,
                attn_metadata=attn_metadata,
                swa_metadata=swa_metadata,
            )
            return
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
                    cache_dtype=self.kv_cache_dtype,
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
                cache_dtype=self.kv_cache_dtype,
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

    def _forward_prefill_flash(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata,
    ) -> None:
        """Native flash_mla fused sparse-MLA prefill (whole-cache dequant + tensor cores).

        ``flash_mla.flash_sparse_mla_prefill`` mirrors the decode interface:
        it consumes the paged fp8_ds_mla caches directly with per-query-token
        GLOBAL slot indices (compact-left, -1 padded). The adapter below
        builds those indices from the block tables — reusing the mixed
        sparse-index builder that already serves the FlashInfer backend —
        instead of dequant+gather staging a bf16 workspace like the Triton
        path (the flash_mla op stages internally).
        """
        swa_only = attn_metadata is None

        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_reqs = num_decodes + num_prefills
        num_tokens = num_decode_tokens + num_prefill_tokens

        assert swa_metadata.query_start_loc is not None
        assert swa_metadata.token_to_req_indices is not None
        assert swa_metadata.seq_lens is not None
        assert swa_metadata.block_table is not None

        # Local (sequence-space) top-k indices, same sources as the Triton
        # path: the C4A indexer buffer or the C128A metadata builder.
        if not swa_only:
            assert attn_metadata is not None
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[
                    num_decode_tokens:num_tokens
                ]
            else:
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            top_k = topk_indices.shape[-1]
            compressed_block_table = attn_metadata.block_table[
                num_decodes:num_reqs
            ]
            compressed_block_size = attn_metadata.block_size // self.compress_ratio
        else:
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[
                num_decode_tokens:num_tokens, :0
            ]
            top_k = 0
            compressed_block_table = None
            compressed_block_size = swa_metadata.block_size

        # Rebase request-indexed metadata to the prefill rows so the mixed
        # builder runs without the decode rows (decode has its own path).
        query_start_loc = swa_metadata.query_start_loc[num_decodes : num_reqs + 1]
        query_start_loc = query_start_loc - query_start_loc[0]
        token_to_req_indices = (
            swa_metadata.token_to_req_indices[num_decode_tokens:num_tokens]
            - num_decodes
        )
        seq_lens = swa_metadata.seq_lens[num_decodes:num_reqs]
        swa_block_table = swa_metadata.block_table[num_decodes:num_reqs]

        empty_decode_swa = torch.empty(
            (0, self.window_size), dtype=torch.int32, device=q.device
        )
        mixed_indices, mixed_lens = build_flashinfer_mixed_sparse_indices(
            empty_decode_swa,
            None,
            None,
            topk_indices[:num_prefill_tokens],
            query_start_loc,
            seq_lens,
            token_to_req_indices,
            swa_block_table,
            swa_metadata.block_size,
            compressed_block_table,
            compressed_block_size,
            self.window_size,
            self.compress_ratio,
            top_k,
        )

        # Column layout: [0, window) = SWA global slots (compact-left, -1
        # padded), [window, window+padded_topk) = compressed global slots.
        # mixed_lens = window_size + topk_len for prefill rows.
        swa_indices = mixed_indices[:, : self.window_size].contiguous()
        swa_lens = (
            torch.clamp(positions + 1, max=self.window_size).to(torch.int32)
        )
        extra_cache = None
        extra_indices = None
        extra_lens = None
        if not swa_only and top_k > 0:
            extra_cache = compressed_k_cache
            extra_indices = mixed_indices[:, self.window_size :].contiguous()
            extra_lens = (mixed_lens - self.window_size).to(torch.int32)

        if self.kv_cache_dtype == "int8_ds_mla":
            # Same fused native prefill, int8 variant: the 528-byte-token paged
            # caches are consumed through strided (int8 rows, fp32 scales)
            # views; the kernel takes the row stride at runtime.
            swa_rows, swa_scales = get_int8_ds_mla_cache_views(
                swa_k_cache, swa_metadata.block_size
            )
            extra_rows = None
            extra_scales = None
            if extra_cache is not None:
                extra_rows, extra_scales = get_int8_ds_mla_cache_views(
                    extra_cache, compressed_block_size
                )
            dump_dir = os.environ.get("APPMANA_DSV4_PREFILL_DUMP_DIR")
            if dump_dir:
                # Diagnostic capture for the deterministic first-prefill IMA on
                # the PP=10 chain (2026-07-17): synthetic replays of this call
                # pass, so save the REAL argument tensors and cache-view
                # geometry before the kernel launch. The last dump written
                # before the fault identifies the faulting call exactly; the
                # saved payload replays locally against zero caches of
                # identical geometry (IMA is addressing, not values).
                idx = getattr(self, "_prefill_dump_idx", 0)
                self._prefill_dump_idx = idx + 1
                torch.cuda.synchronize()
                payload = {
                    "prefix": self.prefix if hasattr(self, "prefix") else "?",
                    "q": q.cpu(),
                    "swa_indices": swa_indices.cpu(),
                    "swa_lens": swa_lens.cpu(),
                    "extra_indices": None if extra_indices is None else extra_indices.cpu(),
                    "extra_lens": None if extra_lens is None else extra_lens.cpu(),
                    "attn_sink": self.attn_sink.cpu(),
                    "scale": self.scale,
                    "swa_rows_shape": tuple(swa_rows.shape),
                    "swa_rows_stride": tuple(swa_rows.stride()),
                    "swa_scales_shape": tuple(swa_scales.shape),
                    "swa_scales_stride": tuple(swa_scales.stride()),
                    "extra_rows_shape": None if extra_rows is None else tuple(extra_rows.shape),
                    "extra_rows_stride": None if extra_rows is None else tuple(extra_rows.stride()),
                    "extra_scales_shape": None if extra_scales is None else tuple(extra_scales.shape),
                    "extra_scales_stride": None if extra_scales is None else tuple(extra_scales.stride()),
                    "q_stride": tuple(q.stride()),
                    "output_shape": tuple(output.shape),
                    "output_stride": tuple(output.stride()),
                }
                pod = os.environ.get("POD_NAME", "pod")
                torch.save(payload, f"{dump_dir}/prefill-{pod}-{idx:03d}.pt")
                print(
                    f"[prefill-dump] idx={idx} prefix={payload['prefix']} "
                    f"q={tuple(q.shape)}/{tuple(q.stride())} "
                    f"swa_idx={tuple(swa_indices.shape)} "
                    f"swa_rows={payload['swa_rows_shape']}/{payload['swa_rows_stride']} "
                    f"extra_rows={payload['extra_rows_shape']}/{payload['extra_rows_stride']} "
                    f"out={payload['output_shape']}/{payload['output_stride']}",
                    flush=True,
                )
            out = sparse_int8_mla_prefill(
                q=q,
                swa_cache=swa_rows,
                swa_scale=swa_scales,
                swa_indices=swa_indices,
                swa_lens=swa_lens,
                scale=self.scale,
                attn_sink=self.attn_sink,
                extra_cache=extra_rows,
                extra_scale=extra_scales,
                extra_indices=extra_indices,
                extra_lens=extra_lens,
            )
        else:
            out = flash_sparse_mla_prefill(
                q=q,
                swa_cache=swa_k_cache,
                swa_indices=swa_indices,
                swa_lens=swa_lens,
                scale=self.scale,
                attn_sink=self.attn_sink,
                extra_cache=extra_cache,
                extra_indices=extra_indices,
                extra_lens=extra_lens,
            )
        output.copy_(out)
        if output.shape[1] > self.n_local_heads:
            output[:, self.n_local_heads :].zero_()


DeepseekV4SM86Attention = DeepseekV4TritonSM86Attention
