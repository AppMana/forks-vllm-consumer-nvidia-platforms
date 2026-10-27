# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 shared-KV metadata with native paged INT8 sparse MLA kernels."""

from typing import ClassVar

from vllm.config.cache import CacheDType
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.common.ops.cache_utils import get_int8_ds_mla_cache_views
from vllm.models.deepseek_v41.common.ops import compute_global_topk_indices_and_lens
from vllm.models.deepseek_v41.nvidia.flashmla import DeepseekV4FlashMLAAttention
from vllm.models.deepseek_v41.sparse_mla import DeepseekV4FlashMLABackend


class DeepseekV41Int8Backend(DeepseekV4FlashMLABackend):
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["int8_ds_mla"]

    @staticmethod
    def get_name():
        return "NATIVE_INT8_MLA_SPARSE_DSV41"

    @classmethod
    def supports_compute_capability(cls, capability):
        return capability.major >= 8


class DeepseekV41Int8Attention(DeepseekV4FlashMLAAttention):
    """Consume V4.1's selected shared-cache slots without whole-cache expansion.

    Prefill and decode use the existing native INT8 implementations. This is
    an execution candidate; workload benchmarks must determine final policy.
    """

    backend_cls = DeepseekV41Int8Backend

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.kv_cache_dtype != "int8_ds_mla":
            raise ValueError("Native V4.1 INT8 attention requires int8_ds_mla cache")
        from flash_mla import sparse_mla_decode_int8, sparse_mla_prefill_int8

        self._decode = sparse_mla_decode_int8
        self._prefill = sparse_mla_prefill_int8

    @classmethod
    def get_padded_num_q_heads(cls, num_heads):
        return num_heads

    def forward_mqa(self, q, kv, positions, output):
        if get_forward_context().attn_metadata is None:
            # These kernels read compact pages directly; no BF16 gather arena.
            output.zero_()
            return
        super().forward_mqa(q, kv, positions, output)

    def _compressed_indices(self, metadata, swa, start, count):
        if self.compress_ratio == 0:
            return None, None
        if metadata is None or self.topk_indices_buffer is None:
            raise ValueError(
                "Compressed V4.1 INT8 attention requires shared-cache metadata"
            )
        indices, lengths = compute_global_topk_indices_and_lens(
            self.topk_indices_buffer[start : start + count],
            swa.token_to_req_indices[start : start + count],
            metadata.block_table,
            metadata.block_size // self.compress_ratio,
            swa.is_valid_token[start : start + count],
        )
        return indices.reshape(count, -1).contiguous(), lengths

    def _run_int8(
        self,
        kernel,
        q,
        output,
        swa_cache,
        swa_indices,
        swa_lens,
        swa,
        cache,
        metadata,
        start,
        count,
    ):
        if count == 0:
            return
        if min(q.shape[0], output.shape[0]) < count:
            raise ValueError("INT8 attention metadata exceeds query/output capacity")
        indices, lengths = self._compressed_indices(metadata, swa, start, count)
        rows, scales = get_int8_ds_mla_cache_views(swa_cache, swa.block_size)
        extra_rows = extra_scales = None
        if self.compress_ratio:
            if cache is None:
                raise ValueError("Missing shared compressed INT8 cache")
            extra_rows, extra_scales = get_int8_ds_mla_cache_views(
                cache, metadata.block_size // self.compress_ratio
            )
        result = kernel(
            q=q[:count],
            swa_cache=rows,
            swa_scale=scales,
            swa_indices=swa_indices[:count].reshape(count, -1).contiguous(),
            swa_lens=swa_lens[:count],
            scale=self.scale,
            attn_sink=self.attn_sink,
            extra_cache=extra_rows,
            extra_scale=extra_scales,
            extra_indices=indices,
            extra_lens=lengths,
        )
        output[:count].copy_(result)

    def _forward_decode(
        self, q, kv_cache, swa_metadata, attn_metadata, swa_only, output
    ):
        if swa_only != (self.compress_ratio == 0):
            raise ValueError("V4.1 SWA metadata disagrees with layer geometry")
        self._run_int8(
            self._decode,
            q,
            output,
            self.swa_cache_layer.kv_cache,
            swa_metadata.decode_swa_indices,
            swa_metadata.decode_swa_lens,
            swa_metadata,
            kv_cache,
            attn_metadata,
            0,
            swa_metadata.num_decode_tokens,
        )

    def _forward_prefill(
        self,
        q,
        positions,
        compressed_k_cache,
        swa_k_cache,
        output,
        attn_metadata,
        swa_metadata,
    ):
        del positions  # Causality and bounded replay are encoded by the SWA builder.
        self._run_int8(
            self._prefill,
            q,
            output,
            swa_k_cache,
            swa_metadata.prefill_swa_indices,
            swa_metadata.prefill_swa_lens,
            swa_metadata,
            compressed_k_cache,
            attn_metadata,
            swa_metadata.num_decode_tokens,
            swa_metadata.num_prefill_tokens,
        )
