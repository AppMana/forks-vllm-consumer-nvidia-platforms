# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GB10 (sm_12x) DeepSeek-V4 sparse-MLA attention via sparkinfer.

Subclasses the FlashInfer SM120 layer to reuse ALL of its metadata plumbing
(global-slot index building, decode/prefill split, query padding) and swaps
only the kernel launch: sparkinfer's ``attention.compressed_mla`` — a
dual-cache (SWA window + compressed top-k) fp8-compute CuTe-DSL kernel that
consumes the fork's paged ``fp8_ds_mla`` layout byte-for-byte
(parity-gated by ``test_sm12x_sparkinfer_compressed_mla_parity``; 5-14x
faster than the portable Triton path per ``bench_sm12x_sparse_mla_decode``).

Kernel selection is the FQN ``"vllm"`` checkpoint block (kernel_config), NOT
env flags: sparkinfer decode/extend are the sm12x defaults; the portable
Triton kernel (bf16-Q, same cache bytes) is the registered fallback, so an
A/B swap is one ``--hf-overrides`` edit with no rebuild.

Cache geometry: vLLM's unpadded ``(num_blocks, block_size, 584)`` blocks
flatten zero-copy into sparkinfer pages — the AppMana sparkinfer fork
accepts the unpadded ``block_size * 584`` page width (SGLang's 576-multiple
round-up is a pool convention, not a kernel requirement) — so the standard
block size 256 (C4 pool 64, C128 pool 2) is used unchanged.
"""

import dataclasses
from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferSM120Attention,
)
from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLABackend
from vllm.models.deepseek_v4.nvidia_sm12x.kernels import (
    sparkinfer_sparse_mla_decode,
    sparkinfer_sparse_mla_extend,
)
from vllm.platforms.interface import DeviceCapability
from vllm.transformers_utils.configs.dsv4.kernel_config import (
    ROLE_SPARSE_MLA_DECODE_FP8,
    ROLE_SPARSE_MLA_PREFILL,
    SPARSE_MLA_DECODE_FP8_SPARKINFER,
    SPARSE_MLA_DECODE_FP8_TRITON,
    SPARSE_MLA_PREFILL_SPARKINFER,
    SPARSE_MLA_PREFILL_TRITON,
    ResolvedKernelConfig,
    resolve_kernel_config_from_hf_config,
    resolved_proof_line,
)

logger = init_logger(__name__)

_SM12X_DECODE_SYMBOLS = (
    SPARSE_MLA_DECODE_FP8_SPARKINFER,
    SPARSE_MLA_DECODE_FP8_TRITON,
)
_SM12X_PREFILL_SYMBOLS = (
    SPARSE_MLA_PREFILL_SPARKINFER,
    SPARSE_MLA_PREFILL_TRITON,
)


def validate_sm12x_kernel_selection(
    resolved: ResolvedKernelConfig, *, kv_cache_dtype: str
) -> tuple[str, str]:
    """Resolve the effective (decode, prefill) symbols for sm12x.

    An explicit block must name sm12x-capable symbols (sparkinfer or the
    portable Triton kernels); anything else — e.g. the sm86 flash_mla
    defaults that blockless resolution produces — maps to sparkinfer, the
    sm12x primary. Fails closed on explicit non-sm12x selections.
    """
    if kv_cache_dtype != "fp8_ds_mla":
        raise ValueError(
            "DeepSeek V4 sm12x sparse MLA requires the fp8_ds_mla KV cache, "
            f"got {kv_cache_dtype}"
        )
    if not resolved.explicit:
        # No "vllm" kernels list: sparkinfer is the sm12x primary.
        return SPARSE_MLA_DECODE_FP8_SPARKINFER, SPARSE_MLA_PREFILL_SPARKINFER

    def _effective(role: str, allowed: tuple[str, ...], default: str) -> str:
        if role not in resolved.listed_roles:
            # Role not listed in the block: the global (sm86-flavored)
            # default filled it in; on sm12x unspecified means sparkinfer.
            return default
        symbol = resolved.roles[role]
        if symbol in allowed:
            return symbol
        # An explicitly listed non-sm12x kernel: fail closed rather than
        # silently substituting.
        raise ValueError(
            f"Kernel {symbol!r} ({role}) is not an sm12x kernel; "
            f"sm12x supports {allowed}"
        )

    decode = _effective(
        ROLE_SPARSE_MLA_DECODE_FP8,
        _SM12X_DECODE_SYMBOLS,
        SPARSE_MLA_DECODE_FP8_SPARKINFER,
    )
    prefill = _effective(
        ROLE_SPARSE_MLA_PREFILL,
        _SM12X_PREFILL_SYMBOLS,
        SPARSE_MLA_PREFILL_SPARKINFER,
    )
    return decode, prefill


class DeepseekV4SparkInferMLABackend(DeepseekV4FlashMLABackend):
    """Paged fp8_ds_mla backend for the sparkinfer sm12x kernels."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["fp8_ds_mla"]

    @staticmethod
    def get_name() -> str:
        return "SPARKINFER_MLA_SPARSE_DSV4"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        # The standard DSV4 block size, shared with the other sparse-MLA
        # backends (kv manager groups require one common size). The AppMana
        # sparkinfer fork consumes vLLM's unpadded pages at any block size.
        return [256]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 12


class DeepseekV4SparkInferSM12xAttention(DeepseekV4FlashInferSM120Attention):
    """DeepSeek V4 sparse MLA through sparkinfer's GB10 CuTe kernels."""

    backend_cls = DeepseekV4SparkInferMLABackend

    def _check_kernel_available(self) -> None:
        try:
            from sparkinfer.attention import compressed_mla
        except ImportError as exc:  # pragma: no cover - environment specific
            raise RuntimeError(
                "DeepSeek V4 sm12x attention requires sparkinfer "
                "(install from the AppMana/sparkinfer fork)."
            ) from exc
        if not compressed_mla.is_supported():
            raise RuntimeError(
                "sparkinfer compressed_mla reports unsupported on this device "
                "(needs SM120/SM121, nvidia-cutlass-dsl >= 4.6, triton)."
            )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Unified kernel selection: the "vllm" checkpoint config block
        # (fail-closed), same machinery as nvidia_sm86.
        resolved = resolve_kernel_config_from_hf_config(self.config)
        self.decode_symbol, self.prefill_symbol = validate_sm12x_kernel_selection(
            resolved, kv_cache_dtype=self.kv_cache_dtype
        )
        self._sink_by_heads: dict[int, torch.Tensor] = {}
        logger.info_once(
            "DeepSeek V4 SM12x sparse MLA kernels: kv_cache_dtype=%s, "
            "decode=%s, prefill=%s",
            self.kv_cache_dtype,
            self.decode_symbol,
            self.prefill_symbol,
        )
        # The proof line must state what actually runs: substitute the sm12x
        # EFFECTIVE decode/prefill symbols (blockless/unlisted roles resolve
        # to sparkinfer here, not the global sm86-flavored defaults).
        effective = dataclasses.replace(
            resolved,
            roles={
                **resolved.roles,
                ROLE_SPARSE_MLA_DECODE_FP8: self.decode_symbol,
                ROLE_SPARSE_MLA_PREFILL: self.prefill_symbol,
            },
        )
        logger.info_once(
            "%s", resolved_proof_line(effective, kv_cache_dtype=self.kv_cache_dtype)
        )

    def _reserve_empty_forward_workspace(self) -> None:
        # No FlashInfer workspace; sparkinfer plans/scratch are cached on
        # first launch (vLLM's eager dummy runs precede CUDA-graph capture,
        # so JIT compilation never lands inside a capture).
        return

    def _padded_sink(self, heads: int) -> torch.Tensor | None:
        """attn_sink padded to the query's padded head count (padded heads
        attend to shared KV and are sliced off downstream; their sink value
        is irrelevant, 0 keeps the math finite). Cached: launches happen
        inside CUDA-graph capture and must not allocate fresh addresses per
        call."""
        sink = self.attn_sink
        if sink is None:
            return None
        cached = self._sink_by_heads.get(heads)
        if cached is None:
            cached = torch.zeros(heads, dtype=torch.float32, device=sink.device)
            cached[: sink.shape[0]] = sink.detach().float()
            self._sink_by_heads[heads] = cached
        return cached

    def _launch_sparse(
        self,
        *,
        q: torch.Tensor,
        swa_cache: torch.Tensor,
        swa_indices: torch.Tensor,
        swa_lens: torch.Tensor,
        extra_cache: torch.Tensor | None,
        extra_indices: torch.Tensor | None,
        extra_lens: torch.Tensor | None,
        output: torch.Tensor,
        mode: str,
    ) -> None:
        rows = q.shape[0]
        symbol = self.decode_symbol if mode == "decode" else self.prefill_symbol
        if extra_indices is not None and extra_indices.dim() == 3:
            extra_indices = extra_indices.reshape(rows, -1)
        if symbol in (
            SPARSE_MLA_DECODE_FP8_SPARKINFER,
            SPARSE_MLA_PREFILL_SPARKINFER,
        ):
            fn = (
                sparkinfer_sparse_mla_decode
                if mode == "decode"
                else sparkinfer_sparse_mla_extend
            )
            fn(
                q=q,
                swa_cache=swa_cache,
                swa_indices=swa_indices,
                swa_lens=swa_lens,
                extra_cache=extra_cache,
                extra_indices=extra_indices,
                extra_lens=extra_lens,
                sm_scale=self.scale,
                attn_sink=self._padded_sink(q.shape[1]),
                out=output,
                max_q_rows=self.max_num_batched_tokens,
            )
            return
        # Portable Triton fallback (bf16-Q over the same fp8_ds_mla bytes);
        # per-row independent, so it serves prefill rows unchanged.
        from vllm.models.deepseek_v4.nvidia_sm86.triton_kernels import (
            decode_sparse_attention_triton,
        )

        decode_sparse_attention_triton(
            q=q,
            swa_cache=swa_cache,
            swa_indices=swa_indices,
            swa_lens=swa_lens,
            scale=self.scale,
            attn_sink=self._padded_sink(q.shape[1]),
            out=output,
            extra_cache=extra_cache,
            extra_indices=extra_indices,
            extra_lens=extra_lens,
        )
