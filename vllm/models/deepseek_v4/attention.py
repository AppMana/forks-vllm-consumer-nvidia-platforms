# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DeepseekV4 MLA Attention Layer
"""

import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DeepseekV2Config, DeepseekV3Config

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.models.common.ops import fused_q_kv_rmsnorm
from vllm.models.deepseek_v4.common.ops import (
    fused_indexer_q_rope_quant,
    fused_qnorm_rope_kv_int8_ds_mla_insert,
)
from vllm.models.deepseek_v4.common.ops.fused_indexer_q import (
    MXFP4_BLOCK_SIZE,
    _supports_fp8e4nv_in_triton,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import (
        DeepseekSparseSWAMetadata,
    )

from vllm.config import (
    CacheConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.utils import extract_layer_index
from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope
from vllm.models.deepseek_v4.compressor import DeepseekCompressor
from vllm.transformers_utils.configs.dsv4.kernel_config import (
    activate_kernel_config,
    resolve_kernel_config_from_hf_config,
)
from vllm.triton_utils import tl, triton
from vllm.utils.multi_stream_utils import (
    execute_in_parallel,
    maybe_execute_in_parallel,
)
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV4IndexerBackend,
    dsa_indexer_uses_fp4,
    get_max_prefill_buffer_size,
)
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    get_kv_quant_mode,
)

logger = init_logger(__name__)

# Sparse-MLA backends that reach the shared top-k / SWA-index Triton helpers.
# Every one of them must be warmed at startup or that arch JIT-compiles them on
# the first live request.
_DEEPSEEK_V4_SPARSE_MLA_BACKENDS = (
    "FLASHMLA_SPARSE_DSV4",
    "ROCM_FLASHMLA_SPARSE_DSV4",
    "SPARKINFER_MLA_SPARSE_DSV4",
    "XPU_V4_MLA_SPARSE",
)

# Diagnostic: serialize the attention stages and log per-stage GPU wall time.
# Read once at import; never part of traced code (the op interior is opaque).
_STEP_TIMING = os.environ.get("APPMANA_DSV4_STEP_TIMING", "0") == "1"
_STEP_TIMING_STATS: dict[str, list[float]] = {}
_STEP_TIMING_CALLS = 0


def _step_timing_record(section: str, seconds: float) -> None:
    bucket = _STEP_TIMING_STATS.setdefault(section, [0.0, 0.0])
    bucket[0] += seconds
    bucket[1] += 1.0


def _step_timing_maybe_log() -> None:
    global _STEP_TIMING_CALLS
    _STEP_TIMING_CALLS += 1
    if _STEP_TIMING_CALLS % 256 != 0:
        return
    parts = [
        f"{name}={total * 1000.0 / max(count, 1.0):.2f}ms(n={int(count)})"
        for name, (total, count) in sorted(_STEP_TIMING_STATS.items())
    ]
    logger.info("DSV4_STEP_TIMING %s", " ".join(parts))
    _STEP_TIMING_STATS.clear()


def use_compilation_safe_attn_gemm_overlap(num_tokens: int) -> bool:
    """Use auxiliary GEMM streams only outside a torch-compiled region.

    Explicit stream/event orchestration inside the compiled model body can be
    reordered across the graph break before ``attention_impl``. Keep the
    compiled path sequential; the eager attention implementation retains its
    own safe overlap after the break.
    """
    return (
        not torch.compiler.is_compiling()
        and num_tokens <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD
    )


@triton.jit
def _fill_short_context_topk_indices(
    output,
    positions,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
):
    # small triton kernel that selects every candidate, -1 otherwise
    row = tl.program_id(0)
    offsets = tl.arange(0, PADDED_TOP_K)
    num_compressed = (tl.load(positions + row) + 1) // COMPRESS_RATIO
    tl.store(
        output + row * TOP_K + offsets,
        tl.where(offsets < num_compressed, offsets, -1),
        mask=offsets < TOP_K,
    )


def _resolve_dsv4_kv_cache_dtype(
    use_fp8_ds_mla_layout: bool,
    kv_cache_dtype: str,
    cache_config: CacheConfig | None,
) -> tuple[str, torch.dtype]:
    """Map ``(layout, --kv-cache-dtype)`` to ``(cache_dtype_str, torch_dtype)``.

    Both layouts are paged; they differ in the per-token block format. The
    ``fp8_ds_mla`` format is UE8M0 block-scaled fp8 packed as ``uint8`` (the
    canonical ``fp8_ds_mla`` string is written back onto ``cache_config`` so the
    page-size specs pick the 576B per-token slot). ``int8_ds_mla`` is signed
    int8 rows plus fp32 row scales packed in a ``uint8`` backing tensor. Plain
    row backends store each token's KV row in its element dtype: bf16 or
    per-tensor FP8 E4M3.
    """
    if kv_cache_dtype == "int8_ds_mla":
        logger.info_once("Using DeepSeek's int8_ds_mla KV cache format.")
        return kv_cache_dtype, torch.uint8

    if use_fp8_ds_mla_layout:
        # fp8_ds_mla block format: UE8M0 block-scaled fp8 packed as uint8.
        if kv_cache_dtype == "auto":
            kv_cache_dtype = "fp8"
        if not kv_cache_dtype.startswith("fp8"):
            raise ValueError(
                "DeepseekV4 fp8_ds_mla layout only supports fp8 "
                f"kv-cache, got {kv_cache_dtype}. Please set "
                "`--kv-cache-dtype fp8` or select a backend that supports "
                "bfloat16 KV cache."
            )
        if kv_cache_dtype != "fp8_ds_mla":
            if cache_config is not None:
                cache_config.cache_dtype = "fp8_ds_mla"
            kv_cache_dtype = "fp8_ds_mla"
            logger.info_once("Using DeepSeek's fp8_ds_mla KV cache format.")
        return kv_cache_dtype, torch.uint8

    # Plain bf16 / per-tensor fp8 KV row (FlashInfer).
    if kv_cache_dtype.startswith("fp8"):
        return kv_cache_dtype, torch.float8_e4m3fn
    # auto / bfloat16 -> plain bf16 KV row.
    return kv_cache_dtype, torch.bfloat16


def resolve_layer_compress_ratio(config, layer_id: int) -> tuple[int, bool]:
    """Resolve the cache ratio and rotary embedding mode for a layer."""
    if layer_id < config.num_hidden_layers:
        return max(1, config.compress_ratios[layer_id]), False
    if layer_id < len(config.compress_ratios):
        raw_compress_ratio = config.compress_ratios[layer_id]
        return max(1, raw_compress_ratio), raw_compress_ratio == 0
    return 1, False


class DeepseekV4Attention(nn.Module, AttentionLayerBase, ABC):
    """DeepseekV4 MLA attention layer.

    The platform-specific sparse-MLA forward (``forward_mqa`` /
    ``get_padded_num_q_heads`` / ``_o_proj`` / ``backend_cls``) is provided by a
    subclass: ``DeepseekV4FlashMLAAttention`` /
    ``DeepseekV4FlashInferSM120Attention`` /
    ``DeepseekV4FlashInferMLAAttention`` (CUDA) or
    ``DeepseekV4ROCMAiterMLAAttention`` (ROCm), selected by the platform-specific
    deepseek_v4 model module. The base is never instantiated directly.
    """

    # Provided by the platform subclass.
    backend_cls: ClassVar[type[AttentionBackend]]
    # Backend for the SWA cache layer; None uses the default SWA backend.
    swa_backend_cls: ClassVar[type[AttentionBackend] | None] = None
    # KV-cache per-token block format (both layouts are paged). True (default)
    # = fp8_ds_mla (UE8M0 block-scaled fp8 packed as uint8); False = plain
    # bf16 / per-tensor fp8 KV row. Backends can override the instance hook when
    # a single attention class dispatches across arch-specific layouts.
    use_fp8_ds_mla_layout: ClassVar[bool] = True
    # Prefill is processed in fixed-size chunks; this bounds the bf16 kv-gather
    # workspace allocated in _forward_prefill and is also read by the dummy-run
    # path to pre-reserve that workspace.
    PREFILL_CHUNK_SIZE: ClassVar[int] = 4

    @classmethod
    @abstractmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        """Q head count the q/output buffers are allocated at.

        The layer allocates the q/output buffers at
        ``[N, get_padded_num_q_heads(n_local_heads), head_dim]``. Must satisfy
        ``result >= num_heads``. Backends with no padding constraint return
        ``num_heads``.
        """
        raise NotImplementedError

    @abstractmethod
    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Platform-specific sparse MLA forward; writes attention into ``output``."""
        raise NotImplementedError

    @abstractmethod
    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Inverse-RoPE + wo_a + wo_b output projection (platform-specific)."""
        raise NotImplementedError

    def _uses_fp8_ds_mla_layout(self) -> bool:
        """Return whether this instance stores fp8 KV in fp8_ds_mla layout."""
        return self.use_fp8_ds_mla_layout

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        tp_size = get_tensor_model_parallel_world_size()
        layer_id = extract_layer_index(prefix)

        self.config = config
        # Resolve + activate the unified AppMana kernel config ("vllm"
        # block in the checkpoint config.json) before any consumer (indexer,
        # compressor, backend dispatch) reads the kernel gates. Runs in every
        # worker process at model build; idempotent by value.
        activate_kernel_config(resolve_kernel_config_from_hf_config(config))
        self.prefix = prefix  # Alias for compatibility with compressor
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        assert self.n_heads % tp_size == 0
        self.n_local_heads = self.n_heads // tp_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // tp_size
        self.window_size = config.sliding_window
        # Vision variant: image spans are visible bidirectionally, widening
        # prefill SWA index rows by up to max_image_tokens columns.
        self.max_image_tokens = (
            getattr(config, "vision_max_n_token", 0)
            if getattr(config, "vision_n_layers", 0) > 0
            else 0
        )
        self.compress_ratio, use_unscaled_rope = resolve_layer_compress_ratio(
            config, layer_id
        )
        self.eps = config.rms_norm_eps
        self.scale = self.head_dim**-0.5

        # Padded Q head count is dictated by the platform subclass.
        self.padded_heads = self.get_padded_num_q_heads(self.n_local_heads)
        # Sink padded to the same head count, initialized to -inf (no sink
        # effect). Weight loading fills the first n_local_heads slots.
        self.attn_sink = nn.Parameter(
            torch.full((self.padded_heads,), -float("inf"), dtype=torch.float32),
            requires_grad=False,
        )

        self.fused_wqa_wkv = MergedColumnParallelLinear(
            self.hidden_size,
            [self.q_lora_rank, self.head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_wqa_wkv",
            disable_tp=True,  # fused ReplicatedLinear
        )
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wq_b",
        )

        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_a",
        )
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.n_local_groups
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
        )

        # Initialize rotary embedding before the indexer/compressor consume it.
        self.rotary_emb = build_deepseek_v4_rope(
            config,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            compress_ratio=self.compress_ratio,
            use_unscaled_rope=use_unscaled_rope,
        )
        self.indexer_rotary_emb = self.rotary_emb
        self.topk_indices_buffer = topk_indices_buffer

        self.indexer = None
        if self.compress_ratio == 4:
            # Only C4A uses sparse attention and hence has indexer.
            # aux_stream_list[2] is free here (outer GEMMs joined) for the inner
            # overlap of wq_b+fused_indexer_q_rope_quant vs compressor. None on
            # ROCm, where aux_stream_list is None.
            indexer_aux_stream = (
                aux_stream_list[2] if aux_stream_list is not None else None
            )
            self.indexer = DeepseekV4Indexer(
                vllm_config,
                config=config,
                hidden_size=self.hidden_size,
                q_lora_rank=self.q_lora_rank,
                quant_config=quant_config,
                cache_config=cache_config,
                topk_indices_buffer=topk_indices_buffer,
                compress_ratio=self.compress_ratio,
                prefix=f"{prefix}.indexer",
                aux_stream=indexer_aux_stream,
            )

        self._prepare_and_attn_fn = self._prepare_and_attn
        if not vllm_config.use_v2_model_runner:
            # MRV1's piecewise capture only tolerates the wide eager region: with
            # the narrow one the attention input preparation stays in the captured
            # graph and MRV1 produces garbage (#51430).
            self._prepare_and_attn_fn = self._prepare_and_attn_eager

        # Will be None on ROCm for now.
        self.aux_stream_list = aux_stream_list
        # [0]: GEMM start / post-GEMM event0. [1..3]: GEMM done events;
        # [1] doubles as post-GEMM event1. Reuse is safe: GEMM fully joins
        # before post-GEMM starts.
        self.ln_events = [torch.cuda.Event() for _ in range(4)]

        assert cache_config is not None, "DeepseekV4 attention requires cache_config"
        # ---- Attention / KV-cache setup ----
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len

        # Resolve the kv-cache dtype from this backend's block format. The same
        # resolution drives the SWA cache tensor dtype below.
        self.kv_cache_dtype, self.kv_cache_torch_dtype = _resolve_dsv4_kv_cache_dtype(
            self._uses_fp8_ds_mla_layout(), cache_config.cache_dtype, cache_config
        )

        self.swa_cache_layer = DeepseekV4SWACache(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=self.kv_cache_torch_dtype,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
            backend_cls=self.swa_backend_cls,
        )

        # Register with compilation context for metadata lookup.
        compilation_config = vllm_config.compilation_config
        if prefix and prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        if prefix:
            compilation_config.static_forward_context[prefix] = self
        self.kv_cache = torch.tensor([])

        # Create the compressor for layers with compress_ratio > 1; after the
        # attention setup above so its KV-cache prefix (self.prefix) is set.
        self.compressor = None
        if self.compress_ratio > 1:
            self.compressor = DeepseekCompressor(
                vllm_config=vllm_config,
                compress_ratio=self.compress_ratio,
                hidden_size=self.hidden_size,
                head_dim=self.head_dim,
                rotate=True,
                prefix=f"{prefix}.compressor",
                k_cache_prefix=self.prefix,
            )

        if vllm_config.kernel_config.enable_jit_warmup:
            from vllm.v1.attention.backends.mla.sparse_swa import (
                _COMPUTE_PREFILL_METADATA_KERNEL,
                _COMPUTE_SWA_INDICES_AND_LENS_KERNEL,
            )

            _COMPUTE_PREFILL_METADATA_KERNEL.register_warmup()
            _COMPUTE_SWA_INDICES_AND_LENS_KERNEL.register_warmup(
                window_size=self.window_size,
                block_size=self.swa_cache_layer.block_size,
                max_image_tokens=self.max_image_tokens,
            )

            if self.compress_ratio > 1:
                from vllm.v1.attention.backends.mla.compressor_utils import (
                    _COMPRESSED_SLOT_MAPPING_KERNEL,
                )

                _COMPRESSED_SLOT_MAPPING_KERNEL.register_warmup()

            if self.indexer is not None:
                from vllm.v1.attention.backends.mla.indexer import (
                    _BUILD_PREFILL_CHUNK_METADATA_KERNEL,
                    _PREPARE_UNIFORM_DECODE_KERNEL,
                )

                _PREPARE_UNIFORM_DECODE_KERNEL.register_warmup()
                _BUILD_PREFILL_CHUNK_METADATA_KERNEL.register_warmup()

            spec_config = vllm_config.speculative_config
            if spec_config is not None and spec_config.use_dspark():
                from vllm.v1.attention.backends.mla.sparse_swa import (
                    _COMPUTE_DSPARK_NONCAUSAL_SWA_INDICES_KERNEL,
                )

                _COMPUTE_DSPARK_NONCAUSAL_SWA_INDICES_KERNEL.register_warmup(
                    window_size=self.window_size,
                    num_speculative_tokens=spec_config.num_speculative_tokens,
                    block_size=self.swa_cache_layer.block_size,
                )

            if self.backend_cls.get_name() in _DEEPSEEK_V4_SPARSE_MLA_BACKENDS:
                from vllm.models.deepseek_v4.common.ops.cache_utils import (
                    _COMBINE_TOPK_SWA_INDICES_KERNEL,
                )

                _COMBINE_TOPK_SWA_INDICES_KERNEL.register_warmup()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-allocate attention output with FlashMLA-padded head count.
        # The op writes into `o_padded`; we slice to n_local_heads after.
        num_tokens = hidden_states.shape[0]
        o_padded = torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Metadata-independent input GEMMs + RMSNorm stay in the captured
        # graph; the metadata-dependent rest (q up-proj + kv-insert, indexer,
        # compressor, MLA attention) runs behind the custom op below.
        qr_kv, kv_score, indexer_kv_score, indexer_weights = (
            self._run_parallel_input_projections(hidden_states)
        )
        qr, qr_scale, kv = self._split_qkv_and_norm(qr_kv)

        # The metadata-dependent attention stack must stay opaque to
        # torch.compile: it reads live forward-context state (attention
        # metadata, KV caches, shared top-k buffers) that Dynamo would
        # otherwise bake in at trace time, and the traced body is reused
        # for every batch with all guards dropped. The custom op executes
        # this Python at runtime; under breakable CUDA graph capture it is
        # also where the capture breaks. Inside the op, the layer dispatches
        # through _prepare_and_attn_fn, which keeps upstream's model-runner
        # version selection.
        torch.ops.vllm.deepseek_v4_attention(
            hidden_states,
            qr,
            kv,
            qr_scale,
            kv_score,
            indexer_kv_score,
            indexer_weights,
            positions,
            o_padded,
            _encode_layer_name(self.prefix),
        )
        o = o_padded[:, : self.n_local_heads, :]

        # Inverse-RoPE + wo_a + wo_b output projection (platform-specific).
        return self._o_proj(o, positions)

    def _split_qkv_and_norm(
        self, qr_kv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Split the fused q-lora / kv projection and RMSNorm both halves.

        ``qr_scale`` is None on the shared path. The ROCm subclass returns
        a pre-quantized fp8 ``qr`` together with its per-1x128 fp32 scales,
        which the downstream wq_b projections consume directly.
        """
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        qr, kv = fused_q_kv_rmsnorm(
            qr,
            kv,
            self.q_norm.weight.data,
            self.kv_norm.weight.data,
            self.eps,
        )
        return qr, None, kv

    @eager_break_during_capture
    def _prepare_and_attn_eager(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        qr_scale: torch.Tensor | None,
        kv_score: torch.Tensor,
        indexer_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        o_padded: torch.Tensor,
    ) -> None:
        """Wide eager region: the whole of ``_prepare_and_attn`` runs eagerly.

        The nested ``_sparse_indexer_and_attn`` break runs inline, since
        ``add_eager`` clears ``_capturing`` before invoking this.
        """
        self._prepare_and_attn(
            hidden_states,
            qr,
            kv,
            qr_scale,
            kv_score,
            indexer_kv_score,
            indexer_weights,
            positions,
            o_padded,
        )

    def _prepare_and_attn(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        qr_scale: torch.Tensor | None,
        kv_score: torch.Tensor,
        indexer_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        o_padded: torch.Tensor,
    ) -> None:
        """Attention input preparation followed by the sparse indexer and MLA.

        Only the latter runs in the eager break.
        """
        attn_metadata = get_forward_context().attn_metadata
        indexer = self.indexer
        compressor = self.compressor
        aux_streams = self.aux_stream_list

        def project_query_and_cache_kv() -> torch.Tensor:
            q = self._wq_b_proj(qr, qr_scale).view(
                -1, self.n_local_heads, self.head_dim
            )
            return self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)

        if _STEP_TIMING and indexer is not None:
            # Serial stages with sync fences: attributes per-stage GPU wall
            # time at the cost of losing the aux-stream overlap. Diagnostic
            # only; guarded by APPMANA_DSV4_STEP_TIMING=1.
            import time as _time

            assert compressor is not None
            torch.cuda.synchronize()
            t0 = _time.perf_counter()
            q = project_query_and_cache_kv()
            torch.cuda.synchronize()
            t1 = _time.perf_counter()
            compressor(kv_score, positions, self.rotary_emb)
            torch.cuda.synchronize()
            t2 = _time.perf_counter()
            self._run_indexer_op(
                hidden_states,
                *indexer(
                    hidden_states,
                    qr,
                    indexer_kv_score,
                    indexer_weights,
                    positions,
                    self.indexer_rotary_emb,
                    qr_scale,
                ),
            )
            torch.cuda.synchronize()
            t3 = _time.perf_counter()
            self.forward_mqa(q, kv, positions, o_padded)
            torch.cuda.synchronize()
            t4 = _time.perf_counter()
            _step_timing_record("wq_b_kv_insert", t1 - t0)
            _step_timing_record("compressor", t2 - t1)
            _step_timing_record("indexer", t3 - t2)
            _step_timing_record("forward_mqa", t4 - t3)
            _step_timing_maybe_log()
            return

        index_q: torch.Tensor | None = None
        index_q_scale: torch.Tensor | None = None
        index_weights_out: torch.Tensor | None = None

        # Keep Q projection and KV insertion on the default stream. The indexer
        # and MLA compressor use aux streams 0 and 1; aux 2 is internal to the
        # indexer. ROCm runs the same work sequentially without aux streams.
        if indexer is not None:
            assert compressor is not None
            q, (indexer_inputs, _) = execute_in_parallel(
                project_query_and_cache_kv,
                [
                    lambda: indexer(
                        hidden_states,
                        qr,
                        indexer_kv_score,
                        indexer_weights,
                        positions,
                        self.indexer_rotary_emb,
                        qr_scale,
                    ),
                    lambda: compressor(kv_score, positions, self.rotary_emb),
                ],
                self.ln_events[0],
                [self.ln_events[1], self.ln_events[2]],
                [aux_streams[0], aux_streams[1]] if aux_streams is not None else None,
                enable=aux_streams is not None
                and hidden_states.shape[0]
                <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD,
            )
            index_q, index_q_scale, index_weights_out = indexer_inputs
        elif compressor is not None:
            aux_stream = aux_streams[0] if aux_streams is not None else None
            q, _ = maybe_execute_in_parallel(
                project_query_and_cache_kv,
                lambda: compressor(kv_score, positions, self.rotary_emb),
                self.ln_events[0],
                self.ln_events[1],
                aux_stream,
            )
        else:
            q = project_query_and_cache_kv()

        self._sparse_indexer_and_attn(
            hidden_states,
            index_q,
            index_q_scale,
            index_weights_out,
            q,
            kv,
            positions,
            o_padded,
        )

    def _fused_wqa_wkv_gemm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Override point: the ROCm layer preshuffles this weight in place, so
        # it cannot go through fused_wqa_wkv directly.
        # MergedColumnParallelLinear returns (output, bias); bias is None.
        qr_kv, _ = self.fused_wqa_wkv(hidden_states)
        return qr_kv

    def _wq_b_proj(
        self, qr: torch.Tensor, qr_scale: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Project the q-lora input to the query heads.

        ``qr_scale`` is only set by the ROCm fused path, where ``qr`` is
        already fp8-quantized with per-1x128 scales and must bypass the
        linear's internal re-quantization (apply_weights would quantize
        the fp8 input again).
        """
        if qr_scale is None:
            return self.wq_b(qr)
        from vllm.models.deepseek_v4.amd.rocm import (
            apply_pre_quantized_block_scaled_mm,
        )

        return apply_pre_quantized_block_scaled_mm(self.wq_b, qr, qr_scale)

    def _run_parallel_input_projections(
        self, hidden_states: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        aux_streams = self.aux_stream_list
        if aux_streams is not None:
            assert len(aux_streams) >= 3
            aux_streams = aux_streams[:3]

        # fused_wqa_wkv (heaviest) on default; the three lighter input GEMMs
        # on aux streams 0..2 when their owning module exists. ln_events[0]
        # is the fan-out start event; ln_events[1..3] are per-aux done events.
        # On ROCm, aux_streams is None and execute_in_parallel runs serially.
        aux_fns: list[Callable[[], Any] | None] = [None, None, None]

        if self.compressor is not None:
            # Local ref so the closure keeps a non-None type for mypy.
            compressor = self.compressor

            def compressor_kv_score() -> torch.Tensor:
                return torch.mm(
                    hidden_states,
                    compressor.fused_wkv_wgate.weight.T,
                    out_dtype=torch.float32,
                )

            aux_fns[0] = compressor_kv_score

        if self.indexer is not None:
            indexer = self.indexer

            def indexer_weights_proj() -> torch.Tensor:
                # ReplicatedLinear returns (output, bias); bias is None.
                weights, _ = indexer.weights_proj(hidden_states)
                return weights

            def indexer_compressor_kv_score() -> torch.Tensor:
                return torch.mm(
                    hidden_states,
                    indexer.compressor.fused_wkv_wgate.weight.T,
                    out_dtype=torch.float32,
                )

            aux_fns[1] = indexer_weights_proj
            aux_fns[2] = indexer_compressor_kv_score

        qr_kv, (kv_score, indexer_weights, indexer_kv_score) = execute_in_parallel(
            lambda: self._fused_wqa_wkv_gemm(hidden_states),
            aux_fns,
            self.ln_events[0],
            self.ln_events[1:4],
            aux_streams,
            enable=use_compilation_safe_attn_gemm_overlap(hidden_states.shape[0]),
        )

        return qr_kv, kv_score, indexer_kv_score, indexer_weights

    def attention_impl(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        kv_score: torch.Tensor | None,
        indexer_kv_score: torch.Tensor | None,
        indexer_weights: torch.Tensor | None,
        positions: torch.Tensor,
        out: torch.Tensor,
        qr_scale: torch.Tensor | None = None,
    ) -> None:
        """Body of the opaque ``deepseek_v4_attention`` custom op."""
        if not isinstance(get_forward_context().attn_metadata, dict):
            # Profile/dummy runs may intentionally omit attention metadata
            # to skip attention while still exercising the surrounding model.
            out.zero_()
            return

        self._prepare_and_attn_fn(
            hidden_states,
            qr,
            kv,
            qr_scale,
            kv_score,
            indexer_kv_score,
            indexer_weights,
            positions,
            out,
        )

    @eager_break_during_capture
    def _sparse_indexer_and_attn(
        self,
        hidden_states: torch.Tensor,
        index_q: torch.Tensor | None,
        index_q_scale: torch.Tensor | None,
        index_weights: torch.Tensor | None,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        self._run_indexer_op(hidden_states, index_q, index_q_scale, index_weights)

        # MLA attention writes into the pre-allocated `out` buffer
        # ([num_tokens, padded_heads, head_dim]).
        self.forward_mqa(q, kv, positions, out)

    def _run_indexer_op(
        self,
        hidden_states: torch.Tensor,
        index_q: torch.Tensor | None,
        index_q_scale: torch.Tensor | None,
        index_weights: torch.Tensor | None,
    ) -> None:
        if self.indexer is None or index_q is None:
            return
        assert index_weights is not None
        q_quant = (index_q, index_q_scale) if index_q_scale is not None else index_q
        self.indexer.indexer_op(
            hidden_states,
            q_quant,
            None,
            index_weights,
        )

    def _fused_qnorm_rope_kv_insert(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: (
            dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None
        ),
    ) -> torch.Tensor:
        if not isinstance(attn_metadata, dict):
            # Profile run: kernel doesn't fire; produce a padded tensor so
            # downstream FlashMLA gets the right shape.
            if self.n_local_heads < self.padded_heads:
                return F.pad(
                    q,
                    (0, 0, 0, self.padded_heads - self.n_local_heads),
                    value=0.0,
                )
            return q

        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_kv_cache = self.swa_cache_layer.kv_cache
        # The fused insert ops require int64 position_ids; the runner's positions
        # buffer is already int64, so no cast is needed.
        assert positions.dtype == torch.int64
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        cache_dtype = swa_kv_cache.dtype

        # kv is unchanged; attention reads kv solely via swa_kv_cache.
        if cache_dtype == torch.uint8 and self.kv_cache_dtype == "int8_ds_mla":
            # int8_ds_mla paged path: 512 signed-int8 bytes + fp32 row scale
            # per token (528B stride). MUST NOT fall through to the csrc
            # fp8_ds_mla writer below: that kernel writes the 576/584-byte
            # UE8M0 layout with 16B uint4 stores, which both corrupts the
            # packed 528B pages and faults with "CUDA error: misaligned
            # address" when the packed block stride is not a 16B multiple.
            return fused_qnorm_rope_kv_int8_ds_mla_insert(
                q,
                kv,
                swa_kv_cache,
                swa_metadata.slot_mapping,
                positions,
                cos_sin_cache,
                self.padded_heads,
                self.eps,
                swa_metadata.block_size,
            )
        if cache_dtype == torch.uint8:
            # fp8_ds_mla UE8M0 paged path. Horizontally fused:
            #   Q side:  per-head RMSNorm (no weight) + GPT-J RoPE, zero-filling
            #            the padding head slots; the kernel allocates and returns
            #            the padded q tensor.
            #   KV side: GPT-J RoPE + UE8M0 FP8 quant + paged cache insert.
            swa_kv_cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)
            return torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
                q,
                kv,
                swa_kv_cache_2d,
                swa_metadata.slot_mapping,
                positions,
                cos_sin_cache,
                self.padded_heads,
                self.eps,
                swa_metadata.block_size,
            )

        # Plain-row path: the [num_blocks, block_size, 512] cache stores the KV
        # row in its element dtype (no Q padding). bf16 rewrites q in place;
        # per-tensor fp8 writes a separately-allocated fp8 q and quantizes the
        # KV row.
        block_size = swa_metadata.block_size
        assert swa_kv_cache.shape[1:] == (block_size, self.head_dim)
        swa_kv_cache_3d = swa_kv_cache
        if cache_dtype == torch.bfloat16:
            torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
                q,
                kv,
                swa_kv_cache_3d,
                swa_metadata.slot_mapping,
                positions,
                cos_sin_cache,
                self.eps,
                block_size,
            )
            return q

        # per-tensor fp8 (torch.float8_e4m3fn)
        q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert(
            q,
            kv,
            q_fp8,
            swa_kv_cache_3d,
            swa_metadata.slot_mapping,
            positions,
            cos_sin_cache,
            self._flashinfer_fp8_kv_scale,
            self._flashinfer_fp8_q_scale_inv,
            self.eps,
            block_size,
        )
        return q_fp8

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # [B, H=1, N, C] -> [B, N, C]
        self.kv_cache = kv_cache.squeeze(1)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.backend_cls

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        if (
            self.compress_ratio <= 1
        ):  # SWA part. Allocated separately as DeepseekV4SWACache.
            return None
        # fp8_ds_mla is a UE8M0 block-scaled uint8 layout and needs 576B
        # alignment; int8_ds_mla packs a 528B page; plain bf16 / per-tensor fp8
        # rows use natural element-size pages.
        uses_fp8_ds_mla_layout = self.kv_cache_dtype == "fp8_ds_mla"
        uses_int8_ds_mla_layout = self.kv_cache_dtype == "int8_ds_mla"
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8 if uses_fp8_ds_mla_layout else self.kv_cache_torch_dtype,
            tokens_per_state=self.compress_ratio,
            cache_dtype_str=self.kv_cache_dtype,
            alignment=576
            if uses_fp8_ds_mla_layout
            else (528 if uses_int8_ds_mla_layout else 512),
            model_version="deepseek_v4",
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
            # DeepseekV4 fp8_ds_mla: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B
            # per token. int8_ds_mla: 512B signed-int8 row + fp32 row scale +
            # 12B pad = 528B per token. head_size stays semantic (512).
            state_content_bytes=584
            if uses_fp8_ds_mla_layout
            else (528 if uses_int8_ds_mla_layout else None),
        )


class DeepseekV4IndexerCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
        compress_ratio: int = 1,
    ):
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        self.compress_ratio = compress_ratio
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # [B, H=1, N, C] -> [B, N, C]
        self.kv_cache = kv_cache.squeeze(1)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # head_dim already carries the fp8 scale padding
        # tokens_per_state=1 for V3.2, >1 for DeepseekV4; same cache layout.
        uses_fp8_ds_mla_layout = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            tokens_per_state=self.compress_ratio,
            # 576B for FlashMLA packing; 512B for FlashInfer sparse (#44577).
            alignment=576 if uses_fp8_ds_mla_layout else 512,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekV4IndexerBackend


class DeepseekV4Indexer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        compress_ratio: int = 1,
        prefix: str = "",
        aux_stream: torch.cuda.Stream | None = None,
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        # self.indexer_cfg = config.attn_module_list_cfg[0]["attn_index"]
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.q_lora_rank = q_lora_rank  # 1536
        self.compress_ratio = compress_ratio
        self.use_fp4_kv = dsa_indexer_uses_fp4(vllm_config)
        # Log the TRUE payload format (mxfp4/fp8/int8) and query mode so
        # benchmark rows can be validated from the logs.
        from vllm.models.deepseek_v4.nvidia_imma.triton_kernels import (
            indexer_cache_is_int8,
            indexer_imma_enabled,
        )

        if self.use_fp4_kv:
            _indexer_cache_fmt = "mxfp4"
            _indexer_query_mode = "mxfp4"
        else:
            _indexer_cache_fmt = "int8" if indexer_cache_is_int8() else "fp8"
            _indexer_query_mode = "int8-imma" if indexer_imma_enabled() else "fp8"
        self._indexer_query_is_int8 = indexer_imma_enabled() and not self.use_fp4_kv
        self._indexer_use_cutedsl = has_cutedsl()
        self._indexer_is_xpu = current_platform.is_xpu()
        self._indexer_fp8_dtype = current_platform.fp8_dtype()
        self._indexer_supports_fp8e4nv_in_triton = _supports_fp8e4nv_in_triton()
        logger.info_once(
            "Lightning Indexer: cache_payload=%s query=%s",
            _indexer_cache_fmt,
            _indexer_query_mode,
        )

        # no tensor parallel, just replicated
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.weights_proj = ReplicatedLinear(
            hidden_size,
            self.n_head,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
        )
        self.softmax_scale = self.head_dim**-0.5

        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128  # TODO: get from config
        self.topk_indices_buffer = topk_indices_buffer

        self.max_model_len = (
            vllm_config.model_config.max_model_len // self.compress_ratio
        )
        self.prefix = prefix

        # Rows reserved in the shared workspace for the prefill K-gather;
        # matches the metadata builder's chunk-planner budget exactly.
        self.max_total_seq_len = get_max_prefill_buffer_size(
            vllm_config, self.compress_ratio
        )

        assert cache_config is not None, "Deepseek V4 indexer requires cache_config"
        if self.use_fp4_kv:
            # MXFP4 stores two values per byte plus one UE8M0 byte per 32 values.
            # head_dim bytes = 64 packed values + 4 UE8M0 scales = 68.
            k_cache_head_dim = self.head_dim // 2 + self.head_dim // MXFP4_BLOCK_SIZE
        else:
            # NOTE(yifan): FP8 indexer cache uses the same layout as V3.2:
            # head_dim bytes = 128 fp8 + 4 fp32 scale = 132.
            k_cache_head_dim = (
                self.head_dim + self.head_dim // self.quant_block_size * 4
            )
        self.k_cache = DeepseekV4IndexerCache(
            head_dim=k_cache_head_dim,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
            compress_ratio=self.compress_ratio,
        )
        self.compressor = DeepseekCompressor(
            vllm_config=vllm_config,
            compress_ratio=self.compress_ratio,
            hidden_size=hidden_size,
            head_dim=self.head_dim,
            rotate=True,
            prefix=f"{prefix}.compressor",
            k_cache_prefix=self.k_cache.prefix,
            use_fp4_cache=self.use_fp4_kv,
        )

        self.indexer_op = SparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            skip_k_cache_insert=True,
            use_fp4_cache=self.use_fp4_kv,
            compress_ratio=self.compress_ratio,
        )

        # None on ROCm: maybe_execute_in_parallel falls back to sequential.
        self.aux_stream = aux_stream
        self.ln_events: list[torch.cuda.Event] = [
            torch.cuda.Event(),
            torch.cuda.Event(),
        ]

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        compressed_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
        qr_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        compressor = self.compressor

        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, dict):
            indexer_metadata = cast(Any, attn_metadata[self.k_cache.prefix])
            if (
                indexer_metadata.max_seq_len // self.compress_ratio <= self.topk_tokens
                and not torch.cuda.is_current_stream_capturing()
            ):
                # candidates num smaller than topk, every candidate is selected
                # but we still need to build k cache
                compressor(compressed_kv_score, positions, rotary_emb)
                assert self.topk_indices_buffer is not None
                num_tokens = (
                    indexer_metadata.num_decode_tokens
                    + indexer_metadata.num_prefill_tokens
                )
                if num_tokens > 0:
                    _fill_short_context_topk_indices[(num_tokens,)](
                        self.topk_indices_buffer,
                        positions,
                        TOP_K=self.topk_tokens,
                        COMPRESS_RATIO=self.compress_ratio,
                        PADDED_TOP_K=triton.next_power_of_2(self.topk_tokens),
                        num_warps=8,
                    )
                return None, None, None

        def wq_b_and_q_quant():
            # INT8 IMMA indexer query: emit a symmetric INT8 query (scale folded
            # into indexer_weights) so the logits run as s8 x s8 integer-MMA.
            q = self._wq_b_proj(qr, qr_scale)
            q = q.view(-1, self.n_head, self.head_dim)
            return fused_indexer_q_rope_quant(
                positions,
                q,
                rotary_emb.cos_sin_cache,
                indexer_weights,
                self.softmax_scale,
                self.n_head**-0.5,
                use_fp4=self.use_fp4_kv,
                q_is_int8=self._indexer_query_is_int8,
                use_cutedsl=self._indexer_use_cutedsl,
                is_xpu=self._indexer_is_xpu,
                fp8_dtype=self._indexer_fp8_dtype,
                supports_fp8e4nv_in_triton=(self._indexer_supports_fp8e4nv_in_triton),
            )

        # compressor returns None and writes K to the indexer KV cache; the
        # join orders that write before indexer_op (skip_k_cache_insert=True).
        (q_quant, weights), _ = maybe_execute_in_parallel(
            wq_b_and_q_quant,
            lambda: compressor(compressed_kv_score, positions, rotary_emb),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
        if isinstance(q_quant, tuple):
            q, q_scale = q_quant
        else:
            q, q_scale = q_quant, None
        return q, q_scale, weights

    def _wq_b_proj(
        self, qr: torch.Tensor, qr_scale: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Project the q-lora input to the indexer query heads.

        Same bypass as ``DeepseekV4Attention._wq_b_proj``: on ROCm the fused
        norm path hands over pre-quantized fp8 ``qr`` with per-1x128 scales,
        so the linear's internal re-quantization must be skipped.
        """
        if qr_scale is None:
            # ReplicatedLinear returns (output, bias); bias is None.
            q, _ = self.wq_b(qr)
            return q
        from vllm.models.deepseek_v4.amd.rocm import (
            apply_pre_quantized_block_scaled_mm,
        )

        return apply_pre_quantized_block_scaled_mm(self.wq_b, qr, qr_scale)


@eager_break_during_capture
def deepseek_v4_attention(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    kv: torch.Tensor,
    qr_scale: torch.Tensor | None,
    kv_score: torch.Tensor | None,
    indexer_kv_score: torch.Tensor | None,
    indexer_weights: torch.Tensor | None,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Opaque DSV4 attention: q up-projection, indexer, compressor, KV-cache
    insert, and sparse MLA. Everything behind this op reads forward-context
    state (attention metadata, KV caches, the shared top-k indices buffer)
    at runtime, so it must never be traced into the compiled model body."""
    layer_name = _resolve_layer_name(layer_name)
    layer = get_forward_context().no_compile_layers[layer_name]
    layer.attention_impl(
        hidden_states,
        qr,
        kv,
        kv_score,
        indexer_kv_score,
        indexer_weights,
        positions,
        out,
        qr_scale,
    )


def deepseek_v4_attention_fake(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    kv: torch.Tensor,
    qr_scale: torch.Tensor | None,
    kv_score: torch.Tensor | None,
    indexer_kv_score: torch.Tensor | None,
    indexer_weights: torch.Tensor | None,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    return


direct_register_custom_op(
    op_name="deepseek_v4_attention",
    op_func=deepseek_v4_attention,
    mutates_args=["out"],
    fake_impl=deepseek_v4_attention_fake,
)
