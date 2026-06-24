# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DeepseekV4 MLA Attention Layer
"""

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
import os
import time
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DeepseekV2Config, DeepseekV3Config

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
)
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.models.deepseek_v4.common.ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
    dequantize_combined_sparse_mla_decode_kv,
    dequantize_global_slots_k_cache,
    fused_indexer_q_rope_quant,
    fused_inv_rope_fp8_quant,
    fused_q_kv_rmsnorm,
    sparse_prefill_combined_topk_size,
)
from vllm.models.deepseek_v4.common.ops.fp8_einsum import (
    deepseek_v4_sm12x_fp8_einsum,
)
from vllm.utils.deep_gemm import fp8_einsum, use_deepgemm_sm12x_kernels
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    rocm_inv_rope_einsum,
    rocm_sparse_attn_prefill,
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
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.input_quant_fp8 import (
    QuantFP8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
)
from vllm.model_executor.layers.deepseek_v4_triton_kernels import (
    decode_sparse_attention_triton,
)
from vllm.model_executor.layers.rotary_embedding.common import (
    rotate_gptj,
    rotate_neox,
)
from vllm.models.deepseek_v4.compressor import DeepseekCompressor
from vllm.platforms import current_platform
from vllm.utils.multi_stream_utils import (
    execute_in_parallel,
    maybe_execute_in_parallel,
)
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    DeepseekV4FlashMLASparseBackend,
    FlashMLASparseBackend,
    FlashMLASparseMetadata,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV4IndexerBackend,
    get_max_prefill_buffer_size,
)
from vllm.v1.attention.backends.mla.sparse_mla_env import (
    disable_triton_sparse_mla_cudagraphs_if_enabled,
    is_triton_sparse_mla_enabled,
    is_triton_sparse_mla_enabled_for_platform,
    triton_sparse_mla_matmul_decode_enabled,
    triton_sparse_mla_matmul_prefill_enabled,
    triton_sparse_mla_query_chunk_size,
    triton_sparse_mla_topk_chunk_size,
)
from vllm.v1.attention.backends.mla.sparse_mla_kernels import (
    accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead,
    accumulate_fp8ds_paged_sparse_mla_attention_chunk_multihead,
    accumulate_indexed_sparse_mla_attention_chunk,
    accumulate_indexed_sparse_mla_attention_chunk_multihead,
    build_combined_sparse_mla_decode_valid_mask,
    finish_sparse_mla_attention_with_sink,
    finish_two_sparse_mla_attention_states_with_sink,
    fp8ds_global_paged_sparse_mla_attention_with_sink_multihead,
    fp8ds_paged_sparse_mla_attention_with_sink_multihead,
    matmul_sparse_mla_attention_with_sink,
    sparse_mla_decode_head_block_size,
)
from vllm.v1.attention.backends.mla.ampere_flashmla_decode import (
    ampere_flashmla_decode_enabled,
    ampere_flashmla_decode_min_tokens,
    ampere_flashmla_sparse_decode,
    ampere_flashmla_supports,
)
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)


def _dsv4_debug_timings_enabled() -> bool:
    return os.getenv("VLLM_DEEPSEEK_V4_DEBUG_TIMINGS", "0") == "1"


def _dsv4_mtp_debug_indexer_enabled() -> bool:
    return os.getenv("VLLM_DSV4_MTP_DEBUG_INDEXER", "0") != "0"


def _dsv4_mtp_debug_verify_enabled() -> bool:
    return os.getenv("VLLM_DSV4_MTP_DEBUG_VERIFY", "0") != "0"


_DSV4_MTP_DECODE_DEBUG_LOGS = 0


def _dsv4_mtp_decode_debug_max_decodes() -> int:
    try:
        return int(os.getenv("VLLM_DSV4_MTP_DEBUG_MAX_DECODES", "1"))
    except ValueError:
        return 1


def _dsv4_mtp_decode_debug_budget() -> int:
    try:
        return int(os.getenv("VLLM_DSV4_MTP_DEBUG_DECODE_LOGS", "240"))
    except ValueError:
        return 240


def _dsv4_mtp_debug_decode_should_log(num_decodes: int, num_decode_tokens: int) -> bool:
    global _DSV4_MTP_DECODE_DEBUG_LOGS
    if not _dsv4_mtp_debug_verify_enabled():
        return False
    if num_decode_tokens <= 0 or num_decode_tokens == num_decodes:
        return False
    if num_decodes > _dsv4_mtp_decode_debug_max_decodes():
        return False
    if _DSV4_MTP_DECODE_DEBUG_LOGS >= _dsv4_mtp_decode_debug_budget():
        return False
    _DSV4_MTP_DECODE_DEBUG_LOGS += 1
    return True


def _dsv4_tensor_summary(name: str, tensor: object | None) -> str:
    if tensor is None:
        return f"{name}=None"
    if isinstance(tensor, dict):
        return f"{name}=dict(keys={sorted(str(key) for key in tensor.keys())})"
    if not isinstance(tensor, torch.Tensor):
        return f"{name}=type({type(tensor).__name__})"
    try:
        if tensor.numel() == 0:
            return f"{name}=shape{tuple(tensor.shape)} empty dtype={tensor.dtype}"
        view = tensor.detach()
        if view.is_cuda:
            torch.cuda.synchronize(view.device)
        if view.dtype == torch.bool:
            return (
                f"{name}=shape{tuple(view.shape)} dtype={view.dtype} "
                f"true={int(view.sum().item())}"
            )
        return (
            f"{name}=shape{tuple(view.shape)} dtype={view.dtype} "
            f"min={int(view.min().item())} max={int(view.max().item())}"
        )
    except Exception as exc:
        shape = getattr(tensor, "shape", "?")
        dtype = getattr(tensor, "dtype", "?")
        return f"{name}=shape{tuple(shape)} dtype={dtype} error={exc!r}"


def _dsv4_tensor_values(name: str, tensor: object | None, limit: int = 8) -> str:
    if tensor is None:
        return f"{name}=None"
    if not isinstance(tensor, torch.Tensor):
        return f"{name}=type({type(tensor).__name__})"
    try:
        view = tensor.detach()
        if view.is_cuda:
            torch.cuda.synchronize(view.device)
        flat = view.reshape(-1)[:limit].cpu().tolist()
        return f"{name}=shape{tuple(view.shape)} dtype={view.dtype} values={flat}"
    except Exception as exc:
        shape = getattr(tensor, "shape", "?")
        dtype = getattr(tensor, "dtype", "?")
        return f"{name}=shape{tuple(shape)} dtype={dtype} error={exc!r}"


def _dsv4_tensor_row_stats(name: str, tensor: object | None, rows: int = 2) -> str:
    if tensor is None:
        return f"{name}=None"
    if not isinstance(tensor, torch.Tensor):
        return f"{name}=type({type(tensor).__name__})"
    try:
        view = tensor.detach()
        if view.numel() == 0:
            return f"{name}=shape{tuple(view.shape)} empty dtype={view.dtype}"
        if view.is_cuda:
            torch.cuda.synchronize(view.device)
        row_view = view.reshape(view.shape[0], -1)[:rows].float()
        norms = torch.linalg.vector_norm(row_view, dim=1).cpu().tolist()
        mins = row_view.min(dim=1).values.cpu().tolist()
        maxs = row_view.max(dim=1).values.cpu().tolist()
        finite = torch.isfinite(row_view).all(dim=1).cpu().tolist()
        return (
            f"{name}=shape{tuple(view.shape)} dtype={view.dtype} "
            f"norms={norms} min={mins} max={maxs} finite={finite}"
        )
    except Exception as exc:
        shape = getattr(tensor, "shape", "?")
        dtype = getattr(tensor, "dtype", "?")
        return f"{name}=shape{tuple(shape)} dtype={dtype} error={exc!r}"


def _dsv4_debug_layer_id(module: object) -> object:
    layer_id = getattr(module, "layer_id", None)
    if layer_id is not None:
        return int(layer_id)
    return getattr(module, "prefix", type(module).__name__)


@contextmanager
def _dsv4_debug_timing(label: str, **fields: object):
    if not _dsv4_debug_timings_enabled():
        yield
        return

    start = time.perf_counter()
    logger.warning("DSV4_DEBUG_START %s %s", label, fields)
    if torch.cuda.is_available():
        try:
            torch.cuda.nvtx.range_push(label)
        except Exception:
            pass
    try:
        yield
    finally:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass
        logger.warning(
            "DSV4_DEBUG_END %s %.6fs %s",
            label,
            time.perf_counter() - start,
            fields,
        )


def _sparse_mla_prefill_workspace_bounds(
    seq_lens_cpu: torch.Tensor,
    gather_lens_cpu: torch.Tensor,
    compress_ratio: int,
    swa_only: bool,
) -> tuple[int, int]:
    if seq_lens_cpu.numel() == 0:
        return 0, 0

    max_gather_len = int(gather_lens_cpu.max().item())
    if swa_only:
        return 0, max_gather_len

    compressed_region_size = int((seq_lens_cpu // compress_ratio).max().item())
    return compressed_region_size, compressed_region_size + max_gather_len


def _sparse_mla_prefill_gather_len_upper_bound(
    *,
    max_model_len: int,
    max_num_batched_tokens: int,
    window_size: int,
) -> tuple[int, int]:
    max_query_chunk_tokens = max(1, min(max_model_len, max_num_batched_tokens))
    max_prefix_len = max(max_model_len - max_query_chunk_tokens, 0)
    max_gather_len = max_query_chunk_tokens + min(
        max_prefix_len,
        max(window_size - 1, 0),
    )
    return max_query_chunk_tokens, max_gather_len


def _deepseek_v4_fp8_einsum_config(
    capability_major: int,
) -> tuple[tuple[int, int, int], bool]:
    if capability_major == 10:
        return (1, 1, 128), True
    return (1, 128, 128), False


def _use_deepseek_v4_sm12x_triton_fp8_einsum(
    equation: str,
    recipe: list[int],
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
) -> bool:
    capability = current_platform.get_device_capability()
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    supported_scale_dtypes = (torch.float32, e8m0_dtype)
    has_e8m0_scale = e8m0_dtype is not None and (
        a_scale.dtype == e8m0_dtype or b_scale.dtype == e8m0_dtype
    )
    return (
        capability is not None
        # Ampere/Ada (sm_8x) routes through the same portable Triton kernel
        # as SM12x — DeepGEMM is Hopper/Blackwell-only and the SM12x Triton
        # kernel is sm_80+ compatible per PR #40991 author's note.
        and capability.major in (8, 12)
        and equation == "bhr,hdr->bhd"
        and tuple(recipe) == (1, 128, 128)
        and a_scale.dtype in supported_scale_dtypes
        and b_scale.dtype in supported_scale_dtypes
        and (not use_deepgemm_sm12x_kernels() or has_e8m0_scale)
    )


def _allocate_deepseek_v4_wo_a_output(
    num_tokens: int,
    num_groups: int,
    output_rank: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    shape = (num_tokens, num_groups, output_rank)
    if torch.compiler.is_compiling():
        # Workspace growth can call torch.accelerator.empty_cache(), which
        # Dynamo intentionally refuses to trace. During compilation this is a
        # normal graph allocation, matching the o_padded allocation above.
        return torch.empty(shape, dtype=dtype, device=device)

    (output,) = current_workspace_manager().get_simultaneous(
        (shape, dtype),
    )
    return output


# Prefill is processed in fixed-size chunks; this bounds the bf16 kv-gather
# workspace allocated at _forward_prefill (and the matching profile-time
# reservation in attention_impl's dummy-run branch).
PREFILL_CHUNK_SIZE = 4
_DEFAULT_SPARSE_MLA_TOPK_TOKENS = 2048


@dataclass
class DeepseekV4MLAModules:
    """Modules used in DeepseekV4 MLA."""

    vllm_config: VllmConfig
    fused_wqa_wkv: torch.nn.Module
    q_norm: torch.nn.Module
    wq_b: torch.nn.Module
    kv_norm: torch.nn.Module
    wo_a: torch.nn.Module
    wo_b: torch.nn.Module
    attn_sink: torch.nn.Module
    rotary_emb: torch.nn.Module
    indexer: torch.nn.Module | None
    indexer_rotary_emb: torch.nn.Module
    topk_indices_buffer: torch.Tensor | None
    aux_stream_list: list[torch.cuda.Stream] | None = None


# --8<-- [start:multi_head_latent_attention]
@PluggableLayer.register("deepseek_v4_multi_head_latent_attention")
class DeepseekV4MultiHeadLatentAttentionWrapper(PluggableLayer):
    """Pluggable MLA layer which allows OOT backends to add
    custom implementations of the outer MLA layer (including rope & o_proj).
    Note that currently oot platforms can still use CustomOp.register_oot to
    replace MLA layer entirely, although we use PluggableLayer to register
    this layer now.

    This class takes positions and hidden_states as input.
    The input tensors can either contain prefill tokens or decode tokens.
    The class does the following:

    1. MLA Preprocess.
    2. Perform multi-head attention to prefill tokens and
       multi-query attention to decode tokens separately.
    3. Return the output tensor.
    """

    # --8<-- [end:multi_head_latent_attention]

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        o_lora_rank: int | None,
        mla_modules: DeepseekV4MLAModules,
        window_size: int,
        compress_ratio: int | None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_local_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale

        # FlashMLA sparse kernel only supports 64 or 128 heads; pad up to the
        # next supported size. Must match DeepseekV4MLAAttention.padded_heads.
        if num_heads <= 64:
            self.padded_heads = 64
        elif num_heads <= 128:
            self.padded_heads = 128
        else:
            raise ValueError(
                f"DeepseekV4 attention does not support {num_heads} heads "
                "(must be <= 128)."
            )

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.window_size = window_size
        self.compress_ratio = compress_ratio if compress_ratio is not None else 1
        self.prefix = prefix

        disable_triton_sparse_mla_cudagraphs_if_enabled(mla_modules.vllm_config)

        # Extract config from vllm_config
        config = mla_modules.vllm_config.model_config.hf_config
        tp_size = get_tensor_model_parallel_world_size()

        # DeepseekV4-specific attributes (num_heads is already TP-adjusted)
        self.eps = config.rms_norm_eps
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = head_dim - self.rope_head_dim
        self.n_local_groups = config.o_groups // tp_size
        self.o_lora_rank = config.o_lora_rank

        # Store projection modules
        self.fused_wqa_wkv = mla_modules.fused_wqa_wkv
        self.q_norm = mla_modules.q_norm
        self.wq_b = mla_modules.wq_b

        self.kv_norm = mla_modules.kv_norm
        self.wo_a = mla_modules.wo_a

        self._wo_a_act_quant = QuantFP8(
            static=False,
            group_shape=GroupShape(1, 128),
            use_ue8m0=True,
        )
        # Bypass packed-for-deepgemm path — we need FP32 scales (not packed
        # INT32) so fp8_einsum can handle layout transform internally.
        self._wo_a_act_quant.use_deep_gemm_supported = False
        self.wo_b = mla_modules.wo_b

        # Pick fp8_einsum recipe based on GPU arch:
        # SM90/SM120: FP32 block scales stay [g, r/128, d/128].
        # SM100: INT32 packed scales become [g, r, ...].
        cap = current_platform.get_device_capability()
        assert cap is not None, "DeepseekV4 attention requires a CUDA device"
        self._einsum_recipe, self._tma_aligned_scales = _deepseek_v4_fp8_einsum_config(
            cap.major
        )

        self.rotary_emb = mla_modules.rotary_emb
        self.indexer_rotary_emb = mla_modules.indexer_rotary_emb
        self.topk_indices_buffer = mla_modules.topk_indices_buffer

        self.indexer = mla_modules.indexer

        # Per-head RMS normalization for Q (no learnable weights)
        self.q_head_norm = RMSNorm(head_dim, eps=self.eps, has_weight=False)

        # TODO(yifan): currently hardcoded for FP8 sparse, make it more generic
        head_bytes = (
            self.nope_head_dim  # 448 fp8 NoPE
            + self.rope_head_dim * 2  # 64 bf16 RoPE
            + self.nope_head_dim // 64  # 7B scale factors
            + 1  # 1B pad
        )

        # Will be None on ROCm for now.
        self.aux_stream_list = mla_modules.aux_stream_list
        # [0]: GEMM start / post-GEMM event0. [1..3]: GEMM done events;
        # [1] doubles as post-GEMM event1. Reuse is safe: GEMM fully joins
        # before post-GEMM starts.
        self.ln_events = [torch.cuda.Event() for _ in range(4)]

        assert cache_config is not None, "DeepseekV4 attention requires cache_config"
        self.swa_cache_layer = DeepseekV4SWACache(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=torch.uint8,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
        )

        self.mla_attn = DeepseekV4MLAAttention(
            num_heads=self.n_local_heads,
            head_dim=self.head_dim,
            scale=self.scale,
            qk_nope_head_dim=self.nope_head_dim,
            qk_rope_head_dim=self.rope_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            compress_ratio=self.compress_ratio,
            window_size=self.window_size,
            head_bytes=head_bytes,
            swa_cache_layer=self.swa_cache_layer,
            attn_sink=mla_modules.attn_sink,  # already padded with -inf
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            indexer=self.indexer,
            topk_indices_buffer=self.topk_indices_buffer,
        )
        # Register this layer in the compilation config's static forward context
        # This allows the custom op to retrieve the layer during execution
        compilation_config = mla_modules.vllm_config.compilation_config
        # HACK
        self.layer_name = prefix + ".deepseek_v4_multi_head_latent_attention"
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self

        # Create the compressor for layers with compress_ratio > 1; after
        # creating the DeepseekV4MLAAttention layer to get its cache.
        self.compressor = None
        if self.compress_ratio > 1:
            self.compressor = DeepseekCompressor(
                vllm_config=mla_modules.vllm_config,
                compress_ratio=self.compress_ratio,
                hidden_size=self.hidden_size,
                head_dim=self.head_dim,
                rotate=True,
                prefix=f"{prefix}.compressor",
                k_cache_prefix=self.mla_attn.prefix,
            )

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

        # Attention (inside custom op for torch.compile boundary)
        torch.ops.vllm.deepseek_v4_attention(
            hidden_states,
            positions,
            o_padded,
            self.layer_name,
        )
        o = o_padded[:, : self.n_local_heads, :]

        # Keep ROCm and AOT INT8-dequanted WO_A on the BF16 reference path.
        # The dsv4_int checkpoint stores WO_A as INT8 and dequants it once to
        # BF16 during loading, so it must not be sent through fp8_einsum.
        if current_platform.is_rocm() or getattr(
            self.wo_a, "_dsv4_int_dequanted", False
        ):
            if (
                current_platform.is_cuda()
                and getattr(self.wo_a, "_dsv4_int_dequanted", False)
                and o.is_cuda
            ):
                z = torch.empty(
                    (num_tokens, self.n_local_groups, self.o_lora_rank),
                    dtype=torch.bfloat16,
                    device=o.device,
                )
                torch.ops.vllm.deepseek_v4_inv_rope_woa(
                    o,
                    positions,
                    self.rotary_emb.cos_sin_cache,
                    self.wo_a.weight,
                    z,
                    self.rope_head_dim,
                    self.n_local_groups,
                    self.o_lora_rank,
                    bool(getattr(self.rotary_emb, "is_neox_style", False)),
                )
                return self.wo_b(z.flatten(1))
            z = rocm_inv_rope_einsum(
                self.rotary_emb,
                o,
                positions,
                self.rope_head_dim,
                self.n_local_groups,
                self.o_lora_rank,
                self.wo_a,
            )
            return self.wo_b(z.flatten(1))

        # O projection: inverse RoPE + FP8 quant + einsum + wo_b
        o_fp8, o_scale = fused_inv_rope_fp8_quant(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            tma_aligned_scales=self._tma_aligned_scales,
        )

        wo_a_fp8 = self.wo_a.weight
        wo_a_scale = self.wo_a.weight_scale_inv

        z = _allocate_deepseek_v4_wo_a_output(
            num_tokens,
            self.n_local_groups,
            self.o_lora_rank,
            torch.bfloat16,
            hidden_states.device,
        )
        torch.ops.vllm.deepseek_v4_fp8_einsum(
            o_fp8,
            o_scale,
            wo_a_fp8,
            wo_a_scale,
            z,
            "bhr,hdr->bhd",
            list(self._einsum_recipe),
        )

        return self.wo_b(z.flatten(1))

    def attn_gemm_parallel_execute(self, hidden_states) -> tuple[Any, ...]:
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

        def fused_wqa_wkv() -> torch.Tensor:
            # MergedColumnParallelLinear returns (output, bias); bias is None.
            qr_kv, _ = self.fused_wqa_wkv(hidden_states)
            return qr_kv

        qr_kv, (kv_score, indexer_weights, indexer_kv_score) = execute_in_parallel(
            fused_wqa_wkv,
            aux_fns,
            self.ln_events[0],
            self.ln_events[1:4],
            aux_streams,
            enable=hidden_states.shape[0]
            <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD,
        )

        return qr_kv, kv_score, indexer_kv_score, indexer_weights

    def attention_impl(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,  # [num_tokens, padded_heads, head_dim], written in place
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        with _dsv4_debug_timing(
            "attention_gemm_parallel_execute",
            layer_id=_dsv4_debug_layer_id(self),
            hidden_shape=tuple(hidden_states.shape),
        ):
            qr_kv, kv_score, indexer_kv_score, indexer_weights = (
                self.attn_gemm_parallel_execute(hidden_states)
            )

        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        with _dsv4_debug_timing(
            "attention_q_kv_rmsnorm",
            layer_id=_dsv4_debug_layer_id(self),
            qr_shape=tuple(qr.shape),
            kv_shape=tuple(kv.shape),
        ):
            qr, kv = fused_q_kv_rmsnorm(
                qr,
                kv,
                self.q_norm.weight.data,
                self.kv_norm.weight.data,
                self.eps,
            )

        # wq_b + kv_insert (+ MLA compressor when an indexer is present) ride
        # on the default stream so q stays on its consumer stream (mla_attn
        # downstream reads q on default). Indexer/compressor go on aux for
        # overlap with default's GEMM + cache write.
        if self.indexer is not None:
            aux_stream = (
                self.aux_stream_list[0] if self.aux_stream_list is not None else None
            )
            indexer = self.indexer
            # Local ref so the closure keeps a non-None type for mypy.
            assert self.compressor is not None
            compressor = self.compressor

            def wq_b_kv_insert_and_compress() -> torch.Tensor:
                q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
                self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
                compressor(kv_score, positions, self.rotary_emb)
                return q

            with _dsv4_debug_timing(
                "attention_wq_cache_indexer_parallel",
                layer_id=_dsv4_debug_layer_id(self),
                hidden_shape=tuple(hidden_states.shape),
            ):
                if _dsv4_mtp_debug_indexer_enabled():
                    forward_context = get_forward_context()
                    logger.warning(
                        "DSV4_MTP_INDEXER before layer=%s prefix=%s "
                        "compress_ratio=%s %s %s %s %s %s num_tokens=%s",
                        _dsv4_debug_layer_id(self),
                        self.prefix,
                        getattr(self, "compress_ratio", None),
                        _dsv4_tensor_summary("hidden_states", hidden_states),
                        _dsv4_tensor_summary("qr", qr),
                        _dsv4_tensor_summary("positions", positions),
                        _dsv4_tensor_summary(
                            "slot_mapping",
                            getattr(forward_context, "slot_mapping", None),
                        ),
                        _dsv4_tensor_summary(
                            "indexer_weights", indexer_weights
                        ),
                        getattr(forward_context, "num_tokens", None),
                    )
                    q = wq_b_kv_insert_and_compress()
                    _ = indexer(
                        hidden_states,
                        qr,
                        indexer_kv_score,
                        indexer_weights,
                        positions,
                        self.indexer_rotary_emb,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(hidden_states.device)
                    logger.warning(
                        "DSV4_MTP_INDEXER after layer=%s prefix=%s",
                        _dsv4_debug_layer_id(self),
                        self.prefix,
                    )
                else:
                    q, _ = maybe_execute_in_parallel(
                        wq_b_kv_insert_and_compress,
                        lambda: indexer(
                            hidden_states,
                            qr,
                            indexer_kv_score,
                            indexer_weights,
                            positions,
                            self.indexer_rotary_emb,
                        ),
                        self.ln_events[0],
                        self.ln_events[1],
                        aux_stream,
                    )
        elif self.compressor is not None:
            # wq_b + kv_insert on default, compressor on aux.
            aux_stream = (
                self.aux_stream_list[0] if self.aux_stream_list is not None else None
            )
            compressor = self.compressor

            def wq_b_kv_insert() -> torch.Tensor:
                q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
                self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
                return q

            with _dsv4_debug_timing(
                "attention_wq_cache_compressor_parallel",
                layer_id=_dsv4_debug_layer_id(self),
                hidden_shape=tuple(hidden_states.shape),
            ):
                q, _ = maybe_execute_in_parallel(
                    wq_b_kv_insert,
                    lambda: compressor(kv_score, positions, self.rotary_emb),
                    self.ln_events[0],
                    self.ln_events[1],
                    aux_stream,
                )
        else:
            # SWA-only layer: no compressor, no overlap.
            q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
            self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)

        # Handle dummy run (no metadata).
        if not isinstance(attn_metadata, dict):
            out.zero_()
            self.mla_attn._reserve_prefill_workspace()
            return

        # Pad q to FlashMLA-required head count (64 or 128)
        if self.n_local_heads < self.padded_heads:
            pad_size = self.padded_heads - self.n_local_heads
            q = F.pad(q, (0, 0, 0, pad_size), value=0.0)

        # MLA attention writes into the pre-allocated `out` buffer
        # ([num_tokens, padded_heads, head_dim]).
        with _dsv4_debug_timing(
            "attention_mla_attn",
            layer_id=_dsv4_debug_layer_id(self),
            q_shape=tuple(q.shape),
            kv_shape=tuple(kv.shape),
            positions_shape=tuple(positions.shape),
            out_shape=tuple(out.shape),
        ):
            self.mla_attn(q, kv, positions, output=out)

    def _fused_qnorm_rope_kv_insert(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: (
            dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None
        ),
    ) -> None:
        if not isinstance(attn_metadata, dict):
            return

        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_kv_cache = self.swa_cache_layer.kv_cache
        swa_kv_cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)

        # Horizontally fused:
        #   Q side:  q_head_norm (per-head RMSNorm, no weight) + GPT-J RoPE
        #   KV side: GPT-J RoPE + UE8M0 FP8 quant + paged cache insert
        # kv is unchanged; mla_attn reads kv solely via swa_kv_cache.
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q,
            kv,
            swa_kv_cache_2d,
            swa_metadata.slot_mapping,
            positions.to(torch.int64),
            self.rotary_emb.cos_sin_cache,
            self.eps,
            swa_metadata.block_size,
        )


@eager_break_during_capture
def deepseek_v4_attention(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self.attention_impl(hidden_states, positions, out)


def deepseek_v4_attention_fake(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    return None


direct_register_custom_op(
    op_name="deepseek_v4_attention",
    op_func=deepseek_v4_attention,
    mutates_args=["out"],
    fake_impl=deepseek_v4_attention_fake,
)


def deepseek_v4_inv_rope_woa(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a_weight: torch.Tensor,
    out: torch.Tensor,
    rope_head_dim: int,
    n_local_groups: int,
    o_lora_rank: int,
    is_neox_style: bool,
) -> None:
    head_size = o.shape[-1]
    nope_dim = head_size - rope_head_dim
    o_pass = o[..., :nope_dim] if nope_dim > 0 else None
    o_rot = o[..., nope_dim:]

    safe_positions = positions.clamp(0, cos_sin_cache.shape[0] - 1)
    cos_sin = cos_sin_cache[safe_positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    if is_neox_style:
        cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2)
        sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2)
        rotate_fn = rotate_neox
    else:
        cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
        sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)
        rotate_fn = rotate_gptj
    o_rot = (o_rot.float() * cos - rotate_fn(o_rot.float()) * sin).to(o.dtype)
    o_ref = torch.cat((o_pass, o_rot), dim=-1) if o_pass is not None else o_rot
    o_ref = o_ref.view(o.shape[0], n_local_groups, -1).to(torch.bfloat16)
    wo_a = wo_a_weight.view(n_local_groups, o_lora_rank, o_ref.shape[-1]).to(
        torch.bfloat16
    )
    out.copy_(torch.einsum("tgd,grd->tgr", o_ref, wo_a))


def deepseek_v4_inv_rope_woa_fake(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a_weight: torch.Tensor,
    out: torch.Tensor,
    rope_head_dim: int,
    n_local_groups: int,
    o_lora_rank: int,
    is_neox_style: bool,
) -> None:
    return None


direct_register_custom_op(
    op_name="deepseek_v4_inv_rope_woa",
    op_func=deepseek_v4_inv_rope_woa,
    mutates_args=["out"],
    fake_impl=deepseek_v4_inv_rope_woa_fake,
)


def deepseek_v4_fp8_einsum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> None:
    if equation == "bhr,hdr->bhd" and b.dim() == 2:
        num_groups = out.shape[1]
        out_rank = out.shape[2]
        hidden_size = a.shape[2]
        if b.shape[0] % out_rank != 0:
            raise RuntimeError(
                "DeepSeek V4 fp8 einsum weight rows must be divisible by "
                f"out_rank={out_rank}, got {b.shape[0]}"
            )
        b_groups = b.shape[0] // out_rank
        group_start = 0
        if b_groups != num_groups:
            if b_groups % num_groups != 0:
                raise RuntimeError(
                    "DeepSeek V4 fp8 einsum weight groups must match the "
                    "TP-local output groups or be an integer multiple of "
                    f"them, got weight_groups={b_groups}, "
                    f"output_groups={num_groups}"
                )
            group_partitions = b_groups // num_groups
            group_start = (
                get_tensor_model_parallel_rank() % group_partitions
            ) * num_groups
        b = b.view(b_groups, out_rank, hidden_size)
        if group_start != 0 or b_groups != num_groups:
            b = b.narrow(0, group_start, num_groups)

        if b_scale.dim() == 2:
            scale_mn = recipe[1]
            scale_k_pack = 4 if b_scale.dtype == torch.int32 else 1
            scale_k = recipe[2] * scale_k_pack
            scale_out_blocks = (out_rank + scale_mn - 1) // scale_mn
            scale_hidden_blocks = (hidden_size + scale_k - 1) // scale_k
            if b_scale.shape[0] % scale_out_blocks != 0:
                raise RuntimeError(
                    "DeepSeek V4 fp8 einsum scale rows must be divisible by "
                    f"scale_out_blocks={scale_out_blocks}, "
                    f"got {b_scale.shape[0]}"
                )
            scale_groups = b_scale.shape[0] // scale_out_blocks
            if scale_groups not in (num_groups, b_groups):
                raise RuntimeError(
                    "DeepSeek V4 fp8 einsum scale groups must match the "
                    "TP-local output groups or weight groups, got "
                    f"scale_groups={scale_groups}, output_groups={num_groups}, "
                    f"weight_groups={b_groups}"
                )
            b_scale = b_scale.view(
                scale_groups,
                scale_out_blocks,
                scale_hidden_blocks,
            )
            if scale_groups == b_groups and scale_groups != num_groups:
                b_scale = b_scale.narrow(0, group_start, num_groups)
        elif b_scale.dim() == 3 and b_scale.shape[0] == b_groups:
            if b_groups != num_groups:
                b_scale = b_scale.narrow(0, group_start, num_groups)

        if _use_deepseek_v4_sm12x_triton_fp8_einsum(equation, recipe, a_scale, b_scale):
            deepseek_v4_sm12x_fp8_einsum(a, a_scale, b, b_scale, out)
            return

    fp8_einsum(equation, (a, a_scale), (b, b_scale), out, recipe=tuple(recipe))


def deepseek_v4_fp8_einsum_fake(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> None:
    return None


direct_register_custom_op(
    op_name="deepseek_v4_fp8_einsum",
    op_func=deepseek_v4_fp8_einsum,
    mutates_args=["out"],
    fake_impl=deepseek_v4_fp8_einsum_fake,
)


class DeepseekV4MLAAttention(nn.Module, AttentionLayerBase):
    # FlashMLA FP8 sparse only supports 64 or 128 heads
    SUPPORTED_HEAD_COUNTS = (64, 128)

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        compress_ratio: int,
        window_size: int,
        head_bytes: int,
        swa_cache_layer: DeepseekV4SWACache,
        attn_sink: torch.Tensor,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        # Sparse MLA Args
        indexer: object | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream: torch.cuda.Stream | None = None,
        **extra_impl_args,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = 1
        self.head_dim = head_dim
        self.scale = scale
        self.window_size = window_size
        self.head_bytes = head_bytes
        self.compress_ratio = compress_ratio
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.nope_head_dim = qk_nope_head_dim
        self.rope_head_dim = qk_rope_head_dim
        self.indexer = indexer
        self.topk_indices_buffer = topk_indices_buffer

        self.prefix = prefix  # Alias for compatibility with compressor

        self.aux_stream = aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

        # Determine padded head count for FlashMLA
        if num_heads not in self.SUPPORTED_HEAD_COUNTS:
            if num_heads < 64:
                self.padded_heads = 64
            elif num_heads < 128:
                self.padded_heads = 128
            else:
                raise ValueError(
                    f"DeepseekV4MLAAttention does not support {num_heads} heads. "
                    f"Supported: <= 128 (will be padded to 64 or 128)"
                )
        else:
            self.padded_heads = num_heads

        # Store attention sink
        assert attn_sink is not None
        self.attn_sink: torch.Tensor = attn_sink
        # Store SWA cache
        assert swa_cache_layer is not None
        self.swa_cache_layer: DeepseekV4SWACache = swa_cache_layer

        # Get vllm config for cache setup
        vllm_config = get_current_vllm_config()
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        # DeepseekV4 only supports fp8 kv-cache format for now.
        kv_cache_dtype = cache_config.cache_dtype if cache_config is not None else "fp8"

        assert kv_cache_dtype.startswith("fp8"), (
            f"DeepseekV4 only supports fp8 kv-cache format for now, "
            f"got {kv_cache_dtype}"
        )
        assert issubclass(self.get_attn_backend(), FlashMLASparseBackend), (
            "Only FlashMLA Sparse Attention backend is supported for DeepseekV4 for now"
        )
        # FlashMLA Sparse Attention fp8 backend uses "fp8_ds_mla" kv-cache format
        # Automatically convert fp8 kv-cache format to "fp8_ds_mla"
        if (
            issubclass(self.get_attn_backend(), FlashMLASparseBackend)
            and kv_cache_dtype.startswith("fp8")
            and kv_cache_dtype != "fp8_ds_mla"
        ):
            assert cache_config is not None
            cache_config.cache_dtype = "fp8_ds_mla"
            kv_cache_dtype = "fp8_ds_mla"
            logger.info_once(
                "Using DeepSeek's fp8_ds_mla KV cache format. To use standard "
                "fp8 kv-cache format, please set `--attention-backend "
                "FLASHINFER_MLA_SPARSE`"
            )

        self.kv_cache_dtype = kv_cache_dtype

        # Register with compilation context for metadata lookup
        compilation_config = vllm_config.compilation_config
        if prefix and prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        if prefix:
            compilation_config.static_forward_context[prefix] = self

        self.kv_cache = torch.tensor([])

    def _prefill_workspace_topk_bound(self) -> int:
        if self.compress_ratio <= 1:
            return 0
        if (
            self.topk_indices_buffer is not None
            and self.topk_indices_buffer.ndim > 0
            and self.topk_indices_buffer.shape[-1] > 0
        ):
            return int(self.topk_indices_buffer.shape[-1])
        indexer_topk = getattr(self.indexer, "topk_tokens", None)
        if indexer_topk is not None:
            return int(indexer_topk)
        return _DEFAULT_SPARSE_MLA_TOPK_TOKENS

    def _prefill_workspace_reservation_specs(
        self,
    ) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        max_model_len = max(1, int(self.max_model_len))
        max_num_batched_tokens = max(1, int(self.max_num_batched_tokens))
        window_size = max(1, int(self.window_size))
        compress_ratio = max(1, int(self.compress_ratio))
        head_dim = int(self.head_dim)
        num_heads = int(self.num_heads)

        max_query_chunk_tokens, max_gather_len = (
            _sparse_mla_prefill_gather_len_upper_bound(
                max_model_len=max_model_len,
                max_num_batched_tokens=max_num_batched_tokens,
                window_size=window_size,
            )
        )
        if compress_ratio <= 1:
            m_bound = max_gather_len
        else:
            compressed_region_size = max_model_len // compress_ratio
            m_bound = compressed_region_size + max_gather_len

        combined_topk = sparse_prefill_combined_topk_size(
            DeepseekV4MLAAttention._prefill_workspace_topk_bound(self),
            window_size,
        )
        specs: list[tuple[tuple[int, ...], torch.dtype]] = [
            ((PREFILL_CHUNK_SIZE, m_bound, head_dim), torch.bfloat16),
            ((max_query_chunk_tokens, combined_topk), torch.int32),
            ((max_query_chunk_tokens,), torch.int32),
        ]
        if is_triton_sparse_mla_enabled_for_platform():
            query_chunk_size = min(
                max_query_chunk_tokens,
                triton_sparse_mla_query_chunk_size(),
            )
            specs.extend(
                [
                    ((query_chunk_size, num_heads), torch.float32),
                    ((query_chunk_size, num_heads), torch.float32),
                    ((query_chunk_size, num_heads, head_dim), torch.float32),
                ]
            )
        return tuple(specs)

    def _reserve_prefill_workspace(self) -> None:
        try:
            workspace_manager = current_workspace_manager()
        except AssertionError:
            return
        workspace_manager.get_simultaneous(*self._prefill_workspace_reservation_specs())

    def get_attn_backend(self) -> type[AttentionBackend]:
        if current_platform.is_rocm():
            from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse_dsv4 import (
                DeepseekV4ROCMAiterMLASparseBackend,
            )

            return DeepseekV4ROCMAiterMLASparseBackend
        return DeepseekV4FlashMLASparseBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        if (
            self.compress_ratio <= 1
        ):  # SWA part. Allocated separately as DeepseekV4SWACache.
            return None
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8,
            compress_ratio=self.compress_ratio,
            cache_dtype_str=self.kv_cache_dtype,
            alignment=576,  # NOTE: FlashMLA requires 576B alignment
            model_version="deepseek_v4",
        )

    def _forward_sparse_mla_swa_decode_triton(
        self,
        q: torch.Tensor,
        swa_k_cache: torch.Tensor,
        swa_metadata: "DeepseekSparseSWAMetadata",
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        swa_lens = swa_metadata.decode_swa_lens[:num_decode_tokens]
        swa_indices = swa_metadata.decode_swa_indices[:num_decode_tokens]
        max_swa_len = swa_metadata.decode_swa_indices.shape[-1]
        head_block_size = sparse_mla_decode_head_block_size(num_decode_tokens)
        if num_decode_tokens > 1:
            # The portable Triton decode kernel is correct for a single token
            # row. Until the multi-row launch is fixed, launch it per row so
            # batched decode does not corrupt rows after the first.
            q_rows = q[:, 0] if q.dim() == 4 else q
            for row in range(num_decode_tokens):
                row_slice = slice(row, row + 1)
                decode_sparse_attention_triton(
                    q=q_rows[row_slice],
                    swa_cache=swa_k_cache,
                    swa_indices=swa_indices[row_slice],
                    swa_lens=swa_lens[row_slice],
                    scale=self.scale,
                    attn_sink=self.attn_sink,
                    out=output[row_slice],
                )
            if output.shape[1] > self.num_heads:
                output[:, self.num_heads :].zero_()
            return

        fp8ds_paged_sparse_mla_attention_with_sink_multihead(
            q=q,
            k_cache=swa_k_cache,
            seq_lens=swa_metadata.seq_lens[:num_decodes],
            gather_lens=swa_lens,
            block_table=swa_metadata.block_table[:num_decodes],
            block_size=swa_metadata.block_size,
            candidate_offset=0,
            num_candidates=max_swa_len,
            scale=self.scale,
            attn_sink=self.attn_sink,
            output=output,
            head_block_size=head_block_size,
            num_heads=self.num_heads,
        )
        if output.shape[1] > self.num_heads:
            output[:, self.num_heads :].zero_()
        return

    def _forward_sparse_mla_compressed_decode_triton(
        self,
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor,
        swa_k_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_lens: torch.Tensor,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: FlashMLASparseMetadata,
        output: torch.Tensor,
    ) -> None:
        if self.compress_ratio not in (4, 128):
            raise NotImplementedError(
                "Triton sparse MLA compressed decode currently supports "
                f"compress_ratio=4 or 128, got {self.compress_ratio}"
            )

        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        mtp_decode = num_decode_tokens != num_decodes

        compressed_block_size = attn_metadata.block_size // self.compress_ratio
        compressed_topk = topk_indices.shape[-1]
        topk_chunk_size = min(
            compressed_topk,
            triton_sparse_mla_topk_chunk_size(),
        )
        compressed_slot_ids = topk_indices[:, 0, :]
        swa_lens = swa_metadata.decode_swa_lens[:num_decode_tokens]
        swa_indices = swa_metadata.decode_swa_indices[:num_decode_tokens]
        head_block_size = sparse_mla_decode_head_block_size(num_decode_tokens)
        max_swa_len = swa_metadata.decode_swa_indices.shape[-1]

        # sm_86 flash-MLA decode tail (VLLM_AMPERE_FLASHMLA_DECODE=1):
        # materialize the combined candidate workspace exactly like the
        # matmul path below, then run the dense kernel per prefix-valid
        # section with an lse/sink merge. Handles batched decode natively
        # (the Triton paths below fall back to a per-row launch for T > 1).
        if (
            compressed_topk <= topk_chunk_size
            and num_decode_tokens >= ampere_flashmla_decode_min_tokens()
            and ampere_flashmla_decode_enabled()
            and ampere_flashmla_supports(
                compressed_topk + max_swa_len, compressed_topk, q.shape[-1]
            )
        ):
            total_candidates = compressed_topk + max_swa_len
            (combined_kv,) = current_workspace_manager().get_simultaneous(
                (
                    (num_decode_tokens, total_candidates, q.shape[-1]),
                    torch.bfloat16,
                ),
            )
            if mtp_decode:
                dequantize_global_slots_k_cache(
                    combined_kv[:, :compressed_topk],
                    compressed_k_cache,
                    compressed_slot_ids,
                    compressed_block_size,
                )
                dequantize_global_slots_k_cache(
                    combined_kv[:, compressed_topk:],
                    swa_k_cache,
                    swa_indices,
                    swa_metadata.block_size,
                )
            else:
                dequantize_combined_sparse_mla_decode_kv(
                    combined_kv,
                    compressed_k_cache,
                    compressed_slot_ids,
                    compressed_block_size,
                    swa_k_cache,
                    swa_metadata.seq_lens[:num_decodes],
                    swa_lens,
                    swa_metadata.block_table[:num_decodes],
                    swa_metadata.block_size,
                )
            ampere_flashmla_sparse_decode(
                q=q,
                combined_kv=combined_kv,
                topk_lens=topk_lens,
                swa_lens=swa_lens,
                compressed_topk=compressed_topk,
                scale=self.scale,
                attn_sink=self.attn_sink,
                output=output,
                num_heads=self.num_heads,
            )
            return

        if num_decode_tokens > 1:
            # The portable Triton decode kernel is correct for a single token
            # row. Until the multi-row launch is fixed, launch it per row so
            # batched decode does not corrupt rows after the first.
            q_rows = q[:, 0] if q.dim() == 4 else q
            for row in range(num_decode_tokens):
                row_slice = slice(row, row + 1)
                decode_sparse_attention_triton(
                    q=q_rows[row_slice],
                    swa_cache=swa_k_cache,
                    swa_indices=swa_indices[row_slice],
                    swa_lens=swa_lens[row_slice],
                    scale=self.scale,
                    attn_sink=self.attn_sink,
                    out=output[row_slice],
                    extra_cache=compressed_k_cache,
                    extra_indices=topk_indices[row_slice],
                    extra_lens=topk_lens[row_slice],
                )
            if output.shape[1] > self.num_heads:
                output[:, self.num_heads :].zero_()
            return

        if (
            compressed_topk <= topk_chunk_size
            and triton_sparse_mla_matmul_decode_enabled()
        ):
            total_candidates = compressed_topk + max_swa_len
            (
                combined_kv,
                valid_tokens,
                score_buffer,
            ) = current_workspace_manager().get_simultaneous(
                (
                    (num_decode_tokens, total_candidates, q.shape[-1]),
                    torch.bfloat16,
                ),
                ((num_decode_tokens, total_candidates), torch.bool),
                ((num_decode_tokens, self.num_heads, total_candidates), torch.bfloat16),
            )
            if mtp_decode:
                dequantize_global_slots_k_cache(
                    combined_kv[:, :compressed_topk],
                    compressed_k_cache,
                    compressed_slot_ids,
                    compressed_block_size,
                )
                dequantize_global_slots_k_cache(
                    combined_kv[:, compressed_topk:],
                    swa_k_cache,
                    swa_indices,
                    swa_metadata.block_size,
                )
            else:
                dequantize_combined_sparse_mla_decode_kv(
                    combined_kv,
                    compressed_k_cache,
                    compressed_slot_ids,
                    compressed_block_size,
                    swa_k_cache,
                    swa_metadata.seq_lens[:num_decodes],
                    swa_lens,
                    swa_metadata.block_table[:num_decodes],
                    swa_metadata.block_size,
                )

            build_combined_sparse_mla_decode_valid_mask(
                valid_tokens,
                compressed_slot_ids,
                topk_lens,
                swa_lens,
            )
            use_dot_finish = num_decode_tokens <= 16
            matmul_sparse_mla_attention_with_sink(
                q=q,
                kv=combined_kv,
                valid_tokens=valid_tokens,
                scale=self.scale,
                attn_sink=self.attn_sink,
                output=output,
                num_heads=self.num_heads,
                score_buffer=score_buffer,
                value_block_size=512 if use_dot_finish else 256,
                candidate_block_size=128 if use_dot_finish else None,
            )
            return

        if not mtp_decode and compressed_topk <= topk_chunk_size:
            fp8ds_global_paged_sparse_mla_attention_with_sink_multihead(
                q=q,
                compressed_k_cache=compressed_k_cache,
                slot_ids=compressed_slot_ids,
                topk_lens=topk_lens,
                compressed_block_size=compressed_block_size,
                swa_k_cache=swa_k_cache,
                seq_lens=swa_metadata.seq_lens[:num_decodes],
                gather_lens=swa_lens,
                block_table=swa_metadata.block_table[:num_decodes],
                swa_block_size=swa_metadata.block_size,
                num_compressed_candidates=compressed_topk,
                num_swa_candidates=max_swa_len,
                scale=self.scale,
                attn_sink=self.attn_sink,
                output=output,
                head_block_size=head_block_size,
                num_heads=self.num_heads,
            )
            if output.shape[1] > self.num_heads:
                output[:, self.num_heads :].zero_()
            return

        (
            comp_max_score,
            comp_denom,
            comp_acc,
            swa_max_score,
            swa_denom,
            swa_acc,
        ) = current_workspace_manager().get_simultaneous(
            ((num_decode_tokens, self.num_heads), torch.float32),
            ((num_decode_tokens, self.num_heads), torch.float32),
            ((num_decode_tokens, self.num_heads, q.shape[-1]), torch.float32),
            ((num_decode_tokens, self.num_heads), torch.float32),
            ((num_decode_tokens, self.num_heads), torch.float32),
            ((num_decode_tokens, self.num_heads, q.shape[-1]), torch.float32),
        )
        comp_max_score.fill_(float("-inf"))
        comp_denom.zero_()
        comp_acc.zero_()
        swa_max_score.fill_(float("-inf"))
        swa_denom.zero_()
        swa_acc.zero_()

        for chunk_start in range(0, compressed_topk, topk_chunk_size):
            chunk_end = min(chunk_start + topk_chunk_size, compressed_topk)
            accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead(
                q=q,
                k_cache=compressed_k_cache,
                slot_ids=compressed_slot_ids[:, chunk_start:chunk_end],
                lens=topk_lens,
                block_size=compressed_block_size,
                candidate_offset=chunk_start,
                scale=self.scale,
                max_score=comp_max_score,
                denom=comp_denom,
                acc=comp_acc,
                head_block_size=head_block_size,
            )
        if mtp_decode:
            accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead(
                q=q,
                k_cache=swa_k_cache,
                slot_ids=swa_indices,
                lens=swa_lens,
                block_size=swa_metadata.block_size,
                scale=self.scale,
                max_score=swa_max_score,
                denom=swa_denom,
                acc=swa_acc,
                head_block_size=head_block_size,
            )
        else:
            accumulate_fp8ds_paged_sparse_mla_attention_chunk_multihead(
                q=q,
                k_cache=swa_k_cache,
                seq_lens=swa_metadata.seq_lens[:num_decodes],
                gather_lens=swa_lens,
                block_table=swa_metadata.block_table[:num_decodes],
                block_size=swa_metadata.block_size,
                candidate_offset=0,
                num_candidates=max_swa_len,
                scale=self.scale,
                max_score=swa_max_score,
                denom=swa_denom,
                acc=swa_acc,
                head_block_size=head_block_size,
            )
        finish_two_sparse_mla_attention_states_with_sink(
            comp_max_score,
            comp_denom,
            comp_acc,
            swa_max_score,
            swa_denom,
            swa_acc,
            self.attn_sink,
            output=output,
        )
        if output.shape[1] > self.num_heads:
            output[:, self.num_heads :].zero_()

    def _forward_sparse_mla_prefill_triton(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        combined_indices: torch.Tensor,
        combined_lens: torch.Tensor,
        output: torch.Tensor,
        state_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        with _dsv4_debug_timing(
            "sparse_mla_prefill_triton",
            layer=self.prefix,
            q_shape=tuple(q.shape),
            kv_shape=tuple(kv.shape),
            indices_shape=tuple(combined_indices.shape),
            lens_shape=tuple(combined_lens.shape),
        ):
            kv_flat = kv.reshape(-1, q.shape[-1])
            topk_chunk_size = min(
                combined_indices.shape[-1],
                triton_sparse_mla_topk_chunk_size(),
            )
            query_chunk_size = min(
                q.shape[0],
                triton_sparse_mla_query_chunk_size(),
            )
            if state_buffers is None:
                (
                    max_score_buffer,
                    denom_buffer,
                    output_buffer,
                ) = current_workspace_manager().get_simultaneous(
                    ((query_chunk_size, self.num_heads), torch.float32),
                    ((query_chunk_size, self.num_heads), torch.float32),
                    ((query_chunk_size, self.num_heads, q.shape[-1]), torch.float32),
                )
            else:
                max_score_buffer, denom_buffer, output_buffer = state_buffers

            for token_start in range(0, q.shape[0], query_chunk_size):
                token_end = min(token_start + query_chunk_size, q.shape[0])
                q_chunk = q[token_start:token_end]
                indices_chunk_full = combined_indices[token_start:token_end]
                lens_chunk = combined_lens[token_start:token_end]
                num_tokens = token_end - token_start
                max_score = max_score_buffer[:num_tokens]
                denom = denom_buffer[:num_tokens]
                subset_acc = output_buffer[:num_tokens]
                max_score.fill_(float("-inf"))
                denom.zero_()
                subset_acc.zero_()

                for index_start in range(
                    0, combined_indices.shape[-1], topk_chunk_size
                ):
                    index_end = min(
                        index_start + topk_chunk_size,
                        combined_indices.shape[-1],
                    )
                    with _dsv4_debug_timing(
                        "sparse_mla_prefill_accumulate",
                        layer=self.prefix,
                        token_start=int(token_start),
                        token_end=int(token_end),
                        index_start=int(index_start),
                        index_end=int(index_end),
                    ):
                        indices_chunk = indices_chunk_full[:, index_start:index_end]
                        if triton_sparse_mla_matmul_prefill_enabled():
                            accumulate_indexed_sparse_mla_attention_chunk_multihead(
                                q=q_chunk,
                                kv_flat=kv_flat,
                                indices=indices_chunk,
                                lens=lens_chunk,
                                candidate_offset=index_start,
                                scale=self.scale,
                                max_score=max_score,
                                denom=denom,
                                acc=subset_acc,
                                head_block_size=sparse_mla_decode_head_block_size(
                                    num_tokens
                                ),
                            )
                        else:
                            accumulate_indexed_sparse_mla_attention_chunk(
                                q=q_chunk,
                                kv_flat=kv_flat,
                                indices=indices_chunk,
                                lens=lens_chunk,
                                candidate_offset=index_start,
                                scale=self.scale,
                                max_score=max_score,
                                denom=denom,
                                acc=subset_acc,
                            )

                with _dsv4_debug_timing(
                    "sparse_mla_prefill_finish",
                    layer=self.prefix,
                    token_start=int(token_start),
                    token_end=int(token_end),
                ):
                    finish_sparse_mla_attention_with_sink(
                        max_score,
                        denom,
                        subset_acc,
                        self.attn_sink,
                        output=output[token_start:token_end],
                    )
                if output.shape[1] > self.num_heads:
                    output[token_start:token_end, self.num_heads :].zero_()

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        if current_platform.is_rocm():
            from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse_dsv4 import (
                DeepseekV4ROCMAiterMLASparseImpl,
            )

            DeepseekV4ROCMAiterMLASparseImpl.forward(self, q, kv, positions, output)
            return

        # Get SWA and indexer metadata from forward context
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            FlashMLASparseMetadata | None, attn_metadata.get(self.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        # SWA-only layers (compress_ratio <= 1) don't have their own KV cache
        # allocation, so self.kv_cache may be empty after profiling cleanup.
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        # Split prefill and decode
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                positions=positions[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def _forward_decode(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: FlashMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        debug_mtp_decode = _dsv4_mtp_debug_decode_should_log(
            num_decodes, num_decode_tokens
        )

        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                # C4A: local indices differ per layer (filled by Indexer).
                assert self.topk_indices_buffer is not None
                local_topk_indices = self.topk_indices_buffer[:num_decode_tokens]
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    local_topk_indices,
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                    global_topk_indices=local_topk_indices,
                )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                # C128A: pre-computed during metadata build.
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens

        if debug_mtp_decode:
            logger.warning(
                "DSV4_MTP_DECODE_ENTER layer=%s prefix=%s compress_ratio=%s "
                "swa_only=%s num_decodes=%s num_decode_tokens=%s "
                "%s %s %s %s %s %s %s",
                _dsv4_debug_layer_id(self),
                self.prefix,
                self.compress_ratio,
                swa_only,
                num_decodes,
                num_decode_tokens,
                _dsv4_tensor_values("positions", positions[:num_decode_tokens], 8),
                _dsv4_tensor_values(
                    "token_to_req",
                    swa_metadata.token_to_req_indices[:num_decode_tokens]
                    if swa_metadata.token_to_req_indices is not None
                    else None,
                    8,
                ),
                _dsv4_tensor_values(
                    "is_valid",
                    swa_metadata.is_valid_token[:num_decode_tokens]
                    if swa_metadata.is_valid_token is not None
                    else None,
                    8,
                ),
                _dsv4_tensor_values("swa_lens", swa_lens[:num_decode_tokens], 8),
                _dsv4_tensor_values(
                    "swa_indices0",
                    swa_indices[: min(num_decode_tokens, 2), :8],
                    16,
                ),
                _dsv4_tensor_values(
                    "topk_lens",
                    topk_lens[:num_decode_tokens] if topk_lens is not None else None,
                    8,
                ),
                _dsv4_tensor_values(
                    "topk_indices0",
                    topk_indices[: min(num_decode_tokens, 2), :, :8]
                    if topk_indices is not None
                    else None,
                    16,
                ),
            )

        # We treat queries in the same seq as different queries
        # and later we only attend by generated indices.
        # q arrives pre-padded to self.padded_heads by the outer wrapper.
        q = q.unsqueeze(1)

        # Prepare SWA cache (num_blocks, swa_block_size, 1, head_bytes)
        # Use unsqueeze to preserve strides (handles padded blocks correctly)
        swa_cache = self.swa_cache_layer.kv_cache.unsqueeze(-2)
        # Reshape KV cache to (num_blocks, block_size, 1, head_bytes)
        compressed_k_cache = kv_cache
        if kv_cache is not None:
            kv_cache = kv_cache.unsqueeze(-2)

        if is_triton_sparse_mla_enabled(q.device):
            if swa_only:
                self._forward_sparse_mla_swa_decode_triton(
                    q=q,
                    swa_k_cache=self.swa_cache_layer.kv_cache,
                    swa_metadata=swa_metadata,
                    output=output,
                )
                if debug_mtp_decode:
                    logger.warning(
                        "DSV4_MTP_DECODE_EXIT layer=%s prefix=%s path=triton_swa "
                        "%s",
                        _dsv4_debug_layer_id(self),
                        self.prefix,
                        _dsv4_tensor_row_stats("output", output, 2),
                    )
                return
            if self.compress_ratio in (4, 128):
                assert compressed_k_cache is not None
                assert attn_metadata is not None
                assert topk_indices is not None
                assert topk_lens is not None
                self._forward_sparse_mla_compressed_decode_triton(
                    q=q,
                    compressed_k_cache=compressed_k_cache,
                    swa_k_cache=self.swa_cache_layer.kv_cache,
                    topk_indices=topk_indices,
                    topk_lens=topk_lens,
                    swa_metadata=swa_metadata,
                    attn_metadata=attn_metadata,
                    output=output,
                )
                if debug_mtp_decode:
                    logger.warning(
                        "DSV4_MTP_DECODE_EXIT layer=%s prefix=%s "
                        "path=triton_compressed %s",
                        _dsv4_debug_layer_id(self),
                        self.prefix,
                        _dsv4_tensor_row_stats("output", output, 2),
                    )
                return
        # One FlashMLASchedMeta per layer type, shared across all same-type
        # layers within this decode step. The first forward call per type
        # triggers the in-kernel planner (allocating tile_scheduler_metadata
        # and num_splits via PyTorch's graph-aware allocator so CUDA graph
        # capture reuses the same addresses on replay); subsequent same-type
        # layers see have_initialized=True and skip the planner.
        if self.compress_ratio <= 1:
            tile_metadata = swa_metadata.tile_sched_swaonly
        elif self.compress_ratio == 4:
            tile_metadata = swa_metadata.tile_sched_c4a
        elif self.compress_ratio == 128:
            tile_metadata = swa_metadata.tile_sched_c128a
        else:
            raise ValueError(
                f"Unsupported compress_ratio={self.compress_ratio}; "
                "expected 1, 4, or 128."
            )
        assert tile_metadata is not None, (
            "swa_metadata missing tile_sched entry for "
            f"compress_ratio={self.compress_ratio}; "
            "DeepseekSparseSWAMetadataBuilder.build_tile_scheduler did not "
            "allocate one for this layer type."
        )

        out, _ = flash_mla_with_kvcache(
            q=q,
            k_cache=swa_cache,
            block_table=None,
            head_dim_v=512,
            tile_scheduler_metadata=tile_metadata,
            cache_seqlens=None,
            is_fp8_kvcache=True,
            indices=swa_indices,
            topk_length=swa_lens,
            softmax_scale=self.scale,
            attn_sink=self.attn_sink,
            extra_k_cache=kv_cache if not swa_only else None,
            extra_indices_in_kvcache=topk_indices,
            extra_topk_length=topk_lens,
            out=output.unsqueeze(1),
        )
        if debug_mtp_decode:
            logger.warning(
                "DSV4_MTP_DECODE_EXIT layer=%s prefix=%s path=flashmla %s",
                _dsv4_debug_layer_id(self),
                self.prefix,
                _dsv4_tensor_row_stats("output", output, 2),
            )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = attn_metadata is None

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        # Use pre-computed prefill metadata.
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        seq_lens_cpu = swa_metadata.prefill_seq_lens_cpu
        gather_lens_cpu = swa_metadata.prefill_gather_lens_cpu
        assert seq_lens is not None
        assert gather_lens is not None
        assert seq_lens_cpu is not None
        assert gather_lens_cpu is not None

        # Derive prefill-local token offsets from the full query_start_loc_cpu.
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
                # C128A: pre-computed during metadata build.
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            top_k = topk_indices.shape[-1]
        else:
            # NOTE(woosuk): topk_indices will not be used for SWA-only layers.
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0

        N, M = _sparse_mla_prefill_workspace_bounds(
            seq_lens_cpu=seq_lens_cpu,
            gather_lens_cpu=gather_lens_cpu,
            compress_ratio=self.compress_ratio,
            swa_only=swa_only,
        )
        # FULL_DECODE_ONLY capture can exercise this path for MTP verification
        # batches (query_len > 1). Avoid device-to-host syncs while the CUDA
        # stream is capturing; the CPU metadata is already prepared outside the
        # graph and is the source of the workspace upper bounds above.
        actual_max_gather_len = int(gather_lens_cpu.max().item())
        if swa_only:
            M = max(M, actual_max_gather_len)
        else:
            actual_compressed_region_size = int(
                (seq_lens_cpu // self.compress_ratio).max().item()
            )
            N = max(N, actual_compressed_region_size)
            M = max(M, N + actual_max_gather_len)
        num_chunks = (num_prefills + PREFILL_CHUNK_SIZE - 1) // PREFILL_CHUNK_SIZE
        max_query_chunk_tokens = 0
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + PREFILL_CHUNK_SIZE, num_prefills)
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )
            max_query_chunk_tokens = max(
                max_query_chunk_tokens, int(query_end - query_start)
            )
        combined_topk = sparse_prefill_combined_topk_size(top_k, self.window_size)

        workspace_manager = current_workspace_manager()
        triton_sparse_mla_enabled = is_triton_sparse_mla_enabled(q.device)
        if triton_sparse_mla_enabled:
            query_chunk_size = min(q.shape[0], triton_sparse_mla_query_chunk_size())
            (
                kv,
                combined_indices_buffer,
                combined_lens_buffer,
                max_score_buffer,
                denom_buffer,
                output_buffer,
            ) = workspace_manager.get_simultaneous(
                ((PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
                ((max_query_chunk_tokens, combined_topk), torch.int32),
                ((max_query_chunk_tokens,), torch.int32),
                ((query_chunk_size, self.num_heads), torch.float32),
                ((query_chunk_size, self.num_heads), torch.float32),
                ((query_chunk_size, self.num_heads, q.shape[-1]), torch.float32),
            )
            prefill_state_buffers = (
                max_score_buffer,
                denom_buffer,
                output_buffer,
            )
        else:
            (
                kv,
                combined_indices_buffer,
                combined_lens_buffer,
            ) = workspace_manager.get_simultaneous(
                ((PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
                ((max_query_chunk_tokens, combined_topk), torch.int32),
                ((max_query_chunk_tokens,), torch.int32),
            )
            prefill_state_buffers = None
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + PREFILL_CHUNK_SIZE, num_prefills)
            chunk_size = chunk_end - chunk_start
            if not swa_only:
                # Gather compressed KV
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                with _dsv4_debug_timing(
                    "prefill_gather_compressed_kv",
                    layer=self.prefix,
                    chunk_idx=int(chunk_idx),
                    chunk_size=int(chunk_size),
                    block_size=int(attn_metadata.block_size // self.compress_ratio),
                    kv_shape=tuple(kv[:chunk_size].shape),
                ):
                    dequantize_and_gather_k_cache(
                        kv[:chunk_size],
                        compressed_k_cache,
                        seq_lens=seq_lens[chunk_start:chunk_end]
                        // self.compress_ratio,
                        gather_lens=None,
                        block_table=block_table[chunk_start:chunk_end],
                        block_size=attn_metadata.block_size // self.compress_ratio,
                        offset=0,
                    )

            # Gather SWA KV
            swa_block_table = swa_metadata.block_table[num_decodes:]
            with _dsv4_debug_timing(
                "prefill_gather_swa_kv",
                layer=self.prefix,
                chunk_idx=int(chunk_idx),
                chunk_size=int(chunk_size),
                block_size=int(swa_metadata.block_size),
                offset=int(N),
                kv_shape=tuple(kv[:chunk_size].shape),
            ):
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    swa_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end],
                    gather_lens=gather_lens[chunk_start:chunk_end],
                    block_table=swa_block_table[chunk_start:chunk_end],
                    block_size=swa_metadata.block_size,
                    offset=N,
                )

            # Combine the topk indices and SWA indices for gathered KV cache
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            query_tokens = query_end - query_start
            with _dsv4_debug_timing(
                "prefill_combine_topk_swa",
                layer=self.prefix,
                chunk_idx=int(chunk_idx),
                query_start=int(query_start),
                query_end=int(query_end),
                query_tokens=int(query_tokens),
                top_k=int(top_k),
                window_size=int(self.window_size),
                compress_ratio=int(self.compress_ratio),
                m_bound=int(M),
                n_bound=int(N),
            ):
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
                    M,
                    N,
                    combined_indices=combined_indices_buffer[:query_tokens],
                    combined_lens=combined_lens_buffer[:query_tokens],
                )

            if triton_sparse_mla_enabled:
                with _dsv4_debug_timing(
                    "prefill_sparse_mla_dispatch",
                    layer=self.prefix,
                    chunk_idx=int(chunk_idx),
                    query_start=int(query_start),
                    query_end=int(query_end),
                ):
                    self._forward_sparse_mla_prefill_triton(
                        q=q[query_start:query_end],
                        kv=kv[:chunk_size],
                        combined_indices=combined_indices,
                        combined_lens=combined_lens,
                        output=output[query_start:query_end],
                        state_buffers=prefill_state_buffers,
                    )
                continue

            if current_platform.is_rocm():
                rocm_sparse_attn_prefill(
                    q=q[query_start:query_end],
                    kv=kv.view(-1, 1, q.shape[-1]),
                    indices=combined_indices.unsqueeze(1),
                    topk_length=combined_lens,
                    scale=self.scale,
                    head_dim=self.head_dim,
                    attn_sink=self.attn_sink,
                    output=output[query_start:query_end],
                )
                continue

            output_chunk, _, _ = flash_mla_sparse_fwd(
                q=q[query_start:query_end],
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices.unsqueeze(1),
                sm_scale=self.scale,
                attn_sink=self.attn_sink,
                topk_length=combined_lens,
                out=output[query_start:query_end],
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

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # head_dim already carries the fp8 scale padding
        # compress_ratio=1 for V3.2, >1 for DeepseekV4; both use the same cache layout.
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            compress_ratio=self.compress_ratio,
            # DeepseekV4 aligns indexer pages to FlashMLA's 576B so they can pack with
            # the indexer's compressor state cache. V3.2 keeps the legacy layout.
            alignment=576,
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
        self.use_fp4_kv = self.vllm_config.attention_config.use_fp4_indexer_cache
        logger.info_once(
            "Using %s indexer cache for Lightning Indexer.",
            "MXFP4" if self.use_fp4_kv else "FP8",
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

        self.max_total_seq_len = (
            get_max_prefill_buffer_size(vllm_config) // self.compress_ratio
        )

        assert cache_config is not None, "Deepseek V4 indexer requires cache_config"
        # NOTE(yifan): FP8 indxer cache use the same layout as V3.2:
        # head_dim bytes = 128 fp8 + 4 fp32 scale = 132.
        # For FP4 indexer cache, we still allocate the same amount of memory as FP8,
        # but only use the first half of the memory.
        k_cache_head_dim = self.head_dim + self.head_dim // self.quant_block_size * 4
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
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        compressed_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        # ReplicatedLinear returns (output, bias); bias is None.
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        k = self.compressor(compressed_kv_score, positions, rotary_emb)
        # INT8 integer-MMA indexer query on Ampere: when the int8 K cache +
        # APPMANA_DSV4_INDEXER_IMMA are on (and not the MXFP4 path), emit a
        # symmetric int8 query so the indexer logits run on s8 x s8 tensor cores
        # instead of arithmetic fp8. Downstream (deep_gemm / paged + workspace
        # logits kernels) detect int8 by q dtype.
        from vllm.model_executor.layers.deepseek_v4_triton_kernels import (
            indexer_imma_enabled,
        )
        q_is_int8 = indexer_imma_enabled() and not self.use_fp4_kv
        q_quant, weights = fused_indexer_q_rope_quant(
            positions,
            q,
            rotary_emb.cos_sin_cache,
            indexer_weights,
            self.softmax_scale,
            self.n_head**-0.5,
            use_fp4=self.use_fp4_kv,
            q_is_int8=q_is_int8,
        )
        return self.indexer_op(hidden_states, q_quant, k, weights)
