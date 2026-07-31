# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.attention.pcp import maybe_gather_indexer_k
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.dsv4.kernel_config import (
    indexer_prefill_topk_slab_rows_override,
    indexer_streaming_topk_prefill_enabled,
)
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    has_deep_gemm,
)
from vllm.utils.import_utils import has_cutedsl
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

SM120_SHORT_ROW_TOPK_ALWAYS_WIDTH = 4096
# GB10's DSV4 C4 indexer is at most 65,536 compressed rows for the planned
# 250k-token admission limit. Keep its decode on the row-local kernel we
# authored for consumer Blackwell instead of re-entering persistent_topk.
SM120_SHORT_ROW_TOPK_MAX_WIDTH = 65536
SM120_SHORT_ROW_TOPK_MAX_ROWS = 16


def _should_use_sm120_short_row_topk_decode(
    topk_tokens: int,
    logits_width: int,
    num_rows: int,
    is_cuda_sm120: bool,
) -> bool:
    if not is_cuda_sm120 or topk_tokens != 512:
        return False
    if logits_width <= SM120_SHORT_ROW_TOPK_ALWAYS_WIDTH:
        return True
    return (
        logits_width < SM120_SHORT_ROW_TOPK_MAX_WIDTH
        and num_rows <= SM120_SHORT_ROW_TOPK_MAX_ROWS
    )


def _use_sm120_short_row_topk_decode(
    logits: torch.Tensor,
    topk_tokens: int,
) -> bool:
    return _should_use_sm120_short_row_topk_decode(
        topk_tokens,
        logits.shape[1],
        logits.shape[0],
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(120),
    )


RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

# Column alignment for the per-step decode logits width. 128 is the Triton
# paged-logits kernels' largest N tile (rowwise path), so an aligned
# token_count keeps the sliced logits contiguous (stride(0) == shape[1]) for
# the top-k kernels.
DECODE_LOGITS_WIDTH_ALIGNMENT = 128


def _decode_logits_token_count(max_context_len: int, max_model_len: int) -> int:
    """Width of the decode logits scratch: the batch's max context rounded up
    to the kernel tile, clamped to [alignment, max_model_len].

    Cudagraph-safe: capture-time metadata carries max_context_len ==
    max_model_len (see DeepSeekV32IndexerDecodeMetadata.max_context_len), so
    captured graphs keep the full width and replays serve any context.
    """
    return min(
        round_up(max(max_context_len, 1), DECODE_LOGITS_WIDTH_ALIGNMENT),
        max_model_len,
    )


def _decode_logits_token_count_for_platform(
    max_context_len: int,
    max_model_len: int,
    is_cuda_sm120: bool,
) -> int:
    """Keep GB10 on the known-safe power-of-two/full-width decode layout.

    The adaptive width is a useful memory optimization on Ampere, but the
    SM12x paged indexer faults at live non-power-of-two widths even though its
    isolated kernels pass. The production 65k configuration reserves only one
    16,384-float row per decode request, so full width is inexpensive.
    """
    if is_cuda_sm120:
        return max_model_len
    return _decode_logits_token_count(max_context_len, max_model_len)


# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32

# ---------------------------------------------------------------------------
# Activation chunking (streaming top-k) for the prefill indexer.
#
# The one-shot prefill path materializes logits[M, N] fp32 with N = the whole
# gathered compressed context, i.e. O(chunk x window) peak memory (~1 GiB per
# sub-chunk at chunk 1024 / 1M-token window / CSA-4; the chunk loop keeps the
# previous sub-chunk's logits referenced while the next one is computed, so up
# to two are live). The streaming path tiles N into slabs and maintains a
# running top-k merged through a strict total order over (score, column)
# packed into UNIQUE int64 keys:
#
#   key = (order_preserving_u32(score_bits) - 2^31) * 2^32
#         + (2^32 - 1 - global_column)
#
# Uniqueness (the column is part of the key) means top-k over keys has no
# ties, and top-k of a totally ordered set is associative:
# topk(A u B) == topk(topk(A) u B). The per-slab scores are bit-identical to
# the corresponding one-shot columns because the sm86 logits kernel computes
# every output column independently (per-(row, n-block) program, fixed
# BLOCK_H/BLOCK_D accumulation order, no atomics). Hence the streamed
# selection is EXACTLY the one-shot selection -- no approximation.
#
# Tie semantics: among equal scores the key order selects the SMALLER column
# first. The production one-shot kernel (top_k_per_row_prefill) resolves ties
# at the k-th boundary in an unspecified order (atomicAdd collection order in
# shared memory), so on data with boundary ties its selected set is not even
# deterministic run-to-run; the key order is a deterministic refinement.
#
# Gated by the checkpoint "vllm" kernel-config block
# (vllm/transformers_utils/configs/dsv4/kernel_config.py): toggle role
# "indexer_streaming_topk_prefill", activated by listing the symbol
# "vllm.model_executor.layers.sparse_attn_indexer.streaming_prefill_topk" in
# "vllm.kernels". Absent block/symbol = OFF (one-shot path, bit-for-bit).
#
# Default context-slab width in K rows (compressed tokens); override via the
# "vllm.indexer_prefill_topk_slab_rows" config key. At M=1024, K=2048,
# slab=16384 the transient footprint is ~231 MiB (64 MiB slab logits fp32 +
# 144 MiB int64 candidate keys + topk temporaries) vs O(M x window) one-shot.
INDEXER_PREFILL_TOPK_SLAB_ROWS = 16384


def _resolved_prefill_topk_slab_rows() -> int:
    override = indexer_prefill_topk_slab_rows_override()
    return INDEXER_PREFILL_TOPK_SLAB_ROWS if override is None else override


def should_use_prefill_streaming_topk(
    dcp_world_size: int,
    use_fp4_cache: bool,
) -> bool:
    """Gate for the streaming prefill top-k path.

    The toggle comes from the checkpoint "vllm" kernel-config block
    (indexer_streaming_topk_prefill role); the remaining guards are the
    DCP/FP4/platform constraints of the streaming implementation.
    """
    if not indexer_streaming_topk_prefill_enabled():
        return False
    # Everything below is a constraint of the streaming implementation, not of
    # the toggle. The toggle was explicitly asked for, so a refusal here has to
    # say so: silently running the one-shot O(M*N) path while the proof line
    # prints indexer_streaming_topk_prefill=<symbol> is how a 1M-context run
    # can OOM for a reason the log denies.
    capability = current_platform.get_device_capability()
    reasons = []
    if dcp_world_size > 1:
        reasons.append(f"decode context parallel is on (dcp={dcp_world_size})")
    if use_fp4_cache:
        reasons.append("the fp4 indexer cache is in use")
    if not current_platform.is_cuda():
        reasons.append("the platform is not CUDA")
    # A floor, not an enumeration: this is a Triton kernel over the same
    # logits, and listing sm_8x and sm_12x excluded sm_90 and sm_100 for no
    # stated reason.
    elif capability is not None and capability.major < 8:
        reasons.append(
            f"compute capability {capability.major}.{capability.minor} < 8.0"
        )
    if reasons:
        logger.warning_once(
            "Streaming prefill top-k was requested by the checkpoint's "
            "kernel config but cannot run: %s. Falling back to the one-shot "
            "prefill top-k, whose memory scales with context length.",
            "; ".join(reasons),
        )
        return False
    return True


def should_stream_prefill_topk_for_context(
    dcp_world_size: int,
    use_fp4_cache: bool,
    context_rows: int,
) -> bool:
    """Use streaming exactly when the gathered context needs multiple slabs."""
    return (
        should_use_prefill_streaming_topk(dcp_world_size, use_fp4_cache)
        and context_rows > _resolved_prefill_topk_slab_rows()
    )


# Centered order-preserving key of float32 -inf (bits 0xFF800000): columns
# masked out by the logits kernel are exactly -inf; every in-range score is a
# finite sum, so "key value-part > this" <=> "column is selectable".
_NEG_INF_CENTERED_KEY = (1 << 31) - 1 - 0xFF800000  # == -2139095041


@triton.jit(do_not_specialize=["num_cols", "col_offset", "stride_lm", "stride_om"])
def _pack_sort_keys_kernel(
    logits_ptr,
    out_ptr,
    num_cols,
    col_offset,
    stride_lm: tl.int64,
    stride_om: tl.int64,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < num_cols
    bits32 = tl.load(logits_ptr + row * stride_lm + offs, mask=mask, other=0.0).to(
        tl.int32, bitcast=True
    )
    bits = bits32.to(tl.int64) & 0xFFFFFFFF
    c = tl.where(bits < (1 << 31), bits, (1 << 31) - 1 - bits)
    key = (c << 32) + ((1 << 32) - 1 - (col_offset + offs).to(tl.int64))
    tl.store(out_ptr + row * stride_om + offs, key, mask=mask)


def _fp32_sort_keys_into(
    logits: torch.Tensor,  # [M, S] fp32
    col_offset: int,
    out: torch.Tensor,  # [M, S] int64
) -> None:
    """Pack fp32 scores and global column ids into strictly ordered, unique
    int64 keys (score-major descending float order, then ascending column).

    Order-preserving uint32 map of the IEEE-754 bit pattern:
    positive floats:  u = bits + 2^31   (in [2^31, 2^32))
    negative floats:  u = ~bits         (in [0, 2^31))
    centered to c = u - 2^31 so the packed key fits int64 exactly.
    """
    m, s = logits.shape
    assert out.shape[0] == m and out.shape[1] == s
    assert logits.stride(1) == 1 and out.stride(1) == 1
    if m == 0 or s == 0:
        return
    BLOCK_N = 1024
    _pack_sort_keys_kernel[(m, triton.cdiv(s, BLOCK_N))](
        logits,
        out,
        s,
        col_offset,
        logits.stride(0),
        out.stride(0),
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )


def oneshot_prefill_topk_reference(
    q_cast: torch.Tensor,  # [M, H, D] fp8-bytes or int8
    kv: tuple[torch.Tensor, torch.Tensor],  # ([N, D], [N] fp32 scale)
    weights: torch.Tensor,  # [M, H] fp32
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_tokens: int,
    qk_int8: bool = False,
) -> torch.Tensor:
    """One-shot reference: full [M, N] logits, single top-k over the same
    unique-key total order as the streaming path. Returns [M, topk] int32
    request-LOCAL indices (column - cu_seqlen_ks), -1 padded."""
    from vllm.models.deepseek_v4.nvidia_imma.triton_kernels import (
        mqa_logits_workspace_triton,
    )

    m = q_cast.shape[0]
    n = kv[0].shape[0]
    logits = mqa_logits_workspace_triton(
        q_cast, kv, weights, cu_seqlen_ks, cu_seqlen_ke, qk_int8=qk_int8
    )
    keys = torch.empty((m, n), dtype=torch.int64, device=q_cast.device)
    _fp32_sort_keys_into(logits, 0, keys)
    top_keys, _ = torch.topk(keys, topk_tokens, dim=1, largest=True, sorted=True)
    return _decode_topk_keys(top_keys, cu_seqlen_ks)


def _decode_topk_keys(
    top_keys: torch.Tensor,  # [M, K] int64
    cu_seqlen_ks: torch.Tensor,  # [M]
) -> torch.Tensor:
    """Unpack selected keys into request-local int32 columns; -1 where the
    key's value part is -inf / sentinel (row shorter than top-k)."""
    value_part = top_keys >> 32
    cols = ((1 << 32) - 1 - (top_keys & 0xFFFFFFFF)).to(torch.int32)
    local = cols - cu_seqlen_ks.to(torch.int32).unsqueeze(1)
    return torch.where(value_part > _NEG_INF_CENTERED_KEY, local, local.new_tensor(-1))


def streaming_prefill_topk(
    q_cast: torch.Tensor,  # [M, H, D] fp8-bytes or int8
    kv: tuple[torch.Tensor, torch.Tensor],  # ([N, D], [N] fp32 scale)
    weights: torch.Tensor,  # [M, H] fp32
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_indices_out: torch.Tensor,  # [M, topk] int32, written in place
    topk_tokens: int,
    slab_rows: int | None = None,
    qk_int8: bool | None = None,
) -> None:
    """Native slab-tiled replacement for full-logits + prefill top-k.

    Each slab's candidates and every running merge are selected by the same
    CUDA radix/histogram implementation as the one-shot prefill path. Two
    ping-pong buffers retain ``(score, global_index)`` candidates, so peak
    memory is O(M x slab_rows) without a generic ``torch.topk`` launch.
    """
    from vllm.models.deepseek_v4.nvidia_imma.triton_kernels import (
        indexer_imma_enabled,
        mqa_logits_workspace_triton,
    )

    k_values, k_scales = kv
    m = q_cast.shape[0]
    n = k_values.shape[0]
    k_top = topk_tokens
    if qk_int8 is None:
        qk_int8 = (
            q_cast.dtype == torch.int8
            and k_values.dtype == torch.int8
            and indexer_imma_enabled()
        )
    if slab_rows is None:
        slab_rows = _resolved_prefill_topk_slab_rows()
    device = q_cast.device
    ks32 = cu_seqlen_ks.to(torch.int32)
    ke32 = cu_seqlen_ke.to(torch.int32)

    # The first half of the current buffer holds the running candidates. The
    # next slab is written into its second half, then the native merge writes
    # the new running set into the other buffer's first half.
    candidate_indices = [
        torch.empty((m, 2 * k_top), dtype=torch.int32, device=device) for _ in range(2)
    ]
    candidate_values = [
        torch.empty((m, 2 * k_top), dtype=torch.float32, device=device)
        for _ in range(2)
    ]
    current = 0
    first = True
    for n0 in range(0, n, slab_rows):
        n1 = min(n0 + slab_rows, n)
        s = n1 - n0
        ks_local = torch.clamp(ks32 - n0, 0, s)
        ke_local = torch.clamp(ke32 - n0, 0, s)
        slab_logits = mqa_logits_workspace_triton(
            q_cast,
            (k_values[n0:n1], k_scales[n0:n1]),
            weights,
            ks_local,
            ke_local,
            qk_int8=qk_int8,
        )
        half = 0 if first else k_top
        slab_indices = candidate_indices[current][:, half : half + k_top]
        slab_values = candidate_values[current][:, half : half + k_top]
        ops.top_k_per_row_prefill_candidates(
            slab_logits,
            ks_local,
            ke_local,
            slab_indices,
            slab_values,
            k_top,
            n0,
        )
        if first:
            first = False
            continue

        other = 1 - current
        ops.top_k_per_row_merge_candidates(
            candidate_values[current],
            candidate_indices[current],
            candidate_indices[other][:, :k_top],
            candidate_values[other][:, :k_top],
            k_top,
        )
        current = other

    final = candidate_indices[current][:, :k_top]
    local = final - ks32.unsqueeze(1)
    topk_indices_out.copy_(torch.where(final >= 0, local, final))


def _assert_cutedsl_dcp_merge_supported(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    k: int,
) -> None:
    # The DCP merge only supports the CuteDSL path (Triton pack kernel + CuteDSL
    # stable-topk selector); there is no PyTorch fallback. The first cut targets
    # Blackwell/Hopper with index_topk in (512, 1024, 2048) (the selector's radix
    # sizing); the Triton pack itself has no shape/topk constraints.
    if not has_cutedsl():
        raise RuntimeError(
            "DCP sparse-indexer merge requires CuteDSL; install it or disable DCP."
        )
    if logits.device.type != "cuda":
        raise RuntimeError("DCP sparse-indexer merge requires CUDA tensors.")
    if logits.dtype != torch.float32 or topk_indices.dtype != torch.int32:
        raise RuntimeError(
            "DCP sparse-indexer merge requires fp32 logits and int32 indices."
        )
    if k not in (512, 1024, 2048):
        raise RuntimeError(
            f"DCP sparse-indexer merge requires index_topk in (512, 1024, 2048); "
            f"got {k}."
        )


def _merge_dcp_topk_global(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    dcp_rank: int,
    dcp_world_size: int,
    cp_interleave: int,
    row_starts: torch.Tensor | None = None,
) -> None:
    """Merge each DCP rank's local top-K into the global top-K.

    ``topk_indices`` are this rank's local top-K positions into its 1/N KV
    shard. A token in the global top-K must also be in its owning rank's local
    top-K (at most ``topk_tokens - 1`` tokens rank globally above it, hence at
    most that many on its own rank), so exchanging only the per-rank local
    candidates is exact -- equivalent to all-gathering the full logit matrix,
    but it ships ``dcp_world_size * topk_tokens`` candidates instead of the whole
    score row. Overwrites ``topk_indices`` with global token ids (``-1`` for
    padding); the attention backend localizes them back to physical slots per
    rank.
    """
    if dcp_world_size <= 1:
        return

    # CuteDSL-only path (no PyTorch fallback): Triton-pack each rank's
    # (score, global_id) candidates on-device, all-gather, then the CuteDSL
    # stable-topk selector.
    _assert_cutedsl_dcp_merge_supported(logits, topk_indices, topk_tokens)
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        pack_dcp_topk_candidates_cutedsl,
        stable_topk_from_gathered_candidates_cutedsl,
    )

    packed = torch.empty(
        (*topk_indices.shape, 2),
        dtype=torch.float32,
        device=topk_indices.device,
    )
    pack_dcp_topk_candidates_cutedsl(
        logits,
        topk_indices,
        packed,
        dcp_rank,
        dcp_world_size,
        cp_interleave,
        row_starts,
    )
    gathered = get_dcp_group().all_gather(packed, dim=1)
    stable_topk_from_gathered_candidates_cutedsl(
        gathered, topk_tokens, out=topk_indices
    )


@triton.jit
def _fused_indexer_q_rope_quant_kernel(
    positions,
    q,
    q_s0,
    q_s1,
    cos_sin_cache,
    cos_sin_s0,
    q_fp8,
    q_fp8_s0,
    q_fp8_s1,
    weights,
    weights_s0,
    weights_s1,
    weights_out,
    weights_out_s0,
    weights_out_s1,
    softmax_scale,
    head_scale,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    is_neox: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    offs32 = tl.arange(0, 32)
    offs64 = tl.arange(0, 64)

    pos = tl.load(positions + token)
    cos = tl.load(cos_sin_cache + pos * cos_sin_s0 + offs32).to(tl.float32)
    sin = tl.load(cos_sin_cache + pos * cos_sin_s0 + 32 + offs32).to(tl.float32)
    q_base = q + token * q_s0 + head * q_s1
    out_base = q_fp8 + token * q_fp8_s0 + head * q_fp8_s1

    if is_neox:
        # NeoX layout, x0 = q[0:32], x1 = q[32:64]
        x0 = tl.load(q_base + offs32).to(tl.float32)
        x1 = tl.load(q_base + 32 + offs32).to(tl.float32)
    else:
        # interleaved layout
        # x0 = q[0, 2, 4, ...], x1 = q[1, 3, 5, ...]
        x0 = tl.load(q_base + offs32 * 2).to(tl.float32)
        x1 = tl.load(q_base + offs32 * 2 + 1).to(tl.float32)
    r0 = (x0 * cos - x1 * sin).to(tl.bfloat16).to(tl.float32)
    r1 = (x1 * cos + x0 * sin).to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(r0)), tl.max(tl.abs(r1)))

    q_nope = tl.load(q_base + 64 + offs64).to(tl.float32)
    amax = tl.maximum(amax, tl.max(tl.abs(q_nope)))
    scale_raw = tl.maximum(amax, 1e-10) * (1.0 / fp8_max)
    # e8m0 format
    q_scale = tl.math.exp2(tl.ceil(tl.log2(scale_raw)))

    if is_neox:
        tl.store(
            out_base + offs32,
            tl.clamp(r0 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
        tl.store(
            out_base + 32 + offs32,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
    else:
        tl.store(
            out_base + offs32 * 2,
            tl.clamp(r0 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
        tl.store(
            out_base + offs32 * 2 + 1,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
    tl.store(
        out_base + 64 + offs64,
        tl.clamp(q_nope / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
    )

    weight = tl.load(weights + token * weights_s0 + head * weights_s1).to(tl.float32)
    tl.store(
        weights_out + token * weights_out_s0 + head * weights_out_s1,
        weight * q_scale * softmax_scale * head_scale,
    )


def fused_indexer_q_rope_quant(
    positions: torch.Tensor,
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert current_platform.is_cuda()
    assert q.dtype == torch.bfloat16
    assert q.shape[-1] == 128
    assert cos_sin_cache.shape[-1] == 64
    assert weights.shape == q.shape[:2]

    q_fp8 = torch.empty_like(q, dtype=current_platform.fp8_dtype())
    weights_out = torch.empty_like(weights, dtype=torch.float32)
    fp8_min, fp8_max = get_fp8_min_max()
    _fused_indexer_q_rope_quant_kernel[(q.shape[0], q.shape[1])](
        positions,
        q,
        q.stride(0),
        q.stride(1),
        cos_sin_cache,
        cos_sin_cache.stride(0),
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        weights,
        weights.stride(0),
        weights.stride(1),
        weights_out,
        weights_out.stride(0),
        weights_out.stride(1),
        softmax_scale,
        head_scale,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        is_neox=is_neox,
        num_warps=1,
    )
    return q_fp8, weights_out


def _gather_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
    use_fp4_cache: bool,
) -> tuple[tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]]:
    """Return ((values_shape, values_dtype), (scales_shape, scales_dtype)) for
    the K-gather workspace. FP8 path: (T, head_dim) fp8 + (T, 4) uint8 fp32
    scales. MXFP4 path: (T, head_dim // 2) uint8 packed mxfp4 +
    (T, head_dim // MXFP4_BLOCK_SIZE) uint8 ue8m0 scales."""
    if use_fp4_cache:
        return (
            ((total_seq_lens, head_dim // 2), torch.uint8),
            ((total_seq_lens, head_dim // MXFP4_BLOCK_SIZE), torch.uint8),
        )
    return (
        ((total_seq_lens, head_dim), fp8_dtype),
        ((total_seq_lens, 4), torch.uint8),
    )


def _reserve_prefill_gather_workspace(
    total_seq_lens: int,
    max_total_seq_len: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
    use_fp4_cache: bool,
) -> None:
    reserve_seq_lens = max(total_seq_lens, max_total_seq_len)
    if reserve_seq_lens <= 0:
        return
    values_spec, scales_spec = _gather_workspace_shapes(
        reserve_seq_lens, head_dim, fp8_dtype, use_fp4_cache
    )
    current_workspace_manager().get_simultaneous(
        values_spec,
        scales_spec,
        ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
    )


def kv_cache_as_quant_view(
    kv_cache: torch.Tensor,
    head_dim: int,
    use_fp4_cache: bool,
) -> torch.Tensor:
    """4D ``[num_blocks, block_size, 1, head_width]`` view expected by
    DeepGEMM, from the 3D indexer kv-cache allocation."""
    if use_fp4_cache:
        assert kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8
        num_blocks, block_size, _ = kv_cache.shape
        page_bytes = int(kv_cache.stride(0))
        fp4_bytes = head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE
        return torch.as_strided(
            kv_cache,
            size=(num_blocks, block_size, 1, fp4_bytes),
            stride=(page_bytes, fp4_bytes, fp4_bytes, 1),
        )
    return kv_cache.unsqueeze(-2)


@eager_break_during_capture
def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_pcp: bool,
    use_fp4_cache: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    skip_topk_buffer_clear: bool = False,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        _reserve_prefill_gather_workspace(
            total_seq_lens=total_seq_lens,
            max_total_seq_len=total_seq_lens,
            head_dim=head_dim,
            fp8_dtype=fp8_dtype,
            use_fp4_cache=use_fp4_cache,
        )

        # Dummy allocation to simulate for peak logits tensor memory during inference.
        # FP8 elements so elements == bytes
        max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        return sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            q_scale,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            use_pcp,
            use_fp4_cache,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    # CUDA graph capture may be decode-only. Reserve the prefill gather workspace
    # here as well so the shared workspace is large enough before it is locked.
    _reserve_prefill_gather_workspace(
        total_seq_lens=total_seq_lens,
        max_total_seq_len=total_seq_lens,
        head_dim=head_dim,
        fp8_dtype=fp8_dtype,
        use_fp4_cache=use_fp4_cache,
    )
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    # Keep PCP padding so every rank contributes the same all-gather shape.
    num_tokens = slot_mapping.shape[0]
    if use_pcp:
        num_tokens //= get_pcp_group().world_size
    if k is not None:
        k = k[:num_tokens]

    # INT8 indexer cache mode (99.48% mean top-512 recall vs fp8 on real
    # tensors). The writer stores symmetric
    # INT8 bit patterns with plain fp32 absmax/127 scales; decode kernels branch
    # on K_IS_INT8 and prefill consumers receive an int8-viewed gather, so their
    # generic float conversions stay correct.
    from vllm.models.deepseek_v4.nvidia_imma.triton_kernels import (
        indexer_cache_is_int8,
    )

    _indexer_int8 = indexer_cache_is_int8() and not use_fp4_cache
    if not skip_k_cache_insert:
        assert k is not None
        k, slot_mapping_for_cache = maybe_gather_indexer_k(
            k,
            slot_mapping,
            num_decode_tokens,
            use_pcp,
        )
        # scale_fmt can be None, but the function expects str
        assert scale_fmt is not None
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        if _indexer_int8:
            ops.indexer_k_quant_and_cache_int8(
                k, kv_cache, slot_mapping_for_cache, quant_block_size
            )
        else:
            ops.indexer_k_quant_and_cache(
                k, kv_cache, slot_mapping_for_cache, quant_block_size, scale_fmt
            )

    # The buffer must be pre-filled with -1 (the "no token" sentinel) before the
    # top-k kernels scatter valid indices into it. On the fused deepseek_v32
    # nvidia path, _fused_norm_rope_kernel already cleared the same
    # [:num_tokens, :topk] region earlier in this forward, so skip the redundant
    # fill.
    if not skip_topk_buffer_clear:
        topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Get the full shared workspace buffers once (will allocate on first use).
        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache.
        workspace_manager = current_workspace_manager()
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
            values_spec,
            scales_spec,
        )
        for chunk in prefill_metadata.chunks:
            cu_seqlen_ks = chunk.cu_seqlen_ks
            cu_seqlen_ke = chunk.cu_seqlen_ke
            assert chunk.local_cu_seq_lens is not None
            k_quant = k_quant_full[: chunk.max_local_total_seq_lens]
            k_scale = k_scale_full[: chunk.max_local_total_seq_lens]
            if not chunk.skip_kv_gather and chunk.local_total_seq_lens > 0:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.local_cu_seq_lens,
                )

            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]

            if chunk.local_total_seq_lens == 0:
                logits = q_slice.new_empty((q_slice.shape[0], 0), dtype=torch.float32)
                topk_indices.fill_(-1)
            else:
                # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
                # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
                if use_fp4_cache:
                    q_slice_cast = q_slice.view(torch.int8)
                    k_quant_cast = k_quant.view(torch.int8)
                    k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
                else:
                    q_slice_cast = q_slice
                    k_quant_cast = (
                        k_quant.view(torch.int8) if _indexer_int8 else k_quant
                    )
                    k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
                # A single slab has no memory advantage and pays the extra
                # candidate-value write of the streaming merge interface.
                # Keep the established one-shot kernel until the gathered
                # context actually exceeds one slab.
                use_streaming_topk = should_stream_prefill_topk_for_context(
                    dcp_world_size,
                    use_fp4_cache,
                    k_quant_cast.shape[0],
                )
                if use_streaming_topk:
                    # Activation chunking: O(M x slab) streaming top-k instead
                    # of the O(M x window) logits materialization below. Exact
                    # selection (see the module comment for the argument).
                    streaming_prefill_topk(
                        q_slice_cast,
                        (k_quant_cast, k_scale_cast),
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        topk_indices,
                        topk_tokens,
                    )
                    # DCP merge below is a no-op at dcp_world_size <= 1; give
                    # it an empty logits tensor.
                    logits = q_slice.new_empty(
                        (q_slice.shape[0], 0), dtype=torch.float32
                    )
                elif current_platform.is_xpu():
                    if q_scale_slice is not None:
                        raise RuntimeError("XPU fp8_mqa_logits does not support FP4 Q")
                    logits = torch.ops.vllm.xpu_fp8_mqa_logits(
                        q_slice_cast,
                        k_quant_cast,
                        k_scale_cast,
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                    )
                else:
                    logits = fp8_fp4_mqa_logits(
                        (q_slice_cast, q_scale_slice),
                        (k_quant_cast, k_scale_cast),
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        clean_logits=False,
                    )
                if not use_streaming_topk:
                    num_rows = logits.shape[0]
                    ops.top_k_per_row_prefill(
                        logits,
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        topk_indices,
                        num_rows,
                        logits.stride(0),
                        logits.stride(1),
                        topk_tokens,
                    )

            _merge_dcp_topk_global(
                logits,
                topk_indices,
                topk_tokens,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
                row_starts=chunk.cu_seqlen_ks,
            )
            # Release before the next chunk's kernel allocates its logits;
            # otherwise two window-proportional [M, N] fp32 buffers are live
            # at once during every prefill step.
            logits = None

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)
        decode_lens = decode_metadata.decode_lens
        if num_decode_tokens == 0:
            padded_q_quant_decode_tokens = q_quant[:1].reshape(1, 1, *q_quant.shape[1:])
            padded_q_scale = (
                q_scale[:1].reshape(1, 1, *q_scale.shape[1:])
                if q_scale is not None
                else None
            )
        elif decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens).
            # FP8 Q is float8_e4m3fn (pack_seq_triton's fp32 pad path is OK,
            # downstream context_lens masks stale slots). MXFP4 Q is two
            # uint8 tensors (values + ue8m0 scales); use the dedicated uint8
            # packer with pad_byte=0 so padded slots dequantize to 0 and
            # can't produce NaN/Inf in the logits kernel.
            if q_scale is not None:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
            else:
                padded_q_scale = None
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )
        if current_platform.is_xpu():
            if padded_q_scale is not None:
                raise RuntimeError("XPU fp8_paged_mqa_logits does not support FP4 Q")
            seq_lens_xpu = (
                seq_lens[:, -1].contiguous() if seq_lens.ndim == 2 else seq_lens
            )
            logits = torch.ops.vllm.xpu_fp8_paged_mqa_logits(
                padded_q_quant_cast,
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens_xpu,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len,
            )
        else:
            # Size the per-step logits scratch by the batch's actual max
            # context (padded to the kernel tile) instead of max_model_len:
            # the tail columns are all -inf and the top-k below is
            # seq_lens-bounded, so narrowing cannot change the selection.
            # Honored by the sm_8x/sm_12x Triton fallback; the Hopper
            # DeepGEMM kernel keeps the full width.
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
                token_count=_decode_logits_token_count_for_platform(
                    decode_metadata.max_context_len,
                    max_model_len,
                    current_platform.is_cuda()
                    and current_platform.is_device_capability_family(120),
                ),
            )
        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        use_cooperative_topk = (
            current_platform.is_cuda()
            and topk_tokens in (512, 1024, 2048)
            and num_rows <= 32
            and logits.stride(0) % 4 == 0  # TMA 16-byte alignment
            and current_platform.has_device_capability(90)
            and not current_platform.is_device_capability_family(120)
        )
        use_persistent_topk = current_platform.is_cuda() and topk_tokens in (
            512,
            1024,
            2048,
        )
        if _use_sm120_short_row_topk_decode(logits, topk_tokens):
            ops.top_k_per_row_decode(
                logits,
                next_n,
                seq_lens,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )
        elif use_cooperative_topk:
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.cooperative_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                attn_metadata_narrowed.max_seq_len,
            )
        elif use_persistent_topk:
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                logits.shape[1],
            )
        else:
            ops.top_k_per_row_decode(
                logits,
                next_n,
                seq_lens,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

        if decode_metadata.global_seq_lens is not None:
            _merge_dcp_topk_global(
                logits,
                topk_indices,
                topk_tokens,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
            )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
    use_pcp: bool,
    use_fp4_cache: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    skip_topk_buffer_clear: bool = False,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer")
class SparseAttnIndexer(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        # DCP scalars are constant for the run; resolve them here (config is set
        # during model construction) and pass them into the custom op, rather
        # than threading them through per-step metadata.
        parallel_config = get_current_vllm_config().parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size
        self.use_pcp = parallel_config.prefill_context_parallel_size > 1
        # DeepGEMM is Hopper-only. On Ampere (sm_8x) and consumer Blackwell
        # (sm_12x) the paged-MQA-logits path is served by the Triton fallback in
        # vllm.utils.deep_gemm.fp8_fp4_paged_mqa_logits (families 80/120), so the
        # indexer op is available there even without DeepGEMM. Mirror that exact
        # capability gate rather than hard-requiring DeepGEMM.
        if (
            current_platform.is_cuda()
            and not has_deep_gemm()
            and not (
                current_platform.is_device_capability_family(80)
                or current_platform.is_device_capability_family(120)
            )
        ):
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM support in "
                "the current vLLM environment."
            )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(hidden_states, q_quant, k, weights)
        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_quant, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_pcp,
            self.use_fp4_cache,
            self.dcp_rank,
            self.dcp_world_size,
            self.cp_kv_cache_interleave_size,
        )

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        return self.forward_cuda(hidden_states, q_fp8, k, weights)

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        assert not self.use_fp4_cache, "AMD platform doesn't support fp4 cache yet"
        assert isinstance(q_quant, torch.Tensor), (
            "AMD sparse_attn_indexer expects a single FP8 q_quant tensor"
        )
        if rocm_aiter_ops.is_enabled():
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                _encode_layer_name(self.k_cache.prefix),
                self.k_cache.kv_cache,
                q_quant,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
                skip_k_cache_insert=self.skip_k_cache_insert,
            )
        raise RuntimeError(
            "Sparse attention indexer ROCm path is only supported on AITER. "
            "Please enable aiter with VLLM_ROCM_USE_AITER=1"
        )
