# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any, ClassVar, cast

import torch
from torch import nn

from vllm.config import CUDAGraphMode, VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    DispatchSpec,
    TritonWarmupTensor,
    triton_kernel_dispatcher_with_warmup,
)
from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    compress_norm_rope_store_triton,
    compress_norm_rope_store_two_stage_triton,
)
from vllm.models.deepseek_v4.common.ops.fused_indexer_q import MXFP4_BLOCK_SIZE
from vllm.models.deepseek_v4.common.ops.save_partial_states import (
    _SAVE_PARTIAL_STATES_KERNEL,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.import_utils import has_cutedsl
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

_C128_RATIO = 128


def _c128_ring_capacity(num_speculative_tokens: int) -> int:
    span = _C128_RATIO + num_speculative_tokens
    return _C128_RATIO * ((span + _C128_RATIO - 1) // _C128_RATIO)


def _compressor_ring_capacity(
    compress_ratio: int, head_dim: int, vllm_config: VllmConfig
) -> int | None:
    """Rows of the per-request circular compressor state, or None for the
    paged sliding-window state.

    CUDA keeps a ring for C128: the upstream CuTe DSL kernels, and the Triton
    kernels of sm_8x / sm_12x, which need one micro-batch at a time.
    """
    if not current_platform.is_cuda():
        return None
    num_speculative_tokens = vllm_config.num_speculative_tokens
    cutedsl = _uses_cutedsl_compressor(head_dim)
    if compress_ratio == _C128_RATIO and cutedsl:
        return _c128_ring_capacity(num_speculative_tokens)
    if cutedsl or vllm_config.parallel_config.use_ubatching:
        # The Triton ring stages each step's rows in one shared buffer, which
        # overlapping micro-batches would both write.
        return None
    if compress_ratio == _C128_RATIO:
        return _c128_ring_capacity(num_speculative_tokens)
    return None


def _state_block_size(compress_ratio: int) -> int:
    # Block size is constrained by tensor sharing between compressor states
    # and KV blocks: the states pack into the same pool slot, so a block of
    # state rows must fit the slot the MLA pages need.
    # - C4 compressor block shape [4, 2*512*2*4] -> block_size = 4
    # - C128 compressor block shape [8, 512*2*4] -> block_size = 8
    if compress_ratio == 4:
        return 4
    if compress_ratio == _C128_RATIO:
        return 8
    raise ValueError(f"Invalid compress ratio: {compress_ratio}")


def _uses_cutedsl_compressor(head_dim: int) -> bool:
    # The CuTe DSL compressor module cannot import on sm_8x and sm_12x; those
    # platforms run the Triton compressor kernels.
    return (
        current_platform.is_cuda()
        and head_dim == 512
        and has_cutedsl()
        and not current_platform.is_device_capability_family(80)
        and not current_platform.is_device_capability_family(120)
    )


@triton.jit(
    do_not_specialize=[
        "block_table_stride",
        "num_actual_tokens",
        "num_tokens",
        "num_reqs",
    ],
    do_not_specialize_on_alignment=["block_table_ptr", "positions_ptr"],
)
def _build_c128_ring_metadata_kernel(
    block_table_ptr,
    block_table_stride,
    query_start_loc_ptr,
    token_to_req_ptr,
    positions_ptr,
    slot_mapping_ptr,
    tail_slot_mapping_ptr,
    num_actual_tokens,
    num_tokens,
    num_reqs,
    CAPACITY: tl.constexpr,
    RING_BLOCK_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    query_batch_end = tl.load(query_start_loc_ptr + num_reqs)
    valid = (offsets < num_actual_tokens) & (offsets < query_batch_end)
    req = tl.load(token_to_req_ptr + offsets, mask=valid, other=0).to(tl.int64)
    query_end = tl.load(query_start_loc_ptr + req + 1, mask=valid, other=0)
    position = tl.load(positions_ptr + offsets, mask=valid, other=0)
    # Ring row -> (ring block, row in block); one block when the ring is a
    # single block of CAPACITY rows.
    row = position % CAPACITY
    block = tl.load(
        block_table_ptr + req * block_table_stride + row // RING_BLOCK_SIZE,
        mask=valid,
        other=-1,
    )
    valid &= block >= 0
    slot = block.to(tl.int64) * RING_BLOCK_SIZE + row % RING_BLOCK_SIZE
    keep_tail = valid & (offsets + CAPACITY >= query_end)
    store = offsets < num_tokens
    tl.store(slot_mapping_ptr + offsets, tl.where(valid, slot, -1), mask=store)
    tl.store(tail_slot_mapping_ptr + offsets, tl.where(keep_tail, slot, -1), mask=store)


def _build_c128_ring_metadata_warmup_inputs(
    *, capacity: int, block_size: int | None = None
) -> dict[str, Any]:
    int32 = TritonWarmupTensor(torch.int32)
    int64 = TritonWarmupTensor(torch.int64)
    return dict(
        block_table=int32,
        query_start_loc=int32,
        token_to_req=int32,
        positions=int64,
        slot_mapping=int64,
        tail_slot_mapping=int64,
        num_actual_tokens=2,
        num_tokens=2,
        num_reqs=1,
        capacity=capacity,
        block_size=capacity if block_size is None else block_size,
    )


@triton_kernel_dispatcher_with_warmup(
    kernel=_build_c128_ring_metadata_kernel,
    warmup_inputs=_build_c128_ring_metadata_warmup_inputs,
)
def _build_c128_ring_metadata(
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    token_to_req: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    tail_slot_mapping: torch.Tensor,
    num_actual_tokens: int,
    num_tokens: int,
    num_reqs: int,
    capacity: int,
    block_size: int,
) -> DispatchSpec:
    return (triton.cdiv(num_tokens, 256),), dict(
        block_table_stride=block_table.stride(0),
        CAPACITY=capacity,
        RING_BLOCK_SIZE=block_size,
        BLOCK=256,
    )


def build_c128_ring_metadata(
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    num_actual_tokens: int,
    num_reqs: int,
    capacity: int,
    token_to_req_indices: torch.Tensor | None = None,
    out_slots: torch.Tensor | None = None,
    out_tail_slots: torch.Tensor | None = None,
    block_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map compressor rows to a per-request ring and select the saved suffix.

    The ring holds ``capacity`` rows in ``capacity // block_size`` blocks of
    the request's block table row (one block by default).
    """
    if block_size is None:
        block_size = capacity
    if capacity < block_size or capacity % block_size != 0:
        raise ValueError(
            f"Ring capacity {capacity} must be a multiple of its block size "
            f"{block_size}"
        )
    if out_slots is None:
        out_slots = torch.empty_like(slot_mapping)
    if out_tail_slots is None:
        out_tail_slots = torch.empty_like(slot_mapping)
    num_tokens = slot_mapping.numel()
    if num_actual_tokens == 0 or num_reqs == 0:
        out_slots.fill_(-1)
        out_tail_slots.fill_(-1)
        return out_slots, out_tail_slots

    if token_to_req_indices is None:
        tokens = torch.arange(num_actual_tokens, device=slot_mapping.device)
        token_to_req_indices = torch.searchsorted(
            query_start_loc[1 : num_reqs + 1], tokens, right=True
        ).to(torch.int32)

    if positions.is_cuda:
        _build_c128_ring_metadata(
            block_table,
            query_start_loc,
            token_to_req_indices,
            positions,
            out_slots,
            out_tail_slots,
            num_actual_tokens,
            num_tokens,
            num_reqs,
            capacity,
            block_size,
        )
        return out_slots, out_tail_slots

    out_slots.fill_(-1)
    out_tail_slots.fill_(-1)
    rows = torch.arange(num_actual_tokens, device=slot_mapping.device)
    query_batch_end = query_start_loc[num_reqs]
    valid_rows = rows < query_batch_end
    req = token_to_req_indices[:num_actual_tokens].long().clamp(max=num_reqs - 1)
    ring_rows = positions[:num_actual_tokens].long().remainder(capacity)
    blocks = block_table[req, ring_rows // block_size].long()
    valid = valid_rows & (blocks >= 0)
    slots = blocks * block_size + ring_rows.remainder(block_size)
    out_slots[:num_actual_tokens] = torch.where(valid, slots, -1)
    query_ends = query_start_loc[1 : num_reqs + 1].index_select(0, req)
    keep_tail = valid & (rows + capacity >= query_ends)
    out_tail_slots[:num_actual_tokens] = torch.where(keep_tail, slots, -1)
    return out_slots, out_tail_slots


def _ring_stage_num_blocks(
    max_num_tokens: int, max_num_reqs: int, window: int, block_size: int
) -> int:
    """Stage blocks for one step: each request stages its chunk and the
    ``window - 1`` rows before it, from the block holding the first of them."""
    return cdiv(max_num_tokens + max_num_reqs * (window - 1), block_size) + (
        2 * max_num_reqs
    )


# Stage block stride pad: keeps a stage block stride off a multiple of 16
# floats when the ring's is.
_STAGE_PAD = 4
_RING_STAGE_STORAGE: dict[tuple[torch.device, int], torch.Tensor] = {}


def _shared_ring_stage_storage(
    num_floats: int, device_type: str, row_width: int
) -> torch.Tensor:
    """fp32 stage shared by the compressor layers of one row width. Layers run
    one after another; the main and indexer compressors of a layer may overlap
    on separate streams, and their row widths differ."""
    device = torch.device(device_type, torch.accelerator.current_device_index())
    key = (device, row_width)
    storage = _RING_STAGE_STORAGE.get(key)
    if storage is None or storage.numel() < num_floats:
        storage = torch.empty(num_floats, dtype=torch.float32, device=device)
        _RING_STAGE_STORAGE[key] = storage
    return storage


def _ring_stage_table_width(vllm_config: VllmConfig, block_size: int) -> int:
    # The width the paged state's block table has, so the paged kernels see
    # the same block-table stride (and specialize the same way) on the stage.
    from vllm.v1.worker.block_table import get_block_table_width

    return get_block_table_width(
        cdiv(vllm_config.model_config.max_model_len, block_size), block_size
    )


@triton.jit
def _fill_ring_stage_kernel(
    query_start_loc_ptr,
    positions_ptr,
    stage_base_ptr,
    stage_lo_ptr,
    stage_count_ptr,
    stage_block_table_ptr,
    stage_block_table_stride,
    stage_slot_mapping_ptr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    base = tl.load(stage_base_ptr + req)
    lo = tl.load(stage_lo_ptr + req)
    count = tl.load(stage_count_ptr + req)
    row = stage_block_table_ptr + req.to(tl.int64) * stage_block_table_stride
    for j in range(0, count, BLOCK):
        offs = j + tl.arange(0, BLOCK)
        tl.store(row + lo + offs, (base + offs).to(tl.int32), mask=offs < count)
    query_start = tl.load(query_start_loc_ptr + req)
    query_end = tl.load(query_start_loc_ptr + req + 1)
    for t in range(query_start, query_end, BLOCK):
        offs = t + tl.arange(0, BLOCK)
        mask = offs < query_end
        pos = tl.load(positions_ptr + offs, mask=mask, other=0).to(tl.int64)
        slot = (base + pos // BLOCK_SIZE - lo) * BLOCK_SIZE + pos % BLOCK_SIZE
        tl.store(stage_slot_mapping_ptr + offs, slot, mask=mask)


def build_ring_stage_metadata(
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    num_reqs: int,
    window: int,
    block_size: int,
    out_block_table: torch.Tensor,
    out_slots: torch.Tensor,
    out_base: torch.Tensor,
    out_lo: torch.Tensor,
) -> None:
    """Paged-state addressing of one step's staged rows.

    Request ``r`` stages logical blocks ``[lo, lo + count)`` (its chunk and the
    ``window - 1`` rows before it) at stage blocks ``[base, base + count)``;
    ``out_block_table[r, lo + j] = base + j``, and ``out_slots`` maps each
    chunk row to its stage slot (-1 for padding). The compressor kernels then
    read the stage exactly as they read the paged state.
    """
    out_slots.fill_(-1)
    if num_reqs == 0:
        return
    qsl = query_start_loc[: num_reqs + 1].long()
    lens = qsl[1:] - qsl[:-1]
    has_rows = lens > 0
    last_token = max(positions.numel() - 1, 0)
    first = positions[qsl[:-1].clamp(max=last_token)].long()
    last = positions[(qsl[1:] - 1).clamp(min=0, max=last_token)].long()
    lo = (first - (window - 1)).clamp(min=0) // block_size
    hi = (last + block_size) // block_size
    count = torch.where(has_rows, hi - lo, 0)
    out_base[:num_reqs] = torch.cumsum(count, 0) - count
    out_lo[:num_reqs] = lo
    stage_count = count.to(torch.int64)
    _fill_ring_stage_kernel[(num_reqs,)](
        query_start_loc,
        positions,
        out_base,
        out_lo,
        stage_count,
        out_block_table,
        out_block_table.stride(0),
        out_slots,
        BLOCK_SIZE=block_size,
        BLOCK=256,
    )


@triton.jit
def _stage_ring_history_kernel(
    ring_ptr,
    ring_stride0,
    ring_stride1,
    stage_ptr,
    stage_stride0,
    stage_stride1,
    ring_block_table_ptr,
    ring_block_table_stride,
    query_start_loc_ptr,
    positions_ptr,
    stage_base_ptr,
    stage_lo_ptr,
    ROW_WIDTH: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WINDOW: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    COLS: tl.constexpr,
):
    """Copy the ``WINDOW - 1`` ring rows before a chunk into its stage."""
    req = tl.program_id(0)
    offset = tl.program_id(1)
    query_start = tl.load(query_start_loc_ptr + req)
    query_end = tl.load(query_start_loc_ptr + req + 1)
    if query_start >= query_end:
        return
    start = tl.load(positions_ptr + query_start).to(tl.int64)
    end = tl.load(positions_ptr + query_end - 1).to(tl.int64) + 1
    # Only compression boundaries inside the chunk read history rows.
    if start % COMPRESS_RATIO + (end - start) < COMPRESS_RATIO:
        return
    pos = start - (WINDOW - 1) + offset
    if pos < 0:
        return
    ring_row = pos % CAPACITY
    ring_block = tl.load(
        ring_block_table_ptr
        + req.to(tl.int64) * ring_block_table_stride
        + ring_row // BLOCK_SIZE
    )
    if ring_block < 0:
        return
    stage_block = (
        tl.load(stage_base_ptr + req) + pos // BLOCK_SIZE - tl.load(stage_lo_ptr + req)
    )
    src = (
        ring_ptr
        + ring_block.to(tl.int64) * ring_stride0
        + (ring_row % BLOCK_SIZE) * ring_stride1
    )
    dst = stage_ptr + stage_block * stage_stride0 + (pos % BLOCK_SIZE) * stage_stride1
    for c in range(0, ROW_WIDTH, COLS):
        cols = c + tl.arange(0, COLS)
        mask = cols < ROW_WIDTH
        tl.store(dst + cols, tl.load(src + cols, mask=mask), mask=mask)


def stage_ring_history(
    ring_cache: torch.Tensor,
    stage_cache: torch.Tensor,
    ring_block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    ring_stage: "RingStage",
    capacity: int,
    block_size: int,
    compress_ratio: int,
) -> None:
    """Copy each request's ``window - 1`` ring rows before its chunk into the
    stage, for requests whose chunk holds a compression boundary."""
    row_width = ring_cache.shape[-1]
    _stage_ring_history_kernel[(ring_stage.lo.shape[0], ring_stage.window - 1)](
        ring_cache,
        ring_cache.stride(0),
        ring_cache.stride(1),
        stage_cache,
        stage_cache.stride(0),
        stage_cache.stride(1),
        ring_block_table,
        ring_block_table.stride(0),
        query_start_loc,
        positions,
        ring_stage.base,
        ring_stage.lo,
        ROW_WIDTH=row_width,
        CAPACITY=capacity,
        BLOCK_SIZE=block_size,
        WINDOW=ring_stage.window,
        COMPRESS_RATIO=compress_ratio,
        COLS=min(1024, triton.next_power_of_2(row_width)),
    )


@dataclass
class RingStage:
    """One step's staged rows for the Triton compressor kernels (see
    ``CompressorStateCache.stage``): the stage is addressed like the paged
    state, so the unchanged paged kernels compress from it."""

    block_table: torch.Tensor  # [num_reqs, width] stage block per logical block
    slot_mapping: torch.Tensor  # [num_tokens] stage slot of each chunk row
    base: torch.Tensor  # [num_reqs] first stage block of each request
    lo: torch.Tensor  # [num_reqs] first staged logical block of each request
    window: int


def _prefer_two_stage_compressor() -> bool:
    # Platforms that favor the triton variant of two-stage compressor split.
    # Currently only tested on ROCm
    return current_platform.is_rocm()


def _get_c128_boundary(metadata: CommonAttentionMetadata) -> bool | None:
    seq_lens_cpu = metadata.seq_lens_cpu_upper_bound
    if seq_lens_cpu is None:
        return None

    query_lens = metadata.query_start_loc_cpu[1:] - metadata.query_start_loc_cpu[:-1]
    starts = seq_lens_cpu - query_lens
    starts_list = starts.tolist()
    query_start_loc = metadata.query_start_loc_cpu.tolist()
    return any(
        start % 128 + query_start_loc[i + 1] - query_start_loc[i] >= 128
        for i, start in enumerate(starts_list)
    )


class CompressorBackend(AttentionBackend):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_name() -> str:
        return "CompressorBackend"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512, 1024]

    @staticmethod
    def get_builder_cls() -> type["CompressorMetadataBuilder"]:
        return CompressorMetadataBuilder


@dataclass
class CompressorMetadata:
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int
    token_to_req_indices: torch.Tensor  # [num_tokens]
    tail_slot_mapping: torch.Tensor  # [num_tokens]
    query_start_loc: torch.Tensor  # [num_reqs + 1]
    is_circular: bool
    num_decode_tokens: int | None = None
    c128_boundary: bool | None = None
    # Rows of the circular state (``block_size`` rows per ring block).
    ring_capacity: int | None = None
    # Multi-block (Triton) rings compress from a per-step stage.
    ring_stage: RingStage | None = None


class CompressorMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(
            self.kv_cache_spec,
            SlidingWindowMLASpec | MLAAttentionSpec | CircularBufferSpec,
        )
        mla_spec = cast(
            SlidingWindowMLASpec | MLAAttentionSpec | CircularBufferSpec,
            self.kv_cache_spec,
        )
        self.block_size = mla_spec.block_size
        self.is_circular = isinstance(mla_spec, CircularBufferSpec)
        self.ring_capacity = (
            mla_spec.ring_capacity if isinstance(mla_spec, CircularBufferSpec) else None
        )
        max_num_batched_tokens = (
            self.vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.token_to_req_indices = torch.zeros(
            max_num_batched_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        self.slot_mapping_buffer = (
            torch.empty(max_num_batched_tokens, dtype=torch.int64, device=self.device)
            if self.is_circular
            else None
        )
        self.tail_slot_mapping_buffer = (
            torch.empty(max_num_batched_tokens, dtype=torch.int64, device=self.device)
            if self.is_circular
            else None
        )
        # A ring spread over several blocks is the Triton layout: the step's
        # rows are staged and compressed by the paged kernels.
        self.stage_window: int | None = None
        if isinstance(mla_spec, CircularBufferSpec) and mla_spec.num_ring_blocks > 1:
            forward_context = self.vllm_config.compilation_config.static_forward_context
            windows = {
                forward_context[name].sliding_window for name in self.layer_names
            }
            assert len(windows) == 1, windows
            self.stage_window = windows.pop()
            scheduler_config = self.vllm_config.scheduler_config
            max_num_reqs = scheduler_config.max_num_seqs
            self.stage_block_table = torch.zeros(
                max_num_reqs,
                _ring_stage_table_width(self.vllm_config, self.block_size),
                dtype=torch.int32,
                device=self.device,
            )
            self.stage_slot_mapping = torch.empty(
                max_num_batched_tokens, dtype=torch.int64, device=self.device
            )
            self.stage_base = torch.zeros(
                max_num_reqs, dtype=torch.int64, device=self.device
            )
            self.stage_lo = torch.zeros(
                max_num_reqs, dtype=torch.int64, device=self.device
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CompressorMetadata:
        token_to_req_indices = common_attn_metadata.token_to_req_indices(
            self.token_to_req_indices
        )
        num_tokens = common_attn_metadata.slot_mapping.numel()
        if self.is_circular:
            assert self.slot_mapping_buffer is not None
            assert self.tail_slot_mapping_buffer is not None
            assert self.ring_capacity is not None
            positions = common_attn_metadata.positions
            assert positions is not None
            slot_mapping, tail_slot_mapping = build_c128_ring_metadata(
                common_attn_metadata.slot_mapping,
                common_attn_metadata.block_table_tensor,
                common_attn_metadata.query_start_loc,
                positions,
                common_attn_metadata.num_actual_tokens,
                common_attn_metadata.num_reqs,
                self.ring_capacity,
                token_to_req_indices=token_to_req_indices,
                out_slots=self.slot_mapping_buffer[:num_tokens],
                out_tail_slots=self.tail_slot_mapping_buffer[:num_tokens],
                block_size=self.block_size,
            )
        else:
            slot_mapping = common_attn_metadata.slot_mapping
            tail_slot_mapping = slot_mapping
        num_decode_tokens = None
        if _prefer_two_stage_compressor():
            _, _, num_decode_tokens, _ = split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=1,
                treat_short_extends_as_decodes=(
                    common_attn_metadata.is_prefilling is None
                ),
            )
        async_spec_decode = (
            self.vllm_config.scheduler_config.async_scheduling
            and self.vllm_config.speculative_config is not None
        )
        block_table = common_attn_metadata.block_table_tensor
        if not self.is_circular:
            block_table = block_table.clamp_(min=0)
        return CompressorMetadata(
            block_table=block_table,
            slot_mapping=slot_mapping,
            block_size=self.block_size,
            token_to_req_indices=token_to_req_indices,
            tail_slot_mapping=tail_slot_mapping,
            query_start_loc=common_attn_metadata.query_start_loc,
            is_circular=self.is_circular,
            num_decode_tokens=num_decode_tokens,
            c128_boundary=(
                _get_c128_boundary(common_attn_metadata)
                if self.is_circular and not async_spec_decode
                else None
            ),
            ring_capacity=self.ring_capacity,
            ring_stage=self._build_ring_stage(common_attn_metadata, num_tokens),
        )

    def _build_ring_stage(
        self, common_attn_metadata: CommonAttentionMetadata, num_tokens: int
    ) -> RingStage | None:
        if self.stage_window is None:
            return None
        positions = common_attn_metadata.positions
        assert positions is not None
        num_reqs = common_attn_metadata.num_reqs
        build_ring_stage_metadata(
            common_attn_metadata.query_start_loc,
            positions,
            num_reqs,
            self.stage_window,
            self.block_size,
            out_block_table=self.stage_block_table,
            out_slots=self.stage_slot_mapping[:num_tokens],
            out_base=self.stage_base,
            out_lo=self.stage_lo,
        )
        return RingStage(
            block_table=self.stage_block_table[:num_reqs],
            slot_mapping=self.stage_slot_mapping[:num_tokens],
            base=self.stage_base[:num_reqs],
            lo=self.stage_lo[:num_reqs],
            window=self.stage_window,
        )


class CompressorStateCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        prefix: str,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.dtype = dtype
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        assert self.dtype == torch.float32
        assert compress_ratio in [4, 128]
        self.compress_ratio = compress_ratio
        coff = 1 + (compress_ratio == 4)
        self.sliding_window = coff * compress_ratio
        self.head_dim = state_dim // (2 * coff)
        # Rows per block of the paged state, and of each block of a Triton ring.
        # The CuTe DSL C128 ring is one block of its whole capacity instead.
        self.block_size = _state_block_size(compress_ratio)

        self.stage = torch.tensor([])
        self._stage_storage: torch.Tensor | None = None

    def allocate_stage(self, vllm_config: VllmConfig) -> None:
        """A Triton ring keeps only the rows later steps read. The step's own
        rows go to a stage addressed like the paged state, so the paged kernels
        compress from it unchanged (see ``RingStage``). Allocated with the
        model, so memory profiling counts it."""
        scheduler_config = vllm_config.scheduler_config
        self._stage_blocks = _ring_stage_num_blocks(
            scheduler_config.max_num_batched_tokens,
            scheduler_config.max_num_seqs,
            self.sliding_window,
            self.block_size,
        )
        self._stage_storage = _shared_ring_stage_storage(
            self._stage_blocks * (self.block_size * self.state_dim + _STAGE_PAD),
            current_platform.device_type,
            self.state_dim,
        )

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # [B, H=1, N, C] -> [B, N, C]
        self.kv_cache = kv_cache.squeeze(1)
        if self._stage_storage is not None:
            # Block stride as divisible by 16 as the ring's, so the paged
            # kernels specialize (and round) on the stage as on the paged state.
            block_stride = self.block_size * self.state_dim
            if self.kv_cache.stride(0) % 16 != 0:
                block_stride += _STAGE_PAD
            self.stage = torch.as_strided(
                self._stage_storage,
                (self._stage_blocks, self.block_size, self.state_dim),
                (block_stride, self.state_dim, 1),
            )

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        capacity = _compressor_ring_capacity(
            self.compress_ratio, self.head_dim, vllm_config
        )
        if capacity is not None and _uses_cutedsl_compressor(self.head_dim):
            return CircularBufferSpec(
                block_size=capacity,
                num_kv_heads=1,
                head_size=self.state_dim,
                head_size_v=0,
                dtype=self.dtype,
            )
        if capacity is not None:
            # The ring is spread over blocks of the paged state's size, so it
            # packs into the pool slot the attention groups already need; a
            # single-block C128 ring (1 MiB with a speculative step) would
            # widen every block of a few-layer rank.
            return CircularBufferSpec(
                block_size=self.block_size,
                num_ring_blocks=capacity // self.block_size,
                num_kv_heads=1,
                head_size=self.state_dim,
                head_size_v=0,
                dtype=self.dtype,
            )
        # This cache stores FP32 compressor state, not packed MLA KV rows.
        # Keep the physical page aligned with the cache it overlays, but do
        # not select that cache's packed row-size formula.
        uses_fp8_ds_mla_layout = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
        uses_int8_ds_mla_layout = vllm_config.cache_config.cache_dtype == "int8_ds_mla"
        return SlidingWindowMLASpec(  # only has one vector instead of K + V
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            alignment=576
            if uses_fp8_ds_mla_layout
            else (528 if uses_int8_ds_mla_layout else 512),
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return CompressorBackend


class DeepseekCompressor(nn.Module):
    """DeepSeek V4 KV/score compressor.

    Owns the linear / norm / state-cache / ape state and the shared forward
    prologue (kv/score split, save_partial_states launch). The
    compress → norm → RoPE → store step is dispatched to a triton kernel
    (``compress_norm_rope_store_triton``) by default, except for the NVIDIA
    head_dim=512 path which uses the CuTeDSL compressor kernels for better
    performance.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        compress_ratio: int,
        hidden_size: int,
        head_dim: int,
        rotate: bool = False,
        prefix: str = "",
        k_cache_prefix="",
        use_fp4_cache: bool = False,
    ):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.rotate = rotate
        self.prefix = prefix
        self.k_cache_prefix = k_cache_prefix
        self.use_fp4_cache = use_fp4_cache

        config = vllm_config.model_config.hf_config
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.rms_norm_eps = config.rms_norm_eps
        self.device = current_platform.device_type
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = vllm_config.model_config.max_model_len

        self.overlap = compress_ratio == 4
        self.coff = 1 + self.overlap

        # The head=512 cr>=128 no-overlap deep gather uses the two-stage
        # compressor, which needs an fp32 scratch [max_batched, 512] for
        # the intermediate compressed_kv.
        # Currently only tested on ROCm
        self._use_two_stage_fused_compressor = (
            _prefer_two_stage_compressor() and head_dim == 512 and not self.overlap
        )
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self._compress_scratch: torch.Tensor | None = None
        if self._use_two_stage_fused_compressor:
            self._compress_scratch = torch.empty(
                self.max_num_batched_tokens,
                self.head_dim,
                dtype=torch.float32,
                device=self.device,
            )

        state_dtype = torch.float32
        self.ape = nn.Parameter(
            torch.empty(
                (compress_ratio, self.coff * self.head_dim),
                dtype=state_dtype,
                device=self.device,
            ),
            requires_grad=False,
        )

        self.fused_wkv_wgate = MergedColumnParallelLinear(
            self.hidden_size,
            [self.coff * self.head_dim, self.coff * self.head_dim],
            bias=False,
            return_bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.fused_wkv_wgate",
        )
        self.norm = RMSNorm(self.head_dim, self.rms_norm_eps)

        self.state_cache = CompressorStateCache(
            state_dim=2 * self.coff * self.head_dim,  # kv_state + score_state
            dtype=state_dtype,
            compress_ratio=compress_ratio,
            prefix=f"{prefix}.state_cache",
        )
        self._use_cutedsl_compressor = _uses_cutedsl_compressor(self.head_dim)
        if (
            not self._use_cutedsl_compressor
            and _compressor_ring_capacity(compress_ratio, head_dim, vllm_config)
            is not None
        ):
            self.state_cache.allocate_stage(vllm_config)

        # Save reference to static_forward_context for forward-time KV cache lookup.
        # get_current_vllm_config() is only available during __init__, not forward.
        self._static_forward_context = (
            vllm_config.compilation_config.static_forward_context
        )

        self._int8_ds_mla = vllm_config.cache_config.cache_dtype == "int8_ds_mla"
        self._pdl_kwargs = (
            {}
            if current_platform.is_rocm() or current_platform.is_xpu()
            else {"launch_pdl": False}
        )
        self._cuda_c128_boundary_shortcut = (
            current_platform.is_cuda()
            and self.head_dim == 512
            and self.compress_ratio == 128
        )
        if self._use_cutedsl_compressor:
            from .nvidia.ops.sparse_attn_compress_cutedsl import (
                _SPARSE_ATTN_COMPRESSOR_CUTEDSL_KERNEL,
            )

            self._compress_norm_rope_store_fn = _SPARSE_ATTN_COMPRESSOR_CUTEDSL_KERNEL
        elif self._use_two_stage_fused_compressor:
            self._compress_norm_rope_store_fn = (
                compress_norm_rope_store_two_stage_triton
            )
        else:
            self._compress_norm_rope_store_fn = compress_norm_rope_store_triton

        if self.head_dim == 512:
            assert not use_fp4_cache, (
                "MXFP4 cache is only supported for indexer (head=128)"
            )
            if self._int8_ds_mla:
                self._quant_block = 512
                self._token_stride = 528
                self._scale_dim = 4
            else:
                self._quant_block = 64
                self._token_stride = self.nope_head_dim + self.rope_head_dim * 2
                self._scale_dim = self.nope_head_dim // 64 + 1  # 7 real + 1 pad
        elif self.head_dim == 128:
            if use_fp4_cache:
                self._quant_block = MXFP4_BLOCK_SIZE
                self._token_stride = self.head_dim // 2
                self._scale_dim = self.head_dim // MXFP4_BLOCK_SIZE
            else:
                self._quant_block = 128
                self._token_stride = self.head_dim
                self._scale_dim = 4  # single float32 scale
        else:
            raise ValueError(
                f"Unsupported head_dim for fused quant+cache: {self.head_dim}"
            )

        if vllm_config.kernel_config.enable_jit_warmup:
            ring_capacity = _compressor_ring_capacity(
                self.compress_ratio, self.head_dim, vllm_config
            )
            if self._use_cutedsl_compressor:
                if ring_capacity is not None:
                    _build_c128_ring_metadata.register_warmup(capacity=ring_capacity)
                _SAVE_PARTIAL_STATES_KERNEL.register_warmup(
                    head_dim=self.head_dim,
                    compress_ratio=self.compress_ratio,
                )
            else:
                if ring_capacity is not None:
                    _build_c128_ring_metadata.register_warmup(
                        capacity=ring_capacity,
                        block_size=_state_block_size(self.compress_ratio),
                    )
                _SAVE_PARTIAL_STATES_KERNEL.register_warmup(
                    head_dim=self.head_dim,
                    compress_ratio=self.compress_ratio,
                    block_size=_state_block_size(self.compress_ratio),
                )
            # Same gate as the live kernel above: the CuTe DSL module cannot
            # even import on sm_8x and sm_12x, let alone warm up.
            if self._use_cutedsl_compressor:
                from vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl import (  # noqa: E501
                    _SPARSE_ATTN_COMPRESS_C128_RING_KERNEL,
                    _SPARSE_ATTN_COMPRESS_NORM_ROPE_STORE_C4_KERNEL,
                    _SPARSE_ATTN_COMPRESS_NORM_ROPE_STORE_FULL_C4_KERNEL,
                    _SPARSE_ATTN_NORM_ROPE_STORE_FULL_KERNEL,
                    _SPARSE_ATTN_NORM_ROPE_STORE_KERNEL,
                )

                store_full_kv = vllm_config.cache_config.cache_dtype != "fp8_ds_mla"
                if self.compress_ratio == 4:
                    (
                        _SPARSE_ATTN_COMPRESS_NORM_ROPE_STORE_FULL_C4_KERNEL
                        if store_full_kv
                        else _SPARSE_ATTN_COMPRESS_NORM_ROPE_STORE_C4_KERNEL
                    ).register_warmup()
                else:
                    _SPARSE_ATTN_COMPRESS_C128_RING_KERNEL.register_warmup()
                    if store_full_kv:
                        _SPARSE_ATTN_NORM_ROPE_STORE_FULL_KERNEL.register_warmup()
                    else:
                        _SPARSE_ATTN_NORM_ROPE_STORE_KERNEL.register_warmup(
                            vllm_config,
                            k_cache_prefix=self.k_cache_prefix,
                            compress_ratio=self.compress_ratio,
                        )
            else:
                from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (  # noqa: E501
                    _FUSED_KV_COMPRESS_NORM_ROPE_INSERT_INDEXER_TRITON_KERNEL,
                )

                _FUSED_KV_COMPRESS_NORM_ROPE_INSERT_INDEXER_TRITON_KERNEL.register_warmup()

    def forward(
        self,
        # [num_tokens, 2 * self.coff * self.head_dim]
        kv_score: torch.Tensor,
        # [num_tokens]
        positions: torch.Tensor,
        rotary_emb,
    ) -> None:
        # Each of shape [num_tokens, coff * self.head_dim]
        # input bf16, output are fp32
        kv, score = kv_score.split(
            [self.coff * self.head_dim, self.coff * self.head_dim], dim=-1
        )

        # Get the metadata and handle dummy profiling run.
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if not isinstance(attn_metadata, dict):
            return

        state_metadata = cast(
            CompressorMetadata, attn_metadata[self.state_cache.prefix]
        )
        token_to_req_indices = state_metadata.token_to_req_indices
        slot_mapping = state_metadata.slot_mapping
        tail_slot_mapping = state_metadata.tail_slot_mapping
        num_actual = slot_mapping.shape[0]
        block_table = state_metadata.block_table
        block_size = state_metadata.block_size

        # [num_blocks, block_size, kv_dim+score_dim], where kv_dim == score_dim
        state_cache = self.state_cache.kv_cache
        # kv_state stored in first half, score_state stored in second half
        state_width = state_cache.shape[-1] // 2
        # The ring the tail is saved into after compression.
        ring_cache = state_cache
        ring_stage = state_metadata.ring_stage

        # full graph cannot branch on per-step CPU metadata after capture
        skip_compress = (
            self._cuda_c128_boundary_shortcut
            and forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL
            and state_metadata.c128_boundary is False
        )
        if ring_stage is not None and not skip_compress:
            # Triton ring: stage the rows the step's boundaries read (the
            # ring rows before the chunk, then the chunk) and compress from
            # the stage with the paged kernels.
            assert state_metadata.ring_capacity is not None
            state_cache = self.state_cache.stage
            stage_ring_history(
                ring_cache,
                state_cache,
                block_table,
                state_metadata.query_start_loc,
                positions,
                ring_stage,
                state_metadata.ring_capacity,
                block_size,
                self.compress_ratio,
            )
            slot_mapping = ring_stage.slot_mapping
            block_table = ring_stage.block_table

        # Paged state (and the stage) stores before compression. The CuTe DSL
        # ring reads current-chunk rows directly. Circular state saves only the
        # ring tail after compression, so a long chunk cannot overwrite an
        # earlier group before it is consumed.
        # NOTE: PDL is disabled: both this kernel and the compress kernels
        # below depend on preceding kernel outputs (kv/score from the cublas
        # GEMM; state_cache from this kernel) but neither emits/waits on PDL
        # grid dependency primitives, so launch_pdl=True caused a
        # read-after-write race and non-deterministic output.
        if not state_metadata.is_circular or (
            ring_stage is not None and not skip_compress
        ):
            _SAVE_PARTIAL_STATES_KERNEL(
                kv=kv,
                score=score,
                ape=self.ape,
                positions=positions,
                state_cache=state_cache,
                slot_mapping=slot_mapping,
                block_size=block_size,
                state_width=state_width,
                compress_ratio=self.compress_ratio,
                pdl_kwargs=self._pdl_kwargs,
            )

        if skip_compress:
            _SAVE_PARTIAL_STATES_KERNEL(
                kv=kv,
                score=score,
                ape=self.ape,
                positions=positions,
                state_cache=state_cache,
                slot_mapping=tail_slot_mapping,
                block_size=block_size,
                state_width=state_width,
                compress_ratio=self.compress_ratio,
                pdl_kwargs=self._pdl_kwargs,
            )
            return

        # Fused: compress → RMSNorm → RoPE → FP8 quant → KV cache write.
        # RoPE requirements (kernel applies forward GPT-J style rotation):
        # - is_neox_style=False (interleaved pairs, NOT split-half)
        # - cos_sin_cache layout: [max_pos, rope_head_dim] with first half cos,
        #   second half sin (per-pair, length rope_head_dim // 2 each)
        # - applied to LAST rope_head_dim elements of head_dim
        # - position used: (positions // compress_ratio) * compress_ratio
        cos_sin_cache = rotary_emb.cos_sin_cache
        k_cache_metadata = cast(Any, attn_metadata[self.k_cache_prefix])
        k_cache_layer = self._static_forward_context[self.k_cache_prefix]
        kv_cache = k_cache_layer.kv_cache

        # Plain-row V4 reads a contiguous bf16 / per-tensor fp8 cache row.
        # Packed byte layouts use their own paged uint8 writers.
        store_full_kv = self.head_dim == 512 and kv_cache.dtype != torch.uint8
        store_full_fp8 = kv_cache.dtype == torch.float8_e4m3fn
        fp8_scale = (
            getattr(k_cache_layer, "_flashinfer_fp8_kv_scale", None)
            if store_full_fp8
            else None
        )

        # cutedsl (head=512) accepts the full-cache flags; triton (indexer/AMD)
        # does not, so the two callables have different signatures.
        if self._use_cutedsl_compressor:
            # head=512 on CUDA always uses cutedsl, for both the fp8_ds_mla
            # layout and the plain full-cache layout. The full-cache flags
            # are consumed only here.
            extra_kwargs: dict[str, Any] = dict(
                kv=kv,
                score=score,
                ape=self.ape,
                query_start_loc=state_metadata.query_start_loc,
                is_circular=state_metadata.is_circular,
                store_full_kv=store_full_kv,
                store_full_fp8=store_full_fp8,
                fp8_scale=fp8_scale,
            )
        elif self._use_two_stage_fused_compressor:
            # head=512 cr>=128 (no overlap): two-pass split compressor on the
            # prefill suffix, single-pass on the decode prefix.
            assert state_metadata.num_decode_tokens is not None
            assert not state_metadata.is_circular
            extra_kwargs = {
                "num_decode_tokens": state_metadata.num_decode_tokens,
                "compress_scratch": self._compress_scratch,
            }
        else:
            # Indexer path (head_dim == 128), sm_8x / sm_12x, or non-CUDA GPUs.
            extra_kwargs = {}

        self._compress_norm_rope_store_fn(
            state_cache=state_cache,
            num_actual=num_actual,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            slot_mapping=slot_mapping,
            block_table=block_table,
            block_size=block_size,
            state_width=state_width,
            cos_sin_cache=cos_sin_cache,
            kv_cache=kv_cache,
            k_cache_metadata=k_cache_metadata,
            pdl_kwargs=self._pdl_kwargs,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            compress_ratio=self.compress_ratio,
            overlap=self.overlap,
            use_fp4_cache=self.use_fp4_cache,
            rms_norm_weight=self.norm.weight,
            rms_norm_eps=self.rms_norm_eps,
            quant_block=self._quant_block,
            token_stride=self._token_stride,
            scale_dim=self._scale_dim,
            int8_ds_mla=self._int8_ds_mla and self.head_dim == 512,
            **extra_kwargs,
        )
        if state_metadata.is_circular:
            _SAVE_PARTIAL_STATES_KERNEL(
                kv=kv,
                score=score,
                ape=self.ape,
                positions=positions,
                state_cache=ring_cache,
                slot_mapping=tail_slot_mapping,
                block_size=block_size,
                state_width=state_width,
                compress_ratio=self.compress_ratio,
                pdl_kwargs=self._pdl_kwargs,
            )
