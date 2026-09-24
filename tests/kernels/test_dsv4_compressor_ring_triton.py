# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton compressor kernels on the circular state (sm_8x / sm_12x path).

Each case runs the same step sequence twice: once on the paged sliding-window
state (save every row, then compress) and once on the multi-block ring (stage
the ring rows before each chunk and the chunk, compress from the stage, then
save the chunk tail into the ring). Both compress with the same paged kernels,
so the compressed KV written to the cache must be identical byte for byte.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    compress_norm_rope_store_triton,
)
from vllm.models.deepseek_v4.common.ops.save_partial_states import (
    _SAVE_PARTIAL_STATES_KERNEL,
)
from vllm.models.deepseek_v4.compressor import (
    _STAGE_PAD,
    RingStage,
    _c128_ring_capacity,
    _ring_stage_num_blocks,
    _ring_stage_table_width,
    build_c128_ring_metadata,
    build_ring_stage_metadata,
    stage_ring_history,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton compressor kernels need a GPU"
)

ROPE_DIM = 64
NUM_SPEC = 5
SLOTS_PER_REQ = 4096
KV_BLOCK_SIZE = 4
MAX_MODEL_LEN = 4096


@dataclass
class _Layer:
    head_dim: int
    compress_ratio: int

    @property
    def coff(self) -> int:
        return 2 if self.compress_ratio == 4 else 1

    @property
    def state_width(self) -> int:
        return self.coff * self.head_dim

    @property
    def block_size(self) -> int:
        return 4 if self.compress_ratio == 4 else 8

    @property
    def capacity(self) -> int:
        return _c128_ring_capacity(NUM_SPEC)

    @property
    def cache_format(self) -> dict:
        if self.head_dim == 512:
            # int8_ds_mla: 512 signed-int8 + fp32 row scale + pad.
            return dict(
                quant_block=512, token_stride=528, scale_dim=4, int8_ds_mla=True
            )
        # Indexer: 128 fp8 + fp32 scale.
        return dict(quant_block=128, token_stride=128, scale_dim=4)

    def new_kv_cache(self, device) -> torch.Tensor:
        num_blocks = 4 * SLOTS_PER_REQ // KV_BLOCK_SIZE
        if self.head_dim == 512:
            shape = (num_blocks, KV_BLOCK_SIZE, 528)
        else:
            shape = (num_blocks, KV_BLOCK_SIZE, 132)
        return torch.zeros(shape, dtype=torch.uint8, device=device)


class _Weights:
    def __init__(self, layer: _Layer, max_pos: int, device) -> None:
        g = torch.Generator(device=device).manual_seed(1234)
        self.ape = torch.randn(
            layer.compress_ratio,
            layer.state_width,
            generator=g,
            device=device,
            dtype=torch.float32,
        )
        self.rms_weight = torch.randn(
            layer.head_dim, generator=g, device=device, dtype=torch.float32
        ).to(torch.bfloat16)
        self.cos_sin_cache = torch.randn(
            max_pos, ROPE_DIM, generator=g, device=device, dtype=torch.float32
        )


def _step_inputs(layer, chunks, device):
    """Kv / score rows of one step. Each chunk draws its own rows, so a
    re-scheduled position (rejected drafts) gets fresh values."""
    positions = torch.cat([torch.arange(s, s + n) for _, s, n in chunks]).to(device)
    reqs = torch.cat([torch.full((n,), r, dtype=torch.int32) for r, _, n in chunks]).to(
        device
    )
    kv_score = torch.cat(
        [
            torch.randn(
                n,
                2 * layer.state_width,
                generator=torch.Generator(device=device).manual_seed(
                    (r * 1_000_003 + s) * 1009 + n
                ),
                device=device,
                dtype=torch.float32,
            )
            for r, s, n in chunks
        ]
    )
    query_start_loc = torch.zeros(len(chunks) + 1, dtype=torch.int32)
    query_start_loc[1:] = torch.cumsum(torch.tensor([n for _, _, n in chunks]), 0)
    return positions, reqs, kv_score, query_start_loc.to(device)


def _kv_slots(layer, positions, reqs, first_written_pos=None):
    slots = reqs.long() * SLOTS_PER_REQ + positions // layer.compress_ratio
    if first_written_pos is not None:
        # Prefix replay: the replayed rows' cache slots are padded.
        for req, first in first_written_pos.items():
            slots = torch.where((reqs == req) & (positions < first), -1, slots)
    return slots


def _compress(layer, weights, state, kv_cache, **kwargs):
    compress_norm_rope_store_triton(
        state_cache=state,
        state_width=layer.state_width,
        cos_sin_cache=weights.cos_sin_cache,
        kv_cache=kv_cache,
        pdl_kwargs={},
        head_dim=layer.head_dim,
        rope_head_dim=ROPE_DIM,
        compress_ratio=layer.compress_ratio,
        overlap=layer.compress_ratio == 4,
        use_fp4_cache=False,
        rms_norm_weight=weights.rms_weight,
        rms_norm_eps=1e-6,
        **layer.cache_format,
        **kwargs,
    )


def _save(layer, weights, state, kv, score, positions, slots):
    _SAVE_PARTIAL_STATES_KERNEL(
        kv=kv,
        score=score,
        ape=weights.ape,
        positions=positions,
        state_cache=state,
        slot_mapping=slots,
        block_size=layer.block_size,
        state_width=layer.state_width,
        compress_ratio=layer.compress_ratio,
        pdl_kwargs={},
    )


def _state(num_blocks, layer, pool_stride, fill, device):
    """[num_blocks, block_size, 2 * state_width] fp32 view with a pool-like
    block stride (the packed KV pool strides blocks by the whole slot)."""
    row = 2 * layer.state_width
    stride = pool_stride or layer.block_size * row
    storage = torch.full((num_blocks * stride,), fill, device=device)
    return torch.as_strided(
        storage, (num_blocks, layer.block_size, row), (stride, row, 1)
    )


def _run(layer, steps, *, ring: bool, replay=None, pool_stride=None, device="cuda"):
    """Run ``steps`` (lists of (req, start, length) chunks) and return the
    compressed cache. ``replay`` maps a request to its prefix-hit position: the
    ring starts empty there (NaN-filled) and rows below it are not written.

    Paged: save every row into the paged state, then compress (trunk). Ring:
    compress from a stage holding the ring rows before each chunk and the chunk
    itself, then save the chunk tail into the ring (the Triton ring path of
    ``DeepseekCompressor.forward``)."""
    max_pos = max(s + n for chunks in steps for _, s, n in chunks) + 1
    weights = _Weights(layer, max_pos, device)
    kv_cache = layer.new_kv_cache(device)
    num_reqs = 1 + max(r for chunks in steps for r, _, _ in chunks)
    bs = layer.block_size
    config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=MAX_MODEL_LEN))
    # The paged state's block table width, as the worker sizes it.
    paged_width = _ring_stage_table_width(config, bs)
    if ring:
        width = layer.capacity // bs
        state = _state(num_reqs * width + 1, layer, pool_stride, float("nan"), device)
        # Scatter the ring blocks so neighbouring requests interleave.
        block_ids = torch.randperm(num_reqs * width, generator=torch.Generator())
        block_table = (1 + block_ids.view(num_reqs, width)).to(torch.int32)
        max_tokens = max(sum(n for _, _, n in chunks) for chunks in steps)
        stage_blocks = _ring_stage_num_blocks(
            max_tokens, num_reqs, layer.coff * layer.compress_ratio, bs
        )
        stage = _state(stage_blocks, layer, None, float("nan"), device)
        if pool_stride is not None and pool_stride % 16:
            stage = _state(
                stage_blocks,
                layer,
                bs * 2 * layer.state_width + _STAGE_PAD,
                float("nan"),
                device,
            )
        stage_table = torch.zeros(
            num_reqs, paged_width, dtype=torch.int32, device=device
        )
        stage_base = torch.zeros(num_reqs, dtype=torch.int64, device=device)
        stage_lo = torch.zeros(num_reqs, dtype=torch.int64, device=device)
    else:
        state = _state(num_reqs * paged_width + 1, layer, pool_stride, 0.0, device)
        block_table = 1 + torch.arange(num_reqs * paged_width, dtype=torch.int32).view(
            num_reqs, -1
        )
    block_table = block_table.to(device)

    for chunks in steps:
        positions, reqs, kv_score, query_start_loc = _step_inputs(layer, chunks, device)
        kv, score = kv_score.split([layer.state_width, layer.state_width], dim=-1)
        kv_slots = _kv_slots(layer, positions, reqs, replay)
        common = dict(
            num_actual=positions.numel(),
            token_to_req_indices=reqs,
            positions=positions,
            block_size=bs,
            k_cache_metadata=SimpleNamespace(slot_mapping=kv_slots),
        )
        if ring:
            _, tail_slots = build_c128_ring_metadata(
                torch.full_like(positions, -1),
                block_table,
                query_start_loc,
                positions,
                positions.numel(),
                len(chunks),
                layer.capacity,
                token_to_req_indices=reqs,
                block_size=bs,
            )
            stage_slots = torch.empty_like(positions)
            window = layer.coff * layer.compress_ratio
            build_ring_stage_metadata(
                query_start_loc,
                positions,
                len(chunks),
                window,
                bs,
                out_block_table=stage_table,
                out_slots=stage_slots,
                out_base=stage_base,
                out_lo=stage_lo,
            )
            ring_stage = RingStage(
                block_table=stage_table[: len(chunks)],
                slot_mapping=stage_slots,
                base=stage_base[: len(chunks)],
                lo=stage_lo[: len(chunks)],
                window=window,
            )
            stage_ring_history(
                state,
                stage,
                block_table,
                query_start_loc,
                positions,
                ring_stage,
                layer.capacity,
                bs,
                layer.compress_ratio,
            )
            _save(layer, weights, stage, kv, score, positions, stage_slots)
            _compress(
                layer,
                weights,
                stage,
                kv_cache,
                slot_mapping=stage_slots,
                block_table=ring_stage.block_table,
                **common,
            )
            _save(layer, weights, state, kv, score, positions, tail_slots)
        else:
            rows = positions.long()
            slots = block_table[reqs.long(), rows // bs].long() * bs + rows % bs
            _save(layer, weights, state, kv, score, positions, slots)
            _compress(
                layer,
                weights,
                state,
                kv_cache,
                slot_mapping=slots,
                block_table=block_table,
                **common,
            )
    torch.accelerator.synchronize()
    return kv_cache


# Two requests: chunked prefill (one chunk longer than the C128 ring), decode
# steps with DSpark-sized spec blocks, rejected drafts rescheduled from an
# earlier position, and a long chunk after decode.
STEPS = [
    [(0, 0, 300), (1, 0, 37)],
    [(0, 300, 6), (1, 37, 6)],
    [(0, 302, 6), (1, 43, 6)],
    [(0, 308, 600), (1, 45, 6)],
    [(0, 908, 6), (1, 51, 250)],
    [(0, 910, 6), (1, 301, 6)],
]

LAYERS = [
    pytest.param(_Layer(512, 128), id="c128-main"),
]


@pytest.mark.parametrize("layer", LAYERS)
@pytest.mark.parametrize(
    "pool_stride",
    # Contiguous blocks, and the 43,552 B slot of the 0731 mini (10,888 floats).
    [None, 10888],
)
def test_ring_matches_paged_state(layer: _Layer, pool_stride) -> None:
    paged = _run(layer, STEPS, ring=False, pool_stride=pool_stride)
    ring = _run(layer, STEPS, ring=True, pool_stride=pool_stride)
    assert paged.any()
    assert torch.equal(ring, paged)


@pytest.mark.parametrize("layer", LAYERS)
def test_ring_prefix_replay_rebuilds_the_open_group(layer: _Layer) -> None:
    """A prefix hit at 512 (a multiple of the scheduler block) hands the
    request an empty ring. Replaying the hit's last ``window - ratio`` tokens
    with their cache slots padded restores every later compressed row."""
    hit = 512
    replay = layer.coff * layer.compress_ratio - layer.compress_ratio
    history = [[(0, 0, hit)]]
    tail = [
        [(0, hit - replay, 200 + replay)],
        [(0, hit + 200, 6)],
        [(0, hit + 203, 6)],
        [(0, hit + 209, 300)],
    ]
    reference = _run(layer, history + tail, ring=False)
    resumed = _run(layer, tail, ring=True, replay={0: hit})
    # Cache blocks holding compressed rows at or after the hit.
    blocks = slice(hit // layer.compress_ratio // KV_BLOCK_SIZE, None)
    assert resumed[blocks].any()
    assert torch.equal(resumed[blocks], reference[blocks])
    # Nothing below the hit is rewritten.
    assert not resumed[: blocks.start].any()
