# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.compressor import (
    CompressorMetadataBuilder,
    CompressorStateCache,
    _c128_ring_capacity,
    build_c128_ring_metadata,
)
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import CircularBufferSpec, SlidingWindowMLASpec

pytestmark = pytest.mark.skip_global_cleanup


def _state_cache(compress_ratio: int, head_dim: int = 512) -> CompressorStateCache:
    cache = object.__new__(CompressorStateCache)
    coff = 2 if compress_ratio == 4 else 1
    cache.compress_ratio = compress_ratio
    cache.head_dim = head_dim
    cache.state_dim = 2 * coff * head_dim
    cache.dtype = torch.float32
    cache.block_size = 4 if compress_ratio == 4 else 8
    cache.sliding_window = coff * compress_ratio
    return cache


def _config(
    num_speculative_tokens: int = 0,
    enable_prefix_caching: bool = False,
    use_v2_model_runner: bool = True,
):
    return SimpleNamespace(
        num_speculative_tokens=num_speculative_tokens,
        cache_config=SimpleNamespace(
            cache_dtype="auto", enable_prefix_caching=enable_prefix_caching
        ),
        use_v2_model_runner=use_v2_model_runner,
        parallel_config=SimpleNamespace(use_ubatching=False),
    )


def _platform(monkeypatch, platform: str) -> None:
    """cutedsl: CUDA with the CuTe DSL head-512 compressor (sm_90 / sm_100).
    triton: CUDA on the Triton compressor kernels (sm_8x / sm_12x)."""
    from vllm.models.deepseek_v4 import compressor

    monkeypatch.setattr(
        compressor.current_platform, "is_cuda", lambda: platform != "cpu"
    )
    monkeypatch.setattr(
        compressor,
        "_uses_cutedsl_compressor",
        lambda head_dim: platform == "cutedsl" and head_dim == 512,
    )


@pytest.mark.parametrize("platform", ["cutedsl", "triton", "cpu"])
def test_c128_spec_is_circular_only_on_cuda(monkeypatch, platform: str) -> None:
    _platform(monkeypatch, platform)
    c128_spec = _state_cache(128).get_kv_cache_spec(_config(5))
    if platform == "cpu":
        assert isinstance(c128_spec, SlidingWindowMLASpec)
        assert c128_spec.block_size == 8
        assert c128_spec.sliding_window == 128
    else:
        assert isinstance(c128_spec, CircularBufferSpec)
        assert c128_spec.prefix_cacheable is False
        assert c128_spec.uses_slot_mapping is False
        assert c128_spec.ring_capacity == 256
        if platform == "cutedsl":
            # Upstream layout: one block holding the whole ring.
            assert (c128_spec.block_size, c128_spec.num_ring_blocks) == (256, 1)
        else:
            # Rings of paged-state-sized blocks keep the pool slot unchanged.
            assert (c128_spec.block_size, c128_spec.num_ring_blocks) == (8, 32)

    c4_spec = _state_cache(4).get_kv_cache_spec(_config(5))
    assert isinstance(c4_spec, SlidingWindowMLASpec)
    assert c4_spec.block_size == 4
    assert c4_spec.sliding_window == 8


def test_c128_capacity_covers_one_speculative_step() -> None:
    assert [_c128_ring_capacity(n) for n in (0, 1, 127, 128, 129)] == [
        128,
        256,
        256,
        256,
        384,
    ]


@pytest.mark.parametrize("group_phase", range(128))
def test_c128_ring_survives_one_speculative_step(group_phase: int) -> None:
    """Draft writes must not overwrite committed rows in the open group."""
    num_spec = 5
    capacity = _c128_ring_capacity(num_spec)
    query_len = num_spec + 1
    positions = torch.arange(group_phase, group_phase + query_len)
    slots, _ = build_c128_ring_metadata(
        torch.full((query_len,), -1, dtype=torch.int64),
        torch.tensor([[0]], dtype=torch.int32),
        torch.tensor([0, query_len], dtype=torch.int32),
        positions,
        query_len,
        1,
        capacity,
    )

    committed = torch.arange(group_phase - group_phase % 128, group_phase)
    assert set(slots.tolist()).isdisjoint((committed % capacity).tolist())


@pytest.mark.parametrize(
    ("is_cuda", "is_rocm", "expected"),
    [(True, False, [{"capacity": 256, "block_size": 8}]), (False, True, [])],
)
def test_c128_model_registers_ring_metadata_warmup_only_on_cuda(
    monkeypatch, is_cuda: bool, is_rocm: bool, expected: list[dict]
) -> None:
    from vllm.models.deepseek_v4 import compressor
    from vllm.models.deepseek_v4.common.ops import fused_compress_quant_cache

    registered = []
    monkeypatch.setattr(
        compressor,
        "current_platform",
        SimpleNamespace(
            device_type="cpu",
            is_cuda=lambda: is_cuda,
            is_rocm=lambda: is_rocm,
            is_xpu=lambda: False,
            is_device_capability_family=lambda family: family == 80,
        ),
    )
    monkeypatch.setattr(
        compressor._build_c128_ring_metadata,
        "register_warmup",
        lambda **kwargs: registered.append(kwargs),
    )
    monkeypatch.setattr(
        compressor._SAVE_PARTIAL_STATES_KERNEL, "register_warmup", lambda **kwargs: None
    )
    indexer_kernel_name = "_FUSED_KV_COMPRESS_NORM_ROPE_INSERT_INDEXER_TRITON_KERNEL"
    indexer_kernel = getattr(fused_compress_quant_cache, indexer_kernel_name)
    monkeypatch.setattr(indexer_kernel, "register_warmup", lambda **kwargs: None)
    monkeypatch.setattr(
        compressor,
        "MergedColumnParallelLinear",
        lambda *args, **kwargs: torch.nn.Identity(),
    )
    monkeypatch.setattr(
        compressor, "RMSNorm", lambda *args, **kwargs: torch.nn.Identity()
    )

    class _StateCache(torch.nn.Identity):
        def allocate_stage(self, vllm_config) -> None:
            pass

    monkeypatch.setattr(
        compressor, "CompressorStateCache", lambda *args, **kwargs: _StateCache()
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(qk_rope_head_dim=64, rms_norm_eps=1e-6),
            max_model_len=1024,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=1,
            max_num_batched_tokens=128,
        ),
        compilation_config=SimpleNamespace(static_forward_context={}),
        kernel_config=SimpleNamespace(enable_jit_warmup=True),
        cache_config=SimpleNamespace(cache_dtype="auto", enable_prefix_caching=False),
        parallel_config=SimpleNamespace(use_ubatching=False),
        num_speculative_tokens=5,
    )

    compressor.DeepseekCompressor(config, 128, 128, 128)

    assert registered == expected


def test_only_circular_c128_allocates_ring_metadata_buffers(monkeypatch) -> None:
    from vllm.models.deepseek_v4 import compressor

    _platform(monkeypatch, "cutedsl")
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=127,
            async_scheduling=False,
        ),
        speculative_config=None,
    )

    def build(spec):
        builder = CompressorMetadataBuilder(
            spec, ["state"], config, torch.device("cpu")
        )
        query_start_loc = torch.tensor([0, 127], dtype=torch.int32)
        common = CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc,
            seq_lens=torch.tensor([127], dtype=torch.int32),
            seq_lens_cpu_upper_bound=torch.tensor([127], dtype=torch.int32),
            num_reqs=1,
            num_actual_tokens=127,
            max_query_len=127,
            max_seq_len=127,
            block_table_tensor=torch.tensor([[0]], dtype=torch.int32),
            slot_mapping=torch.full((127,), -1, dtype=torch.int64),
            positions=torch.arange(127),
        )
        common._token_to_req_indices_cache = torch.zeros(127, dtype=torch.int32)
        return builder, builder.build(0, common)

    circular = _state_cache(128).get_kv_cache_spec(_config())
    circular_builder, circular_metadata = build(circular)
    assert circular_builder.slot_mapping_buffer is not None
    assert circular_builder.tail_slot_mapping_buffer is not None
    assert circular_builder.slot_mapping_buffer.shape == (127,)
    assert circular_builder.tail_slot_mapping_buffer.shape == (127,)
    assert circular_builder.slot_mapping_buffer.data_ptr() % 16 == 0
    assert circular_builder.tail_slot_mapping_buffer.data_ptr() % 16 == 0
    assert circular_metadata.c128_boundary is False

    c4 = _state_cache(4).get_kv_cache_spec(_config())
    c4_builder, c4_metadata = build(c4)
    assert c4_builder.slot_mapping_buffer is None
    assert c4_builder.tail_slot_mapping_buffer is None
    assert c4_metadata.c128_boundary is None

    monkeypatch.setattr(compressor.current_platform, "is_cuda", lambda: False)
    paged_c128 = _state_cache(128).get_kv_cache_spec(_config())
    paged_c128_builder, paged_c128_metadata = build(paged_c128)
    assert paged_c128_builder.slot_mapping_buffer is None
    assert paged_c128_builder.tail_slot_mapping_buffer is None
    assert paged_c128_metadata.c128_boundary is None


def test_c128_ring_mapping_and_tail_for_nonuniform_batch() -> None:
    per_req = [torch.arange(120, 400), torch.arange(250, 390)]
    positions = torch.cat(per_req)
    query_start_loc = torch.tensor([0, len(per_req[0]), positions.numel()])
    num_actual = positions.numel()
    padded = num_actual + 7
    common_slots = torch.full((padded,), -1, dtype=torch.int64)
    block_table = torch.tensor([[11, -1], [3, -1]], dtype=torch.int32)

    slots, tail_slots = build_c128_ring_metadata(
        common_slots,
        block_table,
        query_start_loc,
        positions,
        num_actual,
        2,
        256,
    )

    expected_slots = torch.cat(
        [11 * 256 + per_req[0] % 256, 3 * 256 + per_req[1] % 256]
    )
    assert torch.equal(slots[:num_actual], expected_slots)
    assert slots[num_actual:].tolist() == [-1] * 7
    assert tail_slots[:24].tolist() == [-1] * 24
    assert torch.equal(tail_slots[24 : len(per_req[0])], expected_slots[24:280])
    assert torch.equal(tail_slots[280:num_actual], expected_slots[280:num_actual])
    assert tail_slots[num_actual:].tolist() == [-1] * 7


def test_c128_ring_mapping_masks_padded_actual_tokens() -> None:
    slots, tail_slots = build_c128_ring_metadata(
        torch.full((4,), -1, dtype=torch.int64),
        torch.tensor([[2]], dtype=torch.int32),
        torch.tensor([0, 3], dtype=torch.int32),
        torch.tensor([10, 11, 12, 0]),
        num_actual_tokens=4,
        num_reqs=1,
        capacity=128,
        token_to_req_indices=torch.zeros(4, dtype=torch.int32),
    )

    assert slots.tolist() == [266, 267, 268, -1]
    assert tail_slots.tolist() == [266, 267, 268, -1]


def test_c128_null_block_keeps_all_slots_invalid() -> None:
    positions = torch.arange(120, 140)
    common_slots = torch.full((24,), -1, dtype=torch.int64)
    slots, tail_slots = build_c128_ring_metadata(
        common_slots,
        torch.tensor([[-1]], dtype=torch.int32),
        torch.tensor([0, 20]),
        positions,
        20,
        1,
        256,
    )
    assert slots.tolist() == [-1] * 24
    assert tail_slots.tolist() == [-1] * 24


@pytest.mark.parametrize("on_cuda", [False, True])
def test_multi_block_ring_mapping(on_cuda: bool) -> None:
    """A ring of capacity 16 in blocks of 4: row ``pos % 16`` lives in block
    table entry ``row // 4`` at offset ``row % 4``; the tail keeps the last
    ``capacity`` rows of each request."""
    if on_cuda and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = "cuda" if on_cuda else "cpu"
    per_req = [torch.arange(0, 21), torch.arange(37, 40)]
    positions = torch.cat(per_req).to(device)
    num_actual = positions.numel()
    query_start_loc = torch.tensor(
        [0, len(per_req[0]), num_actual], dtype=torch.int32, device=device
    )
    block_table = torch.tensor(
        [[5, 9, 2, 7], [1, 3, 4, 8]], dtype=torch.int32, device=device
    )
    slots, tail_slots = build_c128_ring_metadata(
        torch.full((num_actual + 3,), -1, dtype=torch.int64, device=device),
        block_table,
        query_start_loc,
        positions,
        num_actual,
        2,
        16,
        token_to_req_indices=torch.tensor(
            [0] * len(per_req[0]) + [1] * len(per_req[1]),
            dtype=torch.int32,
            device=device,
        ),
        block_size=4,
    )

    expected = []
    for req, pos in enumerate(per_req):
        row = pos % 16
        expected.append(block_table[req].cpu()[row // 4].long() * 4 + row % 4)
    expected_slots = torch.cat(expected)
    assert torch.equal(slots[:num_actual].cpu(), expected_slots)
    assert slots[num_actual:].tolist() == [-1] * 3
    # Request 0 has 21 rows: the first 5 are overwritten within the chunk.
    assert tail_slots[:5].tolist() == [-1] * 5
    assert torch.equal(tail_slots[5:num_actual].cpu(), expected_slots[5:])


def test_ring_rejects_capacity_not_a_multiple_of_block() -> None:
    with pytest.raises(ValueError):
        build_c128_ring_metadata(
            torch.full((1,), -1, dtype=torch.int64),
            torch.zeros((1, 4), dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0]),
            1,
            1,
            18,
            block_size=4,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_ring_builder_stages_each_chunk_with_its_history(monkeypatch) -> None:
    """The multi-block ring stages each request's chunk and the window - 1 rows
    before it in consecutive stage blocks, addressed through a block table as
    wide as the paged state's."""
    _platform(monkeypatch, "triton")
    spec = _state_cache(128).get_kv_cache_spec(_config(5))
    layer = SimpleNamespace(sliding_window=128)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=512, max_num_seqs=4, async_scheduling=False
        ),
        speculative_config=None,
        compilation_config=SimpleNamespace(static_forward_context={"state": layer}),
        model_config=SimpleNamespace(max_model_len=4096),
    )
    device = torch.device("cuda")
    builder = CompressorMetadataBuilder(spec, ["state"], config, device)
    chunks = [(300, 200), (0, 7), (1000, 6)]
    positions = torch.cat([torch.arange(s, s + n) for s, n in chunks]).to(device)
    query_start_loc = torch.tensor([0, 200, 207, 213], dtype=torch.int32)
    seq_lens = torch.tensor([500, 7, 1006], dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=query_start_loc.to(device),
        query_start_loc_cpu=query_start_loc,
        seq_lens=seq_lens.to(device),
        seq_lens_cpu_upper_bound=seq_lens,
        num_reqs=3,
        num_actual_tokens=213,
        max_query_len=200,
        max_seq_len=1006,
        block_table_tensor=torch.arange(96, dtype=torch.int32, device=device).view(
            3, 32
        ),
        slot_mapping=torch.full((216,), -1, dtype=torch.int64, device=device),
        positions=torch.cat(
            [positions, torch.zeros(3, dtype=torch.int64, device=device)]
        ),
    )
    stage = builder.build(0, common).ring_stage
    assert stage is not None and stage.window == 128
    assert stage.block_table.shape[1] == 512  # cdiv(4096, 8), 16-block aligned
    # Request 0 stages positions [173, 500): logical blocks 21..62. Request 1
    # stages [0, 7): block 0. Request 2 stages [873, 1006): blocks 109..125.
    assert stage.lo.tolist() == [21, 0, 109]
    assert stage.base.tolist() == [0, 42, 43]
    table = stage.block_table.cpu()
    assert table[0, 21:63].tolist() == list(range(42))
    assert table[1, 0].item() == 42
    assert table[2, 109:126].tolist() == list(range(43, 60))
    slots = stage.slot_mapping.cpu()
    pos = positions.cpu()
    for req, (lo, base) in enumerate(zip([21, 0, 109], [0, 42, 43])):
        rows = slice(*query_start_loc[req : req + 2].tolist())
        expected = (base + pos[rows] // 8 - lo) * 8 + pos[rows] % 8
        assert torch.equal(slots[rows], expected)
    assert slots[213:].tolist() == [-1] * 3
