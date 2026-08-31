# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLAMetadataBuilder
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla import indexer as indexer_module
from vllm.v1.attention.backends.mla import sparse_swa as sparse_swa_module
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadataBuilder
from vllm.v1.kv_cache_interface import MLAAttentionSpec


@pytest.mark.parametrize(
    ("is_prefilling", "expected_decodes"),
    [(True, 0), (None, 1)],
)
def test_indexer_phase_selects_runtime_path_and_capture_fallback(
    monkeypatch, is_prefilling, expected_decodes
):
    """Runtime honors prompt phase; synthetic capture retains width fallback."""
    query_len = 4
    seq_len = 8004
    common = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, query_len], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, query_len], dtype=torch.int32),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([seq_len], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=query_len,
        max_query_len=query_len,
        max_seq_len=seq_len,
        block_table_tensor=torch.zeros((1, 64), dtype=torch.int32),
        slot_mapping=torch.arange(query_len, dtype=torch.int64),
        causal=True,
        is_prefilling=(
            torch.tensor([is_prefilling]) if is_prefilling is not None else None
        ),
    )

    builder = object.__new__(DeepseekV32IndexerMetadataBuilder)
    builder.decode_threshold = 6
    builder.use_flattening = True
    builder.use_pcp = False
    builder.compress_ratio = 1
    builder.dcp_world_size = 1
    builder.dcp_rank = 0
    builder.cp_kv_cache_interleave_size = 1
    builder.max_prefill_buffer_size = seq_len
    builder.use_fp4_indexer_cache = False
    builder.num_speculative_tokens = 5
    builder.kv_cache_spec = SimpleNamespace(storage_block_size=256, block_size=256)
    builder.decode_lens_buffer = torch.zeros(1024, dtype=torch.int32)
    builder.decode_seq_lens_buffer = torch.zeros(1024, dtype=torch.int32)
    builder.global_decode_seq_lens_buffer = torch.zeros(1024, dtype=torch.int32)
    builder.expanded_block_table_buffer = torch.zeros((1024, 64), dtype=torch.int32)
    builder.arange_buffer = torch.arange(1024, dtype=torch.int32)
    builder.offsets_buffer = torch.arange(6, dtype=torch.int32)
    builder.expanded_seq_lens_buffer = torch.zeros(1024, dtype=torch.int32)
    builder.scheduler_metadata_buffer = torch.empty((0, 2), dtype=torch.int32)

    monkeypatch.setattr(indexer_module.current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(
        builder,
        "_prepare_decode_tensors",
        lambda **kwargs: (
            kwargs["seq_lens"].unsqueeze(-1),
            kwargs["block_table"],
            kwargs["decode_lens"],
            kwargs["num_decodes"],
            False,
        ),
    )
    monkeypatch.setattr(
        indexer_module,
        "build_prefill_chunk_metadata",
        lambda *args, **kwargs: SimpleNamespace(),
    )

    metadata = builder.build(0, common)

    expected_prefills = 1 - expected_decodes
    assert metadata.num_decodes == expected_decodes
    assert metadata.num_decode_tokens == expected_decodes * query_len
    assert metadata.num_prefills == expected_prefills
    assert metadata.num_prefill_tokens == expected_prefills * query_len


def test_sparse_swa_short_prompt_tail_stays_on_prefill_metadata_path(monkeypatch):
    """SWA and indexer metadata must use the same prompt-phase boundary."""
    query_len = 4
    seq_len = 8004
    common = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, query_len], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, query_len], dtype=torch.int32),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([seq_len], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=query_len,
        max_query_len=query_len,
        max_seq_len=seq_len,
        block_table_tensor=torch.zeros((1, 64), dtype=torch.int32),
        slot_mapping=torch.arange(query_len, dtype=torch.int64),
        causal=True,
        is_prefilling=torch.tensor([True]),
    )

    builder = object.__new__(DeepseekSparseSWAMetadataBuilder)
    builder.decode_threshold = 6
    builder.window_size = 128
    builder.block_size = 256
    builder.is_dspark = False
    builder.token_to_req_indices = torch.zeros(1024, dtype=torch.int32)
    builder.is_valid_token = torch.zeros(1024, dtype=torch.bool)
    builder.decode_swa_indices = torch.zeros((1024, 1, 128), dtype=torch.int32)
    builder.decode_swa_lens = torch.zeros(1024, dtype=torch.int32)
    builder.prefill_swa_indices = torch.zeros((1024, 1, 128), dtype=torch.int32)
    builder.prefill_swa_lens = torch.zeros(1024, dtype=torch.int32)
    builder._build_deepseek_v4_metadata = lambda *args: {}
    builder.build_tile_scheduler = lambda _num_tokens: {
        "swaonly": None,
        "c4a": None,
        "c128a": None,
    }

    class NoopKernel:
        def __getitem__(self, _grid):
            return lambda *args, **kwargs: None

    monkeypatch.setattr(
        sparse_swa_module, "_compute_swa_indices_and_lens_kernel", NoopKernel()
    )

    metadata = builder.build(0, common)

    assert metadata.num_decodes == 0
    assert metadata.num_decode_tokens == 0
    assert metadata.num_prefills == 1
    assert metadata.num_prefill_tokens == query_len


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_indexer_builder_deepseek_v4_compressed_slot_mapping_uses_storage_block_size():
    """Regression test: DeepseekV4 compression path must compute slot_mapping from
    compressed positions, not reuse the uncompressed common metadata mapping.
    """
    device = torch.device("cuda")

    # storage_block_size = block_size // compress_ratio = 256 // 4 = 64
    kv_cache_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        compress_ratio=4,
    )
    vllm_config = create_vllm_config(
        model_name="deepseek-ai/DeepSeek-V2-Lite-Chat",
        max_model_len=1024,
        hf_config_override={
            "sliding_window": 128,
            "index_topk": 4,
            "compress_ratios": [4],
        },
    )
    builder = DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["dummy"],
        vllm_config=vllm_config,
        device=device,
    )

    # Construct a single request where:
    # - num_computed = 240 (=> compressed_pos_start = 60)
    # - query_len = 40 (=> num_groups = 10)
    # => compressed positions are 60..69 which cross the storage block boundary at 64.
    query_start_loc = torch.tensor([0, 40], dtype=torch.int32, device=device)
    query_start_loc_cpu = query_start_loc.cpu()
    seq_lens = torch.tensor([280], dtype=torch.int32, device=device)  # 240 + 40

    # Two blocks: compressed positions 0..63 map to block 5, 64..127 map to block 7.
    block_table_tensor = torch.tensor([[5, 7]], dtype=torch.int32, device=device)

    # Dummy uncompressed slot mapping (length == uncompressed num_actual_tokens).
    slot_mapping = torch.full((40,), -123, dtype=torch.int64, device=device)

    common = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens.cpu(),
        num_reqs=1,
        num_actual_tokens=40,
        max_query_len=40,
        max_seq_len=280,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=True,
    )

    md = builder.build(common_prefix_len=0, common_attn_metadata=common)

    # The compressed slot_mapping retains the original uncompressed size (40).
    # Only every compress_ratio-th position gets a valid slot; the rest are -1.
    assert md.slot_mapping.numel() == 40
    valid_slots = md.slot_mapping[md.slot_mapping >= 0]
    assert valid_slots.numel() == 10  # 40 tokens / compress_ratio 4

    storage_bs = kv_cache_spec.storage_block_size  # 64
    # Compressed positions 60..63 land in block 5, positions 64..69 in block 7.
    expected = torch.tensor(
        [
            5 * storage_bs + 60,
            5 * storage_bs + 61,
            5 * storage_bs + 62,
            5 * storage_bs + 63,
        ]
        + [
            7 * storage_bs + 0,
            7 * storage_bs + 1,
            7 * storage_bs + 2,
            7 * storage_bs + 3,
            7 * storage_bs + 4,
            7 * storage_bs + 5,
        ],
        dtype=torch.int64,
        device=device,
    )
    torch.testing.assert_close(valid_slots, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_v4_mtp_verifier_rows_stay_on_decode_metadata_path():
    device = torch.device("cuda")
    kv_cache_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        compress_ratio=4,
        model_version="deepseek_v4",
    )
    vllm_config = create_vllm_config(
        model_name="deepseek-ai/DeepSeek-V2-Lite-Chat",
        max_model_len=1024,
        hf_config_override={
            "sliding_window": 128,
            "index_topk": 4,
            "compress_ratios": [4],
        },
    )
    vllm_config.speculative_config = SimpleNamespace(
        num_speculative_tokens=1,
        parallel_drafting=False,
        use_dspark=lambda: False,
    )

    indexer = DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["dummy_indexer"],
        vllm_config=vllm_config,
        device=device,
    )
    flashmla = DeepseekV4FlashMLAMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["dummy_mla"],
        vllm_config=vllm_config,
        device=device,
    )
    sparse_swa = DeepseekSparseSWAMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["dummy_swa"],
        vllm_config=vllm_config,
        device=device,
    )

    assert indexer.reorder_batch_threshold is None
    assert indexer.decode_threshold == 2
    assert flashmla.reorder_batch_threshold == 2
    assert sparse_swa.decode_threshold == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_v4_pp_mtp_forces_flattened_indexer_decode_metadata():
    device = torch.device("cuda")
    kv_cache_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        compress_ratio=128,
        model_version="deepseek_v4",
    )
    vllm_config = create_vllm_config(
        model_name="deepseek-ai/DeepSeek-V2-Lite-Chat",
        max_model_len=1024,
        hf_config_override={
            "sliding_window": 128,
            "index_topk": 4,
            "compress_ratios": [128],
        },
    )
    vllm_config.speculative_config = SimpleNamespace(
        num_speculative_tokens=4,
        parallel_drafting=False,
    )
    vllm_config.parallel_config.pipeline_parallel_size = 2

    indexer = DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["dummy_indexer"],
        vllm_config=vllm_config,
        device=device,
    )

    assert indexer.use_flattening
