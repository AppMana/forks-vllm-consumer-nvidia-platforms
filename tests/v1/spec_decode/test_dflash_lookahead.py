# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.core.utils import create_requests
from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_module
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

# Matches defaults from tests/v1/spec_decode/test_eagle.py
DFLASH_TARGET_DIR = "Qwen/Qwen3-8B"
DFLASH_DRAFT_DIR = "z-lab/Qwen3-8B-DFlash-b16"

BLOCK_SIZE = 16
NUM_BLOCKS = 8
NUM_SPECULATIVE_TOKENS = 3


def _dflash_speculative_config(num_speculative_tokens: int) -> SpeculativeConfig:
    model_config = ModelConfig(
        model=DFLASH_TARGET_DIR,
        runner="generate",
        max_model_len=100,
        trust_remote_code=True,
    )
    return SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        model=DFLASH_DRAFT_DIR,
        method="dflash",
        num_speculative_tokens=num_speculative_tokens,
    )


def _create_dflash_scheduler(num_speculative_tokens: int) -> Scheduler:
    speculative_config = _dflash_speculative_config(num_speculative_tokens)
    model_config = speculative_config.target_model_config
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=8192,
        max_model_len=model_config.max_model_len,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
        speculative_config=speculative_config,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    cache_config.num_gpu_blocks = NUM_BLOCKS
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK_SIZE,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


def test_dflash_prefill_reserves_lookahead_blocks():
    scheduler = _create_dflash_scheduler(NUM_SPECULATIVE_TOKENS)

    assert scheduler.num_lookahead_tokens == NUM_SPECULATIVE_TOKENS + 1

    (request,) = create_requests(
        num_requests=1,
        num_tokens=BLOCK_SIZE,
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens[request.request_id] == BLOCK_SIZE
    # prefill block + one lookahead block
    assert len(output.scheduled_new_reqs[0].block_ids[0]) == 2


def test_dflash_first_prefill_query_window_fits_allocated_blocks():
    scheduler = _create_dflash_scheduler(NUM_SPECULATIVE_TOKENS)

    (request,) = create_requests(
        num_requests=1,
        num_tokens=BLOCK_SIZE,
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(request)

    output = scheduler.schedule()
    block_ids = output.scheduled_new_reqs[0].block_ids[0]
    query_positions = range(BLOCK_SIZE, BLOCK_SIZE + scheduler.num_lookahead_tokens)

    assert all(pos // BLOCK_SIZE < len(block_ids) for pos in query_positions)


def test_dflash_drafter_window_reserves_bonus_token():
    # DFlash's drafter window is num_spec + 1 (the extra slot is the bonus token),
    # so max_seq_len + num_spec + 1 must stay within the draft model's max len.
    input_fits_in_drafter = GPUModelRunner._input_fits_in_drafter
    dflash_runner = SimpleNamespace(
        num_spec_tokens=NUM_SPECULATIVE_TOKENS,
        effective_drafter_max_model_len=100,
        speculative_config=_dflash_speculative_config(NUM_SPECULATIVE_TOKENS),
    )
    # window = 4, so 96 fits (96 + 4 == 100) but 97 does not (97 + 4 == 101)
    assert input_fits_in_drafter(dflash_runner, SimpleNamespace(max_seq_len=96))
    assert not input_fits_in_drafter(dflash_runner, SimpleNamespace(max_seq_len=97))
    assert not input_fits_in_drafter(dflash_runner, None)  # no metadata

    # Other drafters don't reserve the bonus token, so 97 fits (97 + 3 == 100).
    plain_runner = SimpleNamespace(
        num_spec_tokens=NUM_SPECULATIVE_TOKENS,
        effective_drafter_max_model_len=100,
        speculative_config=SimpleNamespace(use_dflash=lambda: False),
    )
    assert input_fits_in_drafter(plain_runner, SimpleNamespace(max_seq_len=97))


@pytest.mark.parametrize("generate_draft", [False, True])
def test_dflash_context_update_only_drafts_when_requested(
    monkeypatch,
    generate_draft: bool,
) -> None:
    speculator = object.__new__(DFlashSpeculator)
    speculator.num_query_per_req = 5
    speculator.num_speculative_steps = 5
    speculator.max_model_len = 8192
    speculator.max_num_reqs = 1
    speculator.max_num_tokens = 1024
    speculator.parallel_drafting_token_id = 1
    speculator.hidden_states = torch.zeros((1024, 4))
    speculator.context_positions = torch.zeros(1024, dtype=torch.int64)
    speculator.sample_indices = torch.zeros(5, dtype=torch.int64)
    speculator.sample_pos = torch.zeros(5, dtype=torch.int64)
    speculator.sample_idx_mapping = torch.zeros(5, dtype=torch.int32)
    speculator.sample_from_anchor = True
    speculator.input_buffers = SimpleNamespace()
    speculator.block_tables = SimpleNamespace(
        slot_mappings=torch.zeros((1, 1024), dtype=torch.int64),
        input_block_tables=[torch.zeros((1, 64), dtype=torch.int32)],
        kernel_block_sizes=[16],
    )
    speculator.draft_kv_cache_group_id = 0
    speculator.draft_kv_cache_group_ids = [0]
    speculator._context_slot_mappings = [
        torch.zeros(1024, dtype=torch.int64)
    ]
    speculator._layer_group_idx = None
    speculator._group_causal = True
    speculator.model = SimpleNamespace(
        precompute_and_store_context_kv=MagicMock()
    )
    speculator._copy_request_inputs = MagicMock()
    speculator._prepare_eplb_forward = MagicMock()
    speculator._build_draft_attn_metadata = MagicMock(return_value={})
    speculator._generate_draft = MagicMock()
    speculator._prof = None
    speculator.draft_tokens = torch.zeros((1, 5), dtype=torch.int64)
    speculator.query_cudagraph_manager = None
    speculator.dp_size = 1
    speculator.dp_rank = 0
    speculator.kv_cache_config = SimpleNamespace()

    input_batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=896,
        seq_lens_cpu_upper_bound=torch.tensor([896], dtype=torch.int32),
        idx_mapping=torch.tensor([0], dtype=torch.int32),
    )
    monkeypatch.setattr(dflash_module, "prepare_dflash_inputs", MagicMock())
    monkeypatch.setattr(
        dflash_module,
        "dispatch_cg_and_sync_dp",
        MagicMock(
            return_value=(
                SimpleNamespace(
                    num_reqs=None,
                    num_tokens=5,
                    cg_mode=dflash_module.CUDAGraphMode.NONE,
                ),
                None,
            )
        ),
    )
    monkeypatch.setattr(
        dflash_module, "build_slot_mappings_by_layer", MagicMock(return_value={})
    )

    speculator.propose(
        input_batch=input_batch,
        attn_metadata={},
        slot_mappings={},
        last_hidden_states=torch.zeros((896, 4)),
        aux_hidden_states=None,
        num_sampled=torch.zeros(1, dtype=torch.int32),
        num_rejected=torch.zeros(1, dtype=torch.int32),
        last_sampled=torch.zeros(1, dtype=torch.int64),
        next_prefill_tokens=torch.zeros(1, dtype=torch.int64),
        temperature=torch.ones(1),
        seeds=torch.zeros(1, dtype=torch.int64),
        generate_draft=generate_draft,
    )

    speculator.model.precompute_and_store_context_kv.assert_called_once()
    assert speculator._generate_draft.call_count == int(generate_draft)
