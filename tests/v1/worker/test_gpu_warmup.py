# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

import vllm.v1.worker.gpu.warmup as gpu_warmup


class _FakeKVConnector:
    def set_disabled(self, disabled: bool) -> None:
        self.disabled = disabled


class _FakeBlockTable:
    block_size = 16


class _FakeMultiGroupBlockTable:
    def __init__(self):
        self.block_tables = [_FakeBlockTable()]
        self.calls = []

    def add_row(self, block_ids, row_idx):
        self.calls.append(("add_row", block_ids, row_idx))

    def commit_block_table(self, num_reqs):
        self.calls.append(("commit_block_table", num_reqs))

    def compute_slot_mapping(self, num_reqs, query_start_loc, positions):
        self.calls.append(
            ("compute_slot_mapping", num_reqs, tuple(query_start_loc.tolist()), len(positions))
        )

    def clear_row(self, row_idx):
        self.calls.append(("clear_row", row_idx))


class _FakeModelRunner:
    is_pooling_model = False
    is_last_pp_rank = False
    decode_query_len = 1
    device = None
    kv_connector = _FakeKVConnector()
    kv_cache_config = SimpleNamespace(
        num_blocks=128,
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16)),
        ],
    )
    parallel_config = SimpleNamespace(pipeline_parallel_size=4)
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"])
    )
    scheduler_config = SimpleNamespace(max_num_batched_tokens=16, max_num_seqs=6)


def test_mixed_prefill_decode_warmup_drains_async_pp_slots():
    executed = []

    def execute_model(scheduler_output):
        executed.append(scheduler_output)

    def sample_tokens(_grammar_output):
        return None

    assert gpu_warmup.run_mixed_prefill_decode_warmup(
        _FakeModelRunner(),
        execute_model,
        sample_tokens,
        16,
    )

    scheduled_token_counts = [
        output.total_num_scheduled_tokens for output in executed
    ]
    assert scheduled_token_counts == [2, 16, 0, 0, 0, 0, 0]
    assert executed[-1].finished_req_ids == {
        "_v2_mixed_warmup_decode_",
        "_v2_mixed_warmup_prefill_",
    }


def test_deepseek_v4_long_prefill_warmup_skips_full_model_batch(monkeypatch):
    executed = []
    metadata_warmup = []
    runner = SimpleNamespace(
        is_pooling_model=False,
        device=torch.device("cuda", 0),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
        ),
    )

    monkeypatch.setattr(
        gpu_warmup,
        "warmup_prefill_chunk_metadata_kernel",
        lambda device, *, compress_ratio: metadata_warmup.append(
            (device, compress_ratio)
        ),
    )

    gpu_warmup.warmup_long_prefill_kernels(
        runner,
        lambda scheduler_output: executed.append(scheduler_output),
        lambda _grammar_output: None,
    )

    assert metadata_warmup == [(torch.device("cuda", 0), 4)]
    assert executed == []


def test_deepseek_v4_long_prefill_warmup_directly_warms_slot_mapping(monkeypatch):
    executed = []
    metadata_warmup = []
    block_table = _FakeMultiGroupBlockTable()
    runner = SimpleNamespace(
        is_pooling_model=False,
        device=torch.device("cpu"),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
        ),
        input_batch=SimpleNamespace(block_table=block_table),
    )

    monkeypatch.setattr(
        gpu_warmup,
        "warmup_prefill_chunk_metadata_kernel",
        lambda device, *, compress_ratio: metadata_warmup.append(
            (device, compress_ratio)
        ),
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    gpu_warmup.warmup_long_prefill_kernels(
        runner,
        lambda scheduler_output: executed.append(scheduler_output),
        lambda _grammar_output: None,
    )

    assert metadata_warmup == [(torch.device("cpu"), 4)]
    assert executed == []
    assert block_table.calls == [
        ("add_row", ([1],), 0),
        ("commit_block_table", 1),
        ("compute_slot_mapping", 1, (0, 16), 16),
        ("clear_row", 0),
        ("commit_block_table", 1),
    ]


def test_deepseek_v4_pp_warmup_kernels_skip_generic_execute_model(monkeypatch):
    executed = []
    metadata_warmup = []
    runner = SimpleNamespace(
        is_pooling_model=False,
        parallel_config=SimpleNamespace(pipeline_parallel_size=5),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
        ),
        device=torch.device("cuda", 0),
    )

    monkeypatch.setattr(
        gpu_warmup,
        "warmup_prefill_chunk_metadata_kernel",
        lambda device, *, compress_ratio: metadata_warmup.append(
            (device, compress_ratio)
        ),
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    gpu_warmup.warmup_kernels(
        runner,
        lambda scheduler_output: executed.append(scheduler_output),
        lambda _grammar_output: None,
    )

    assert executed == []
    assert metadata_warmup == [(torch.device("cuda", 0), 4)]


def test_non_deepseek_v4_pp_warmup_kernels_keeps_generic_execute_model(monkeypatch):
    executed = []
    sampled = []
    runner = _FakeModelRunner()
    runner.parallel_config = SimpleNamespace(pipeline_parallel_size=5)
    runner.num_speculative_steps = 0

    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    gpu_warmup.warmup_kernels(
        runner,
        lambda scheduler_output: executed.append(scheduler_output),
        lambda grammar_output: sampled.append(grammar_output),
    )

    assert [output.total_num_scheduled_tokens for output in executed] == [
        12,
        6,
        0,
        2,
        16,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert len(sampled) == 4
