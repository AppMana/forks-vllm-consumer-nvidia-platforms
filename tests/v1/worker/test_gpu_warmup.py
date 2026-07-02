# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

import vllm.v1.worker.gpu.warmup as gpu_warmup


class _FakeKVConnector:
    def set_disabled(self, disabled: bool) -> None:
        self.disabled = disabled


class _FakeModelRunner:
    is_pooling_model = False
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
    scheduler_config = SimpleNamespace(max_num_batched_tokens=16)


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
