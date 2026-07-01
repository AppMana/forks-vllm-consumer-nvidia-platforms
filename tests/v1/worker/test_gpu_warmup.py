# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.v1.worker.gpu.warmup import run_mixed_prefill_decode_warmup


class _FakeKVConnector:
    def set_disabled(self, disabled: bool) -> None:
        self.disabled = disabled


class _FakeModelRunner:
    is_pooling_model = False
    decode_query_len = 1
    kv_connector = _FakeKVConnector()
    kv_cache_config = SimpleNamespace(
        num_blocks=128,
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16)),
        ],
    )
    parallel_config = SimpleNamespace(pipeline_parallel_size=4)


def test_mixed_prefill_decode_warmup_drains_async_pp_slots():
    executed = []

    def execute_model(scheduler_output):
        executed.append(scheduler_output)

    def sample_tokens(_grammar_output):
        return None

    assert run_mixed_prefill_decode_warmup(
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
