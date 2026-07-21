# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step scheduled prefill token accounting.

During a long chunked prefill no request produces outputs, so all
first-token-based prompt counters stay frozen. The scheduler must report
the number of prompt tokens scheduled on EVERY step via SchedulerStats so
that prefill progress is observable live.
"""

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.outputs import ModelRunnerOutput


def _model_runner_output(scheduler_output, sampled=None):
    req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=sampled if sampled is not None else [[] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def test_chunked_prefill_steps_report_scheduled_prefill_tokens():
    """Each chunked-prefill step must carry that step's prompt token count."""
    scheduler = create_scheduler(max_num_batched_tokens=128, max_model_len=512)
    (req,) = create_requests(num_requests=1, num_tokens=300)
    scheduler.add_request(req)

    # Step 1: first 128-token prefill chunk, no output tokens.
    scheduler_output = scheduler.schedule()
    assert scheduler_output.num_scheduled_tokens[req.request_id] == 128
    outputs = scheduler.update_from_output(
        scheduler_output, _model_runner_output(scheduler_output)
    )
    stats = outputs[0].scheduler_stats
    assert stats is not None
    assert stats.num_scheduled_prefill_tokens == 128

    # Step 2: second chunk.
    scheduler_output = scheduler.schedule()
    outputs = scheduler.update_from_output(
        scheduler_output, _model_runner_output(scheduler_output)
    )
    stats = outputs[0].scheduler_stats
    assert stats is not None
    assert stats.num_scheduled_prefill_tokens == 128


def test_decode_steps_report_zero_scheduled_prefill_tokens():
    scheduler = create_scheduler(max_num_batched_tokens=128, max_model_len=512)
    (req,) = create_requests(num_requests=1, num_tokens=64)
    scheduler.add_request(req)

    # Prefill completes in one step and samples the first token.
    scheduler_output = scheduler.schedule()
    outputs = scheduler.update_from_output(
        scheduler_output, _model_runner_output(scheduler_output, sampled=[[0]])
    )
    stats = outputs[0].scheduler_stats
    assert stats is not None
    assert stats.num_scheduled_prefill_tokens == 64

    # Decode step schedules one generation token; no prefill tokens.
    scheduler_output = scheduler.schedule()
    assert scheduler_output.num_scheduled_tokens[req.request_id] == 1
    outputs = scheduler.update_from_output(
        scheduler_output, _model_runner_output(scheduler_output, sampled=[[1]])
    )
    stats = outputs[0].scheduler_stats
    assert stats is not None
    assert stats.num_scheduled_prefill_tokens == 0
