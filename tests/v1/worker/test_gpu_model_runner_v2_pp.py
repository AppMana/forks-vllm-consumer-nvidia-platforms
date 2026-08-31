# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import vllm.v1.worker.gpu.model_runner as model_runner_module
from tests.v1.core.utils import create_requests, create_scheduler
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.request import RequestStatus
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    _is_compatible,
    get_uniform_token_count,
)
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_runner import (
    ExecuteModelState,
    GPUModelRunner,
    _copy_or_reuse_pp_intermediate_tensor,
)
from vllm.v1.worker.gpu.states import RequestState


def test_pp_intermediate_copy_uses_received_length() -> None:
    dst = torch.empty((4096, 8), dtype=torch.bfloat16)
    src = torch.arange(537 * 8, dtype=torch.float32).reshape(537, 8).to(torch.bfloat16)

    actual = _copy_or_reuse_pp_intermediate_tensor(dst, src, num_tokens=4096)

    assert actual.shape == src.shape
    torch.testing.assert_close(actual, src)


def test_pp_intermediate_copy_reuses_matching_view() -> None:
    src = torch.empty((537, 8), dtype=torch.bfloat16)

    actual = _copy_or_reuse_pp_intermediate_tensor(src, src, num_tokens=537)

    assert actual.shape == src.shape
    assert actual.data_ptr() == src.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA input kernels")
def test_pp_deferred_steady_verify_is_one_width_k_plus_one_forward() -> None:
    """A steady K=5 verify is one width-6 target-model transaction.

    The anchor and all five draft tokens must traverse pipeline parallelism in
    the same model forward. The target therefore produces six logits rows,
    while request accounting reserves at most five emitted-token slots.
    """
    num_draft_tokens = 5
    anchor = 99
    draft_tokens = [101, 102, 103, 104, 105]

    scheduler = create_scheduler(
        async_scheduling=True,
        pipeline_parallel_size=2,
        num_speculative_tokens=num_draft_tokens,
        speculative_method="ngram_gpu",
        use_v2_model_runner=True,
        max_num_batched_tokens=128,
        max_model_len=128,
    )
    request = create_requests(1, num_tokens=2, max_tokens=20)[0]
    request.append_output_token_ids(anchor)
    request.num_computed_tokens = 3
    request.spec_token_ids = draft_tokens
    request.status = RequestStatus.RUNNING
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)

    placeholders_before = request.num_output_placeholders
    scheduler_output = scheduler.schedule()

    device = torch.device("cuda")
    req_states = RequestState(
        max_num_reqs=1,
        max_model_len=128,
        max_num_batched_tokens=128,
        num_speculative_steps=num_draft_tokens,
        vocab_size=1000,
        device=device,
    )
    req_states.add_request(
        request.request_id,
        prompt_len=2,
        all_token_ids=[0, 0, anchor],
        num_computed_tokens=3,
        max_tokens=20,
    )
    req_states.last_sampled_tokens[0, 0] = anchor
    req_states.draft_tokens[0] = torch.tensor(draft_tokens, device=device)
    req_states.apply_staged_writes()

    runner = object.__new__(GPUModelRunner)
    runner.decode_query_len = num_draft_tokens + 1
    runner.pp_handler = object()
    runner.device = device
    runner.max_num_reqs = 1
    runner.use_dcp = False
    runner.use_pp = True
    runner.model_config = SimpleNamespace(rswa_window=None)
    runner.input_buffers = InputBuffers(1, 128, device)
    runner.req_states = req_states
    runner.model_state = SimpleNamespace(num_new_sampled_tokens_per_step=1)
    runner.pcp_manager = None
    runner.block_tables = MagicMock()
    runner.update_requests(scheduler_output)

    uniform_token_count = get_uniform_token_count(
        num_reqs=1,
        num_tokens=scheduler_output.total_num_scheduled_tokens,
        max_query_len=max(scheduler_output.num_scheduled_tokens.values()),
    )
    batch_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=scheduler_output.total_num_scheduled_tokens,
        num_reqs=1,
        uniform_token_count=uniform_token_count,
    )
    input_batch = runner.prepare_inputs(scheduler_output, batch_desc)
    torch.cuda.synchronize()

    observed = (
        scheduler_output.num_scheduled_tokens[request.request_id],
        input_batch.input_ids[: input_batch.num_tokens].tolist(),
        input_batch.positions[: input_batch.num_tokens].tolist(),
        input_batch.logits_indices.tolist(),
        request.num_output_placeholders - placeholders_before,
        scheduler_output.replayed_pp_anchor_req_ids,
    )
    expected = (
        num_draft_tokens + 1,
        [anchor, *draft_tokens],
        list(range(2, 2 + num_draft_tokens + 1)),
        list(range(num_draft_tokens + 1)),
        num_draft_tokens,
        {request.request_id},
    )
    assert observed == expected
    assert _is_compatible(
        batch_desc,
        num_reqs=1,
        num_tokens=num_draft_tokens + 1,
        uniform_token_count=num_draft_tokens + 1,
        num_active_loras=0,
    )


def _record_profile_scopes(monkeypatch):
    scopes = []

    @contextmanager
    def record_scope(name):
        scopes.append(name)
        yield

    monkeypatch.setattr(
        model_runner_module, "record_function_or_nullcontext", record_scope
    )
    return scopes


def _make_sampling_runner(num_draft_tokens: int):
    runner = object.__new__(GPUModelRunner)
    runner.model = SimpleNamespace(
        compute_logits=MagicMock(return_value=torch.zeros(1, 4))
    )
    sampler_output = SimpleNamespace(
        num_sampled=torch.ones(1, dtype=torch.int32),
        num_rejected=torch.zeros(1, dtype=torch.int32),
    )
    runner.sampler = MagicMock(return_value=sampler_output)
    runner.rejection_sampler = None
    runner.speculator = None
    runner.structured_outputs_worker = None
    runner.pp_handler = None
    runner.req_states = SimpleNamespace(last_sampled_tokens=torch.zeros(1))
    input_batch = SimpleNamespace(
        logits_indices=torch.tensor([0]),
        num_draft_tokens=num_draft_tokens,
    )
    return runner, input_batch, sampler_output


def test_sampling_trace_separates_logits_from_sampling(monkeypatch) -> None:
    scopes = _record_profile_scopes(monkeypatch)
    runner, input_batch, sampler_output = _make_sampling_runner(0)

    output = runner.sample(torch.zeros(1, 4), input_batch, None)

    assert output[0] is sampler_output
    assert scopes == [
        "gpu_model_runner: target_logits",
        "gpu_model_runner: sample",
    ]


def test_sampling_trace_identifies_speculative_verification(monkeypatch) -> None:
    scopes = _record_profile_scopes(monkeypatch)
    runner, input_batch, sampler_output = _make_sampling_runner(1)
    runner.speculator = SimpleNamespace(draft_logits=None)
    runner.rejection_sampler = MagicMock(return_value=sampler_output)

    output = runner.sample(torch.zeros(1, 4), input_batch, None)

    assert output[0] is sampler_output
    assert scopes == [
        "gpu_model_runner: target_logits",
        "gpu_model_runner: spec_verify",
    ]


def test_deferred_draft_update_does_not_reduce_a_cuda_mask_on_the_host(
    monkeypatch,
) -> None:
    """Deferred PP bookkeeping must remain asynchronous with the broadcast.

    Reducing the validity mask to a Python bool synchronizes the host with the
    side-stream sampled-token broadcast. Different pipeline stages then reach
    the next worker command at different times and lose pipeline overlap.
    """
    runner = object.__new__(GPUModelRunner)
    runner.is_last_pp_rank = False
    runner.sampler = None
    runner.req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=torch.zeros(2, dtype=torch.int32)),
        last_sampled_tokens=torch.zeros(2, dtype=torch.int64),
        all_token_ids=SimpleNamespace(gpu=torch.zeros((2, 8), dtype=torch.int64)),
        total_len=SimpleNamespace(gpu=torch.zeros(2, dtype=torch.int32)),
        draft_tokens=torch.zeros((2, 2), dtype=torch.int64),
    )
    runner.model_state = SimpleNamespace(postprocess_state=MagicMock())

    monkeypatch.setattr(model_runner_module, "post_update", MagicMock())
    scatter_draft_tokens = MagicMock()
    monkeypatch.setattr(
        model_runner_module,
        "scatter_draft_tokens",
        scatter_draft_tokens,
        raising=False,
    )

    def fail_on_any(_self, *args, **kwargs):
        raise AssertionError("deferred PP path reduced a device mask on the host")

    monkeypatch.setattr(torch.Tensor, "any", fail_on_any)

    idx_mapping = torch.tensor([0, -1], dtype=torch.int32)
    proposed_tokens = torch.tensor([[11, 12], [21, 22]], dtype=torch.int64)
    runner.postprocess_sampled(
        idx_mapping=idx_mapping,
        sampled_tokens=torch.zeros((2, 3), dtype=torch.int64),
        num_sampled=torch.ones(2, dtype=torch.int32),
        num_rejected=torch.zeros(2, dtype=torch.int32),
        proposed_tokens=proposed_tokens,
    )

    scatter_draft_tokens.assert_called_once_with(
        runner.req_states.draft_tokens,
        idx_mapping,
        proposed_tokens,
    )


@pytest.mark.parametrize(
    ("num_computed_tokens", "num_scheduled_tokens", "should_generate"),
    [(0, 896, False), (7168, 832, True)],
)
def test_chunked_prefill_drafts_only_after_final_chunk(
    monkeypatch,
    num_computed_tokens: int,
    num_scheduled_tokens: int,
    should_generate: bool,
) -> None:
    """Intermediate chunks update draft context without generating tokens."""
    runner = object.__new__(GPUModelRunner)
    input_batch = SimpleNamespace(
        req_ids=["request"],
        num_reqs=1,
        idx_mapping=torch.tensor([0]),
        query_start_loc=torch.tensor([0, num_scheduled_tokens]),
        num_scheduled_tokens=np.array([num_scheduled_tokens], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array(
            [num_computed_tokens], dtype=np.int32
        ),
        prefill_len_np=np.array([8000], dtype=np.int32),
    )
    runner.execute_model_state = ExecuteModelState(
        input_batch=input_batch,
        attn_metadata={},
        slot_mappings_by_layer={},
        hidden_states=torch.zeros(num_scheduled_tokens, 4),
        aux_hidden_states=None,
        finished_req_ids=set(),
    )
    runner.is_last_pp_rank = True
    runner.pcp_manager = None
    sampler_output = SimpleNamespace(sampled_token_ids=torch.zeros((1, 1)))
    num_sampled = torch.tensor([int(should_generate)], dtype=torch.int32)
    num_rejected = torch.zeros(1, dtype=torch.int32)
    runner.sample = MagicMock(
        return_value=(sampler_output, num_sampled, num_rejected)
    )
    runner.prompt_logprobs_worker = SimpleNamespace(
        compute_prompt_logprobs=MagicMock(return_value={})
    )
    runner.model = SimpleNamespace(compute_logits=MagicMock())
    runner.__dict__["main_stream"] = object()
    runner.output_copy_stream = object()
    runner.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=torch.zeros((1, 8000))),
        num_computed_tokens=SimpleNamespace(gpu=torch.zeros(1)),
        prompt_len=SimpleNamespace(np=np.array([8000], dtype=np.int32)),
        last_sampled_tokens=torch.zeros(1),
        next_prefill_tokens=torch.zeros(1),
        draft_tokens=torch.zeros((1, 5), dtype=torch.int64),
    )
    runner.model_state = SimpleNamespace(gather_mm_embeddings=MagicMock())
    runner.postprocess_sampled = MagicMock()
    runner.sampler = SimpleNamespace(
        sampling_states=SimpleNamespace(
            temperature=SimpleNamespace(gpu=torch.ones(1)),
            seeds=SimpleNamespace(gpu=torch.zeros(1, dtype=torch.int64)),
        )
    )
    runner.speculator = SimpleNamespace(
        supports_mm_inputs=False,
        propose=MagicMock(return_value=torch.ones((1, 5), dtype=torch.int64)),
    )
    runner.pp_handler = MagicMock()
    runner.num_speculative_steps = 5
    runner.draft_tokens_handler = MagicMock()
    runner.kv_connector = SimpleNamespace(post_forward=MagicMock(return_value=None))
    runner.eplb = SimpleNamespace(step=MagicMock())

    monkeypatch.setattr(model_runner_module, "AsyncOutput", MagicMock())

    runner.sample_tokens(None)

    runner.speculator.propose.assert_called_once()
    assert (
        runner.speculator.propose.call_args.kwargs["generate_draft"]
        is should_generate
    )
    proposed_token_ids = runner.pp_handler.broadcast.call_args.kwargs[
        "proposed_token_ids"
    ]
    if should_generate:
        torch.testing.assert_close(
            proposed_token_ids, torch.ones((1, 5), dtype=torch.int64)
        )
    else:
        assert proposed_token_ids is None
    runner.pp_handler.broadcast.assert_called_once_with(
        sampler_output.sampled_token_ids,
        num_sampled,
        num_rejected,
        input_batch,
        proposed_token_ids=proposed_token_ids,
    )
