# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

import vllm.v1.worker.gpu.model_runner as model_runner_module
from vllm.v1.worker.gpu.model_runner import (
    GPUModelRunner,
    _copy_or_reuse_pp_intermediate_tensor,
)


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
