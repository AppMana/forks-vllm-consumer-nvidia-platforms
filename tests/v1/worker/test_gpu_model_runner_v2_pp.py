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
