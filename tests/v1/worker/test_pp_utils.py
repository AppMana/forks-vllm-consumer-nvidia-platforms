# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

import vllm.v1.worker.gpu.pp_utils as pp_utils_module
from vllm.v1.worker.gpu.pp_utils import PendingRecv, PPHandler


class _FakeMainStream:
    def wait_event(self, event: object) -> None:
        pass


def test_non_speculative_pp_does_not_return_empty_proposed_tokens() -> None:
    """An empty tensor would enter the deferred draft-token update path.

    That path tests a CUDA tensor's truth value and indexes it with two CUDA
    boolean masks, introducing three host synchronizations per decode step
    even when speculative decoding is disabled.
    """
    handler = object.__new__(PPHandler)
    handler.queue = deque()
    handler.queue.append(
        PendingRecv(
            event=object(),  # type: ignore[arg-type]
            payload=torch.tensor([[17, 1, 0]], dtype=torch.int64),
            idx_mapping=torch.tensor([0], dtype=torch.int32),
            idx_mapping_np=np.array([0], dtype=np.int32),
            need_sampled_mask=np.array([True]),
            gen_at_receive_np=np.array([0], dtype=np.int32),
        )
    )
    handler.num_speculative_steps = 0
    handler.max_sample_len = 1
    handler.tokens_width = 1
    handler.req_idx_gen_np = np.zeros(1, dtype=np.int32)
    handler.main_stream = _FakeMainStream()

    output = handler.get_prev_sampled_outputs()

    assert output is not None
    assert output["proposed_tokens"] is None


def test_pp_receive_has_a_profile_scope(monkeypatch) -> None:
    scopes = []

    @contextmanager
    def record_scope(name):
        scopes.append(name)
        yield

    class FakePayload:
        def record_stream(self, stream) -> None:
            pass

    handler = object.__new__(PPHandler)
    handler.is_last_rank = False
    handler.last_rank = 1
    handler.num_speculative_steps = 0
    handler.max_sample_len = 1
    handler.tokens_width = 1
    handler.payload_width = 3
    handler.device = torch.device("cpu")
    handler.main_stream = MagicMock()
    handler.broadcast_stream = MagicMock()
    handler.broadcast_stream.record_event.return_value = object()
    handler.broadcast_group = object()
    handler.queue = deque([None])
    handler.req_idx_gen_np = np.zeros(1, dtype=np.int32)
    input_batch = SimpleNamespace(
        num_reqs=1,
        idx_mapping=torch.tensor([0], dtype=torch.int32),
        idx_mapping_np=np.array([0], dtype=np.int32),
    )

    monkeypatch.setattr(
        pp_utils_module,
        "compute_need_sampled_mask",
        lambda _: np.ones(1, dtype=bool),
    )
    monkeypatch.setattr(
        pp_utils_module,
        "record_function_or_nullcontext",
        record_scope,
        raising=False,
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: FakePayload())
    monkeypatch.setattr(torch.distributed, "broadcast", MagicMock())

    assert handler.receive(input_batch)
    assert scopes == ["gpu_model_runner: pp_receive"]
