# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The PP sampled-token broadcast: which rows it carries, and how it is
received."""

from collections import deque
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import torch

import vllm.v1.worker.gpu.pp_utils as pp_utils_module
from vllm.v1.worker.gpu import pp_utils
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


def test_pp_deferred_output_compacts_cpu_known_excluded_rows() -> None:
    """The receiver already knows invalid rows on the CPU.

    Returning a CUDA mapping containing -1 forces the model runner to build a
    device mask, reduce it to the host, and use dynamic boolean indexing.
    Compact here so downstream scatter remains fixed-shape and asynchronous.
    """
    handler = object.__new__(PPHandler)
    handler.queue = deque()
    handler.queue.append(
        PendingRecv(
            event=object(),  # type: ignore[arg-type]
            payload=torch.tensor(
                [
                    [11, 12, 13, 21, 22, 2, 1],
                    [31, 32, 33, 41, 42, 1, 2],
                ],
                dtype=torch.int64,
            ),
            idx_mapping=torch.tensor([0, 1], dtype=torch.int32),
            idx_mapping_np=np.array([0, 1], dtype=np.int32),
            need_sampled_mask=np.array([True, False]),
            gen_at_receive_np=np.array([0, 0], dtype=np.int32),
        )
    )
    handler.num_speculative_steps = 2
    handler.max_sample_len = 3
    handler.tokens_width = 5
    handler.req_idx_gen_np = np.zeros(2, dtype=np.int32)
    handler.main_stream = _FakeMainStream()
    handler.device = torch.device("cpu")

    output = handler.get_prev_sampled_outputs()

    assert output is not None
    torch.testing.assert_close(
        output["idx_mapping"], torch.tensor([0], dtype=torch.int32)
    )
    assert output["sampled_tokens"].shape == (1, 3)
    assert output["proposed_tokens"].shape == (1, 2)
    torch.testing.assert_close(output["proposed_tokens"], torch.tensor([[21, 22]]))


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


def _batch(num_computed, prefill_len, num_scheduled):
    return Mock(
        num_reqs=len(num_computed),
        num_computed_tokens_np=np.array(num_computed, dtype=np.int32),
        prefill_len_np=np.array(prefill_len, dtype=np.int32),
        num_scheduled_tokens=np.array(num_scheduled, dtype=np.int32),
    )


def test_excludes_non_final_prefill_chunks():
    """Unchanged behaviour: a chunk that does not finish its prefill is skipped."""
    # Row 0 is a middle prefill chunk and produces no sample; row 1 finishes its
    # prefill this step and therefore does.
    batch = _batch(
        num_computed=[512, 1000],
        prefill_len=[4096, 1004],
        num_scheduled=[448, 4],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [False, True]


def test_none_when_no_row_samples():
    """Unchanged behaviour: an all-prefill batch needs no broadcast at all."""
    batch = _batch(
        num_computed=[0, 512],
        prefill_len=[4096, 4096],
        num_scheduled=[448, 448],
    )

    assert pp_utils.compute_need_sampled_mask(batch) is None


def test_keeps_decoding_request_past_its_length_cap():
    """A decoding request must never be dropped from the broadcast.

    Speculative decoding advances `num_computed_tokens` several tokens per step,
    so it can overrun `prompt_len + max_tokens` while the scheduler is still
    running the request. Predicting "this one is finishing" and skipping its
    broadcast freezes the earlier pipeline stages' `last_sampled_tokens` and
    `draft_tokens` while the last rank keeps advancing its own, and the stages
    then diverge permanently.
    """
    batch = _batch(
        # 14176 computed tokens is already past this request's own
        # prompt_len + max_tokens; the scheduler is still running it.
        num_computed=[14176],
        prefill_len=[12175],
        num_scheduled=[8],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True]


def test_decode_row_ahead_of_a_prefill_chunk():
    """Row order does not matter: only whether the row finishes its prefill."""
    batch = _batch(
        num_computed=[10, 512],
        prefill_len=[8, 4096],
        num_scheduled=[1, 448],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True, False]
