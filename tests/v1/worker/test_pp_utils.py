# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The PP sampled-token broadcast: which rows it carries, and how it is
received."""

from collections import deque
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import torch

import vllm.v1.worker.gpu.pp_utils as pp_utils_module
from vllm.v1.worker.gpu import model_runner, pp_utils
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
    handler.disabled = False
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


def _cuda_handler(max_sample_len=6):
    handler = object.__new__(pp_utils.PPHandler)
    handler.is_last_rank = True
    handler.disabled = False
    handler.max_sample_len = max_sample_len
    handler.num_speculative_steps = max_sample_len - 1
    handler.tokens_width = max_sample_len + handler.num_speculative_steps
    handler.payload_width = handler.tokens_width + 2
    handler.last_rank = 1
    handler.broadcast_group = Mock()
    handler.device = torch.device("cuda")
    handler.main_stream = torch.cuda.current_stream()
    handler.broadcast_stream = torch.cuda.Stream()
    return handler


def _batch(num_computed, prefill_len, num_scheduled, idx_mapping=None):
    num_reqs = len(num_computed)
    if idx_mapping is None:
        idx_mapping = list(range(num_reqs))
    return Mock(
        num_reqs=num_reqs,
        num_computed_tokens_np=np.array(num_computed, dtype=np.int32),
        prefill_len_np=np.array(prefill_len, dtype=np.int32),
        num_scheduled_tokens=np.array(num_scheduled, dtype=np.int32),
        idx_mapping=torch.tensor(idx_mapping, dtype=torch.int64),
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


def test_disabled_handler_skips_broadcast_and_receive(monkeypatch):
    """While disabled (warmup), neither side enqueues a broadcast op."""
    sent = []
    monkeypatch.setattr(
        pp_utils.torch.distributed,
        "broadcast",
        lambda *args, **kwargs: sent.append((args, kwargs)),
    )

    handler = object.__new__(pp_utils.PPHandler)
    handler.set_disabled(True)

    handler.is_last_rank = False
    assert handler.receive(Mock()) is False

    handler.is_last_rank = True
    assert handler.broadcast(Mock(), Mock(), Mock(), Mock()) is None

    assert sent == []

    handler.set_disabled(False)
    assert handler.disabled is False


def _payload_handler(max_sample_len: int) -> pp_utils.PPHandler:
    handler = object.__new__(pp_utils.PPHandler)
    handler.num_speculative_steps = max_sample_len - 1
    handler.max_sample_len = max_sample_len
    handler.tokens_width = max_sample_len + handler.num_speculative_steps
    handler.payload_width = handler.tokens_width + 2
    handler.device = torch.device("cpu")
    return handler


def test_unpacked_payload_counts_are_16_byte_aligned():
    """Triton specializes on pointer alignment: an unaligned `num_sampled` or
    `num_rejected` would compile a second `_post_update_kernel` variant at
    serving time, where the in-flight broadcast NCCL kernel can block the
    module load. The single int64 payload's count columns are converted into
    fresh int32 tensors, so they are aligned for every batch size."""
    handler = _payload_handler(max_sample_len=8)
    for num_reqs in range(1, 9):
        payload = torch.zeros(num_reqs, handler.payload_width, dtype=torch.int64)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int64)
        outputs = handler._unpack_payload(payload, idx_mapping)
        assert outputs["num_sampled"].data_ptr() % 16 == 0
        assert outputs["num_rejected"].data_ptr() % 16 == 0
        assert outputs["num_sampled"].dtype == torch.int32
        assert outputs["num_rejected"].dtype == torch.int32
        assert outputs["sampled_tokens"].data_ptr() % 16 == 0


def _post_update_signature(args: tuple) -> list:
    """What triton specializes a `post_update` launch on, per argument.

    Pointers specialize on dtype and 16-byte alignment. The kernel indexes 1-D
    tensors as contiguous and receives the row stride of 2-D tensors as an
    integer, which specializes on being 1 and on divisibility by 16.
    """
    signature: list = []
    for arg in args:
        if arg is None:
            signature.append(None)
        elif isinstance(arg, torch.Tensor):
            key = [arg.dtype, arg.dim(), arg.data_ptr() % 16 == 0]
            if arg.dim() == 1:
                key.append(arg.is_contiguous())
            else:
                row_stride = arg.stride(0)
                key += [row_stride == 1, row_stride % 16 == 0, arg.stride(1) == 1]
            signature.append(tuple(key))
        else:
            signature.append(type(arg))
    return signature


@pytest.mark.parametrize("max_sample_len", [1, 8])
def test_warmup_pp_decode_update_matches_serving_specialization(
    monkeypatch, max_sample_len
):
    """The warmup launch must hit the same triton specialization as serving.

    A mismatch means the first real ``update_pp_decode_requests`` recompiles
    mid-serving, where the in-flight broadcast NCCL kernel blocks the CUDA
    module load and deadlocks the pipeline. Serving consumes a received
    payload through ``get_prev_sampled_outputs``; the warmup must call
    ``post_update`` with identically shaped, strided and typed arguments.
    """
    calls = []
    monkeypatch.setattr(model_runner, "post_update", lambda *args: calls.append(args))
    monkeypatch.setattr(
        model_runner, "scatter_draft_tokens", lambda *args, **kwargs: None
    )

    handler = _payload_handler(max_sample_len)
    handler.queue = deque()
    handler.req_idx_gen_np = np.zeros(4, dtype=np.int32)
    handler.main_stream = _FakeMainStream()
    num_reqs = 3
    handler.queue.append(
        PendingRecv(
            event=object(),  # type: ignore[arg-type]
            payload=torch.zeros(num_reqs, handler.payload_width, dtype=torch.int64),
            idx_mapping=torch.arange(num_reqs, dtype=torch.int64),
            idx_mapping_np=np.arange(num_reqs, dtype=np.intp),
            need_sampled_mask=np.ones(num_reqs, dtype=bool),
            gen_at_receive_np=np.zeros(num_reqs, dtype=np.int32),
        )
    )

    runner = object.__new__(model_runner.GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.is_last_pp_rank = False
    runner.pp_handler = handler
    runner.req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=torch.zeros(4, dtype=torch.int32)),
        last_sampled_tokens=torch.zeros(4, 1, dtype=torch.int64),
        all_token_ids=SimpleNamespace(gpu=torch.zeros(4, 16, dtype=torch.int32)),
        total_len=SimpleNamespace(gpu=torch.zeros(4, dtype=torch.int32)),
        draft_tokens=torch.zeros(4, max(max_sample_len - 1, 1), dtype=torch.int64),
    )
    runner.model_state = Mock()

    runner.update_pp_decode_requests()
    runner.warmup_pp_decode_update()

    assert len(calls) == 2
    serving, warmup = calls
    assert warmup[0].tolist() == [-1]
    assert _post_update_signature(warmup) == _post_update_signature(serving)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA stream")
def test_broadcast_pads_plain_sampler_rows_to_max_sample_len(monkeypatch):
    """The wire shape must not depend on whether the batch carried drafts:
    the receiver always allocates [num_reqs, payload_width], and a NCCL
    broadcast with mismatched counts hangs the receiver. Everything rides in
    one broadcast call."""
    sent = []
    monkeypatch.setattr(
        pp_utils.torch.distributed,
        "broadcast",
        lambda tensor, **kw: sent.append(tensor),
    )
    handler = _cuda_handler()
    batch = _batch(num_computed=[10], prefill_len=[8], num_scheduled=[1])

    handler.broadcast(
        torch.zeros(1, 1, dtype=torch.int64, device="cuda"),  # plain sampler
        torch.ones(1, dtype=torch.int32, device="cuda"),
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        batch,
        proposed_token_ids=torch.zeros(1, 5, dtype=torch.int64, device="cuda"),
    )

    assert len(sent) == 1
    assert sent[0].shape == (1, handler.payload_width)
    torch.accelerator.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA stream")
def test_broadcast_carries_a_fresh_copy_of_the_proposed_block(monkeypatch):
    """The proposed drafts ride in the one payload, copied out of the table:
    propose() overwrites its persistent buffer on the next step, possibly
    before this send completes. An all-prefill batch sends nothing."""
    sent = []
    monkeypatch.setattr(
        pp_utils.torch.distributed,
        "broadcast",
        lambda tensor, **kw: sent.append(tensor),
    )
    handler = _cuda_handler()
    batch = _batch(num_computed=[10], prefill_len=[8], num_scheduled=[1])
    table = torch.arange(20, dtype=torch.int64, device="cuda").view(4, 5)
    proposed = table[2:3]

    handler.broadcast(
        torch.zeros(1, 6, dtype=torch.int64, device="cuda"),
        torch.ones(1, dtype=torch.int32, device="cuda"),
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        batch,
        proposed_token_ids=proposed,
    )

    assert len(sent) == 1
    assert sent[0].data_ptr() != table.data_ptr()
    drafts = sent[0][:, handler.max_sample_len : handler.tokens_width]
    assert drafts.cpu().tolist() == [table[2].cpu().tolist()]

    sent.clear()
    prefill_batch = _batch(num_computed=[0], prefill_len=[4096], num_scheduled=[448])
    handler.broadcast(
        torch.zeros(1, 6, dtype=torch.int64, device="cuda"),
        torch.ones(1, dtype=torch.int32, device="cuda"),
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        prefill_batch,
        proposed_token_ids=proposed,
    )
    assert sent == []
    torch.accelerator.synchronize()
