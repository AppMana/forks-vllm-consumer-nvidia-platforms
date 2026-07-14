# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real multi-process PP>1 coverage for PPHandler (vllm/v1/worker/gpu/pp_utils.py).

This is the mechanism the DSpark PP+speculative-decoding fix landed in: a
dedicated NCCL sibling communicator carries the last rank's sampled tokens,
newly-proposed (not-yet-verified) draft block, and accept/reject counts to
every other rank in exactly ONE `torch.distributed.broadcast` call per step,
consumed `pp_size` steps later. Before this fix, non-last ranks never
received the proposed block at all (their local draft-token buffer was
permanently stale), and a separate bug packed num_sampled/num_rejected into
a non-16-byte-aligned view, which Triton silently recompiled as a distinct,
effectively unwarmable kernel specialization -- see pp_utils.py's docstring
and the "one tensor, one broadcast" rule it encodes. Nothing here should ever
need a second `torch.distributed.broadcast` call; that's the regression this
file guards against most directly (test_receive_and_broadcast_use_exactly_one_collective).
"""

from types import SimpleNamespace

import numpy as np
import pytest
import ray
import torch

from ..utils import init_test_distributed_environment, multi_gpu_test, multi_process_parallel


def _fake_input_batch(
    *,
    num_reqs: int,
    idx_mapping_np: np.ndarray,
    device: torch.device,
    old_computed: int = 5,
    prefill_len: int = 1,
    scheduled: int = 1,
    max_seq_len: int = 100,
) -> SimpleNamespace:
    """Minimal duck-typed stand-in for InputBatch: PPHandler/
    compute_need_sampled_mask only ever touch these fields."""
    return SimpleNamespace(
        num_reqs=num_reqs,
        idx_mapping=torch.from_numpy(idx_mapping_np).to(device),
        idx_mapping_np=idx_mapping_np,
        num_computed_tokens_np=np.full(num_reqs, old_computed, dtype=np.int32),
        prefill_len_np=np.full(num_reqs, prefill_len, dtype=np.int32),
        max_seq_len_np=np.full(num_reqs, max_seq_len, dtype=np.int32),
        num_scheduled_tokens=np.full(num_reqs, scheduled, dtype=np.int32),
    )


@ray.remote(num_gpus=1, max_calls=1)
def pp_handler_broadcast_roundtrip_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
):
    """PP=2, num_speculative_steps=2. Broadcasts 3 successive steps of
    distinct sampled/proposed/accept-reject data from the last rank and
    verifies rank 0 recovers each step's data bit-for-bit exactly
    `pp_size` steps later -- the queue-offset contract PPHandler documents."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

    from vllm.v1.worker.gpu.pp_utils import PPHandler

    num_speculative_steps = 2
    num_reqs = 2
    idx_mapping_np = np.arange(num_reqs, dtype=np.int32)
    handler = PPHandler(
        max_num_reqs=8, num_speculative_steps=num_speculative_steps, device=device
    )
    assert handler.is_last_rank == (rank == pp_size - 1)

    def step_payload(step: int):
        sampled = torch.arange(
            step * 100, step * 100 + num_reqs * (num_speculative_steps + 1),
            dtype=torch.int64, device=device,
        ).reshape(num_reqs, num_speculative_steps + 1)
        proposed = torch.arange(
            step * 1000, step * 1000 + num_reqs * num_speculative_steps,
            dtype=torch.int64, device=device,
        ).reshape(num_reqs, num_speculative_steps)
        num_sampled = torch.full((num_reqs,), step + 1, dtype=torch.int32, device=device)
        num_rejected = torch.full((num_reqs,), step, dtype=torch.int32, device=device)
        return sampled, proposed, num_sampled, num_rejected

    consumed_at_step: dict[int, dict[str, torch.Tensor]] = {}
    for step in range(4):
        input_batch = _fake_input_batch(
            num_reqs=num_reqs, idx_mapping_np=idx_mapping_np, device=device
        )

        if not handler.is_last_rank:
            result = handler.get_prev_sampled_outputs()
            if result is not None:
                consumed_at_step[step] = result
            handler.receive(input_batch)
        else:
            sampled, proposed, num_sampled, num_rejected = step_payload(step)
            handler.broadcast(sampled, num_sampled, num_rejected, input_batch, proposed)

        torch.cuda.synchronize(device)

    if not handler.is_last_rank:
        # First pp_size (=2) consumes are no-ops (queue pre-seeded with None);
        # step 0's broadcast is recoverable starting at step 0 + pp_size = 2.
        assert set(consumed_at_step) == {2, 3}
        for step in (2, 3):
            source_step = step - pp_size
            want_sampled, want_proposed, want_num_sampled, want_num_rejected = step_payload(
                source_step
            )
            got = consumed_at_step[step]
            torch.testing.assert_close(got["sampled_tokens"], want_sampled)
            torch.testing.assert_close(got["proposed_tokens"], want_proposed)
            torch.testing.assert_close(got["num_sampled"], want_num_sampled)
            torch.testing.assert_close(got["num_rejected"], want_num_rejected)
            torch.testing.assert_close(
                got["idx_mapping"].cpu(), torch.from_numpy(idx_mapping_np)
            )


@ray.remote(num_gpus=1, max_calls=1)
def pp_handler_single_collective_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
):
    """Regression guard for the standing pp_utils.py rule: receive()/
    broadcast() must each issue exactly one torch.distributed.broadcast
    call. A second, independently-allocated collective is how the original
    bug (a misaligned num_rejected view triggering a distinct, effectively
    unwarmable Triton specialization) got introduced."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

    from vllm.v1.worker.gpu.pp_utils import PPHandler

    num_reqs = 2
    idx_mapping_np = np.arange(num_reqs, dtype=np.int32)
    handler = PPHandler(max_num_reqs=8, num_speculative_steps=2, device=device)

    call_count = {"n": 0}
    real_broadcast = torch.distributed.broadcast

    def counting_broadcast(*args, **kwargs):
        call_count["n"] += 1
        return real_broadcast(*args, **kwargs)

    monkeypatch.setattr(torch.distributed, "broadcast", counting_broadcast)

    input_batch = _fake_input_batch(
        num_reqs=num_reqs, idx_mapping_np=idx_mapping_np, device=device
    )
    if handler.is_last_rank:
        sampled = torch.zeros(num_reqs, 3, dtype=torch.int64, device=device)
        proposed = torch.zeros(num_reqs, 2, dtype=torch.int64, device=device)
        num_sampled = torch.ones(num_reqs, dtype=torch.int32, device=device)
        num_rejected = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        handler.broadcast(sampled, num_sampled, num_rejected, input_batch, proposed)
    else:
        handler.receive(input_batch)
    torch.cuda.synchronize(device)

    assert call_count["n"] == 1, (
        f"expected exactly one torch.distributed.broadcast call, got {call_count['n']}"
    )


@ray.remote(num_gpus=1, max_calls=1)
def pp_handler_payload_alignment_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
):
    """The concrete bug this session: num_reqs=1 previously produced a
    non-16-byte-aligned view for the trailing num_rejected column when it
    was unbound from a separately-allocated [2, num_reqs] tensor. Assert the
    payload tensor itself, and `sampled_tokens` (the only column-view of it
    that feeds a Triton-compiled kernel, via postprocess_sampled ->
    post_update -- and which starts at column 0, so it always shares the
    payload's own base pointer) are always 16-byte aligned, independent of
    num_reqs. `proposed_tokens` and `num_sampled`/`num_rejected` are exempt:
    the former is consumed only via plain PyTorch indexed assignment (no
    Triton pointer-alignment specialization to break), and the latter are
    always freshly allocated by the `.to(torch.int32)` cast in
    get_prev_sampled_outputs (a genuine dtype change forces a copy)."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

    from vllm.v1.worker.gpu.pp_utils import PPHandler

    num_reqs = 1  # the exact width that triggered the original bug
    idx_mapping_np = np.arange(num_reqs, dtype=np.int32)
    handler = PPHandler(max_num_reqs=8, num_speculative_steps=2, device=device)

    input_batch = _fake_input_batch(
        num_reqs=num_reqs, idx_mapping_np=idx_mapping_np, device=device
    )
    if handler.is_last_rank:
        sampled = torch.zeros(num_reqs, 3, dtype=torch.int64, device=device)
        proposed = torch.zeros(num_reqs, 2, dtype=torch.int64, device=device)
        num_sampled = torch.ones(num_reqs, dtype=torch.int32, device=device)
        num_rejected = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        handler.broadcast(sampled, num_sampled, num_rejected, input_batch, proposed)
    else:
        handler.receive(input_batch)
        torch.cuda.synchronize(device)
        slot = handler.queue[-1]
        assert slot is not None
        assert slot.payload.data_ptr() % 16 == 0
        for _ in range(pp_size - 1):
            handler.get_prev_sampled_outputs()
        result = handler.get_prev_sampled_outputs()
        assert result is not None
        assert result["sampled_tokens"].data_ptr() == slot.payload.data_ptr()
        assert result["sampled_tokens"].data_ptr() % 16 == 0


@ray.remote(num_gpus=1, max_calls=1)
def pp_handler_narrow_sampler_width_padding_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
):
    """The plain (non-speculative) Sampler always returns width-1
    sampled_token_ids (sampled.view(-1, 1)) regardless of max_sample_len --
    e.g. a prefill/first step, before any draft tokens are active yet.
    broadcast() must pad to max_sample_len instead of asserting; receive()
    must still see a full-width, correctly shaped payload."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

    from vllm.v1.worker.gpu.pp_utils import PPHandler

    num_reqs = 2
    num_speculative_steps = 2
    idx_mapping_np = np.arange(num_reqs, dtype=np.int32)
    handler = PPHandler(
        max_num_reqs=8, num_speculative_steps=num_speculative_steps, device=device
    )
    input_batch = _fake_input_batch(
        num_reqs=num_reqs, idx_mapping_np=idx_mapping_np, device=device
    )

    if handler.is_last_rank:
        # Width 1, NOT max_sample_len (=3): the plain-Sampler edge case.
        sampled = torch.full((num_reqs, 1), 7, dtype=torch.int64, device=device)
        proposed = torch.zeros(
            num_reqs, num_speculative_steps, dtype=torch.int64, device=device
        )
        num_sampled = torch.ones(num_reqs, dtype=torch.int32, device=device)
        num_rejected = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        handler.broadcast(sampled, num_sampled, num_rejected, input_batch, proposed)
    else:
        handler.receive(input_batch)
    torch.cuda.synchronize(device)

    if not handler.is_last_rank:
        for _ in range(pp_size - 1):
            handler.get_prev_sampled_outputs()
        result = handler.get_prev_sampled_outputs()
        assert result is not None
        assert result["sampled_tokens"].shape == (num_reqs, num_speculative_steps + 1)
        # post_update only ever reads the first num_sampled[req] columns, so
        # only column 0 (the real sampled token) is asserted here.
        assert torch.all(result["sampled_tokens"][:, 0] == 7)
        assert torch.all(result["num_sampled"] == 1)


@ray.remote(num_gpus=1, max_calls=1)
def pp_handler_freed_request_excluded_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
):
    """A request index freed between receive() and its later consume must be
    excluded (idx_mapping row set to -1) instead of silently postprocessed
    with stale/reused data."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

    from vllm.v1.worker.gpu.pp_utils import PPHandler

    num_reqs = 2
    idx_mapping_np = np.arange(num_reqs, dtype=np.int32)
    handler = PPHandler(max_num_reqs=8, num_speculative_steps=1, device=device)
    input_batch = _fake_input_batch(
        num_reqs=num_reqs, idx_mapping_np=idx_mapping_np, device=device
    )

    if handler.is_last_rank:
        sampled = torch.ones(num_reqs, 2, dtype=torch.int64, device=device)
        proposed = torch.ones(num_reqs, 1, dtype=torch.int64, device=device)
        num_sampled = torch.ones(num_reqs, dtype=torch.int32, device=device)
        num_rejected = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        handler.broadcast(sampled, num_sampled, num_rejected, input_batch, proposed)
    else:
        handler.receive(input_batch)
        # Request index 1 gets freed (e.g. finished/aborted) before its
        # sampled output is ever consumed.
        handler.on_req_idx_freed(1)
    torch.cuda.synchronize(device)

    if not handler.is_last_rank:
        for _ in range(pp_size - 1):
            handler.get_prev_sampled_outputs()
        result = handler.get_prev_sampled_outputs()
        assert result is not None
        idx_mapping = result["idx_mapping"].cpu().numpy()
        assert idx_mapping[0] == 0
        assert idx_mapping[1] == -1


@multi_gpu_test(num_gpus=2)
def test_pp_handler_speculative_broadcast_roundtrip(monkeypatch: pytest.MonkeyPatch):
    multi_process_parallel(monkeypatch, 1, 2, pp_handler_broadcast_roundtrip_worker)


@multi_gpu_test(num_gpus=2)
def test_pp_handler_receive_and_broadcast_use_exactly_one_collective(
    monkeypatch: pytest.MonkeyPatch,
):
    multi_process_parallel(monkeypatch, 1, 2, pp_handler_single_collective_worker)


@multi_gpu_test(num_gpus=2)
def test_pp_handler_payload_always_16_byte_aligned(monkeypatch: pytest.MonkeyPatch):
    multi_process_parallel(monkeypatch, 1, 2, pp_handler_payload_alignment_worker)


@multi_gpu_test(num_gpus=2)
def test_pp_handler_pads_narrow_plain_sampler_width(monkeypatch: pytest.MonkeyPatch):
    multi_process_parallel(
        monkeypatch, 1, 2, pp_handler_narrow_sampler_width_padding_worker
    )


@multi_gpu_test(num_gpus=2)
def test_pp_handler_excludes_freed_request_on_consume(monkeypatch: pytest.MonkeyPatch):
    multi_process_parallel(monkeypatch, 1, 2, pp_handler_freed_request_excluded_worker)


@multi_gpu_test(num_gpus=3)
def test_pp_handler_speculative_broadcast_roundtrip_pp3(monkeypatch: pytest.MonkeyPatch):
    """Same roundtrip contract at pp_size=3, to catch bugs that only manifest
    with more than one non-last (interior) rank relaying the broadcast."""
    multi_process_parallel(monkeypatch, 1, 3, pp_handler_broadcast_roundtrip_worker)
