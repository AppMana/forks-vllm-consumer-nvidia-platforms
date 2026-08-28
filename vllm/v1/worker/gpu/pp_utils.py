# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline Parallelism utils for V2 Model Runner."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from vllm.distributed.parallel_state import get_pp_group
from vllm.platforms import current_platform
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch


@dataclass
class PendingRecv:
    """Per-step slot data for a deferred postprocess on the main stream."""

    event: torch.cuda.Event

    # [num_reqs, combined_width]: one single tensor packing everything sent
    # in the one broadcast call -- verified sampled tokens, the newly
    # proposed (not-yet-verified) next block, and num_sampled/num_rejected
    # as trailing int64 columns. Split back apart in get_prev_sampled_outputs.
    payload: torch.Tensor
    idx_mapping: torch.Tensor  # [num_reqs]
    idx_mapping_np: np.ndarray  # [num_reqs]
    # Records which rows need a deferred postprocess (bool).
    need_sampled_mask: np.ndarray  # [num_reqs]
    # Snapshot of slot generation counters at receive time, used to
    # detect requests aborted since then.
    gen_at_receive_np: np.ndarray  # [num_reqs]


def compute_need_sampled_mask(input_batch: InputBatch) -> np.ndarray | None:
    """Return a bool array of shape `[input_batch.num_reqs]` marking requests
    with outputs that might be needed in a subsequent (decode) step.
    Returns None if no sampled outputs are needed in the requests' next step."""

    old_computed = input_batch.num_computed_tokens_np
    prefill_len = input_batch.prefill_len_np
    max_seq_len = input_batch.max_seq_len_np
    assert max_seq_len is not None  # always populated under PP
    # Exclude non-final prefill chunks (they don't produce a sample).
    produces_sample = old_computed + input_batch.num_scheduled_tokens >= prefill_len
    # Exclude requests that we know are finished.
    not_finishing = np.maximum(old_computed, prefill_len) + 1 < max_seq_len
    need_sampled_mask = produces_sample & not_finishing
    return need_sampled_mask if need_sampled_mask.any() else None


class PPHandler:
    """Runs the PP sampled-token broadcast/recv on a side stream so the
    default stream isn't gated by the matching peer call. Step T's recv is
    consumed at step T+pp_size via `get_prev_sampled_outputs`.

    Uses a dedicated NCCL communicator (sibling of the PP `device_group`)
    for the broadcast so it does not serialize on the wire with the
    inter-stage hidden-state p2p send/recv ops.

    `receive`/`broadcast` must each issue exactly one `torch.distributed
    .broadcast` call. Do not add a second one to carry new data -- pack it
    into the existing payload tensor instead (see `payload_width`). Every
    additional tensor allocated independently and broadcast on its own is
    a real correctness risk: a narrow tensor unbound/sliced from another
    (e.g. row 1 of a stacked [2, N] tensor) can land at a non-16-byte-
    aligned offset, and Triton compiles a distinct, effectively unwarmable
    kernel specialization for an unaligned pointer -- this is what caused
    a real, hours-long "hang" that was actually a live cold compile, not a
    deadlock. One tensor, one broadcast, always aligned.
    """

    def __init__(
        self, max_num_reqs: int, num_speculative_steps: int, device: torch.device
    ):
        self.is_last_rank = get_pp_group().is_last_rank
        self.last_rank = get_pp_group().last_rank
        self.num_speculative_steps = num_speculative_steps
        self.max_sample_len = num_speculative_steps + 1
        # Single-tensor payload width, one broadcast call: verified sampled
        # tokens, the newly-proposed (not-yet-verified) next block, then
        # num_sampled and num_rejected as two trailing int64 columns.
        self.tokens_width = self.max_sample_len + num_speculative_steps
        self.payload_width = self.tokens_width + 2
        self.device = device
        self.main_stream = torch.cuda.current_stream(device)
        self.broadcast_stream = torch.cuda.Stream(device)

        # On non-last ranks, a FIFO with one entry per in-flight step: the entry
        # pushed by step T's `receive` is consumed pp_size steps later. Pre-seeded
        # with pp_size None placeholders so the first pp_size consumes are no-ops.
        # None means no postprocess is pending for that step (broadcast skipped).
        self.queue: deque[PendingRecv | None] = (
            deque() if self.is_last_rank else deque([None] * get_pp_group().world_size)
        )

        # Per req-index generation counter, incremented every time a request
        # index is freed in RequestStats. Used for invalidating freed req data
        # between PP decodes.
        self.req_idx_gen_np = np.zeros(max_num_reqs, dtype=np.int32)

        # Dedicated subgroup for the sampled-token broadcast.
        self.broadcast_group = get_pp_group().make_sibling_device_group(
            group_desc="pp_broadcast"
        )

    def on_req_idx_freed(self, req_idx: int) -> None:
        self.req_idx_gen_np[req_idx] += 1

    def get_prev_sampled_outputs(self) -> dict[str, torch.Tensor | None] | None:
        """Consume the entry from pp_size steps ago and wait for its recv event,
        then filter out entries whose request was freed since `receive`.
        """
        if not self.queue:
            return None
        slot = self.queue.popleft()
        # Reserve this step's slot; `receive` overwrites it if applicable.
        self.queue.append(None)
        if slot is None:
            return None

        # Skip requests which did not need sampled output and/or those already
        # finished. The post_update kernel skips the -1 entries.
        freed = self.req_idx_gen_np[slot.idx_mapping_np] != slot.gen_at_receive_np
        exclude_mask = freed | ~slot.need_sampled_mask
        idx_mapping = slot.idx_mapping
        if exclude_mask.any():
            if exclude_mask.all():
                # No states require update anymore.
                return None
            # Filter excluded request indices.
            idx_mapping_np = np.where(exclude_mask, -1, slot.idx_mapping_np)
            idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)

        self.main_stream.wait_event(slot.event)
        sampled_tokens = slot.payload[:, : self.max_sample_len]
        proposed_tokens = (
            slot.payload[:, self.max_sample_len : self.tokens_width]
            if self.num_speculative_steps > 0
            else None
        )
        num_sampled = slot.payload[:, self.tokens_width].to(torch.int32)
        num_rejected = slot.payload[:, self.tokens_width + 1].to(torch.int32)
        return dict(
            sampled_tokens=sampled_tokens,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            proposed_tokens=proposed_tokens,
            idx_mapping=idx_mapping,
        )

    def receive(self, input_batch: InputBatch) -> bool:
        """Returns True iff sampled tokens need to be gathered from *all*
        requests in the batch."""
        assert not self.is_last_rank
        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if need_sampled_mask is None:
            # Leave this step's reserved slot as None.
            return False

        # Snapshot the per-slot generation counter so a later free of any of
        # these RequestStates request indices is detectable at consume time.
        gen_at_receive_np = self.req_idx_gen_np[input_batch.idx_mapping_np]

        num_reqs = input_batch.num_reqs
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            # One single, freshly-allocated tensor -- naturally aligned, no
            # sub-view offsets -- carrying everything in one broadcast call.
            payload = torch.empty(
                num_reqs, self.payload_width, dtype=torch.int64, device=self.device
            )
            with record_function_or_nullcontext("gpu_model_runner: pp_receive"):
                torch.distributed.broadcast(
                    payload, src=self.last_rank, group=self.broadcast_group
                )
            event = self.broadcast_stream.record_event()
            # Must record_stream since this was allocated on broadcast stream but
            # later used on the main stream.
            payload.record_stream(self.main_stream)
        self.queue[-1] = PendingRecv(
            event,
            payload,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            need_sampled_mask,
            gen_at_receive_np,
        )
        return bool(need_sampled_mask.all())

    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
        # [num_reqs, num_speculative_steps]; the NEW, not-yet-verified block
        # this rank just proposed via propose(). Non-last ranks need this as
        # real embedding input for their own next verification forward pass
        # -- only this rank can compute it (propose() needs aux_hidden_states,
        # which only exists here). Packed into the same payload tensor below.
        proposed_token_ids: torch.Tensor | None = None,
    ) -> None:
        assert self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:
            # No request needs sampled outputs for a subsequent decode step.
            return

        assert sampled_token_ids.dtype == torch.int64
        # The plain (non-speculative) Sampler always returns width 1
        # (sampled.view(-1, 1)), regardless of max_sample_len -- e.g. a
        # prefill/first step, before any draft tokens are active yet. Only
        # the RejectionSampler pads to max_sample_len. receive() allocates a
        # fixed max_sample_len-wide column range, so pad here to match;
        # post_update only ever reads the first num_sampled[req] columns of
        # a row, so the padding value itself is never consumed.
        width = sampled_token_ids.shape[1]
        assert width <= self.max_sample_len
        if width < self.max_sample_len:
            sampled_token_ids = torch.nn.functional.pad(
                sampled_token_ids, (0, self.max_sample_len - width)
            )
        parts = [sampled_token_ids]
        if self.num_speculative_steps > 0:
            assert proposed_token_ids is not None
            assert proposed_token_ids.dtype == torch.int64
            assert proposed_token_ids.shape[1] == self.num_speculative_steps
            parts.append(proposed_token_ids)
        parts.append(num_sampled.to(torch.int64).unsqueeze(1))
        parts.append(num_rejected.to(torch.int64).unsqueeze(1))
        # One single tensor, one broadcast call.
        payload = torch.cat(parts, dim=1).contiguous()

        if current_platform.is_xpu():
            self.main_stream.synchronize()

        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            with record_function_or_nullcontext("gpu_model_runner: pp_broadcast"):
                torch.distributed.broadcast(
                    payload, src=self.last_rank, group=self.broadcast_group
                )
            payload.record_stream(self.broadcast_stream)
