# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import torch

from vllm.config import SpeculativeConfig
from vllm.config.model import PROCESSED_LOGPROBS_MODES
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates
from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    expand_idx_mapping,
    get_num_sampled_and_rejected,
)
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores
from vllm.v1.worker.gpu.sample.output import SamplerOutput, SamplingMaskTensors
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    rejection_sample,
)

# Cap on the FP32 target-logits buffer materialized by apply_sampling_params.
# TODO(mgoin): Chunking is a workaround. The rejection kernels already upcast
# per vocab block on load and apply ops like temperature and gumbel, so folding
# sampling-param application into those kernels would remove this buffer and
# its traffic entirely.
MAX_CHUNK_BYTES = 2**30  # 1GB
_FP32_BYTES = 4


def get_max_chunk_logits(vocab_size: int) -> int:
    """Largest number of logits rows one verification chunk may hold."""
    return max(1, MAX_CHUNK_BYTES // (vocab_size * _FP32_BYTES))


def _iter_request_chunks(
    cu_num_logits: np.ndarray, max_chunk_logits: int
) -> Iterator[tuple[int, int]]:
    """Yield maximally packed request ranges without splitting requests."""
    assert max_chunk_logits > 0
    num_reqs = cu_num_logits.size - 1
    start = 0
    while start < num_reqs:
        max_logit = int(cu_num_logits[start]) + max_chunk_logits
        end = int(np.searchsorted(cu_num_logits, max_logit, side="right") - 1)
        end = min(num_reqs, max(start + 1, end))
        yield start, end
        start = end


@triton.jit
def _flatten_sampled_kernel(
    # [num_logits]
    flat_sampled_ptr,
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    for i in range(num_sampled):
        token_id = tl.load(sampled_ptr + req_idx * sampled_stride + i)
        tl.store(flat_sampled_ptr + start_idx + i, token_id)


@triton.jit
def _gather_draft_sampled_kernel(
    # [num_logits]
    draft_sampled_ptr,
    # [num_logits]
    pos_ptr,
    # [num_tokens]
    input_ids_ptr,
    # [num_tokens]
    positions_ptr,
    # [num_logits]
    logits_indices_ptr,
    # [num_logits]
    expanded_idx_mapping_ptr,
    # [num_logits]
    expanded_local_pos_ptr,
    # [max_num_reqs]
    prefill_len_ptr,
    num_logits,
    BLOCK_SIZE: tl.constexpr,
):
    block = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < num_logits
    token_idx = tl.load(logits_indices_ptr + block, mask=mask)
    token_id = tl.load(input_ids_ptr + token_idx, mask=mask)
    pos = tl.load(positions_ptr + token_idx, mask=mask)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + block, mask=mask)
    local_pos = tl.load(expanded_local_pos_ptr + block, mask=mask)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx, mask=mask)
    # Drafts are only proposed after sampling past the prefill, so any draft
    # slots in a step starting within the prefill are placeholders (e.g. the
    # padded first step after a P/D remote KV load).
    is_placeholder = (local_pos > 0) & (pos - local_pos < prefill_len)
    token_id = tl.where(is_placeholder, -1, token_id)
    tl.store(draft_sampled_ptr + block, token_id, mask=mask)
    tl.store(pos_ptr + block, pos, mask=mask)


def gather_draft_sampled(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    logits_indices: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    prefill_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather the input token and position of each logits row.

    Draft rows of requests that have not yet sampled past their prefill are
    set to -1 so that the rejection kernels reject them.
    """
    num_logits = logits_indices.shape[0]
    draft_sampled = input_ids.new_empty(num_logits)
    pos = positions.new_empty(num_logits)
    BLOCK_SIZE = 1024
    _gather_draft_sampled_kernel[(triton.cdiv(num_logits, BLOCK_SIZE),)](
        draft_sampled,
        pos,
        input_ids,
        positions,
        logits_indices,
        expanded_idx_mapping,
        expanded_local_pos,
        prefill_len,
        num_logits,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return draft_sampled, pos


class RejectionSampler:
    def __init__(
        self,
        sampler: Sampler,
        spec_config: SpeculativeConfig,
        device: torch.device,
        *,
        watermark_key: int | None = None,
    ):
        self.sampler = sampler
        self.watermark_key = watermark_key
        if watermark_key is not None:
            assert isinstance(sampler, GPUWatermarkSampler)
        self.num_speculative_steps = spec_config.num_speculative_tokens
        self.enable_adaptive_verification = spec_config.enable_adaptive_verification
        rejection_sample_method = spec_config.rejection_sample_method
        self.use_block_verification: bool = False
        self.synthetic_conditional_rates: torch.Tensor | None = None
        if rejection_sample_method == "synthetic":
            assert spec_config.synthetic_acceptance_rates is not None
            self.synthetic_conditional_rates = torch.tensor(
                unconditional_to_conditional_rates(
                    spec_config.synthetic_acceptance_rates
                ),
                dtype=torch.float32,
                device=device,
            )
        elif rejection_sample_method == "block":
            self.use_block_verification = True

    def _get_logprobs_tensors(
        self,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        logits: torch.Tensor,
        cu_num_logits: torch.Tensor,
        cu_num_logits_np: np.ndarray,
        max_num_logprobs: int,
    ) -> LogprobsTensors | None:
        if max_num_logprobs == NO_LOGPROBS:
            return None

        num_reqs = cu_num_logits.shape[0] - 1
        num_logits = logits.shape[0]
        flat_sampled = torch.zeros(
            num_logits, dtype=sampled.dtype, device=sampled.device
        )
        _flatten_sampled_kernel[(num_reqs,)](
            flat_sampled,
            sampled,
            sampled.stride(0),
            num_sampled,
            cu_num_logits,
            num_warps=1,
        )
        expanded_logits = num_logits != num_reqs
        cu_num_generated_tokens: list[int] | torch.Tensor | None = None
        if expanded_logits:
            if self.enable_adaptive_verification:
                # Adaptive verification keeps the true per-request boundaries
                # on device only; cu_num_logits_np holds the pre-compacted layout.
                cu_num_generated_tokens = cu_num_logits.clone()
            else:
                cu_num_generated_tokens = cu_num_logits_np.tolist()
        return compute_topk_scores(
            logits,
            max_num_logprobs,
            flat_sampled,
            cu_num_generated_tokens,
            logits_mode=self.sampler.logprobs_mode
            in ("raw_logits", "processed_logits"),
        )

    def _expand_for_deferred(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        prev_sampled_tokens: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Prepend a virtual anchor row for legacy drafts-only PP frames.

        Current PP-deferred scheduling supplies the real [anchor, drafts...]
        rows in one target forward and bypasses this compatibility path. For
        an older drafts-only frame, the anchor position ran and was sampled in
        a prior in-flight step, so only its sample (``last_sampled``) remains.
        A one-hot row at that token preserves the coupled-verification result;
        the caller then drops the virtual slot from emitted outputs.
        """
        device = logits.device
        old_draft_sampled, old_pos = gather_draft_sampled(
            input_batch.input_ids,
            input_batch.positions,
            input_batch.logits_indices,
            input_batch.expanded_idx_mapping,
            input_batch.expanded_local_pos,
            self.sampler.req_states.prefill_len.gpu,
        )
        num_reqs = input_batch.idx_mapping.shape[0]
        cu_np = input_batch.cu_num_logits_np
        counts_np = np.diff(cu_np)
        replayed_np = getattr(input_batch, "replayed_pp_anchor_np", None)
        if replayed_np is not None and replayed_np.all():
            # The scheduler supplied the real anchor row for every request in
            # this batch. Use the target logits directly; constructing virtual
            # one-hot distributions here would both duplicate the anchors and
            # skip their target verification.
            return (
                logits,
                old_draft_sampled,
                old_pos,
                input_batch.cu_num_logits,
                input_batch.expanded_idx_mapping,
                input_batch.expanded_local_pos,
                torch.zeros(num_reqs, dtype=torch.int64, device=device),
            )
        # A request needs a virtual anchor row iff its FIRST logit position is
        # a draft position -- i.e. the real anchor (the last known token) was
        # NOT scheduled in this step. Decide this from the SAME GPU token
        # boundary the combine kernel used to write inputs (total_len = count
        # of known tokens), not from scheduler draft counts: under PP those
        # counts lag the boundary, and the count heuristic (counts == ndpr)
        # wrongly prepended a virtual anchor on the first post-prefill verify
        # -- which DOES schedule the real anchor -- duplicating it (the
        # observed alternating-token output). Row 0's absolute position is
        # seq_len - num_logits; it is a real known token iff that position is
        # below total_len, exactly matching the combine kernel's per-position
        # known/draft split.
        counts_g = torch.as_tensor(counts_np, device=device, dtype=torch.int64)
        first_pos = input_batch.seq_lens[:num_reqs].to(torch.int64) - counts_g
        total_len_g = self.sampler.req_states.total_len.gpu[input_batch.idx_mapping]
        has_virtual_np = (
            (first_pos >= total_len_g.to(torch.int64)).to(torch.int64).cpu().numpy()
        )
        if replayed_np is not None:
            has_virtual_np[replayed_np] = 0
        if not has_virtual_np.any():
            # Classic-only batch: nothing to expand.
            return (
                logits,
                old_draft_sampled,
                old_pos,
                input_batch.cu_num_logits,
                input_batch.expanded_idx_mapping,
                input_batch.expanded_local_pos,
                torch.zeros(num_reqs, dtype=torch.int64, device=device),
            )
        new_cu_np = np.zeros(num_reqs + 1, dtype=np.int32)
        np.cumsum(counts_np + has_virtual_np, out=new_cu_np[1:])
        new_n = int(new_cu_np[-1])
        n = logits.shape[0]

        # Each real row shifts by the number of virtual rows inserted at or
        # before its request; each virtual row sits at its request's new start.
        shift_np = np.cumsum(has_virtual_np)
        dest_np = np.arange(n, dtype=np.int64) + np.repeat(shift_np, counts_np)
        virt_reqs_np = has_virtual_np.astype(bool)
        virt_np = new_cu_np[:-1][virt_reqs_np].astype(np.int64)

        dest = torch.from_numpy(dest_np).to(device, non_blocking=True)
        virt = torch.from_numpy(virt_np).to(device, non_blocking=True)
        virt_reqs = torch.from_numpy(virt_reqs_np).to(device, non_blocking=True)

        # last_sampled_tokens is [max_num_reqs, 1]; flatten after the gather.
        prev_tok = prev_sampled_tokens[input_batch.idx_mapping].reshape(-1)
        virt_tok = prev_tok[virt_reqs]

        new_logits = logits.new_full((new_n, logits.shape[1]), float("-inf"))
        new_logits.index_copy_(0, dest, logits)
        new_logits[virt, virt_tok] = 0.0

        draft_sampled = old_draft_sampled.new_empty(new_n)
        draft_sampled.index_copy_(0, dest, old_draft_sampled)
        draft_sampled[virt] = virt_tok.to(draft_sampled.dtype)

        pos = old_pos.new_empty(new_n)
        pos.index_copy_(0, dest, old_pos)
        first_real = torch.from_numpy(cu_np[:-1][virt_reqs_np].astype(np.int64)).to(
            device, non_blocking=True
        )
        pos[virt] = old_pos[first_real] - 1

        cu_num_logits = torch.from_numpy(new_cu_np).to(device, non_blocking=True)
        expanded_idx_mapping, expanded_local_pos = expand_idx_mapping(
            input_batch.idx_mapping,
            new_n,
            cu_num_logits,
            max_expand_len=self.num_speculative_steps + 2,
        )
        has_virtual = torch.from_numpy(has_virtual_np).to(device, non_blocking=True)
        return (
            new_logits,
            draft_sampled,
            pos,
            cu_num_logits,
            expanded_idx_mapping,
            expanded_local_pos,
            has_virtual,
        )

    def _watermarking_kwargs(
        self,
        draft_sampled: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> dict[str, Any]:
        if self.watermark_key is None:
            return {}
        assert isinstance(self.sampler, GPUWatermarkSampler)
        contexts = self.sampler._get_contexts(
            expanded_idx_mapping, expanded_local_pos, draft_sampled
        )
        watermarking_skip_mask = None
        if self.sampler.deduplicate_contexts != "none":
            watermarking_skip_mask = self.sampler._get_repeated_contexts(
                expanded_idx_mapping, contexts, expanded_local_pos
            )
        return {
            "contexts": contexts,
            "watermarking": self.sampler.watermarking.gpu,
            "watermarking_skip_mask": watermarking_skip_mask,
            "watermark_key": self.watermark_key,
        }

    def _verify(
        self,
        logits: torch.Tensor,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        seq_lens_upper_bound_np: np.ndarray,
        verify_draft_sampled: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        processed_logits = self.sampler.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            draft_sampled,
            expanded_local_pos,
            seq_lens_upper_bound_np,
        )
        sampled, num_sampled = rejection_sample(
            processed_logits,
            draft_logits,
            verify_draft_sampled,
            cu_num_logits,
            pos,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            self.sampler.sampling_states.temperature.gpu,
            self.sampler.sampling_states.seeds.gpu,
            self.num_speculative_steps,
            self.synthetic_conditional_rates,
            use_fp64=self.sampler.use_fp64_gumbel,
            use_block_verification=self.use_block_verification,
            **self._watermarking_kwargs(
                draft_sampled, expanded_idx_mapping, expanded_local_pos
            ),
        )
        return processed_logits, sampled, num_sampled

    def _verify_in_chunks(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        max_chunk_logits: int,
        max_num_logprobs: int,
        verify_draft_sampled: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        LogprobsTensors | None,
        SamplingMaskTensors | None,
    ]:
        if verify_draft_sampled is None:
            verify_draft_sampled = draft_sampled
        cu_num_logits_np = input_batch.cu_num_logits_np
        use_processed_logits = self.sampler.logprobs_mode in PROCESSED_LOGPROBS_MODES
        num_reqs = input_batch.num_reqs

        if logits.shape[0] <= max_chunk_logits:
            # One chunk covers the batch. Adaptive verification compacts the logits
            # without updating cu_num_logits_np (it keeps the pre-compacted layout),
            # so the stale sums must not pick chunk boundaries; its budget cap
            # guarantees the compacted batch always lands here.
            request_chunks: Iterable[tuple[int, int]] = ((0, num_reqs),)
        else:
            assert not self.enable_adaptive_verification
            request_chunks = _iter_request_chunks(cu_num_logits_np, max_chunk_logits)

        sampled_chunks: list[torch.Tensor] = []
        num_sampled_chunks: list[torch.Tensor] = []
        logprobs_chunks: list[LogprobsTensors] = []
        sampling_mask_chunks: list[SamplingMaskTensors] = []
        max_num_kept = None
        if self.sampler.return_sampling_mask:
            max_num_kept = int(
                np.max(
                    self.sampler.sampling_states.top_k.np[input_batch.idx_mapping_np]
                )
            )

        for start, end in request_chunks:
            lo = int(cu_num_logits_np[start])
            hi = int(cu_num_logits_np[end])
            chunk_cu_num_logits_np = cu_num_logits_np[start : end + 1] - lo
            chunk_cu_num_logits = input_batch.cu_num_logits[start : end + 1] - lo
            # draft_logits uses persistent request-state indices and stays global.
            processed_logits, sampled, num_sampled = self._verify(
                logits[lo:hi],
                draft_logits,
                draft_sampled[lo:hi],
                pos[lo:hi],
                chunk_cu_num_logits,
                input_batch.idx_mapping[start:end],
                input_batch.idx_mapping_np[start:end],
                input_batch.expanded_idx_mapping[lo:hi],
                input_batch.expanded_local_pos[lo:hi],
                input_batch.seq_lens_cpu_upper_bound.numpy()[start:end],
                verify_draft_sampled[lo:hi],
            )
            chunk_logprobs = self._get_logprobs_tensors(
                sampled,
                num_sampled,
                processed_logits if use_processed_logits else logits[lo:hi],
                chunk_cu_num_logits,
                chunk_cu_num_logits_np,
                max_num_logprobs,
            )
            if chunk_logprobs is not None:
                logprobs_chunks.append(chunk_logprobs)
            if max_num_kept is not None:
                sampling_mask_chunks.append(
                    SamplingMaskTensors.from_logits(
                        processed_logits,
                        chunk_cu_num_logits,
                        num_sampled,
                        max_num_kept,
                        self.num_speculative_steps + 1,
                    )
                )
            del processed_logits
            sampled_chunks.append(sampled)
            num_sampled_chunks.append(num_sampled)

        if len(sampled_chunks) == 1:
            logprobs_tensors = logprobs_chunks[0] if logprobs_chunks else None
            sampling_mask_tensors = (
                sampling_mask_chunks[0] if sampling_mask_chunks else None
            )
            return (
                sampled_chunks[0],
                num_sampled_chunks[0],
                logprobs_tensors,
                sampling_mask_tensors,
            )

        logprobs_tensors = None
        if logprobs_chunks:
            expanded_logits = logits.shape[0] != input_batch.num_reqs
            logprobs_tensors = LogprobsTensors.cat(
                logprobs_chunks,
                cu_num_generated_tokens=(
                    cu_num_logits_np.tolist() if expanded_logits else None
                ),
            )

        sampled = torch.cat(sampled_chunks)
        num_sampled = torch.cat(num_sampled_chunks)
        sampling_mask_tensors = (
            SamplingMaskTensors.cat(sampling_mask_chunks)
            if sampling_mask_chunks
            else None
        )
        return sampled, num_sampled, logprobs_tensors, sampling_mask_tensors

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None = None,
        invalid_drafts: torch.Tensor | None = None,
        prev_sampled_tokens: torch.Tensor | None = None,
    ) -> SamplerOutput:
        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = get_num_nans(logits) if self.sampler.compute_nans else None

        deferred = prev_sampled_tokens is not None
        if deferred:
            (
                logits_in,
                draft_sampled,
                pos,
                cu_num_logits,
                expanded_idx_mapping,
                expanded_local_pos,
                has_virtual,
            ) = self._expand_for_deferred(logits, input_batch, prev_sampled_tokens)
        else:
            logits_in = logits
            draft_sampled, pos = gather_draft_sampled(
                input_batch.input_ids,
                input_batch.positions,
                input_batch.logits_indices,
                input_batch.expanded_idx_mapping,
                input_batch.expanded_local_pos,
                self.sampler.req_states.prefill_len.gpu,
            )
            cu_num_logits = input_batch.cu_num_logits
            expanded_idx_mapping = input_batch.expanded_idx_mapping
            expanded_local_pos = input_batch.expanded_local_pos
            has_virtual = None

        verify_draft_sampled = draft_sampled
        if invalid_drafts is not None:
            if deferred and logits_in.shape[0] != logits.shape[0]:
                # Insertion shifts every real row; virtual anchors remain valid.
                assert has_virtual is not None
                rows = torch.arange(logits.shape[0], device=logits.device)
                requests = torch.searchsorted(
                    input_batch.cu_num_logits[1:], rows, right=True
                )
                destinations = rows + has_virtual.cumsum(0)[requests]
                expanded_invalid = torch.zeros_like(draft_sampled, dtype=torch.bool)
                expanded_invalid.index_copy_(0, destinations, invalid_drafts)
                invalid_drafts = expanded_invalid
            verify_draft_sampled = draft_sampled.masked_fill(invalid_drafts, -1)

        max_num_logprobs = self.sampler.sampling_states.max_num_logprobs(
            input_batch.idx_mapping_np
        )
        sampling_mask_tensors = None
        if deferred:
            # The expanded layout does not line up with input_batch.cu_num_logits,
            # and the chunking wrapper slices by exactly that, so verify the
            # expanded batch in one pass instead.
            if max_num_logprobs != NO_LOGPROBS:
                raise NotImplementedError(
                    "logprobs are not supported with PP-deferred speculative "
                    "verification"
                )
            processed_logits, sampled, num_sampled = self._verify(
                logits_in,
                draft_logits,
                draft_sampled,
                pos,
                cu_num_logits,
                input_batch.idx_mapping,
                input_batch.idx_mapping_np,
                expanded_idx_mapping,
                expanded_local_pos,
                input_batch.seq_lens_cpu_upper_bound.numpy(),
                verify_draft_sampled,
            )
            logprobs_tensors = None
        else:
            chunk_logit_limit = get_max_chunk_logits(logits.shape[1])
            (
                sampled,
                num_sampled,
                logprobs_tensors,
                sampling_mask_tensors,
            ) = self._verify_in_chunks(
                logits,
                input_batch,
                draft_logits,
                draft_sampled,
                pos,
                chunk_logit_limit,
                max_num_logprobs,
                verify_draft_sampled,
            )
        if deferred:
            assert has_virtual is not None
            # Drop each virtual slot: its token (the anchor position's sample)
            # was already emitted by the step that computed it.
            steps1 = sampled.shape[1]
            gather_idx = (
                torch.arange(steps1, device=sampled.device).unsqueeze(0)
                + has_virtual.unsqueeze(1)
            ).clamp_max_(steps1 - 1)
            sampled = torch.gather(sampled, 1, gather_idx)
            num_sampled = num_sampled - has_virtual.to(num_sampled.dtype)
            replayed_pp_anchor = getattr(input_batch, "replayed_pp_anchor", None)
            if replayed_pp_anchor is not None:
                max_emitted = torch.diff(input_batch.cu_num_logits).to(
                    num_sampled.dtype
                ) - replayed_pp_anchor.to(num_sampled.dtype)
                num_sampled = torch.where(
                    replayed_pp_anchor,
                    torch.minimum(num_sampled, max_emitted),
                    num_sampled,
                )

        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.sampler.req_states.prefill_len.gpu,
        )
        replayed_pp_anchor = getattr(input_batch, "replayed_pp_anchor", None)
        if deferred and replayed_pp_anchor is not None:
            # The replay row is a real target row but not a draft rejection.
            # Exclude it from rejection accounting after capping emission.
            num_rejected = num_rejected - replayed_pp_anchor.to(num_rejected.dtype)

        if deferred and self.sampler.return_sampling_mask:
            # Supports describe emitted tokens; omit the already-emitted virtual anchor.
            mask_starts = cu_num_logits.clone()
            mask_starts[:-1] += has_virtual.to(mask_starts.dtype)
            max_num_kept = int(
                np.max(
                    self.sampler.sampling_states.top_k.np[input_batch.idx_mapping_np]
                )
            )
            sampling_mask_tensors = SamplingMaskTensors.from_logits(
                processed_logits,
                mask_starts,
                num_sampled,
                max_num_kept,
                self.num_speculative_steps + 1,
            )

        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            sampling_mask_tensors=sampling_mask_tensors,
        )
