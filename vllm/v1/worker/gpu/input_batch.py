# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils import random_uuid
from vllm.utils.math_utils import cdiv

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.block_table import BlockTables


class InputBuffers:
    def __init__(
        self,
        max_num_reqs: int,
        max_num_tokens: int,
        device: torch.device,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_num_tokens = max_num_tokens
        self.device = device

        self.input_ids = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        self.positions = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        self.is_padding = torch.zeros(max_num_tokens, dtype=torch.bool, device=device)
        self.query_start_loc = torch.zeros(
            max_num_reqs + 1, dtype=torch.int32, device=device
        )
        self.seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        # DCP: per-request local seq_lens buffer
        self.dcp_local_seq_lens = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )


@dataclass
class InputBatch:
    # batch_idx -> req_id
    req_ids: list[str]
    num_reqs: int
    num_reqs_after_padding: int

    # batch_idx -> req_state_idx
    idx_mapping: torch.Tensor
    idx_mapping_np: np.ndarray
    # Identical to idx_mapping except for spec decoding.
    expanded_idx_mapping: torch.Tensor
    # [total_num_logits] position within request for each logit
    expanded_local_pos: torch.Tensor

    # [num_reqs]
    # batch_idx -> num_scheduled_tokens, (upper bound when using adaptive verification)
    num_scheduled_tokens: np.ndarray
    # number of tokens in the batch,
    #  may be < sum(num_scheduled_tokens) when using adaptive verification
    num_tokens: int
    num_tokens_after_padding: int
    # Sum of draft tokens scheduled across requests.
    num_draft_tokens: int
    # [num_reqs] number of draft tokens scheduled for each request, if any.
    num_draft_tokens_per_req: np.ndarray | None

    # [num_reqs + 1]
    query_start_loc: torch.Tensor
    query_start_loc_np: np.ndarray
    # [num_reqs]
    seq_lens: torch.Tensor
    # [num_reqs] CPU upper bound on seq_lens (see CommonAttentionMetadata).
    seq_lens_cpu_upper_bound: torch.Tensor
    # [num_reqs]
    dcp_local_seq_lens: torch.Tensor | None
    # [num_reqs]
    num_computed_tokens_np: np.ndarray
    # [num_reqs]
    prefill_len_np: np.ndarray
    # [num_reqs]
    num_computed_prefill_tokens_np: np.ndarray
    # [num_reqs] CPU bool array == (num_computed_prefill_tokens_np < prefill_len_np).
    is_prefilling_np: np.ndarray
    # == np.any(is_prefilling_np)
    has_prefill: bool

    # [num_tokens_after_padding]
    input_ids: torch.Tensor
    # [num_tokens_after_padding]
    positions: torch.Tensor
    # [num_tokens_after_padding]
    is_padding: torch.Tensor

    # [total_num_logits]
    logits_indices: torch.Tensor
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor
    cu_num_logits_np: np.ndarray

    # Whether any requests in batch use structured output.
    has_structured_output_reqs: bool

    # [num_reqs] per-request prompt length, only populated for R-SWA.
    prompt_lens: torch.Tensor | None

    # [num_reqs] bool mask for steady PP verifies whose first query row
    # replays the most recently emitted anchor.
    replayed_pp_anchor: torch.Tensor | None = None
    replayed_pp_anchor_np: np.ndarray | None = None

    # Longest query the batch may contain. Set when a cudagraph descriptor promises
    # a query length this batch's own split does not reach, so attention metadata
    # stays valid for every replay the graph serves.
    max_query_len: int | None = None

    @classmethod
    def make_dummy(
        cls,
        num_reqs: int,
        num_tokens: int,
        input_buffers: InputBuffers,
        max_query_len: int | None = None,
    ) -> "InputBatch":
        assert 0 < num_reqs <= num_tokens
        device = input_buffers.device

        req_ids = [f"req_{i}_{random_uuid()}" for i in range(num_reqs)]
        idx_mapping_np = np.arange(num_reqs, dtype=np.intp)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int64, device=device)
        expanded_idx_mapping = idx_mapping
        expanded_local_pos = torch.zeros(num_reqs, dtype=torch.int32, device=device)

        # Distribute the remainder evenly so that no dummy request exceeds
        # ceil(num_tokens / num_reqs) <= max_model_len tokens. Varlen graphs
        # accept any split with non-empty slots, so this shape works for them
        # too; attention metadata is built from the promised max_query_len.
        base_tokens = num_tokens // num_reqs
        num_extra = num_tokens % num_reqs
        assert max_query_len is None or base_tokens + (num_extra > 0) <= max_query_len
        num_scheduled_tokens = np.full(num_reqs, base_tokens, dtype=np.int32)
        if num_extra > 0:
            num_scheduled_tokens[-num_extra:] += 1
        assert int(num_scheduled_tokens.sum()) == num_tokens

        # seq_len equals to query_len
        input_buffers.seq_lens[: num_reqs - num_extra] = base_tokens
        input_buffers.seq_lens[num_reqs - num_extra : num_reqs] = base_tokens + 1
        # Pad for full CUDA graph mode.
        input_buffers.seq_lens[num_reqs:] = 0
        seq_lens = input_buffers.seq_lens[:num_reqs]

        query_start_loc_np = np.empty(num_reqs + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1:])
        input_buffers.query_start_loc[:1] = 0
        torch.cumsum(
            seq_lens, dim=0, out=input_buffers.query_start_loc[1 : num_reqs + 1]
        )
        # Pad for full CUDA graph mode.
        input_buffers.query_start_loc[num_reqs + 1 :] = num_tokens
        query_start_loc = input_buffers.query_start_loc[: num_reqs + 1]

        input_ids = input_buffers.input_ids[:num_tokens].zero_()
        positions = input_buffers.positions[:num_tokens].zero_()

        input_buffers.is_padding[:num_tokens].fill_(True)
        is_padding = input_buffers.is_padding[:num_tokens]

        logits_indices = query_start_loc[1:] - 1
        cu_num_logits = torch.arange(num_reqs + 1, device=device, dtype=torch.int32)
        cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
        # Copy so set_dummy_context can add context in place without touching
        # num_scheduled_tokens.
        seq_lens_cpu_upper_bound = torch.from_numpy(num_scheduled_tokens.copy())
        return cls(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens,
            num_draft_tokens=0,
            num_draft_tokens_per_req=None,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=None,
            num_computed_tokens_np=np.zeros(num_reqs, dtype=np.int32),
            prefill_len_np=np.zeros(num_reqs, dtype=np.int32),
            num_computed_prefill_tokens_np=np.zeros(num_reqs, dtype=np.int32),
            is_prefilling_np=np.zeros(num_reqs, dtype=np.bool_),
            has_prefill=False,
            input_ids=input_ids,
            positions=positions,
            is_padding=is_padding,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=False,
            prompt_lens=None,
            max_query_len=max_query_len,
        )


def set_dummy_context(
    input_batch: InputBatch,
    block_tables: "BlockTables",
    context_len: int,
    num_kv_blocks: int,
    max_model_len: int,
) -> None:
    """Give each dummy request context_len of context, used when profiling step cost."""
    if not block_tables.input_block_tables:
        # Attention-free models have no KV context to fabricate.
        return
    num_reqs = input_batch.num_reqs
    query_len = input_batch.max_query_len or int(input_batch.num_scheduled_tokens.max())
    context_len = max(min(context_len, max_model_len - query_len), 0)
    if not context_len:
        return

    # Decode-like shape: each request continues after context_len
    # already-computed tokens.
    input_batch.seq_lens += context_len
    input_batch.seq_lens_cpu_upper_bound += context_len
    input_batch.num_computed_tokens_np.fill(context_len)
    input_batch.num_computed_prefill_tokens_np.fill(context_len)
    local_pos = np.arange(input_batch.num_tokens, dtype=np.int64) - np.repeat(
        input_batch.query_start_loc_np[:-1], input_batch.num_scheduled_tokens
    )
    input_batch.positions.copy_(torch.from_numpy(local_pos + context_len))

    seq_len = context_len + query_len
    for block_table, block_size, bpk in zip(
        block_tables.input_block_tables,
        block_tables.kernel_block_sizes,
        block_tables.blocks_per_kv_block,
    ):
        num_blocks = min(cdiv(seq_len, block_size), block_table.shape[1])
        # Spans are disjoint until the pool runs out, then they wrap and share
        # blocks: profiling only needs the reads to be realistic, not distinct.
        block_ids = torch.arange(
            num_reqs * num_blocks, dtype=block_table.dtype, device=block_table.device
        ) % (num_kv_blocks * bpk)
        block_table[:num_reqs, :num_blocks] = block_ids.view(num_reqs, num_blocks)


@triton.jit
def _prepare_prefill_inputs_kernel(
    input_ids_ptr,
    next_prefill_tokens_ptr,
    next_prefill_tokens_stride,
    num_lookahead,
    idx_mapping_ptr,
    query_start_loc_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    prefill_lens_ptr,
    num_computed_tokens_ptr,
    BLOCK_SIZE: tl.constexpr,
    LOOKAHEAD_BLOCK: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    prefill_len = tl.load(prefill_lens_ptr + req_state_idx)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    if num_computed >= prefill_len:
        # Not prefill.
        return

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    request_ptr = all_token_ids_ptr + req_state_idx * all_token_ids_stride
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        tokens = tl.load(request_ptr + num_computed + block, mask=mask)
        tl.store(input_ids_ptr + query_start + block, tokens, mask=mask)

    # Store the next num_lookahead prefill tokens.
    lookahead = tl.arange(0, LOOKAHEAD_BLOCK)
    pos = num_computed + query_len + lookahead
    in_lookahead = lookahead < num_lookahead
    tokens = tl.load(
        request_ptr + pos, mask=in_lookahead & (pos < prefill_len), other=0
    )
    tl.store(
        next_prefill_tokens_ptr
        + lookahead * next_prefill_tokens_stride
        + req_state_idx,
        tokens,
        mask=in_lookahead,
    )


def prepare_prefill_inputs(
    input_ids: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    all_token_ids: torch.Tensor,
    prefill_len: torch.Tensor,
    num_computed_tokens: torch.Tensor,
) -> None:
    num_reqs = idx_mapping.shape[0]
    num_lookahead = next_prefill_tokens.shape[0]
    _prepare_prefill_inputs_kernel[(num_reqs,)](
        input_ids,
        next_prefill_tokens,
        next_prefill_tokens.stride(0),
        num_lookahead,
        idx_mapping,
        query_start_loc,
        all_token_ids,
        all_token_ids.stride(0),
        prefill_len,
        num_computed_tokens,
        BLOCK_SIZE=1024,
        LOOKAHEAD_BLOCK=triton.next_power_of_2(num_lookahead),
    )


@triton.jit
def _prepare_pos_seq_lens_kernel(
    pos_ptr,
    seq_lens_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    num_computed_tokens_ptr,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    req_id = tl.program_id(0)
    num_reqs = tl.num_programs(0) - 1
    if req_id == num_reqs:
        # Pad unused seq_lens as 0 for full CUDA graphs.
        for i in tl.range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(seq_lens_ptr + block, 0, mask=mask)
        return

    req_state_idx = tl.load(idx_mapping_ptr + req_id)
    num_computed_tokens = tl.load(num_computed_tokens_ptr + req_state_idx)

    start = tl.load(query_start_loc_ptr + req_id)
    end = tl.load(query_start_loc_ptr + req_id + 1)
    query_len = end - start

    seq_len = num_computed_tokens + query_len
    tl.store(seq_lens_ptr + req_id, seq_len)

    for i in tl.range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        pos = num_computed_tokens + block
        tl.store(pos_ptr + start + block, pos, mask=mask)


def prepare_pos_seq_lens(
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    pos: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    num_reqs = idx_mapping.shape[0]
    # NOTE(woosuk): We do +1 because the last thread block is used
    # to pad unused seq_lens as 0 for full CUDA graphs.
    _prepare_pos_seq_lens_kernel[(num_reqs + 1,)](
        pos,
        seq_lens,
        idx_mapping,
        query_start_loc,
        num_computed_tokens,
        seq_lens.shape[0],
        BLOCK_SIZE=1024,
    )


@triton.jit
def _combine_sampled_and_draft_tokens_kernel(
    input_ids_ptr,
    idx_mapping_ptr,
    last_sampled_tokens_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    prefill_len_ptr,
    draft_tokens_ptr,
    draft_tokens_stride,
    cu_num_logits_ptr,
    logits_indices_ptr,
    num_draft_per_req_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
    BLOCK_SIZE: tl.constexpr,
    NUM_NEW_SAMPLED_TOKENS: tl.constexpr = 1,
    HAS_PER_REQ_DRAFTS: tl.constexpr = False,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    # Get the number of logits and draft tokens.
    cu_num_logits_start = tl.load(cu_num_logits_ptr + batch_idx)
    cu_num_logits_end = tl.load(cu_num_logits_ptr + batch_idx + 1)
    num_logits = cu_num_logits_end - cu_num_logits_start
    if HAS_PER_REQ_DRAFTS:
        # PP-deferred batches can carry a real replayed anchor plus drafts, or
        # a legacy drafts-only frame. Draft-less decode requests in the same
        # batch still carry their single placeholder logit whose input id must
        # be rewritten from last_sampled. The bonus count is per request.
        num_draft_tokens = tl.load(num_draft_per_req_ptr + batch_idx)
        num_bonus_tokens = num_logits - num_draft_tokens
    else:
        num_draft_tokens = num_logits - NUM_NEW_SAMPLED_TOKENS
        num_bonus_tokens = NUM_NEW_SAMPLED_TOKENS

    # Compute the logits indices.
    block = tl.arange(0, BLOCK_SIZE)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    logits_start = query_end - num_logits
    tl.store(
        logits_indices_ptr + cu_num_logits_start + block,
        logits_start + block,
        mask=block < num_logits,
    )

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    if seq_len <= prefill_len:
        # Handling prefill tokens. No sampled or draft tokens.
        return

    if HAS_PER_REQ_DRAFTS:
        # PP path: write each generated-token query slot by POSITION against
        # the request's known-token boundary (total_len = tokens whose ids
        # are known: prompt + emitted outputs incl last_sampled). Positions
        # below the boundary take their KNOWN id from all_token_ids -- this
        # covers the anchor slot AND any deeper post-rewind resume; positions
        # at/above it are draft slots with draft index = position - boundary.
        # Deriving the split from counts instead (bonus = logits - drafts)
        # mis-writes when the scheduler's spec attachment disagrees with the
        # scheduled window (observed: d1 written into the anchor slot).
        total_len = tl.load(total_len_ptr + req_state_idx)
        # The last BLOCK_SIZE query slots cover every draft/known rewrite
        # (drafts <= num_speculative_steps, resume depth <= drafts + 1).
        j = query_end - 1 - block
        pos = seq_len - 1 - block
        in_window = (j >= query_end - num_logits) & (pos >= prefill_len)
        # A current steady verify and the first verify after prefill both carry
        # an anchor logit (num_bonus_tokens==1); only legacy drafts-only frames
        # have zero. On a NON-last PP rank the anchor's just-sampled token may
        # not yet be committed to all_token_ids/total_len at combine time.
        # Read it from last_sampled_tokens (as the non-PP branch does), and
        # shift draft indexing so drafts start one position after it. On the
        # last rank last_sampled_tokens already holds the same value.
        has_anchor = num_bonus_tokens > 0
        anchor_pos = seq_len - num_logits
        is_anchor = in_window & has_anchor & (pos == anchor_pos)
        anchor_tok = tl.load(last_sampled_tokens_ptr + req_state_idx)
        tl.store(input_ids_ptr + j, anchor_tok, mask=is_anchor)
        is_known = in_window & (pos < total_len) & (~is_anchor)
        known_tok = tl.load(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + pos,
            mask=is_known,
            other=0,
        )
        tl.store(input_ids_ptr + j, known_tok, mask=is_known)
        # Draft tokens occupy the LAST num_draft_tokens scheduled positions of
        # the window -- a structural fact of the schedule, independent of the
        # request's commit state. Deriving the base from total_len instead
        # (total_len + 1 when an anchor logit is present) double-counted the
        # anchor once the defer-first-verify scheduler gate guaranteed the
        # anchor is ALREADY committed at verify time (total_len includes it):
        # the first draft slot's index came out -1, was never written, and a
        # stale token id sat in that input row every step -- a guaranteed
        # position-0 rejection (observed as EXACTLY 0% acceptance with the
        # real drafts shifted one row deeper).
        didx = pos - (seq_len - num_draft_tokens)
        is_draft = in_window & (~is_anchor) & (~is_known) & (didx >= 0) & (
            didx < num_draft_tokens
        )
        dtok = tl.load(
            draft_tokens_ptr + req_state_idx * draft_tokens_stride + didx,
            mask=is_draft,
            other=0,
        )
        tl.store(input_ids_ptr + j, dtok, mask=is_draft)
        return

    # Keep prompt-tail slots intact; only rewrite generated-token slots.
    first_logit_seq_pos = seq_len - num_logits
    if num_bonus_tokens > 0 and first_logit_seq_pos >= prefill_len:
        # Write the last sampled token ID to input_ids.
        last_token_id = tl.load(last_sampled_tokens_ptr + req_state_idx)
        tl.store(input_ids_ptr + logits_start, last_token_id)

    # Write the draft tokens (if any) to input_ids.
    if num_draft_tokens > 0:
        mask = block < num_draft_tokens
        draft_tokens = tl.load(
            draft_tokens_ptr + req_state_idx * draft_tokens_stride + block,
            mask=mask,
        )
        tl.store(
            input_ids_ptr + query_end - num_draft_tokens + block,
            draft_tokens,
            mask=mask,
        )


def combine_sampled_and_draft_tokens(
    input_ids: torch.Tensor,
    idx_mapping: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    prefill_len: torch.Tensor,
    draft_tokens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_logits: int,
    num_new_sampled_tokens: int = 1,  # excl accepted draft tokens, a.k.a bonus tokens
    num_draft_per_req: torch.Tensor | None = None,
    all_token_ids: torch.Tensor | None = None,
    total_len: torch.Tensor | None = None,
) -> torch.Tensor:
    assert num_new_sampled_tokens in (0, 1), (
        f"num_new_sampled_tokens must be 0 or 1, got {num_new_sampled_tokens}"
    )
    # use idx_mapping.shape[0] for actual request count
    num_reqs = idx_mapping.shape[0]
    num_speculative_steps = draft_tokens.shape[-1]

    logits_indices = torch.empty(
        num_logits,
        dtype=torch.int64,
        device=input_ids.device,
    )
    _combine_sampled_and_draft_tokens_kernel[(num_reqs,)](
        input_ids,
        idx_mapping,
        last_sampled_tokens,
        query_start_loc,
        seq_lens,
        prefill_len,
        draft_tokens,
        draft_tokens.stride(0),
        cu_num_logits,
        logits_indices,
        num_draft_per_req,
        all_token_ids,
        all_token_ids.stride(0) if all_token_ids is not None else 0,
        total_len,
        NUM_NEW_SAMPLED_TOKENS=num_new_sampled_tokens,
        HAS_PER_REQ_DRAFTS=num_draft_per_req is not None,
        # NOTE(woosuk): Add num_new_sampled_tokens to ensure the block covers the
        # last sampled token in addition to all draft tokens. +1 covers the
        # per-req-drafts mode, where a draft-less request in the batch still
        # has its bonus logit even when num_new_sampled_tokens is 0.
        BLOCK_SIZE=triton.next_power_of_2(
            num_speculative_steps + max(num_new_sampled_tokens, 1)
        ),
    )
    return logits_indices


@triton.jit
def _get_num_sampled_and_rejected_kernel(
    num_sampled_ptr,
    num_rejected_ptr,
    seq_lens_ptr,
    cu_num_logits_ptr,
    idx_mapping_ptr,
    prefill_len_ptr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    is_chunked_prefilling = seq_len < prefill_len

    num_sampled = tl.load(num_sampled_ptr + batch_idx)
    num_sampled = tl.where(is_chunked_prefilling, 0, num_sampled)
    tl.store(num_sampled_ptr + batch_idx, num_sampled)

    logits_start = tl.load(cu_num_logits_ptr + batch_idx)
    logits_end = tl.load(cu_num_logits_ptr + batch_idx + 1)
    num_logits = logits_end - logits_start

    num_rejected = num_logits - num_sampled
    num_rejected = tl.where(is_chunked_prefilling, 0, num_rejected)
    tl.store(num_rejected_ptr + batch_idx, num_rejected)


def get_num_sampled_and_rejected(
    num_sampled: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    prefill_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = idx_mapping.shape[0]
    num_rejected = torch.empty_like(num_sampled)
    _get_num_sampled_and_rejected_kernel[(num_reqs,)](
        num_sampled,
        num_rejected,
        seq_lens,
        cu_num_logits,
        idx_mapping,
        prefill_len,
    )
    return num_sampled, num_rejected


@triton.jit
def _post_update_kernel(
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    last_sampled_tokens_ptr,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    sampled_tokens_ptr,
    sampled_tokens_stride,
    num_sampled_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
):
    req_id = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_id)
    if req_state_idx < 0:
        # Filter rows with negative index entries.
        return

    total_len = tl.load(total_len_ptr + req_state_idx)
    num_sampled = tl.load(num_sampled_ptr + req_id)
    if num_sampled > 0:
        token_id = tl.load(
            sampled_tokens_ptr + req_id * sampled_tokens_stride + num_sampled - 1
        )
        tl.store(last_sampled_tokens_ptr + req_state_idx, token_id)
        tl.store(total_len_ptr + req_state_idx, total_len + num_sampled)

    for i in range(num_sampled):
        token_id = tl.load(sampled_tokens_ptr + req_id * sampled_tokens_stride + i)
        tl.store(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + total_len + i,
            token_id,
        )

        if output_bin_counts_ptr is not None:
            token_ptr = (
                output_bin_counts_ptr
                + req_state_idx * output_bin_counts_stride
                + token_id
            )
            count = tl.load(token_ptr)
            tl.store(token_ptr, count + 1)

    if query_start_loc_ptr is None:
        query_len = 0
    else:
        query_start = tl.load(query_start_loc_ptr + req_id)
        query_end = tl.load(query_start_loc_ptr + req_id + 1)
        query_len = query_end - query_start
    num_rejected = tl.load(num_rejected_ptr + req_id)

    computed_delta = query_len - num_rejected
    if computed_delta != 0:
        num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
        tl.store(num_computed_tokens_ptr + req_state_idx, num_computed + computed_delta)


def post_update(
    # [num_reqs] batch_idx -> req_state_idx; negative index means skip.
    idx_mapping: torch.Tensor,
    # [max_num_reqs]
    num_computed_tokens: torch.Tensor,
    # [max_num_reqs]
    last_sampled_tokens: torch.Tensor,
    # [max_num_reqs, vocab_size]
    output_bin_counts: torch.Tensor | None,
    # [num_reqs, num_speculative_steps + 1]
    sampled_tokens: torch.Tensor,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [num_reqs + 1]
    query_start_loc: torch.Tensor | None,
    # [max_num_reqs, max_model_len]
    all_token_ids: torch.Tensor,
    # [max_num_reqs]
    total_len: torch.Tensor,
) -> None:
    num_reqs = idx_mapping.shape[0]
    _post_update_kernel[(num_reqs,)](
        idx_mapping,
        num_computed_tokens,
        last_sampled_tokens,
        output_bin_counts,
        output_bin_counts.stride(0) if output_bin_counts is not None else 0,
        sampled_tokens,
        sampled_tokens.stride(0),
        num_sampled,
        num_rejected,
        query_start_loc,
        all_token_ids,
        all_token_ids.stride(0),
        total_len,
        num_warps=1,
    )


@triton.jit
def _post_update_num_computed_tokens_kernel(
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    query_start_loc_ptr,
):
    batch_id = tl.program_id(0)
    query_start = tl.load(query_start_loc_ptr + batch_id)
    query_end = tl.load(query_start_loc_ptr + batch_id + 1)
    query_len = query_end - query_start

    req_state_idx = tl.load(idx_mapping_ptr + batch_id)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    tl.store(num_computed_tokens_ptr + req_state_idx, num_computed + query_len)


def post_update_num_computed_tokens(
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [max_num_reqs]
    num_computed_tokens: torch.Tensor,
    # [num_reqs + 1]
    query_start_loc: torch.Tensor,
) -> None:
    num_reqs = idx_mapping.shape[0]
    _post_update_num_computed_tokens_kernel[(num_reqs,)](
        idx_mapping,
        num_computed_tokens,
        query_start_loc,
    )


@triton.jit
def _expand_idx_mapping_kernel(
    idx_mapping_ptr,
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    cu_num_logits_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx

    block = tl.arange(0, BLOCK_SIZE)
    mask = block < num_tokens
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    tl.store(expanded_idx_mapping_ptr + start_idx + block, req_state_idx, mask=mask)
    tl.store(expanded_local_pos_ptr + start_idx + block, block, mask=mask)


def expand_idx_mapping(
    idx_mapping: torch.Tensor,
    total_num_logits: int,
    cu_num_logits: torch.Tensor,
    max_expand_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = idx_mapping.shape[0]
    expanded_idx_mapping = idx_mapping.new_empty(total_num_logits)
    expanded_local_pos = torch.empty(
        total_num_logits, dtype=torch.int32, device=idx_mapping.device
    )
    _expand_idx_mapping_kernel[(num_reqs,)](
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        cu_num_logits,
        BLOCK_SIZE=triton.next_power_of_2(max_expand_len),
    )
    return expanded_idx_mapping, expanded_local_pos
