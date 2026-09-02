# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import numpy as np
import torch

from vllm import PoolingParams, SamplingParams
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.sparse_attn_indexer import (
    warmup_indexer_prefill_gather_kernel,
    warmup_indexer_prefill_logits_kernel,
    warmup_indexer_prefill_topk_kernel,
    warmup_indexer_streaming_topk_kernels,
)
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.triton_utils import HAS_TRITON
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.mla.indexer import (
    warmup_prefill_chunk_metadata_kernel,
)
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.kv_cache_interface import CrossAttentionSpec, MambaSpec
from vllm.v1.request import Request
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)

if HAS_TRITON:
    from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton


def _pp_size(model_runner: GPUModelRunner) -> int:
    return int(
        getattr(
            getattr(model_runner, "parallel_config", None),
            "pipeline_parallel_size",
            1,
        )
        or 1
    )


def pp_ranks_all_agree(model_runner: GPUModelRunner, local_ok: bool, what: str) -> bool:
    """Reduce a per-rank decision to one answer shared by every PP rank.

    Every warmup step that goes through `worker_execute_model` is
    collectively coupled. For a batch with scheduled tokens,
    `GPUWorker.execute_model` posts an UNBOUNDED, BLOCKING metadata
    rendezvous on both sides of each pipeline hop:
    `isend_tensor_dict` -> `send_object` -> `torch.distributed.send` and
    `irecv_tensor_dict` -> `recv_object` -> `torch.distributed.recv`
    (parallel_state.py:1062/852 and 1160/872). "isend" is a misnomer: only
    the payload tensors are async, the metadata handshake is not.

    So a rank that decides on its own to skip a warmup sequence does not
    just lose the warmup, it strands its neighbours forever. Any gate that
    reads RANK-LOCAL state (KV block counts, per-rank projected cache
    groups, anything sized from this rank's own memory) must therefore be
    reduced here, so all ranks either run the sequence or all skip it.

    Gates computed from configuration that is identical in every worker
    process do not need this and must not use it: a rank returning on such
    a gate is guaranteed to have every peer return with it, and routing it
    through a collective would itself be a place to hang. For the same
    reason, everything a caller evaluates before reaching this call must be
    plain attribute reads and arithmetic over state that already exists --
    a rank that raises on the way here strands the ranks that arrived.
    """
    if _pp_size(model_runner) <= 1 or not torch.distributed.is_initialized():
        return local_ok

    flag = torch.tensor([1 if local_ok else 0], dtype=torch.int32, device="cpu")
    torch.distributed.all_reduce(
        flag, op=torch.distributed.ReduceOp.MIN, group=get_pp_group().cpu_group
    )
    agreed = bool(flag.item())
    if local_ok and not agreed:
        logger.info(
            "Skipping %s on every PP rank because at least one rank cannot run "
            "it. The warmup only pays off if all ranks participate; a partial "
            "run deadlocks the pipeline.",
            what,
        )
    return agreed


def run_pp_coupled(
    model_runner: GPUModelRunner, what: str, body: Callable[[], None]
) -> None:
    """Run a PP-coupled warmup sequence, failing fast instead of hanging.

    Once the first transfer of the sequence has been posted, this rank owes
    its neighbours an exact number of matching transfers. Unwinding the
    stack (which under the multiproc executor just returns the worker to
    `worker_busy_loop`'s dequeue, multiproc_executor.py:1021-1039) leaves
    them parked in a rendezvous nothing will ever match, and the engine
    leader blocks reading rank 0's response before it ever sees the
    failure -- the deployment wedges with no error anywhere. Dying is the
    only way to release a blocked peer: it drops the transport and the
    executor's worker monitor fails the engine promptly. Mirrors the same
    reasoning already applied in multiproc_executor.py:1040-1065.
    """
    try:
        body()
    except BaseException:
        if _pp_size(model_runner) <= 1:
            raise
        logger.exception(
            "%s failed after PP transfers were already posted. Peer ranks are "
            "blocked in an unbounded send/recv rendezvous this rank can no "
            "longer match; exiting so they fail fast instead of deadlocking.",
            what,
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


def _is_deepseek_v4_model_runner(model_runner: GPUModelRunner) -> bool:
    model_config = getattr(model_runner, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", None) or ()
    return any("DeepseekV4" in arch or "DeepSeekV4" in arch for arch in architectures)


def _parallel_draft_query_len(model_runner: GPUModelRunner) -> int | None:
    vllm_config = getattr(model_runner, "vllm_config", None)
    speculative_config = getattr(vllm_config, "speculative_config", None)
    method = getattr(speculative_config, "method", None)
    if method not in ("dflash", "dspark"):
        return None

    num_speculative_tokens = int(
        getattr(
            speculative_config,
            "num_speculative_tokens",
            getattr(model_runner, "num_speculative_steps", 0),
        )
    )
    if num_speculative_tokens <= 0:
        return None
    if method == "dflash":
        return 1 + num_speculative_tokens

    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    hf_config = getattr(draft_model_config, "hf_config", None)
    sample_from_anchor = getattr(hf_config, "sample_from_anchor", True)
    return num_speculative_tokens if sample_from_anchor else 1 + num_speculative_tokens


def _prepare_dflash_inputs_block_size(
    max_target_query_len: int, num_query_per_req: int
) -> int:
    span = max(1, max_target_query_len + num_query_per_req)
    return min(256, 1 << (span - 1).bit_length())


def _missing_dflash_prepare_warmup_sizes(
    model_runner: GPUModelRunner, max_tokens: int
) -> set[int]:
    num_query_per_req = _parallel_draft_query_len(model_runner)
    if num_query_per_req is None or max_tokens <= 0:
        return set()

    num_speculative_steps = int(
        getattr(model_runner, "num_speculative_steps", num_query_per_req)
    )
    covered_target_sizes = {
        min(max_tokens, 1 + num_speculative_steps),
        min(max_tokens, 15),
        max(1, max_tokens - 1),
    }
    covered_block_sizes = {
        _prepare_dflash_inputs_block_size(size, num_query_per_req)
        for size in covered_target_sizes
    }

    missing_sizes: set[int] = set()
    for block_size in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        first_span = 1 if block_size == 1 else block_size // 2 + 1
        target_size = max(1, first_span - num_query_per_req)
        if target_size > max_tokens:
            continue
        if (
            _prepare_dflash_inputs_block_size(target_size, num_query_per_req)
            == block_size
            and block_size not in covered_block_sizes
        ):
            missing_sizes.add(target_size)
    return missing_sizes


def warmup_topk_topp_sampler(model_runner: GPUModelRunner) -> bool:
    """Warm the three Triton top-k/top-p constexpr combinations."""
    if not HAS_TRITON or not getattr(model_runner, "is_last_pp_rank", False):
        return False

    device = getattr(model_runner, "device", None)
    model_config = getattr(model_runner, "model_config", None)
    get_vocab_size = getattr(model_config, "get_vocab_size", None)
    if not isinstance(device, torch.device) or not callable(get_vocab_size):
        return False

    vocab_size = int(get_vocab_size())
    if vocab_size <= 0:
        return False
    batch_size = max(8, int(getattr(model_runner, "decode_query_len", 1)))
    logits = torch.zeros(
        (batch_size, vocab_size), dtype=torch.float32, device=device
    )
    top_k = torch.full(
        (batch_size,), min(50, vocab_size), dtype=torch.int32, device=device
    )
    top_p = torch.full((batch_size,), 0.9, dtype=torch.float32, device=device)

    apply_top_k_top_p_triton(logits, None, top_p)
    apply_top_k_top_p_triton(logits, top_k, None)
    apply_top_k_top_p_triton(logits, top_k, top_p)
    torch.accelerator.synchronize()
    logger.info("Triton top-k/top-p sampler warmup completed.")
    return True


def run_mixed_prefill_decode_warmup(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
    num_tokens: int,
    *,
    mixed_step_context: AbstractContextManager[object] | None = None,
    req_id_prefix: str = "_v2_mixed_warmup",
) -> bool:
    """Run a V2 mixed prefill+decode step through normal scheduler inputs."""
    # Config-uniform gates: is_pooling_model comes from
    # model_config.runner_type and max_num_reqs from
    # scheduler_config.max_num_seqs, both identical in every worker process,
    # and num_tokens is derived from scheduler_config by the caller. Every
    # rank returns here or no rank does, so no agreement round is needed.
    if model_runner.is_pooling_model or model_runner.max_num_reqs < 2 or num_tokens < 3:
        return False

    decode_req_id = f"{req_id_prefix}_decode_"
    prefill_req_id = f"{req_id_prefix}_prefill_"
    decode_prompt_len = 2
    decode_scheduled_tokens = 1
    prefill_len = num_tokens - decode_scheduled_tokens
    decode_token_ids = list(range(decode_prompt_len))
    prefill_token_ids = list(range(prefill_len))

    kv_cache_groups = model_runner.kv_cache_config.kv_cache_groups
    num_kv_cache_groups = len(kv_cache_groups)
    group_block_sizes = [g.kv_cache_spec.block_size for g in kv_cache_groups]
    decode_prefill_block_counts = [
        cdiv(decode_prompt_len, block_size) for block_size in group_block_sizes
    ]
    decode_block_counts = [
        cdiv(decode_prompt_len + decode_scheduled_tokens, block_size)
        for block_size in group_block_sizes
    ]
    decode_block_deltas = [
        decode - prefill
        for decode, prefill in zip(decode_block_counts, decode_prefill_block_counts)
    ]
    prefill_block_counts = [
        cdiv(prefill_len, block_size) for block_size in group_block_sizes
    ]
    required_blocks = sum(decode_block_counts) + sum(prefill_block_counts)
    # RANK-LOCAL gate: kv_cache_config is this rank's own config (its
    # projected cache groups, its block budget). Deciding alone here would
    # strand every other rank in the pipeline, so agree first.
    has_blocks = model_runner.kv_cache_config.num_blocks > required_blocks
    if not has_blocks:
        logger.warning(
            "Skipping V2 mixed prefill+decode warmup because only %d KV blocks "
            "are available for %d required warmup blocks.",
            model_runner.kv_cache_config.num_blocks,
            required_blocks,
        )
    if not pp_ranks_all_agree(
        model_runner, has_blocks, "V2 mixed prefill+decode warmup"
    ):
        return False

    next_block_id = 1

    def _alloc_blocks(num_blocks: int) -> list[int]:
        nonlocal next_block_id
        block_ids = list(range(next_block_id, next_block_id + num_blocks))
        next_block_id += num_blocks
        return block_ids

    sampling_params = SamplingParams(max_tokens=2, temperature=0.0)

    decode_prefill_output = SchedulerOutput.make_empty()
    decode_prefill_output.scheduled_new_reqs = [
        NewRequestData(
            req_id=decode_req_id,
            prompt_token_ids=decode_token_ids,
            mm_features=[],
            sampling_params=sampling_params,
            pooling_params=None,
            block_ids=tuple(_alloc_blocks(n) for n in decode_prefill_block_counts),
            num_computed_tokens=0,
            lora_request=None,
            prefill_token_ids=decode_token_ids,
        ),
    ]
    decode_prefill_output.num_scheduled_tokens = {
        decode_req_id: decode_prompt_len,
    }
    decode_prefill_output.total_num_scheduled_tokens = decode_prompt_len
    decode_prefill_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

    decode_new_blocks = tuple(_alloc_blocks(n) for n in decode_block_deltas)
    cached_decode_req = CachedRequestData.make_empty()
    cached_decode_req.req_ids = [decode_req_id]
    cached_decode_req.num_computed_tokens = [decode_prompt_len]
    cached_decode_req.num_output_tokens = [1]
    cached_decode_req.new_block_ids = [
        decode_new_blocks if any(decode_block_deltas) else None
    ]

    mixed_output = SchedulerOutput.make_empty()
    mixed_output.scheduled_cached_reqs = cached_decode_req
    mixed_output.scheduled_new_reqs = [
        NewRequestData(
            req_id=prefill_req_id,
            prompt_token_ids=prefill_token_ids,
            mm_features=[],
            sampling_params=sampling_params,
            pooling_params=None,
            block_ids=tuple(_alloc_blocks(n) for n in prefill_block_counts),
            num_computed_tokens=0,
            lora_request=None,
            prefill_token_ids=prefill_token_ids,
        ),
    ]
    mixed_output.num_scheduled_tokens = {
        decode_req_id: decode_scheduled_tokens,
        prefill_req_id: prefill_len,
    }
    mixed_output.total_num_scheduled_tokens = num_tokens
    mixed_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

    cleanup_output = SchedulerOutput.make_empty()
    cleanup_output.finished_req_ids = {decode_req_id, prefill_req_id}

    context = mixed_step_context or nullcontext()
    pp_size = _pp_size(model_runner)

    def _sequence() -> None:
        def _advance_deferred_state() -> None:
            for _ in range(max(0, pp_size)):
                worker_execute_model(SchedulerOutput.make_empty())

        worker_execute_model(decode_prefill_output)
        worker_sample_tokens(None)
        # The mixed step reuses the decode request. Match scheduler cadence:
        # its sampled state must traverse one pipeline depth first.
        _advance_deferred_state()
        with context:
            worker_execute_model(mixed_output)
            worker_sample_tokens(None)

        # Async PP defers sampled-token postprocessing by one pipeline depth.
        # Drain those slots before finishing the synthetic requests so the
        # deferred post_update path is JIT-warmed while request state is valid.
        # These batches carry zero scheduled tokens, so they post no PP
        # transfers on any rank (see run_spec_verify_warmup).
        _advance_deferred_state()

        worker_execute_model(cleanup_output)

    model_runner.kv_connector.set_disabled(True)
    try:
        run_pp_coupled(model_runner, "V2 mixed prefill+decode warmup", _sequence)
    finally:
        model_runner.kv_connector.set_disabled(False)
    return True


def run_pure_prefill_warmup(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
    num_tokens: int,
    *,
    req_id_prefix: str = "_v2_pure_prefill_warmup",
) -> bool:
    """Warm an intermediate single-request prefill chunk at an exact size."""
    if model_runner.is_pooling_model or num_tokens < 1:
        return False

    req_id = f"{req_id_prefix}_prefill_"
    # One unscheduled prompt token keeps this on the intermediate-prefill path,
    # matching a long request whose current chunk consumes the whole budget.
    prompt_token_ids = list(range(num_tokens + 1))
    kv_cache_groups = model_runner.kv_cache_config.kv_cache_groups
    num_kv_cache_groups = len(kv_cache_groups)
    block_counts = [
        cdiv(num_tokens, group.kv_cache_spec.block_size)
        for group in kv_cache_groups
    ]
    required_blocks = sum(block_counts)
    has_blocks = model_runner.kv_cache_config.num_blocks > required_blocks
    if not has_blocks:
        logger.warning(
            "Skipping V2 pure prefill warmup because only %d KV blocks are "
            "available for %d required warmup blocks.",
            model_runner.kv_cache_config.num_blocks,
            required_blocks,
        )
    if not pp_ranks_all_agree(model_runner, has_blocks, "V2 pure prefill warmup"):
        return False

    next_block_id = 1
    block_ids = []
    for count in block_counts:
        block_ids.append(list(range(next_block_id, next_block_id + count)))
        next_block_id += count

    prefill_output = SchedulerOutput.make_empty()
    prefill_output.scheduled_new_reqs = [
        NewRequestData(
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            mm_features=[],
            sampling_params=SamplingParams(max_tokens=2, temperature=0.0),
            pooling_params=None,
            block_ids=tuple(block_ids),
            num_computed_tokens=0,
            lora_request=None,
            prefill_token_ids=prompt_token_ids,
        )
    ]
    prefill_output.num_scheduled_tokens = {req_id: num_tokens}
    prefill_output.total_num_scheduled_tokens = num_tokens
    prefill_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

    cleanup_output = SchedulerOutput.make_empty()
    cleanup_output.finished_req_ids = {req_id}
    pp_size = _pp_size(model_runner)

    def _sequence() -> None:
        worker_execute_model(prefill_output)
        worker_sample_tokens(None)
        for _ in range(max(0, pp_size)):
            worker_execute_model(SchedulerOutput.make_empty())
        worker_execute_model(cleanup_output)

    model_runner.kv_connector.set_disabled(True)
    try:
        run_pp_coupled(model_runner, "V2 pure prefill warmup", _sequence)
    finally:
        model_runner.kv_connector.set_disabled(False)
    return True


def run_spec_verify_warmup(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
    *,
    req_id_prefix: str = "_v2_spec_warmup",
) -> bool:
    """Run one prefill + both speculative verify shapes through the worker.

    With a speculator enabled, the first live request pays a burst of Triton
    JIT compiles that this warmup exists to absorb: the combine kernel's
    HAS_PER_REQ_DRAFTS specialization, the spec variants of the prepare/
    post-update kernels, the multi-token verify attention metadata kernels,
    and (on the last PP rank) the draft-propose + rejection-sampling stack.

    Two decode steps exercise the scheduler-reachable atomic verification
    layout: the anchor position plus the drafts (1 + T query tokens). State
    bookkeeping for the synthetic request follows the real PP cadence before
    the request is finished and its blocks are dropped.

    Participation is a COLLECTIVE decision. Every non-empty batch below is a
    matched PP send/recv rendezvous, so this either runs on all PP ranks or
    on none: config-uniform gates return outright, rank-local ones go
    through `pp_ranks_all_agree` first.
    """
    # Config-uniform gate. num_speculative_steps is assigned straight from
    # vllm_config.num_speculative_tokens (gpu/model_runner.py:209) and
    # is_pooling_model from model_config.runner_type (gpu/model_runner.py:240);
    # both are identical in every worker process, so a return here is taken
    # by every rank simultaneously and cannot strand a peer.
    num_spec = model_runner.num_speculative_steps
    if num_spec <= 0 or model_runner.is_pooling_model:
        return False

    req_id = f"{req_id_prefix}_req_"
    prompt_len = 2
    prompt_token_ids = list(range(prompt_len))
    # Prefill + anchor + two rounds of drafts, with slack.
    max_len = prompt_len + 2 * (num_spec + 1)

    kv_cache_groups = model_runner.kv_cache_config.kv_cache_groups
    num_kv_cache_groups = len(kv_cache_groups)
    group_block_sizes = [g.kv_cache_spec.block_size for g in kv_cache_groups]
    prefill_block_counts = [
        cdiv(prompt_len, block_size) for block_size in group_block_sizes
    ]
    full_block_counts = [cdiv(max_len, block_size) for block_size in group_block_sizes]
    block_deltas = [
        full - prefill for full, prefill in zip(full_block_counts, prefill_block_counts)
    ]
    # RANK-LOCAL gate: both the block budget and the projected cache groups
    # this sum is taken over belong to THIS rank (a tail rank owning no
    # target layers can even have zero groups, hence a zero requirement).
    # Returning on it alone is what deadlocks the pipeline: the ranks that
    # kept going park in `send_object`/`recv_object` waiting for a peer that
    # already walked away. Agree across the PP group before acting on it.
    has_blocks = model_runner.kv_cache_config.num_blocks > sum(full_block_counts)
    if not has_blocks:
        logger.warning(
            "Skipping V2 spec verify warmup: only %d KV blocks available for "
            "%d required warmup blocks.",
            model_runner.kv_cache_config.num_blocks,
            sum(full_block_counts),
        )
    if not pp_ranks_all_agree(model_runner, has_blocks, "V2 spec verify warmup"):
        return False

    next_block_id = 1

    def _alloc_blocks(num_blocks: int) -> list[int]:
        nonlocal next_block_id
        block_ids = list(range(next_block_id, next_block_id + num_blocks))
        next_block_id += num_blocks
        return block_ids

    # Use a nontrivial value: SamplingStates intentionally skips its Triton
    # temperature kernel when every request has temperature 0 or 1.
    sampling_params = SamplingParams(max_tokens=max_len, temperature=0.5)

    prefill_output = SchedulerOutput.make_empty()
    prefill_output.scheduled_new_reqs = [
        NewRequestData(
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            mm_features=[],
            sampling_params=sampling_params,
            pooling_params=None,
            block_ids=tuple(_alloc_blocks(n) for n in prefill_block_counts),
            num_computed_tokens=0,
            lora_request=None,
            prefill_token_ids=prompt_token_ids,
        ),
    ]
    prefill_output.num_scheduled_tokens = {req_id: prompt_len}
    prefill_output.total_num_scheduled_tokens = prompt_len
    prefill_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

    new_blocks = tuple(_alloc_blocks(n) for n in block_deltas)
    has_new_blocks = any(block_deltas)

    def _cached_step(
        num_computed: int, num_output: int, blocks_first: bool
    ) -> CachedRequestData:
        cached = CachedRequestData.make_empty()
        cached.req_ids = [req_id]
        cached.num_computed_tokens = [num_computed]
        cached.num_output_tokens = [num_output]
        cached.new_block_ids = [new_blocks if (has_new_blocks and blocks_first) else None]
        return cached

    # First verify after prefill: anchor position + T drafts (1 + T tokens).
    first_verify = SchedulerOutput.make_empty()
    first_verify.scheduled_cached_reqs = _cached_step(prompt_len, 1, True)
    first_verify.num_scheduled_tokens = {req_id: 1 + num_spec}
    first_verify.total_num_scheduled_tokens = 1 + num_spec
    first_verify.scheduled_spec_decode_tokens = {req_id: [0] * num_spec}
    first_verify.num_common_prefix_blocks = [0] * num_kv_cache_groups

    # Steady-state verify: anchor position + T drafts (1 + T tokens), matching
    # the scheduler-reachable atomic target-forward shape. A drafts-only
    # synthetic frame exercises a fallback that production scheduling does
    # not enter and can JIT-compile rank by rank inside the coupled pipeline.
    steady_verify = SchedulerOutput.make_empty()
    steady_verify.scheduled_cached_reqs = _cached_step(prompt_len + 1, 2, False)
    steady_verify.num_scheduled_tokens = {req_id: 1 + num_spec}
    steady_verify.total_num_scheduled_tokens = 1 + num_spec
    steady_verify.scheduled_spec_decode_tokens = {req_id: [0] * num_spec}
    steady_verify.replayed_pp_anchor_req_ids = {req_id}
    steady_verify.num_common_prefix_blocks = [0] * num_kv_cache_groups

    cleanup_output = SchedulerOutput.make_empty()
    cleanup_output.finished_req_ids = {req_id}

    pp_size = _pp_size(model_runner)

    def _sequence() -> None:
        # Async PP applies a sampled result to a request only after one full
        # pipeline depth. Zero-token worker calls post no PP transfers, but
        # they advance the deferred queue exactly like intervening scheduler
        # steps, so the same synthetic request is not reused with stale state.
        def _advance_deferred_state() -> None:
            for _ in range(max(0, pp_size)):
                worker_execute_model(SchedulerOutput.make_empty())

        # Three batches with scheduled tokens: each is exactly one PP
        # transfer per hop, identical in count on every rank (a recv on
        # every non-first rank, a send on every non-last rank).
        worker_execute_model(prefill_output)
        worker_sample_tokens(None)
        _advance_deferred_state()
        worker_execute_model(first_verify)
        worker_sample_tokens(None)
        _advance_deferred_state()
        worker_execute_model(steady_verify)
        worker_sample_tokens(None)
        # Async PP defers sampled-token postprocessing by one pipeline depth.
        # Drain those slots so the deferred post_update path is JIT-warmed
        # while the synthetic request's state is still valid.
        #
        # These carry zero scheduled tokens, and so does the cleanup batch
        # below. A zero-token batch posts NO PP transfer on ANY rank:
        # gpu_worker.execute_model computes
        # `forward_pass = total_num_scheduled_tokens > 0` (gpu_worker.py:1036)
        # and guards the irecv on it (gpu_worker.py:1071), while
        # GPUModelRunner.execute_model returns the connector's empty output
        # before producing IntermediateTensors (gpu/model_runner.py:1289-1292)
        # so the isend at gpu_worker.py:1116 is never reached. The drain is
        # therefore collectively symmetric on first, middle and last ranks
        # alike, and stays inside the agreed sequence.
        _advance_deferred_state()
        worker_execute_model(cleanup_output)

    model_runner.kv_connector.set_disabled(True)
    try:
        run_pp_coupled(model_runner, "V2 spec verify warmup", _sequence)
    finally:
        model_runner.kv_connector.set_disabled(False)
    logger.info(
        "V2 spec verify warmup completed (first-verify %d tokens, steady %d tokens).",
        1 + num_spec,
        1 + num_spec,
    )
    return True


def warmup_long_prefill_kernels(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
) -> None:
    """Warm kernels that only appear in long chunked-prefill batches."""
    if model_runner.is_pooling_model:
        return

    device = getattr(model_runner, "device", None)
    if isinstance(device, torch.device):
        # DeepSeek V4 indexer compresses context 4:1. Warming this directly
        # avoids first-request JIT even when a PP rank has no long prefill work.
        warmup_prefill_chunk_metadata_kernel(device, compress_ratio=4)
        warmup_indexer_prefill_gather_kernel(device)
        warmup_indexer_prefill_logits_kernel(device)
        warmup_indexer_prefill_topk_kernel(device)
        warmup_indexer_streaming_topk_kernels(device)
        warmup_block_table_slot_mapping_kernel(model_runner, device)

    scheduler_config = model_runner.scheduler_config
    max_tokens = scheduler_config.max_num_batched_tokens
    mixed_token_sizes = {16, max_tokens}
    pure_prefill_token_sizes: set[int] = set()
    max_scheduled_tokens = getattr(scheduler_config, "max_num_scheduled_tokens", None)
    if max_scheduled_tokens is not None and max_scheduled_tokens != max_tokens:
        pure_prefill_token_sizes.add(max_scheduled_tokens)

    vllm_config = getattr(model_runner, "vllm_config", None)
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is not None:
        draft_slots = speculative_config.max_num_new_slots_for_drafting
        adaptive_budget = max_tokens - draft_slots
        if 0 < adaptive_budget < max_tokens:
            pure_prefill_token_sizes.add(adaptive_budget)
    pure_prefill_token_sizes.update(
        _missing_dflash_prepare_warmup_sizes(model_runner, max_tokens)
    )

    warmed_mixed_sizes: list[int] = []
    for num_tokens in sorted(mixed_token_sizes):
        if num_tokens < 3:
            continue
        if run_mixed_prefill_decode_warmup(
            model_runner,
            worker_execute_model,
            worker_sample_tokens,
            num_tokens,
            req_id_prefix=f"_v2_long_prefill_warmup_{num_tokens}",
        ):
            warmed_mixed_sizes.append(num_tokens)

    warmed_pure_prefill_sizes: list[int] = []
    for num_tokens in sorted(pure_prefill_token_sizes):
        if run_pure_prefill_warmup(
            model_runner,
            worker_execute_model,
            worker_sample_tokens,
            num_tokens,
            req_id_prefix=f"_v2_long_prefill_boundary_warmup_{num_tokens}",
        ):
            warmed_pure_prefill_sizes.append(num_tokens)

    if warmed_mixed_sizes or warmed_pure_prefill_sizes:
        logger.info(
            "V2 long prefill kernel warmup completed with mixed scheduled "
            "tokens %s and pure-prefill scheduled tokens %s.",
            warmed_mixed_sizes,
            warmed_pure_prefill_sizes,
        )


def warmup_block_table_slot_mapping_kernel(
    model_runner: GPUModelRunner,
    device: torch.device,
) -> bool:
    input_batch = getattr(model_runner, "input_batch", None)
    block_tables = getattr(getattr(input_batch, "block_table", None), "block_tables", None)
    if not block_tables:
        return False

    max_tokens = int(model_runner.scheduler_config.max_num_batched_tokens)
    if max_tokens <= 0:
        return False

    multi_group_block_table = input_batch.block_table

    # max_num_batched_tokens is a whole-BATCH token budget (possibly summed
    # across many concurrent requests) and is independent of max_model_len;
    # it can legitimately exceed max_model_len. The block-table row we are
    # about to synthesize below, however, represents a SINGLE request, whose
    # real on-GPU capacity (block_table.max_num_blocks_per_req, in
    # kernel-block units) was sized from max_model_len when the KV-cache
    # manager was built. In production configs max_model_len is large enough
    # that this never matters, but on deliberately small max_model_len
    # configs (testbed shrinks, unit tests) max_num_batched_tokens can be
    # larger than a single row's capacity, and writing
    # cdiv(max_num_batched_tokens, block_size) blocks into that row overflows
    # its backing buffer. Clamp the synthetic row to what each group's table
    # can actually hold; the goal here is only to JIT-warm the kernel, not to
    # exercise the full max_num_batched_tokens budget in one row.
    block_counts = []
    for block_table in block_tables:
        wanted = cdiv(max_tokens, block_table.block_size)
        # append_row() expands each pre-remap block id into
        # blocks_per_kv_block kernel-block ids for hybrid tables, so the
        # capacity ceiling on the PRE-remap count we pass to add_row() is
        # max_num_blocks_per_req // blocks_per_kv_block, not
        # max_num_blocks_per_req directly.
        capacity = max(
            1, block_table.max_num_blocks_per_req // max(1, block_table.blocks_per_kv_block)
        )
        block_counts.append(min(wanted, capacity))

    effective_tokens = min(
        max_tokens,
        min(
            count * block_table.block_size
            for count, block_table in zip(block_counts, block_tables)
        ),
    )
    if effective_tokens <= 0:
        return False

    query_start_loc = torch.tensor(
        [0, effective_tokens], dtype=torch.int32, device=device
    )
    positions = torch.arange(effective_tokens, dtype=torch.int64, device=device)

    try:
        block_ids = tuple(list(range(1, count + 1)) for count in block_counts)
        multi_group_block_table.add_row(block_ids, 0)
        multi_group_block_table.commit_block_table(1)
        multi_group_block_table.compute_slot_mapping(1, query_start_loc, positions)
        torch.accelerator.synchronize()
    finally:
        multi_group_block_table.clear_row(0)
        multi_group_block_table.commit_block_table(1)

    if effective_tokens < max_tokens:
        logger.info(
            "Block-table slot-mapping warmup completed with %d scheduled "
            "tokens (clamped from max_num_batched_tokens=%d to fit each "
            "group's block-table capacity).",
            effective_tokens,
            max_tokens,
        )
    else:
        logger.info(
            "Block-table slot-mapping warmup completed with %d scheduled tokens.",
            effective_tokens,
        )
    return True


def warmup_post_update_num_computed_tokens_kernel(
    model_runner: GPUModelRunner,
    device: torch.device,
) -> bool:
    """Force the first-ever compile of _post_update_num_computed_tokens_kernel
    during warmup, not mid-serving-step.

    Under pipeline_parallel_size > 1, postprocess_num_computed_tokens runs on
    every rank once per step (it's per-rank scheduling bookkeeping, not
    last-rank-only like sampling). If this kernel's first compile happens
    live -- interleaved with another rank's in-flight cross-process
    irecv_tensor_dict handoff for the SAME step -- it can wedge inside the
    CUDA driver's module-load call (triton/compiler/compiler.py:_init_handles
    -> driver.active.utils.load_binary) rather than the compile itself;
    compiling before entering the PP hot loop avoids that timing window
    entirely. Mirrors warmup_block_table_slot_mapping_kernel's synthetic-input
    pattern.
    """
    from vllm.v1.worker.gpu.input_batch import post_update_num_computed_tokens

    req_states = getattr(model_runner, "req_states", None)
    num_computed_tokens = getattr(req_states, "num_computed_tokens", None)
    num_computed_tokens_gpu = getattr(num_computed_tokens, "gpu", None)
    if num_computed_tokens_gpu is None:
        return False

    idx_mapping = torch.zeros(1, dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 0], dtype=torch.int32, device=device)

    post_update_num_computed_tokens(
        idx_mapping,
        num_computed_tokens_gpu,
        query_start_loc,
    )
    torch.accelerator.synchronize()

    logger.info("post_update_num_computed_tokens kernel warmup completed.")
    return True


def warmup_post_update_kernel(
    model_runner: GPUModelRunner,
    device: torch.device,
) -> bool:
    """Force the first-ever compile of _post_update_kernel during warmup.

    Same rationale as warmup_post_update_num_computed_tokens_kernel above:
    postprocess_sampled -> post_update runs on every PP rank once per step
    (accept/reject + all_token_ids bookkeeping), and a first compile
    interleaved with another rank's in-flight cross-process tensor-dict recv
    can wedge inside the CUDA driver's module-load call. Reuses the model
    runner's own real req_states buffers (matching production dtypes/shapes)
    so the compiled kernel is actually reused by the real call, not just a
    differently-specialized one.
    """
    from vllm.v1.worker.gpu.input_batch import post_update

    req_states = getattr(model_runner, "req_states", None)
    if req_states is None:
        return False
    num_computed_tokens_gpu = getattr(
        getattr(req_states, "num_computed_tokens", None), "gpu", None
    )
    all_token_ids_gpu = getattr(getattr(req_states, "all_token_ids", None), "gpu", None)
    total_len_gpu = getattr(getattr(req_states, "total_len", None), "gpu", None)
    last_sampled_tokens = getattr(req_states, "last_sampled_tokens", None)
    if any(
        t is None
        for t in (num_computed_tokens_gpu, all_token_ids_gpu, total_len_gpu, last_sampled_tokens)
    ):
        return False

    num_speculative_steps = int(getattr(model_runner, "num_speculative_steps", 0) or 0)
    idx_mapping = torch.zeros(1, dtype=torch.int32, device=device)
    sampled_tokens = torch.zeros(
        1, num_speculative_steps + 1, dtype=torch.int64, device=device
    )
    num_sampled = torch.zeros(1, dtype=torch.int32, device=device)
    num_rejected = torch.zeros(1, dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 0], dtype=torch.int32, device=device)

    # Two real call sites, two distinct Triton specializations: Triton treats
    # a None-valued pointer argument as a different compiled kernel from a
    # real-tensor argument of the same otherwise-matching shape/dtype (it
    # changes the generated null-check/launcher code, not just a runtime
    # branch inside the kernel body). model_runner.py's synchronous
    # postprocess after this rank's own sampler always passes a real
    # query_start_loc tensor; the ASYNC PP-deferred path (non-last ranks
    # consuming a prior step's broadcast) always passes query_start_loc=None.
    # Both must be warmed independently.
    for warm_query_start_loc in (query_start_loc, None):
        post_update(
            idx_mapping,
            num_computed_tokens_gpu,
            last_sampled_tokens,
            None,  # output_bin_counts: None on non-last-rank in the real call too
            sampled_tokens,
            num_sampled,
            num_rejected,
            warm_query_start_loc,
            all_token_ids_gpu,
            total_len_gpu,
        )
    torch.accelerator.synchronize()

    logger.info("post_update kernel warmup completed (both query_start_loc specializations).")
    return True


@torch.inference_mode()
def warmup_kernels(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
) -> None:
    """Run two execute_model + sample_tokens iterations to JIT compile
    triton kernels. We must call the provided worker's execute_model for
    pipeline parallel coordination.

    The first iteration simulates a prefill with requests of
    decode_query_len + 1 prompt tokens each. The second iteration simulates
    a decode step with all requests generating decode_query_len tokens.
    """
    pp_size = getattr(model_runner.parallel_config, "pipeline_parallel_size", 1)
    if _is_deepseek_v4_model_runner(model_runner) and pp_size > 1:
        logger.info(
            "Skipping generic V2 warmup_kernels for DeepSeek V4 with pipeline "
            "parallel size %d; direct DeepSeek V4 warmups cover the prefill "
            "specializations without startup PP tensor transport.",
            pp_size,
        )
        warmup_long_prefill_kernels(
            model_runner, worker_execute_model, worker_sample_tokens
        )
        # With a speculator, the verify shapes (anchor+drafts and drafts-only)
        # and the draft-propose stack otherwise JIT-compile on the first live
        # request (observed: 19 distinct kernels, ~20s of first-request TTFT).
        run_spec_verify_warmup(
            model_runner, worker_execute_model, worker_sample_tokens
        )
        warmup_topk_topp_sampler(model_runner)
        torch.accelerator.synchronize()
        return

    num_spec_steps = model_runner.num_speculative_steps
    decode_query_len = model_runner.decode_query_len
    # Use decode_query_len + 1 tokens so the prefill batch's per-request query
    # length exceeds decode_query_len, preventing it from being misclassified as
    # a uniform decode batch.
    prompt_len = decode_query_len + 1
    if _is_deepseek_v4_model_runner(model_runner) and num_spec_steps > 0:
        # DFlash input preparation specializes on
        # next_power_of_2(scheduled_tokens + num_query_per_req), capped at
        # 256. Exercise the long-prefill (BLOCK_SIZE=256) compile key during
        # startup instead of compiling it in the first real prefill.
        prompt_len = min(256, model_runner.scheduler_config.max_num_batched_tokens)
    prompt_token_ids = list(range(prompt_len))
    # After prefill, decode generates decode_query_len tokens.
    decode_len = prompt_len + decode_query_len

    kv_cache_groups = model_runner.kv_cache_config.kv_cache_groups
    num_kv_cache_groups = len(kv_cache_groups)

    # Encoder-decoder models: give each warmup request a dummy encoder input so
    # cross-attention warms up over a realistic, non-empty key sequence.
    # The dummy mm_feature is registered in the encoder cache and only its encoder
    # length is read (not the inputs themselves); the encoder itself is not scheduled.
    max_encoder_len = getattr(model_runner.model_state, "max_encoder_len", 0)
    warmup_mm_features: list[MultiModalFeatureSpec] = []
    if model_runner.is_encoder_decoder and max_encoder_len:
        warmup_mm_features = [
            MultiModalFeatureSpec(
                data=None,
                modality="",
                identifier="_warmup_encoder",
                mm_position=PlaceholderRange(offset=0, length=max_encoder_len),
            )
        ]

    # Compute per-request block counts for each KV cache group.
    def _warmup_block_count(num_tokens: int, spec: Any) -> int:
        if isinstance(spec, CrossAttentionSpec):
            num_tokens = max_encoder_len
        num_blocks = cdiv(num_tokens, spec.block_size)
        if isinstance(spec, MambaSpec) and spec.mamba_cache_mode == "align":
            # Align mode reserves extra blocks beyond the token range for the
            # speculative-decode running-state snapshots.
            num_blocks += spec.num_speculative_blocks
        return num_blocks

    kv_cache_specs = [g.kv_cache_spec for g in kv_cache_groups]
    prefill_block_counts = [_warmup_block_count(prompt_len, s) for s in kv_cache_specs]
    decode_block_counts = [_warmup_block_count(decode_len, s) for s in kv_cache_specs]
    decode_block_deltas = [
        d - p for d, p in zip(decode_block_counts, prefill_block_counts)
    ]
    max_blocks_per_req = sum(decode_block_counts)

    num_reqs = min(
        model_runner.scheduler_config.max_num_seqs,
        model_runner.scheduler_config.max_num_batched_tokens
        // max(prompt_len, decode_query_len),
        # Reserve block 0 (null block) and ensure we have enough blocks.
        max(1, (model_runner.kv_cache_config.num_blocks - 1) // max_blocks_per_req),
    )

    req_ids = [f"_warmup_{i}_" for i in range(num_reqs)]

    # SamplingParams exercising all sampling features.
    if model_runner.is_pooling_model:
        sampling_params = None
        pooling_params = PoolingParams()
    else:
        sampling_params = SamplingParams.for_sampler_warmup()
        pooling_params = None

    # Assign distinct block IDs per request per group. 0 null block, start from 1.
    next_block_id = 1

    def _alloc_blocks(num_blocks: int) -> list[int]:
        nonlocal next_block_id
        return list(range(next_block_id, next_block_id := next_block_id + num_blocks))

    # Step 1: Prefill all requests with 1 + decode_query_len prompt tokens each.
    new_reqs = [
        NewRequestData.from_request(
            Request(
                req_ids[i],
                prompt_token_ids,
                sampling_params,
                pooling_params,
                mm_features=warmup_mm_features,
            ),
            block_ids=tuple(_alloc_blocks(n) for n in prefill_block_counts),
            prefill_token_ids=prompt_token_ids,
        )
        for i in range(num_reqs)
    ]

    prefill_output = SchedulerOutput.make_empty()
    prefill_output.scheduled_new_reqs = new_reqs
    prefill_output.num_scheduled_tokens = {rid: prompt_len for rid in req_ids}
    prefill_output.total_num_scheduled_tokens = prompt_len * num_reqs
    prefill_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

    # Disable KV connector for warmup run.
    model_runner.kv_connector.set_disabled(True)
    worker_execute_model(prefill_output)

    if not model_runner.is_pooling_model:
        # Warm up sampler and perform a decode step for non-pooling models.

        grammar_output = None
        if model_runner.is_last_pp_rank:
            # Build a GrammarOutput to exercise the structured output bitmask
            # kernel during the prefill step.
            vocab_size = model_runner.model_config.get_vocab_size()
            bitmask_width = (vocab_size + 31) // 32
            grammar_bitmask = np.full(
                (len(req_ids), bitmask_width), fill_value=-1, dtype=np.int32
            )
            grammar_output = GrammarOutput(
                structured_output_request_ids=req_ids, grammar_bitmask=grammar_bitmask
            )

        worker_sample_tokens(grammar_output)

        # Step 2: Decode all requests with decode_query_len tokens each.
        cached_req_data = CachedRequestData.make_empty()
        cached_req_data.req_ids = list(req_ids)
        cached_req_data.num_computed_tokens = [prompt_len] * num_reqs
        cached_req_data.num_output_tokens = [1] * num_reqs
        new_block = any(decode_block_deltas)
        cached_req_data.new_block_ids = [
            tuple(_alloc_blocks(n) for n in decode_block_deltas) if new_block else None
            for _ in range(num_reqs)
        ]

        decode_output = SchedulerOutput.make_empty()
        decode_output.scheduled_cached_reqs = cached_req_data
        decode_output.num_scheduled_tokens = {
            req_id: decode_query_len for req_id in req_ids
        }
        if num_spec_steps > 0:
            decode_output.scheduled_spec_decode_tokens = {
                req_id: [0] * num_spec_steps for req_id in req_ids
            }
        decode_output.total_num_scheduled_tokens = sum(
            decode_output.num_scheduled_tokens.values()
        )
        decode_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

        worker_execute_model(decode_output)
        worker_sample_tokens(None)

    # Clean up - process finish_req_ids.
    cleanup_output = SchedulerOutput.make_empty()
    cleanup_output.finished_req_ids = set(req_ids)
    worker_execute_model(cleanup_output)
    model_runner.kv_connector.set_disabled(False)
    warmup_long_prefill_kernels(
        model_runner, worker_execute_model, worker_sample_tokens
    )
    if _is_deepseek_v4_model_runner(model_runner):
        warmup_topk_topp_sampler(model_runner)
    torch.accelerator.synchronize()
