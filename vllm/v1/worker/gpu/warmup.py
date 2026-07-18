# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import numpy as np
import torch

from vllm import PoolingParams, SamplingParams
from vllm.logger import init_logger
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.attention.backends.mla.indexer import (
    warmup_prefill_chunk_metadata_kernel,
)
from vllm.utils.math_utils import cdiv
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


def _is_deepseek_v4_model_runner(model_runner: GPUModelRunner) -> bool:
    model_config = getattr(model_runner, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", None) or ()
    return any("DeepseekV4" in arch or "DeepSeekV4" in arch for arch in architectures)


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
    if model_runner.kv_cache_config.num_blocks <= required_blocks:
        logger.warning(
            "Skipping V2 mixed prefill+decode warmup because only %d KV blocks "
            "are available for %d required warmup blocks.",
            model_runner.kv_cache_config.num_blocks,
            required_blocks,
        )
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
    model_runner.kv_connector.set_disabled(True)
    try:
        worker_execute_model(decode_prefill_output)
        worker_sample_tokens(None)
        with context:
            worker_execute_model(mixed_output)
            worker_sample_tokens(None)

        # Async PP defers sampled-token postprocessing by one pipeline depth.
        # Drain those slots before finishing the synthetic requests so the
        # deferred post_update path is JIT-warmed while request state is valid.
        pp_size = getattr(
            getattr(model_runner, "parallel_config", None),
            "pipeline_parallel_size",
            1,
        )
        for _ in range(max(0, pp_size)):
            worker_execute_model(SchedulerOutput.make_empty())

        worker_execute_model(cleanup_output)
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

    Two decode steps cover the two verify layouts the scheduler emits under
    PP-deferred verification: the FIRST verify after a prefill schedules the
    anchor position plus the drafts (1 + T query tokens), while steady-state
    verify batches hold ONLY the draft positions (T query tokens). Both are
    distinct kernel specializations. State bookkeeping for the synthetic
    request only needs to be internally consistent enough to reach the
    kernels; the request is finished and its blocks dropped afterwards.
    """
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
    if model_runner.kv_cache_config.num_blocks <= sum(full_block_counts):
        logger.warning(
            "Skipping V2 spec verify warmup: only %d KV blocks available for "
            "%d required warmup blocks.",
            model_runner.kv_cache_config.num_blocks,
            sum(full_block_counts),
        )
        return False

    next_block_id = 1

    def _alloc_blocks(num_blocks: int) -> list[int]:
        nonlocal next_block_id
        block_ids = list(range(next_block_id, next_block_id + num_blocks))
        next_block_id += num_blocks
        return block_ids

    sampling_params = SamplingParams(max_tokens=max_len, temperature=0.0)

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

    # Steady-state PP-deferred verify: drafts only (T tokens).
    steady_verify = SchedulerOutput.make_empty()
    steady_verify.scheduled_cached_reqs = _cached_step(prompt_len + 1, 2, False)
    steady_verify.num_scheduled_tokens = {req_id: num_spec}
    steady_verify.total_num_scheduled_tokens = num_spec
    steady_verify.scheduled_spec_decode_tokens = {req_id: [0] * num_spec}
    steady_verify.num_common_prefix_blocks = [0] * num_kv_cache_groups

    cleanup_output = SchedulerOutput.make_empty()
    cleanup_output.finished_req_ids = {req_id}

    pp_size = getattr(
        getattr(model_runner, "parallel_config", None), "pipeline_parallel_size", 1
    )

    model_runner.kv_connector.set_disabled(True)
    try:
        worker_execute_model(prefill_output)
        worker_sample_tokens(None)
        worker_execute_model(first_verify)
        worker_sample_tokens(None)
        worker_execute_model(steady_verify)
        worker_sample_tokens(None)
        # Async PP defers sampled-token postprocessing by one pipeline depth.
        # Drain those slots so the deferred post_update path is JIT-warmed
        # while the synthetic request's state is still valid.
        for _ in range(max(0, pp_size)):
            worker_execute_model(SchedulerOutput.make_empty())
        worker_execute_model(cleanup_output)
    finally:
        model_runner.kv_connector.set_disabled(False)
    logger.info(
        "V2 spec verify warmup completed (first-verify %d tokens, steady %d tokens).",
        1 + num_spec,
        num_spec,
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
        warmup_block_table_slot_mapping_kernel(model_runner, device)

    if _is_deepseek_v4_model_runner(model_runner):
        logger.info(
            "Skipping DeepSeek V4 full-model long-prefill warmup; direct sparse "
            "MLA prefill warmups cover the Triton specializations."
        )
        return

    max_tokens = model_runner.scheduler_config.max_num_batched_tokens
    token_sizes = sorted({16, max_tokens})
    warmed_sizes: list[int] = []
    for num_tokens in token_sizes:
        if num_tokens < 3:
            continue
        if run_mixed_prefill_decode_warmup(
            model_runner,
            worker_execute_model,
            worker_sample_tokens,
            num_tokens,
            req_id_prefix=f"_v2_long_prefill_warmup_{num_tokens}",
        ):
            warmed_sizes.append(num_tokens)

    if warmed_sizes:
        logger.info(
            "V2 long prefill kernel warmup completed with scheduled tokens: %s.",
            warmed_sizes,
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
        torch.accelerator.synchronize()
        return

    num_spec_steps = model_runner.num_speculative_steps
    decode_query_len = model_runner.decode_query_len
    # Use decode_query_len + 1 tokens so the prefill batch's per-request query
    # length exceeds decode_query_len, preventing it from being misclassified as
    # a uniform decode batch.
    prompt_len = decode_query_len + 1
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
    torch.accelerator.synchronize()
