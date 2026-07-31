# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warmup kernels used during model execution.
This is useful specifically for JIT'ed kernels as we don't want JIT'ing to
happen during model execution.
"""

from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.warmup.cutedsl_warmup import cutedsl_warmup
from vllm.model_executor.warmup.deep_gemm_warmup import deep_gemm_warmup
from vllm.model_executor.warmup.deepseek_v4_mhc_warmup import (
    deepseek_v4_mhc_warmup,
)
from vllm.model_executor.warmup.fa4_cutedsl_warmup import (
    fa4_cutedsl_warmup,
)
from vllm.model_executor.warmup.flashinfer_autotune_cache import (
    resolve_flashinfer_autotune_file,
    write_flashinfer_autotune_cache,
)
from vllm.model_executor.warmup.flashinfer_sparse_mla_warmup import (
    flashinfer_sparse_mla_decode_autotune_warmup,
)
from vllm.model_executor.warmup.qwen_triton_warmup import qwen_triton_warmup
from vllm.model_executor.warmup.sparse_mla_triton_warmup import (
    sparse_mla_triton_warmup,
)
from vllm.model_executor.warmup.v1_block_table_warmup import (
    warm_v1_block_table_kernels,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.flashinfer import has_flashinfer

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_DEEPSEEK_V4_SPARSE_MLA_PREFILL_WARMUP_TOKENS = 8192
_DEEPSEEK_V4_MLA_HEAD_DIM = 512
_DEEPSEEK_V4_FP8_DS_MLA_TOKEN_DATA_BYTES = 576
_DEEPSEEK_V4_FP8_DS_MLA_SCALE_BYTES = 8
_DEEPSEEK_V4_FP8_DS_MLA_PAGE_TOKEN_BYTES = (
    _DEEPSEEK_V4_FP8_DS_MLA_TOKEN_DATA_BYTES
    + _DEEPSEEK_V4_FP8_DS_MLA_SCALE_BYTES
)
_DEEPSEEK_V4_SYNTHETIC_TOPK = 512
_DEEPSEEK_V4_SYNTHETIC_WINDOW = 128

_LL_BF16_WARMUP_MODEL_SHAPES: tuple[tuple[int, int], ...] = (
    (6144, 264),  # Inkling
    (7168, 256),  # DSV3
    (7168, 384),  # DSV4-Pro
    (14400, 256),  # DSV4-Flash
)
_LL_BF16_WARMUP_M_RANGE = range(1, 17)


def _warmup_ll_bf16_router_gemm() -> None:
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
        is_available as is_ll_bf16_gemm_available,
    )
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
        ll_bf16_gemm_kernel,
    )

    if not is_ll_bf16_gemm_available():
        return

    logger.info("Warming up ll_bf16 router GEMM kernels.")
    ll_bf16_gemm_kernel.warmup(
        shapes=_LL_BF16_WARMUP_MODEL_SHAPES,
        m_values=_LL_BF16_WARMUP_M_RANGE,
    )


def kernel_warmup(worker: "Worker"):
    from vllm.model_executor.warmup.minimax_m3_msa_warmup import (
        minimax_m3_msa_warmup,
    )

    # Pooling models do not use the generation slot-mapping path.
    if not worker.use_v2_model_runner and not worker.model_runner.is_pooling_model:
        warm_v1_block_table_kernels(
            getattr(worker.model_runner, "device", torch.device("cuda")),
            worker.scheduler_config.max_num_batched_tokens,
        )
    qwen_triton_warmup(worker.model_runner, worker.vllm_config.model_config)

    # DSv4 mHC TileLang kernels (hc_pre/hc_post/hc_head_op) run every decoder
    # layer per token; warm them across token sizes first so the first real
    # request doesn't pay JIT cost. No-op for non-DSv4 models (gated inside).
    deepseek_v4_mhc_warmup(
        worker.get_model(),
        max_tokens=worker.scheduler_config.max_num_batched_tokens,
        cudagraph_capture_sizes=(
            worker.vllm_config.compilation_config.cudagraph_capture_sizes or []
        ),
    )

    # Run next so input-prep kernels JIT against pristine runner state.
    flashinfer_sparse_mla_decode_autotune_warmup(worker)
    _deepseek_v4_sparse_mla_prefill_warmup(worker)
    _deepseek_v4_block_table_slot_mapping_warmup(worker)
    _deepseek_v4_marlin_moe_warmup(worker)

    # Deep GEMM warmup
    do_deep_gemm_warmup = (
        envs.VLLM_USE_DEEP_GEMM
        and is_deep_gemm_supported()
        and envs.VLLM_DEEP_GEMM_WARMUP != "skip"
    )
    if do_deep_gemm_warmup:
        model = worker.get_model()
        max_tokens = worker.scheduler_config.max_num_batched_tokens
        deep_gemm_warmup(model, max_tokens)

    minimax_m3_msa_warmup(worker)

    enable_flashinfer_autotune = (
        worker.vllm_config.kernel_config.enable_flashinfer_autotune
    )
    # FlashInfer autotune for Hopper (SM 9.0) and Blackwell (SM 10.0) GPUs
    if enable_flashinfer_autotune is False:
        logger.info("Skipping FlashInfer autotune because it is disabled.")
    elif has_flashinfer() and current_platform.has_device_capability(90):
        flashinfer_autotune(worker.model_runner)

    if current_platform.has_device_capability(90):
        _warmup_ll_bf16_router_gemm()

    # FlashInfer attention warmup
    # Only warmup if the model has FlashInfer attention groups
    # and is not a pooling model
    def _is_flashinfer_backend(backend):
        try:
            return backend.get_name() == "FLASHINFER"
        except NotImplementedError:
            return False

    if (
        not worker.model_runner.is_pooling_model
        and worker.model_runner.attn_groups
        # NOTE: This should be `any` instead of `all` but other hybrid attention
        # backends don't support this dummy run. Once we remove
        # `build_for_cudagraph_capture`, we can change it to `any`.
        and all(
            _is_flashinfer_backend(group.backend)
            for groups in worker.model_runner.attn_groups
            for group in groups
        )
    ):
        logger.info("Warming up FlashInfer attention.")
        # Warmup with mixed batch containing both prefill and decode tokens
        # This is to warm up both prefill and decode attention kernels
        worker.model_runner._dummy_run(
            num_tokens=16,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=True,
        )

    if worker.vllm_config.kernel_config.enable_cutedsl_warmup:
        # TODO(roberto): Remove after registered CuTeDSL warmups are migrated
        # to the shared JIT warmup infrastructure.
        # https://github.com/vllm-project/vllm/pull/47451
        cutedsl_warmup()

    if worker.vllm_config.kernel_config.enable_jit_warmup:
        fa4_cutedsl_warmup(worker)
        sparse_mla_triton_warmup(worker)


def _is_deepseek_v4_worker(worker: "Worker") -> bool:
    architectures = getattr(worker.model_config.hf_config, "architectures", None) or ()
    return any("DeepseekV4" in arch or "DeepSeekV4" in arch for arch in architectures)


def _deepseek_v4_sparse_mla_prefill_warmup(worker: "Worker") -> None:
    """Warm the DSv4 Triton sparse-prefill fallback without PP transport.

    Do not drive this through GPUModelRunner._dummy_run: on PP rank 0 that
    executes a real model prefill before startup and can leave a CUDA kernel
    outstanding while graph capture starts. Warm the kernels directly with a
    bounded synthetic row instead.
    """
    if not _is_deepseek_v4_worker(worker):
        return

    max_tokens = worker.scheduler_config.max_num_batched_tokens
    if worker.model_runner is None or max_tokens <= 0:
        return

    from vllm.transformers_utils.configs.dsv4.kernel_config import (
        ROLE_SPARSE_MLA_PREFILL,
        SPARSE_MLA_PREFILL_TRITON,
        resolve_kernel_config_from_hf_config,
    )

    resolved = resolve_kernel_config_from_hf_config(worker.model_config.hf_config)
    if resolved.symbol(ROLE_SPARSE_MLA_PREFILL) != SPARSE_MLA_PREFILL_TRITON:
        logger.info(
            "Skipping DeepSeek V4 Triton sparse-MLA prefill warmup; "
            "selected prefill kernel is %s.",
            resolved.symbol(ROLE_SPARSE_MLA_PREFILL),
        )
        return

    prefill_tokens = max(
        1, min(max_tokens, _DEEPSEEK_V4_SPARSE_MLA_PREFILL_WARMUP_TOKENS)
    )
    logger.info(
        "Warming DeepSeek V4 Triton sparse-MLA prefill kernel with %d tokens.",
        prefill_tokens,
    )

    _deepseek_v4_sparse_mla_prefill_kernel_warmup(worker)


def _deepseek_v4_block_table_slot_mapping_warmup(worker: "Worker") -> None:
    if not _is_deepseek_v4_worker(worker):
        return

    model_runner = worker.model_runner
    if model_runner is None:
        return

    from vllm.v1.worker.gpu.warmup import (
        warmup_block_table_slot_mapping_kernel,
        warmup_post_update_kernel,
        warmup_post_update_num_computed_tokens_kernel,
    )

    warmup_block_table_slot_mapping_kernel(model_runner, model_runner.device)
    # Runs regardless of speculative_config: postprocess_num_computed_tokens
    # and post_update are generic per-rank v1 scheduling bookkeeping (every
    # PP rank, every step), not DSpark-specific -- but under PP>1 their
    # first-ever compile landing mid-serving-step (interleaved with another
    # rank's in-flight tensor-dict recv) is exactly the class of startup
    # wedge this function already exists to avoid for the block-table kernel
    # above.
    warmup_post_update_num_computed_tokens_kernel(model_runner, model_runner.device)
    warmup_post_update_kernel(model_runner, model_runner.device)


def _deepseek_v4_marlin_moe_warmup(worker: "Worker") -> None:
    """Warm DSv4 int4/int8 Marlin MoE without full-model PP traffic."""
    if not _is_deepseek_v4_worker(worker):
        return

    from vllm.model_executor.layers.quantization.dsv4_int import Dsv4Int4MoEMethod

    model = worker.get_model()
    max_tokens = max(1, int(worker.scheduler_config.max_num_batched_tokens))
    token_sizes = sorted({1, min(16, max_tokens), max_tokens})
    warmed: set[tuple[int, int, int, torch.dtype]] = set()

    for module in model.modules():
        quant_method = getattr(module, "quant_method", None)
        if not isinstance(quant_method, Dsv4Int4MoEMethod):
            continue
        if not all(
            hasattr(module, attr)
            for attr in (
                "w13_weight",
                "w2_weight",
                "w13_weight_scale",
                "w2_weight_scale",
            )
        ):
            continue

        hidden_size = int(getattr(module, "hidden_size", quant_method.hidden_size))
        intermediate_size = int(
            getattr(
                module,
                "intermediate_size_per_partition",
                quant_method.intermediate_size,
            )
        )
        top_k = max(1, int(getattr(module, "top_k", 1)))
        input_dtype = quant_method.input_dtype or torch.bfloat16
        key = (hidden_size, intermediate_size, top_k, input_dtype)
        if key in warmed:
            continue
        warmed.add(key)

        device = module.w13_weight.device
        dtype = getattr(module, "params_dtype", torch.bfloat16)
        expert_ids = _deepseek_v4_marlin_warmup_expert_ids(module, top_k, device)
        topk_weights = torch.full(
            (1, top_k), 1.0 / top_k, dtype=torch.float32, device=device
        )

        for num_tokens in token_sizes:
            x = torch.zeros((num_tokens, hidden_size), dtype=dtype, device=device)
            topk_ids = expert_ids.expand(num_tokens, top_k).contiguous()
            weights = topk_weights.expand(num_tokens, top_k).contiguous()
            quant_method.apply(module, x, weights, topk_ids)
            torch.cuda.synchronize(device)

    if warmed:
        logger.info(
            "DeepSeek V4 Marlin MoE warmup completed for %d unique shapes and "
            "token sizes %s.",
            len(warmed),
            token_sizes,
        )


def _deepseek_v4_marlin_warmup_expert_ids(
    module: torch.nn.Module, top_k: int, device: torch.device
) -> torch.Tensor:
    expert_map = getattr(module, "expert_map", None)
    if callable(expert_map):
        expert_map = expert_map()

    if isinstance(expert_map, torch.Tensor):
        local_globals = torch.nonzero(expert_map >= 0, as_tuple=False).flatten()
        if local_globals.numel() > 0:
            repeats = (top_k + local_globals.numel() - 1) // local_globals.numel()
            return local_globals.repeat(repeats)[:top_k].to(
                device=device, dtype=torch.int32
            ).view(1, top_k)

    local_num_experts = max(1, int(getattr(module, "local_num_experts", 1)))
    ids = torch.arange(top_k, dtype=torch.int32, device=device) % local_num_experts
    return ids.view(1, top_k)


def _deepseek_v4_sparse_mla_prefill_kernel_warmup(worker: "Worker") -> None:
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        combine_topk_swa_indices,
        dequantize_and_gather_k_cache,
    )
    from vllm.models.deepseek_v4.nvidia_imma.triton_kernels import (
        sparse_attention_triton,
    )

    device = worker.model_runner.device
    hf_config = worker.model_config.hf_config
    tp_size = max(1, int(worker.parallel_config.tensor_parallel_size))
    num_heads = int(getattr(hf_config, "num_attention_heads", 128)) // tp_size
    num_heads = max(1, num_heads)
    block_size = 64
    topk = int(getattr(hf_config, "index_topk", _DEEPSEEK_V4_SYNTHETIC_TOPK))
    topk = max(1, topk)
    window = int(getattr(hf_config, "sliding_window", _DEEPSEEK_V4_SYNTHETIC_WINDOW))
    window = max(1, window)
    width = topk + window

    # Native Ampere gather/dequant path over the fp8_ds_mla uint8 cache layout.
    # Each page stores all token payloads first, then 8 scale bytes per token.
    gathered = torch.empty(
        (1, 1, _DEEPSEEK_V4_MLA_HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    k_cache = torch.zeros(
        (1, block_size, _DEEPSEEK_V4_FP8_DS_MLA_PAGE_TOKEN_BYTES),
        dtype=torch.uint8,
        device=device,
    )
    seq_lens = torch.tensor([1], dtype=torch.int32, device=device)
    gather_lens = torch.tensor([1], dtype=torch.int32, device=device)
    block_table = torch.zeros((1, 1), dtype=torch.int32, device=device)
    dequantize_and_gather_k_cache(
        gathered,
        k_cache,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        block_table=block_table,
        block_size=block_size,
        offset=0,
    )
    torch.cuda.synchronize(device)

    # Combined C4A/C128A top-k + SWA index path. Use a single late-position row
    # so both the top-k and SWA portions are active while the launch stays small.
    topk_indices = torch.arange(topk, dtype=torch.int32, device=device).view(1, topk)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)
    context_len = max(topk * 4, window)
    seq_lens = torch.tensor([context_len], dtype=torch.int32, device=device)
    gather_lens = torch.tensor([window], dtype=torch.int32, device=device)
    indices, lengths = combine_topk_swa_indices(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size=window,
        compress_ratio=4,
        topk=topk,
        M=width,
        N=topk,
    )
    torch.cuda.synchronize(device)

    # Sparse attention proper: one query row, real local head count, full
    # top-k+SWA width. num_tokens is pinned off specialization in the Triton JIT,
    # while num_heads and index width are the expensive specializations to warm.
    q = torch.zeros(
        (1, num_heads, _DEEPSEEK_V4_MLA_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    kv = torch.zeros(
        (width, 1, _DEEPSEEK_V4_MLA_HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    out = torch.empty_like(q)
    sink = torch.zeros((num_heads,), dtype=torch.float32, device=device)
    sparse_attention_triton(
        q=q,
        kv=kv,
        indices=indices.unsqueeze(1),
        lengths=lengths,
        scale=1.0,
        attn_sink=sink,
        out=out,
    )
    torch.cuda.synchronize(device)


def _flashinfer_autotune_skip_ops(runner: "GPUModelRunner") -> set[str] | None:
    if envs.VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS is not None:
        return set(envs.VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS) or None

    from vllm.model_executor.kernels.linear import (
        FlashInferCuteDslNvFp4LinearKernel,
    )

    for module in runner.get_model().modules():
        for holder_name in ("quant_method", "scheme"):
            kernel = getattr(getattr(module, holder_name, None), "kernel", None)
            # CuTe-DSL mm_fp4 tuning JIT-compiles every tactic and its
            # fallback is already the heuristic; all mm_fp4 backends share
            # the "fp4_gemm" op name, so skip only when cute-dsl is selected.
            if isinstance(kernel, FlashInferCuteDslNvFp4LinearKernel):
                return {"fp4_gemm"}
    return None


def flashinfer_autotune(runner: "GPUModelRunner") -> None:
    """
    Autotune FlashInfer operations.
    FlashInfer have many implementations for the same operation,
    autotuning runs benchmarks for each implementation and stores
    the results. The results are cached transparently and
    future calls to FlashInfer will use the best implementation.
    Without autotuning, FlashInfer will rely on heuristics, which may
    be significantly slower.

    Tuning is performed only on rank 0. The resulting cache is broadcast
    to every rank so all ranks dispatch the same kernel tactic.
    """
    import vllm.utils.flashinfer as fi_utils
    from vllm.distributed.parallel_state import get_world_group

    autotune_kwargs: dict = {}
    skip_ops = _flashinfer_autotune_skip_ops(runner)
    if skip_ops:
        logger.info(
            "Skipping FlashInfer autotuning for ops %s",
            sorted(skip_ops),
        )
        autotune_kwargs["skip_ops"] = skip_ops

    use_persistent_cache = True

    # When distributed, tune on every rank so the collectives stay synchronized.
    if get_world_group().world_size > 1:
        use_persistent_cache = False

    if not use_persistent_cache:
        with torch.inference_mode(), fi_utils.autotune(**autotune_kwargs):
            runner._dummy_run(
                num_tokens=runner.scheduler_config.max_num_batched_tokens,
                skip_eplb=True,
                is_profile=True,
            )
        get_world_group().barrier()
        return

    world = get_world_group()
    is_leader = world.rank_in_group == 0

    cache_path = resolve_flashinfer_autotune_file(runner)
    if is_leader:
        logger.info("Using FlashInfer autotune cache file: %s", cache_path)

    # We skip EPLB here since we don't want to record dummy metrics.
    # When autotuning with number of tokens m, flashinfer will autotune
    # operations for all number of tokens up to m, so we only need to
    # run with the max number of tokens.
    dummy_run_kwargs = dict(
        num_tokens=runner.scheduler_config.max_num_batched_tokens,
        skip_eplb=True,
        is_profile=True,
    )

    with torch.inference_mode():
        if is_leader:
            with fi_utils.autotune(
                tune_mode=True, cache=str(cache_path), **autotune_kwargs
            ):
                runner._dummy_run(**dummy_run_kwargs)
        else:
            runner._dummy_run(**dummy_run_kwargs)

    # Broadcast autotune cache from rank 0 to all other ranks so every
    # rank loads the same set of chosen tactics.
    tune_results: bytes | None = None
    if is_leader and cache_path.exists():
        with open(cache_path, "rb") as f:
            tune_results = f.read()

    tune_results = world.broadcast_object(tune_results, src=0)

    if tune_results is None:
        logger.warning(
            "No FlashInfer autotune cache entries found."
            "Falling back to default tactics."
        )
    else:
        write_flashinfer_autotune_cache(cache_path, tune_results)
        world.barrier()
        from flashinfer.autotuner import AutoTuner

        AutoTuner.get().load_configs(str(cache_path))
        logger.info(
            "FlashInfer autotune cache loaded on rank %d from %s.",
            world.rank_in_group,
            cache_path,
        )
