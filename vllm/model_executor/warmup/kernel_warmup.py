# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warmup kernels used during model execution.
This is useful specifically for JIT'ed kernels as we don't want JIT'ing to
happen during model execution.
"""

import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.warmup.attention_warmup import (
    mixed_batch_attention_warmup,
)
from vllm.model_executor.warmup.b12x_warmup import b12x_warmup
from vllm.model_executor.warmup.cutedsl_warmup import cutedsl_warmup
from vllm.model_executor.warmup.deep_gemm_warmup import deep_gemm_warmup
from vllm.model_executor.warmup.deepseek_v4_mhc_warmup import (
    deepseek_v4_mhc_warmup,
)
from vllm.model_executor.warmup.flashinfer_autotune_cache import (
    resolve_flashinfer_autotune_file,
)
from vllm.model_executor.warmup.flashinfer_sparse_mla_warmup import (
    autotune_hisparse_flashinfer_attention,
    flashinfer_sparse_mla_decode_autotune_warmup,
)
from vllm.model_executor.warmup.kimi_k3_triton_warmup import (
    kimi_k3_triton_warmup,
)
from vllm.model_executor.warmup.mamba_triton_warmup import mamba_triton_warmup
from vllm.model_executor.warmup.qwen4_exp_qsa_warmup import (
    qwen4_exp_qsa_triton_warmup,
)
from vllm.model_executor.warmup.qwen_triton_warmup import qwen_triton_warmup
from vllm.model_executor.warmup.qwen_vl_triton_warmup import qwen_vl_triton_warmup
from vllm.model_executor.warmup.replayssm_warmup import (
    replayssm_autotune_warmup,
)
from vllm.model_executor.warmup.spec_decode_rejection_warmup import (
    spec_decode_rejection_warmup,
)
from vllm.model_executor.warmup.watermark_sample_warmup import (
    watermark_sample_warmup,
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
    _DEEPSEEK_V4_FP8_DS_MLA_TOKEN_DATA_BYTES + _DEEPSEEK_V4_FP8_DS_MLA_SCALE_BYTES
)
_DEEPSEEK_V4_SYNTHETIC_TOPK = 512
_DEEPSEEK_V4_SYNTHETIC_WINDOW = 128

_LL_BF16_WARMUP_M_RANGE = range(1, 17)


def _ll_bf16_router_shapes_from_model(
    model: torch.nn.Module,
) -> tuple[tuple[int, int], ...]:
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    shapes: set[tuple[int, int]] = set()
    for module in model.modules():
        if not isinstance(module, GateLinear):
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            continue
        if weight.dim() != 2 or weight.dtype != torch.bfloat16:
            continue
        n, k = weight.shape
        if k % 8 == 0:
            shapes.add((int(k), int(n)))
    return tuple(sorted(shapes))


def _warmup_ll_bf16_router_gemm(model: torch.nn.Module) -> None:
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
        is_available as is_ll_bf16_gemm_available,
    )
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
        ll_bf16_gemm_kernel,
    )

    if not is_ll_bf16_gemm_available():
        return

    shapes = _ll_bf16_router_shapes_from_model(model)
    if not shapes:
        logger.debug_once(
            "Skipping ll_bf16 router GEMM warmup: no bf16 GateLinear shapes found."
        )
        return

    logger.info_once("Warming up ll_bf16 router GEMM kernels for shapes: %s.", shapes)
    ll_bf16_gemm_kernel.warmup(
        shapes=shapes,
        m_values=_LL_BF16_WARMUP_M_RANGE,
    )


def _warmup_bf16x3_router_gemm(
    model: torch.nn.Module,
    max_num_tokens: int,
) -> None:
    from vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_cutedsl import (  # noqa: E501
        warmup_bf16x3_router_gemm,
    )
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    gate = next(
        (
            module
            for module in model.modules()
            if isinstance(module, GateLinear) and module.allow_bf16x3_router_gemm
        ),
        None,
    )
    if gate is None:
        logger.debug_once(
            "Skipping BF16x3 router GEMM warmup: no eligible GateLinear found."
        )
        return

    min_num_tokens = gate.FP32_MAX_TOKENS + 1 if gate.allow_fp32_router_gemm else 1
    logger.info_once(
        "Warming up BF16x3 router GEMM for K=%d, M=%d.",
        gate.input_size,
        gate.output_size,
    )
    configs = warmup_bf16x3_router_gemm(
        gate.input_size,
        gate.output_size,
        min_num_tokens,
        max_num_tokens,
    )
    logger.info_once("Warmed up BF16x3 router GEMM configs: %s.", configs)


def _warmup_gemm_rs_ar() -> None:
    # Model construction (Kimi-K3, DeepSeek-V4.1) imports this module only
    # when GEMM-RS/AR is enabled and initializes its singleton before
    # kernel_warmup runs. Avoid importing it here so other models do not
    # compile the RS/AR variants.
    module = sys.modules.get("vllm.model_executor.kernels.linear.cute_dsl.gemm_rs_ar")
    if module is None:
        return
    compiled = module.warmup_gemm_rs_ar()
    if compiled:
        logger.info_once("Warmed up %d GEMM-RS/AR variants.", compiled)


def _autotune_kimi_k3_kda_qkvg(model: torch.nn.Module) -> None:
    module = sys.modules.get("vllm.models.kimi_k3.nvidia.low_latency_gemm")
    if module is not None:
        module.autotune_kda_qkvg(model)


def kernel_warmup(worker: "Worker", *, process_local_only: bool = False):
    from vllm.model_executor.warmup.minimax_m3_msa_warmup import (
        minimax_m3_msa_warmup,
    )

    if not worker.use_v2_model_runner:
        # The KV-block zeroing kernel is driven by the scheduler's
        # `new_block_ids_to_zero`, so no dummy run ever reaches it.
        zeroer = getattr(worker.model_runner, "_kv_block_zeroer", None)
        if zeroer is not None:
            zeroer.warmup(worker.model_runner.kv_cache_config.num_blocks)

    enable_jit_warmup = worker.vllm_config.kernel_config.enable_jit_warmup
    if enable_jit_warmup:
        logger.info("JIT kernel warmup starting.")
        jit_warmup_start = time.perf_counter()
        try:
            registry = (
                worker.model_runner.jit_warmup_registry  # type: ignore[attr-defined]
            )
            registry.warmup()
        except Exception:
            logger.exception(
                "JIT kernel warmup failed after %.2fs.",
                time.perf_counter() - jit_warmup_start,
            )
            raise
        logger.info(
            "JIT kernel warmup finished in %.2fs.",
            time.perf_counter() - jit_warmup_start,
        )

    qwen_triton_warmup(worker.model_runner, worker.vllm_config.model_config)
    qwen_vl_triton_warmup(worker.model_runner)
    mamba_triton_warmup(worker.model_runner)

    compilation_config = worker.vllm_config.compilation_config
    cudagraph_capture_sizes = list(compilation_config.cudagraph_capture_sizes or [])

    # DSv4 mHC TileLang kernels (hc_pre/hc_post/hc_head_op) run every decoder
    # layer per token; warm them across token sizes first so the first real
    # request doesn't pay JIT cost. No-op for non-DSv4 models (gated inside).
    max_batched_tokens = worker.scheduler_config.max_num_batched_tokens
    additional_mhc_token_sizes: list[int] = []
    speculative_config = getattr(worker.vllm_config, "speculative_config", None)
    if speculative_config is not None:
        draft_slots = speculative_config.max_num_new_slots_for_drafting
        adaptive_prefill_tokens = max_batched_tokens - draft_slots
        if 0 < adaptive_prefill_tokens < max_batched_tokens:
            additional_mhc_token_sizes.append(adaptive_prefill_tokens)

    deepseek_v4_mhc_warmup(
        worker.get_model(),
        max_tokens=max_batched_tokens,
        cudagraph_capture_sizes=cudagraph_capture_sizes,
        additional_token_sizes=additional_mhc_token_sizes,
    )

    # Run next so input-prep kernels JIT against pristine runner state.
    if enable_jit_warmup:
        kimi_k3_triton_warmup(worker)
        spec_decode_rejection_warmup(worker)
        watermark_sample_warmup(worker)
        qwen4_exp_qsa_triton_warmup(worker)

    if enable_jit_warmup and current_platform.is_device_capability_family(100):
        _warmup_bf16x3_router_gemm(
            worker.get_model(),
            worker.scheduler_config.max_num_batched_tokens,
        )

    if current_platform.has_device_capability(90):
        _warmup_ll_bf16_router_gemm(worker.get_model())

    _warmup_gemm_rs_ar()

    if worker.vllm_config.kernel_config.enable_cutedsl_warmup:
        # TODO(roberto): Remove after registered CuTeDSL warmups are migrated
        # to the shared JIT warmup infrastructure.
        # https://github.com/vllm-project/vllm/pull/47451
        cutedsl_warmup()

    if process_local_only:
        return

    if current_platform.is_rocm():
        from vllm.model_executor.warmup.rocm_segmented_attn_autotune_warmup import (
            rocm_segmented_attn_autotune_warmup,
        )

        rocm_segmented_attn_autotune_warmup(worker)

    flashinfer_sparse_mla_decode_autotune_warmup(worker)
    _deepseek_v4_sparse_mla_prefill_warmup(worker)
    _deepseek_v4_block_table_slot_mapping_warmup(worker)
    _deepseek_v4_marlin_moe_warmup(worker)
    _deepseek_v4_vision_warmup(worker)

    # Deep GEMM warmup
    do_deep_gemm_warmup = (
        is_deep_gemm_supported() and envs.VLLM_DEEP_GEMM_WARMUP != "skip"
    )
    if do_deep_gemm_warmup:
        model = worker.get_model()
        max_tokens = worker.scheduler_config.max_num_batched_tokens
        deep_gemm_warmup(model, max_tokens)

    b12x_warmup(worker, cudagraph_capture_sizes)

    minimax_m3_msa_warmup(worker)

    # Allocate the exact decode-sized workspace, autotune cache misses, and
    # resolve every CUDA Graph bucket before capture begins.
    # Lazy import: flashinfer_pcie_ipc_all_reduce imports flashinfer.comm,
    # which initializes CUDA at import time and must not run at module scope.
    if envs.VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC:
        from vllm.distributed.device_communicators import (
            flashinfer_pcie_ipc_all_reduce,
        )

        flashinfer_pcie_ipc_all_reduce.warmup_flashinfer_pcie_ipc_allreduce(worker)

    enable_flashinfer_autotune = (
        worker.vllm_config.kernel_config.enable_flashinfer_autotune
    )
    # FlashInfer autotune for Hopper (SM 9.0) and Blackwell (SM 10.0) GPUs
    if enable_flashinfer_autotune is False:
        logger.info_once("Skipping FlashInfer autotune because it is disabled.")
    elif has_flashinfer() and current_platform.has_device_capability(90):
        flashinfer_autotune(worker.model_runner)

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
        logger.info_once("Warming up FlashInfer attention.")
        mixed_batch_attention_warmup(worker)


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
            torch.accelerator.synchronize(device)

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
            return (
                local_globals.repeat(repeats)[:top_k]
                .to(device=device, dtype=torch.int32)
                .view(1, top_k)
            )

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
    torch.accelerator.synchronize(device)

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
    torch.accelerator.synchronize(device)

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
    torch.accelerator.synchronize(device)


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


def _flashinfer_deferred_moe_token_counts(
    runner: "GPUModelRunner",
) -> tuple[int, ...]:
    """Return bounded token counts that exercise deferred MoE finalization."""
    from vllm.model_executor.layers.fused_moe import MoERunner

    max_tokens = runner.scheduler_config.max_num_batched_tokens
    token_counts: list[int] = []
    for module in runner.get_model().modules():
        if not isinstance(module, MoERunner):
            continue

        moe_config = module.moe_config
        max_deferred_tokens = moe_config.defer_moe_finalize_max_num_tokens
        if moe_config.use_deferred_moe_finalize and max_deferred_tokens > 0:
            token_counts.append(min(max_tokens, max_deferred_tokens))

    return tuple(dict.fromkeys(token_counts))


def _flashinfer_autotune_token_counts(runner: "GPUModelRunner") -> tuple[int, ...]:
    max_tokens = runner.scheduler_config.max_num_batched_tokens
    # Tune the widest bucket set first so bounded passes reuse its configs.
    token_counts = [max_tokens]
    token_counts.extend(_flashinfer_deferred_moe_token_counts(runner))
    return tuple(dict.fromkeys(token_counts))


def _run_flashinfer_autotune_dummy_runs(
    runner: "GPUModelRunner", *, skip_attn: bool = False
) -> None:
    import vllm.utils.flashinfer as fi_utils

    dummy_run_kwargs = {"skip_attn": True} if skip_attn else {}
    for num_tokens in _flashinfer_autotune_token_counts(runner):
        # The drafter may use more tokens than this pass in the same dummy run.
        # Include its max M when building the autotune buckets.
        max_tuning_tokens = num_tokens
        speculator = getattr(runner, "speculator", None)
        if speculator is not None:
            num_query_per_req = speculator.num_query_per_req
            num_draft_reqs = min(
                num_tokens,
                runner.max_num_reqs,
                runner.max_num_tokens // num_query_per_req,
            )
            max_tuning_tokens = max(
                max_tuning_tokens, num_draft_reqs * num_query_per_req
            )

        tuning_buckets = fi_utils.flashinfer_get_hybrid_num_tokens_buckets(
            max_tuning_tokens
        )
        logger.info(
            "Running FlashInfer autotune with %d tokens and token buckets %s.",
            num_tokens,
            tuning_buckets,
        )
        # Round M up to match serving-time bucket selection for non-bucket M.
        # See https://github.com/flashinfer-ai/flashinfer/issues/5450
        with fi_utils.autotune(tuning_buckets=tuning_buckets, round_up=True):
            runner._dummy_run(
                num_tokens=num_tokens,
                skip_eplb=True,
                is_profile=True,
                randomize_inputs=True,
                **dummy_run_kwargs,
            )


def _autotune_cache_fingerprint(path: Path) -> tuple[str, int] | None:
    """Identify a saved autotune file by its FlashInfer metadata and size."""
    try:
        configs = json.loads(path.read_text())
        metadata = configs.pop("_metadata", None)
    except (OSError, ValueError, AttributeError):
        return None
    return json.dumps(metadata, sort_keys=True), len(configs)


def _all_ranks_have_matching_cache(path: Path, group) -> bool:
    """True iff every rank in ``group`` has its own, mutually consistent file."""
    fingerprint = _autotune_cache_fingerprint(path)
    if group.world_size == 1:
        return fingerprint is not None
    gathered: list[tuple[str, int] | None] = [None] * group.world_size
    torch.distributed.all_gather_object(gathered, fingerprint, group=group.cpu_group)
    return fingerprint is not None and all(f == fingerprint for f in gathered)


def flashinfer_autotune(runner: "GPUModelRunner", *, skip_attn: bool = False) -> None:
    """Autotune FlashInfer operations.
    FlashInfer have many implementations for the same operation,
    autotuning runs benchmarks for each implementation and stores
    the results. The results are cached transparently and
    future calls to FlashInfer will use the best implementation.
    Without autotuning, FlashInfer will rely on heuristics, which may
    be significantly slower.

    Args:
        runner: The model runner to tune.
        skip_attn: Additionally skip attention in the tuning dummy runs.

    With PP > 1, stages run different layers and may profile different ops,
    so each stage's TP group tunes separately with its own cache file;
    otherwise the world group tunes together. Per-tactic timings are
    averaged over the tuning group so all its ranks select the same tactic.

    Results are persisted per rank: FlashInfer keys MoE entries by tp/ep rank
    (``MoERunner.get_cache_key_extras``), so one rank's file only hits on that
    rank. A rank with a cache hit skips the per-tactic reduce the others block
    in, so ranks keep loaded configs only if every rank in the tuning group
    has a matching file and successfully loads it.

    """
    from flashinfer.autotuner import AutoTuner, set_autotune_process_group

    import vllm.utils.flashinfer as fi_utils
    from vllm.distributed.parallel_state import (
        get_pp_group,
        get_tp_group,
        get_world_group,
    )

    world = get_world_group()
    pp_size = get_pp_group().world_size
    tune_group = get_tp_group() if pp_size > 1 else world
    is_leader = tune_group.rank_in_group == 0
    tuner = AutoTuner.get()

    autotune_kwargs: dict = {}
    skip_ops = _flashinfer_autotune_skip_ops(runner)
    if skip_ops:
        logger.info_once(
            "Skipping FlashInfer autotuning for ops %s",
            tuple(sorted(skip_ops)),
        )
        autotune_kwargs["skip_ops"] = skip_ops

    cache_path = resolve_flashinfer_autotune_file(runner)
    # The world group only spans DP ranks when vLLM folds DP into it.
    dp_rank = runner.vllm_config.parallel_config.data_parallel_rank
    cache_path = cache_path.with_name(
        f"{cache_path.stem}_dp{dp_rank}_rank{world.rank_in_group}{cache_path.suffix}"
    )
    if is_leader:
        logger.info_once("Using FlashInfer autotune cache file: %s", cache_path)

    # We skip EPLB here since we don't want to record dummy metrics.
    # Randomize inputs to avoid every token pick the same experts,
    # which lead to some EP ranks receiving no tokens and skipping their
    # MoE kernel entirely, and cause hang due to all-reduce collective
    # during synchronized autotuning.
    if _all_ranks_have_matching_cache(cache_path, tune_group):
        loaded = tuner.load_configs(str(cache_path))
        if tune_group.world_size > 1:
            loaded_by_rank: list[bool | None] = [None] * tune_group.world_size
            torch.distributed.all_gather_object(
                loaded_by_rank, loaded, group=tune_group.cpu_group
            )
            loaded = all(loaded_by_rank)
        if not loaded:
            tuner.clear_cache()

    group = tune_group.cpu_group if tune_group.world_size > 1 else None
    set_autotune_process_group(group)
    try:
        with (
            torch.inference_mode(),
            fi_utils.autotune(tune_mode=True, **autotune_kwargs),
        ):
            hisparse_enabled = (
                runner.vllm_config.attention_config.hisparse_config is not None
            )
            skip_attn = skip_attn or hisparse_enabled
            if hisparse_enabled:
                # HiSparse hot-buffer attention is bounded by decode batch
                # size, not the prefill-sized batch used for the full model.
                autotune_hisparse_flashinfer_attention(runner)
            _run_flashinfer_autotune_dummy_runs(runner, skip_attn=skip_attn)
            replayssm_autotune_warmup(runner)
            _autotune_kimi_k3_kda_qkvg(runner.get_model())
    finally:
        set_autotune_process_group(None)

    if world.world_size > 1:
        world.barrier()
    # Skip the rewrite when nothing was tuned this start (every entry came from
    # the file). FlashInfer gates its own autotune(cache=...) save the same way.
    if tuner._dirty:
        tuner.save_configs(str(cache_path))


def _deepseek_v4_vision_warmup(worker: "Worker") -> None:
    """Compile the vision tower's kernels before the first image request.

    Only the pipeline rank that owns the tower has one. Token counts are
    runtime arguments of the INT8 vision kernels, so one small synthetic
    image compiles every linear shape and the attention kernel for all
    image sizes.
    """
    if not _is_deepseek_v4_worker(worker) or worker.model_runner is None:
        return
    model = worker.get_model()
    vision = getattr(model, "vision", None)
    aligner = getattr(model, "aligner", None)
    if vision is None or aligner is None:
        return
    projection = vision.patch_embed.proj
    grid = 2 * int(aligner.downsample_ratio)
    device = projection.weight.device
    dtype = getattr(model, "compute_dtype", worker.model_config.dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    patches = torch.randn(
        grid * grid,
        projection.input_size,
        device=device,
        dtype=torch.float32,
        generator=generator,
    ).to(dtype)
    with torch.inference_mode():
        aligner(vision(patches, grid, grid), grid, grid)
    torch.accelerator.synchronize(device)
    logger.info("DeepSeek V4 vision kernels warmed on a %dx%d patch grid.", grid, grid)
