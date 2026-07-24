# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sparkinfer-backed sparse-MLA launchers for GB10 (sm_12x).

These two functions are the salient callables the ``"vllm"`` kernel-config
block selects by FQN (see ``kernel_config.KERNEL_REGISTRY``):

- ``sparkinfer_sparse_mla_decode``: decode rows through sparkinfer's
  ``attention.compressed_mla`` fp8-compute CuTe kernel (mode ``decode``).
- ``sparkinfer_sparse_mla_extend``: prefill rows through the same op in
  scratch mode ``extend`` (the MG prefill kernel).

Contract (parity-gated by ``test_sm12x_sparkinfer_compressed_mla_parity``):
identical fp8_ds_mla byte layout on both sides; caches passed as the raw
paged vLLM tensors ``(num_blocks, block_size, 584)`` — flattened here to
sparkinfer's ``[pages, page_bytes]`` view, valid with ZERO padding exactly
when ``block_size % 72 == 0`` (the backend advertises 288, C4 pool 72);
indices are -1-padded global slot ids; ``attn_sink`` is per padded head.

CUDA-graph safety: ``plan``/scratch allocation is cached per capacity key and
happens on the first (eager warmup) call; ``bind`` is views-only. The kernels
JIT-compile on their first launch, which must therefore happen before graph
capture — vLLM's dummy/profile runs satisfy this.
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# (device_index, heads, max_q_rows, swa_width, extra_width, mode) -> plan.
# Capacities are stable per layer type, so this holds a handful of entries
# per process.
_PLAN_CACHE: dict[tuple, object] = {}
# One shared scratch buffer per device, grown to the largest plan spec seen.
# Growth only happens at plan creation, which vLLM's eager warmup runs drive
# before any CUDA-graph capture, so captured launches see a stable address.
_SCRATCH_BY_DEVICE: dict[int, torch.Tensor] = {}


def _as_page_bytes(cache: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Flatten a paged vLLM fp8_ds_mla cache to sparkinfer's [pages,
    page_bytes] uint8 view and return it with the page size (tokens/page).

    The AppMana sparkinfer fork accepts vLLM's unpadded page width
    (block_size * 584 bytes) directly — the kernel touches payload entries
    and scale footers only, never SGLang's 576-multiple round-up pad — so
    this is zero-copy at any block size.
    """
    if cache.dim() == 4 and cache.shape[-2] == 1:
        cache = cache.squeeze(-2)
    assert cache.dim() == 3, f"expected (blocks, block_size, 584), got {cache.shape}"
    page_size = int(cache.shape[1])
    if cache.dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise TypeError(f"expected byte/fp8 cache storage, got {cache.dtype}")
    view = cache if cache.dtype == torch.uint8 else cache.view(torch.uint8)
    return view.reshape(view.shape[0], -1), page_size


def _get_binding(
    *,
    mode: str,
    q: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    extra_indices: torch.Tensor | None,
    extra_lens: torch.Tensor | None,
    max_q_rows: int,
):
    from sparkinfer.attention import compressed_mla

    heads = int(q.shape[1])
    swa_width = int(swa_indices.shape[-1])
    extra_width = int(extra_indices.shape[-1]) if extra_indices is not None else 0
    max_q_rows = max(int(max_q_rows), int(q.shape[0]))
    key = (q.device.index, heads, max_q_rows, swa_width, extra_width, mode)
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        width = swa_width + extra_width
        plan = compressed_mla.plan(
            compressed_mla.Caps(
                device=q.device,
                dtype=torch.bfloat16,
                kv_dtype=torch.uint8,
                num_q_heads=heads,
                max_width=width,
                max_q_rows=max_q_rows,
                max_batch=max_q_rows,
                max_kv_rows=max_q_rows * width,
                # Scratch scales with max_q_rows * max_chunks_per_row; the
                # default (64) sizes for a worst-case decode split, but a row
                # only ever covers ceil(width/64) index chunks. Bound it (+1
                # slack) or the layout balloons to gigabytes per plan.
                max_chunks_per_row=(width + 63) // 64 + 1,
            )
        )
        (spec,) = plan.scratch_specs()
        scratch = _SCRATCH_BY_DEVICE.get(q.device.index)
        needed = int(spec.shape[0])
        if scratch is not None and scratch.dtype != spec.dtype:
            raise TypeError(
                f"sparkinfer scratch dtype changed across plans: "
                f"{scratch.dtype} vs {spec.dtype}"
            )
        if scratch is None or scratch.numel() < needed:
            scratch = torch.empty(needed, dtype=spec.dtype, device=q.device)
            _SCRATCH_BY_DEVICE[q.device.index] = scratch
        logger.info_once(
            "sparkinfer compressed_mla plan: mode=%s heads=%d max_q_rows=%d "
            "swa_width=%d extra_width=%d scratch_elems=%d (shared)",
            mode,
            heads,
            max_q_rows,
            swa_width,
            extra_width,
            needed,
        )
        _PLAN_CACHE[key] = plan
    scratch = _SCRATCH_BY_DEVICE[q.device.index]
    binding = plan.bind(
        scratch=scratch,
        q=q,
        swa_indices=swa_indices,
        swa_lengths=swa_lens,
        indexed_indices=extra_indices,
        indexed_lengths=extra_lens,
    )
    binding.scratch.mode = mode
    # Fixed pre-planned capacity; live per-step lengths mutate in place — the
    # replay contract test_compressed_mla_shared_core_replays_under_cuda_graph
    # covers this shape of use.
    binding.scratch.use_cuda_graph = True
    return binding


def _run(
    *,
    mode: str,
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    extra_cache: torch.Tensor | None,
    extra_indices: torch.Tensor | None,
    extra_lens: torch.Tensor | None,
    sm_scale: float,
    attn_sink: torch.Tensor | None,
    out: torch.Tensor,
    max_q_rows: int,
) -> None:
    from sparkinfer.attention import compressed_mla

    swa_pages, swa_page_size = _as_page_bytes(swa_cache)
    extra_pages = None
    extra_page_size = None
    if extra_cache is not None:
        extra_pages, extra_page_size = _as_page_bytes(extra_cache)
    binding = _get_binding(
        mode=mode,
        q=q,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        extra_indices=extra_indices,
        extra_lens=extra_lens,
        max_q_rows=max_q_rows,
    )
    compressed_mla.run(
        swa_k_cache=swa_pages,
        binding=binding,
        swa_page_size=swa_page_size,
        indexed_k_cache=extra_pages,
        indexed_page_size=extra_page_size,
        attn_sink=attn_sink,
        sm_scale=sm_scale,
        out=out,
    )


def sparkinfer_sparse_mla_decode(
    *,
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    extra_cache: torch.Tensor | None,
    extra_indices: torch.Tensor | None,
    extra_lens: torch.Tensor | None,
    sm_scale: float,
    attn_sink: torch.Tensor | None,
    out: torch.Tensor,
    max_q_rows: int,
) -> None:
    """Sparse-MLA decode over selected fp8_ds_mla slots (sparkinfer fp8-Q)."""
    _run(
        mode="decode",
        q=q,
        swa_cache=swa_cache,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        extra_lens=extra_lens,
        sm_scale=sm_scale,
        attn_sink=attn_sink,
        out=out,
        max_q_rows=max_q_rows,
    )


def sparkinfer_sparse_mla_extend(
    *,
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    extra_cache: torch.Tensor | None,
    extra_indices: torch.Tensor | None,
    extra_lens: torch.Tensor | None,
    sm_scale: float,
    attn_sink: torch.Tensor | None,
    out: torch.Tensor,
    max_q_rows: int,
) -> None:
    """Sparse-MLA prefill rows via sparkinfer's extend/MG kernel path."""
    _run(
        mode="extend",
        q=q,
        swa_cache=swa_cache,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        extra_lens=extra_lens,
        sm_scale=sm_scale,
        attn_sink=attn_sink,
        out=out,
        max_q_rows=max_q_rows,
    )
