# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SparkInfer mHC adapters for DeepSeek V4 on SM120/SM121.

The public SparkInfer API owns the execution plan. Under Dynamo it emits one
functional custom op, so PyTorch's CUDA graph pool owns the output and scratch
allocations and pins their addresses. Eager execution intentionally uses the
API's temporary outputs; raw eager graph capture fails closed below. A
per-layer binding would pin four large output tensors for every layer and
capture size, defeating the graph pool's global liveness reuse.
"""

from __future__ import annotations

import torch

from vllm.platforms import current_platform

_SUPPORTED_HIDDEN_SIZES = (4096, 7168)
_HC_MULT = 4


def _sparkinfer_mhc_split_k(hidden_size: int, block_k: int) -> int:
    """Return the exact K partition count required by SparkInfer mHC."""
    total_k = _HC_MULT * int(hidden_size)
    block_k = int(block_k)
    if block_k <= 0 or total_k % block_k != 0:
        raise ValueError(
            "SparkInfer mHC requires hc_mult * hidden_size to be divisible by "
            f"block_k, got hc_mult={_HC_MULT}, hidden_size={hidden_size}, "
            f"block_k={block_k}"
        )
    return total_k // block_k


def _fail_on_uncompiled_cuda_graph(tensor: torch.Tensor) -> None:
    if (
        tensor.is_cuda
        and not torch.compiler.is_compiling()
        and torch.cuda.is_current_stream_capturing()
    ):
        raise RuntimeError(
            "SparkInfer mHC requires the compiled functional custom-op path "
            "during CUDA graph capture; use normal vLLM compilation or "
            "--enforce-eager"
        )


def sparkinfer_mhc_unavailable_reason() -> str | None:
    if not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(120)
    ):
        return "requires an SM120/SM121 CUDA device"
    try:
        from sparkinfer.norm import mhc
    except ImportError as exc:
        return f"sparkinfer.norm.mhc is not importable ({exc})"
    if not mhc.is_supported():
        return "sparkinfer.norm.mhc reports unsupported on this device"
    return None


def validate_sparkinfer_mhc_contract(
    *,
    hidden_size: int,
    hc_mult: int,
    rms_eps: float,
    hc_eps: float,
    hc_post_alpha: float,
    sinkhorn_iters: int,
) -> None:
    reason = sparkinfer_mhc_unavailable_reason()
    if reason is not None:
        raise RuntimeError(f"SparkInfer mHC was selected but is unavailable: {reason}")
    if hidden_size not in _SUPPORTED_HIDDEN_SIZES:
        raise ValueError(
            "SparkInfer mHC supports hidden_size in "
            f"{_SUPPORTED_HIDDEN_SIZES}, got {hidden_size}"
        )
    expected = {
        "hc_mult": (hc_mult, 4),
        "rms_eps": (rms_eps, 1.0e-6),
        "hc_eps": (hc_eps, 1.0e-6),
        "hc_post_alpha": (hc_post_alpha, 2.0),
        "sinkhorn_iters": (sinkhorn_iters, 20),
    }
    mismatches = [
        f"{name}={actual!r} (requires {required!r})"
        for name, (actual, required) in expected.items()
        if actual != required
    ]
    if mismatches:
        raise ValueError(
            "SparkInfer mHC does not support this checkpoint contract: "
            + ", ".join(mismatches)
        )


def sparkinfer_mhc_pre_broadcast(
    x: torch.Tensor,
    fn_broadcast: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Broadcast a 2D first-layer input and fuse its following RMSNorm."""
    from sparkinfer.norm import mhc

    _fail_on_uncompiled_cuda_graph(x)
    split_k = _sparkinfer_mhc_split_k(x.shape[-1], mhc.DEFAULT_BLOCK_K)
    residual, post, comb, y = mhc.run_pre(
        x,
        fn_broadcast,
        hc_scale,
        hc_base,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        split_k=split_k,
    )
    return residual, post.unsqueeze(-1), comb, y


def sparkinfer_mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    from sparkinfer.norm import mhc

    _fail_on_uncompiled_cuda_graph(x)
    return mhc.run_post(x, residual, post_layer_mix, comb_res_mix)


def sparkinfer_mhc_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run fused post/pre, optionally including the following RMSNorm."""
    from sparkinfer.norm import mhc

    _fail_on_uncompiled_cuda_graph(x)
    split_k = _sparkinfer_mhc_split_k(residual.shape[-1], mhc.DEFAULT_BLOCK_K)
    residual_cur, post, comb, y = mhc.run_post_pre(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        expected_m=residual.shape[0],
        split_k=split_k,
    )
    return residual_cur, post.unsqueeze(-1), comb, y
