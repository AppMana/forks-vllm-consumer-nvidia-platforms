# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch
import torch.nn.functional as F

# this import will also register the custom ops
import vllm.model_executor.kernels.mhc as mhc_kernels
from vllm._aiter_ops import is_aiter_found_and_supported
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_tilelang

logger = init_logger(__name__)

_MHC_CUDA_BACKENDS = {"auto", "sparkinfer", "tilelang", "triton"}


def _requested_mhc_cuda_backend() -> str:
    backend = os.getenv("VLLM_MHC_CUDA_BACKEND", "auto").strip().lower()
    if backend not in _MHC_CUDA_BACKENDS:
        choices = ", ".join(sorted(_MHC_CUDA_BACKENDS))
        raise ValueError(
            f"Invalid VLLM_MHC_CUDA_BACKEND={backend!r}; expected one of {choices}"
        )
    return backend


def _selected_mhc_cuda_backend() -> str:
    from vllm.transformers_utils.configs.dsv4.kernel_config import (
        MHC_SPARKINFER,
        MHC_VLLM_AUTO,
        ROLE_MHC,
        active_kernel_config,
    )

    config = active_kernel_config()
    if config is not None:
        selected = config.symbol(ROLE_MHC)
        if selected == MHC_SPARKINFER:
            return "sparkinfer"
        if selected == MHC_VLLM_AUTO and ROLE_MHC in config.listed_roles:
            return "auto"
    return _requested_mhc_cuda_backend()


def mhc_uses_sparkinfer() -> bool:
    if torch.compiler.is_compiling():
        return _MHC_SPARKINFER
    return _selected_mhc_cuda_backend() == "sparkinfer"


def _has_tilelang_mhc() -> bool:
    if not has_tilelang():
        return False
    if current_platform.is_cuda():
        return True
    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx942

        # TileLang MHC currently produces incorrect results on gfx942. Keep
        # gfx942 on the existing torch/triton fallbacks until that path is fixed.
        return not on_gfx942()
    return False


# Retained for the amd/xpu backends, which select fused vs unfused MHC by this
# flag. The CUDA sm_8x path below uses the torch/triton fallback dispatch.
HAS_TILELANG_MHC = _has_tilelang_mhc()
HAS_AITER_MHC = is_aiter_found_and_supported()
_MHC_SPARKINFER = _selected_mhc_cuda_backend() == "sparkinfer"


def _should_use_mhc_torch_fallback() -> bool:
    if current_platform.is_rocm():
        return True
    if current_platform.is_cuda():
        requested_backend = _selected_mhc_cuda_backend()
        if requested_backend == "sparkinfer":
            # SparkInfer supplies pre/post/post-pre. Its first-layer 3D and
            # HCHead gaps use the existing Triton fallback, never TileLang.
            return True
        if requested_backend == "triton":
            return True
        if requested_backend == "tilelang":
            if not HAS_TILELANG_MHC:
                raise RuntimeError(
                    "VLLM_MHC_CUDA_BACKEND=tilelang requires tilelang to be installed"
                )
            return False
        capability = current_platform.get_device_capability()
        if capability is None:
            return False
        if capability.major == 8:
            return True
        # Everything else (sm_9x, sm_10x, sm_12x) runs the tilelang kernels --
        # but only if tilelang is actually installed. Without this check the
        # CUDA path calls torch.ops.vllm.mhc_*_tilelang regardless, and
        # tilelang_kernels raises ImportError from inside the op body: op
        # registration succeeds, so the failure lands mid-forward on the first
        # request rather than at startup.
        if not HAS_TILELANG_MHC:
            logger.warning(
                "tilelang is not installed; MHC falls back to the torch/triton "
                "kernels on compute capability %s.",
                capability,
            )
            return True
        return False
    return False


_MHC_TORCH_FALLBACK = _should_use_mhc_torch_fallback()


def mhc_uses_tilelang() -> bool:
    """True when the MHC ops dispatch the tilelang kernels on this device.

    Callers use this to distinguish the default TileLang broadcast path from
    the SparkInfer broadcast and torch/Triton fallback paths.
    """
    if torch.compiler.is_compiling():
        return not _MHC_SPARKINFER and not _MHC_TORCH_FALLBACK
    return not mhc_uses_sparkinfer() and not _should_use_mhc_torch_fallback()


_MHC_PRE_TRITON = (
    _MHC_TORCH_FALLBACK
    and current_platform.is_cuda()
    and torch.cuda.is_available()
    and os.getenv("VLLM_MHC_PRE_TRITON", "1") != "0"
)
_MHC_POST_TRITON = (
    _MHC_TORCH_FALLBACK
    and current_platform.is_cuda()
    and torch.cuda.is_available()
    and os.getenv("VLLM_MHC_POST_TRITON", "1") != "0"
)
_MHC_HEAD_TRITON = (
    _MHC_TORCH_FALLBACK
    and current_platform.is_cuda()
    and torch.cuda.is_available()
    and os.getenv("VLLM_MHC_HEAD_TRITON", "1") != "0"
)


def refresh_mhc_backend_selection() -> None:
    """Cache configured CUDA dispatch for the compiled model forward.

    The checkpoint kernel configuration is activated after this module can be
    imported. Refreshing during model construction makes the values visible as
    Python constants to Dynamo without freezing a stale import-time choice.
    """
    global _MHC_SPARKINFER
    global _MHC_TORCH_FALLBACK
    global _MHC_PRE_TRITON
    global _MHC_POST_TRITON
    global _MHC_HEAD_TRITON

    _MHC_SPARKINFER = _selected_mhc_cuda_backend() == "sparkinfer"
    _MHC_TORCH_FALLBACK = _should_use_mhc_torch_fallback()
    _MHC_PRE_TRITON = (
        _MHC_TORCH_FALLBACK
        and current_platform.is_cuda()
        and torch.cuda.is_available()
        and os.getenv("VLLM_MHC_PRE_TRITON", "1") != "0"
    )
    _MHC_POST_TRITON = (
        _MHC_TORCH_FALLBACK
        and current_platform.is_cuda()
        and torch.cuda.is_available()
        and os.getenv("VLLM_MHC_POST_TRITON", "1") != "0"
    )
    _MHC_HEAD_TRITON = (
        _MHC_TORCH_FALLBACK
        and current_platform.is_cuda()
        and torch.cuda.is_available()
        and os.getenv("VLLM_MHC_HEAD_TRITON", "1") != "0"
    )


def _use_mhc_torch_fallback() -> bool:
    if torch.compiler.is_compiling():
        return _MHC_TORCH_FALLBACK
    return _should_use_mhc_torch_fallback()


def _mhc_torch_fallback_synchronize() -> bool:
    return os.getenv("VLLM_MHC_TORCH_FALLBACK_SYNCHRONIZE", "1") != "0"


def _synchronize_mhc_torch_fallback() -> None:
    if torch.compiler.is_compiling():
        return
    if not _mhc_torch_fallback_synchronize():
        return
    if torch.cuda.is_current_stream_capturing():
        return
    mode = os.getenv("VLLM_MHC_TORCH_FALLBACK_SYNC_MODE", "stream").lower()
    if mode == "none":
        return
    if mode == "device":
        torch.cuda.synchronize()
        return
    if mode != "stream":
        logger.warning_once(
            "Unknown VLLM_MHC_TORCH_FALLBACK_SYNC_MODE=%r; using stream sync.",
            mode,
        )
    torch.cuda.current_stream().synchronize()


def _use_mhc_pre_triton() -> bool:
    if torch.compiler.is_compiling():
        return _MHC_PRE_TRITON
    return (
        _use_mhc_torch_fallback()
        and current_platform.is_cuda()
        and torch.cuda.is_available()
        and os.getenv("VLLM_MHC_PRE_TRITON", "1") != "0"
    )


def _use_mhc_post_triton() -> bool:
    if torch.compiler.is_compiling():
        return _MHC_POST_TRITON
    return (
        _use_mhc_torch_fallback()
        and current_platform.is_cuda()
        and torch.cuda.is_available()
        and os.getenv("VLLM_MHC_POST_TRITON", "1") != "0"
    )


def _use_mhc_head_triton() -> bool:
    if torch.compiler.is_compiling():
        return _MHC_HEAD_TRITON
    return (
        _use_mhc_torch_fallback()
        and current_platform.is_cuda()
        and torch.cuda.is_available()
        and os.getenv("VLLM_MHC_HEAD_TRITON", "1") != "0"
    )


def _mhc_pre_triton(*args, **kwargs):
    return mhc_kernels.mhc_pre_triton(*args, **kwargs)


def _mhc_post_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    out: torch.Tensor,
) -> None:
    out.copy_(mhc_kernels.mhc_post_triton(x, residual, post_layer_mix, comb_res_mix))


def _hc_head_fused_reference(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> None:
    x_flat = hs_flat.flatten(-2)
    x_float = x_flat.float()
    rstd = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + rms_eps)
    x_normed = (x_float * rstd).to(hs_flat.dtype).float()
    mixes = F.linear(x_normed, fn)
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    out.copy_(torch.sum(pre.unsqueeze(-1) * hs_flat.float(), dim=1).to(out.dtype))


def _hc_head_triton(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> None:
    torch.ops.vllm.hc_head_triton(
        hs_flat,
        fn,
        hc_scale,
        hc_base,
        out,
        hidden_size,
        rms_eps,
        hc_eps,
        hc_mult,
    )


def _hc_head_fused_kernel(*args, **kwargs) -> None:
    hs_flat = args[0] if args else kwargs["hs_flat"]
    if _use_mhc_torch_fallback():
        if hs_flat.is_cuda and _use_mhc_head_triton():
            _hc_head_triton(*args, **kwargs)
            return
        _hc_head_fused_reference(*args, **kwargs)
        _synchronize_mhc_torch_fallback()
        return
    torch.ops.vllm.hc_head_fused_kernel_tilelang(*args, **kwargs)


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _should_use_mhc_torch_fallback():
        if residual.is_cuda and _use_mhc_pre_triton():
            return mhc_kernels.mhc_pre_triton(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
            )
        out = mhc_kernels.mhc_pre_torch(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
        )
        _synchronize_mhc_torch_fallback()
        return out
    return torch.ops.vllm.mhc_pre_tilelang(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
    )


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    if mhc_uses_sparkinfer():
        from vllm.models.deepseek_v4.nvidia_sm12x.mhc import (
            sparkinfer_mhc_post,
        )

        return sparkinfer_mhc_post(x, residual, post_layer_mix, comb_res_mix)
    if _should_use_mhc_torch_fallback():
        if x.is_cuda and _use_mhc_post_triton():
            return mhc_kernels.mhc_post_triton(
                x, residual, post_layer_mix, comb_res_mix
            )
        out = mhc_kernels.mhc_post_torch(x, residual, post_layer_mix, comb_res_mix)
        _synchronize_mhc_torch_fallback()
        return out
    return torch.ops.vllm.mhc_post_tilelang(x, residual, post_layer_mix, comb_res_mix)


# --8<-- [start:mhc_pre]
@CustomOp.register("mhc_pre")
class MHCPreOp(CustomOp):
    """MHC pre block.

    Computes mix logits from RMS-normalized HC residual streams, then
    returns post_mix, comb_mix, and
    layer_input = sum_i pre_mix_i * residual_i.
    """

    # --8<-- [end:mhc_pre]
    @classmethod
    def enabled(cls) -> bool:
        return True

    def forward_cuda(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if _use_mhc_torch_fallback():
            if _use_mhc_pre_triton() and residual.is_cuda:
                return torch.ops.vllm.mhc_pre_triton(
                    residual,
                    fn,
                    hc_scale,
                    hc_base,
                    rms_eps,
                    hc_pre_eps,
                    hc_sinkhorn_eps,
                    hc_post_mult_value,
                    sinkhorn_repeat,
                    n_splits,
                )
            out = mhc_kernels.mhc_pre_torch(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
            )
            _synchronize_mhc_torch_fallback()
            return out
        return torch.ops.vllm.mhc_pre_tilelang(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
        )

    def forward_hip(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_size = residual.shape[-1]
        if HAS_AITER_MHC and hidden_size % 256 == 0:
            return torch.ops.vllm.mhc_pre_aiter(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
            )
        elif HAS_TILELANG_MHC:
            return torch.ops.vllm.mhc_pre_tilelang(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
            )
        else:
            return self.forward_native(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
            )

    def forward_native(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return mhc_kernels.mhc_pre_torch(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )


# --8<-- [start:mhc_post]
@CustomOp.register("mhc_post")
class MHCPostOp(CustomOp):
    """MHC post block.

    Combines the layer output with the HC residual streams:
    out_j = post_layer_mix_j * x + sum_i comb_res_mix_ij * residual_i.
    """

    # --8<-- [end:mhc_post]

    @classmethod
    def enabled(cls) -> bool:
        return True

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        if mhc_uses_sparkinfer():
            from vllm.models.deepseek_v4.nvidia_sm12x.mhc import (
                sparkinfer_mhc_post,
            )

            return sparkinfer_mhc_post(x, residual, post_layer_mix, comb_res_mix)
        if _use_mhc_torch_fallback():
            if _use_mhc_post_triton() and x.is_cuda:
                return torch.ops.vllm.mhc_post_triton(
                    x,
                    residual,
                    post_layer_mix,
                    comb_res_mix,
                )
            out = mhc_kernels.mhc_post_torch(
                x,
                residual,
                post_layer_mix,
                comb_res_mix,
            )
            _synchronize_mhc_torch_fallback()
            return out
        return torch.ops.vllm.mhc_post_tilelang(
            x, residual, post_layer_mix, comb_res_mix
        )

    def forward_hip(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        hidden_size = residual.shape[-1]
        if HAS_AITER_MHC and hidden_size % 256 == 0:
            return torch.ops.vllm.mhc_post_aiter(
                x,
                residual,
                post_layer_mix,
                comb_res_mix,
            )
        if HAS_TILELANG_MHC:
            return torch.ops.vllm.mhc_post_tilelang(
                x, residual, post_layer_mix, comb_res_mix
            )
        else:
            return self.forward_native(x, residual, post_layer_mix, comb_res_mix)

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        return mhc_kernels.mhc_post_torch(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
        )


# --8<-- [start:hc_head]
@CustomOp.register("hc_head")
class HCHeadOp(CustomOp):
    """HC head reduction for DeepSeek V4.

    Computes gates from the RMS-normalized flattened HC residual and
    returns out = sum_i gate_i * residual_i, collapsing hc_mult streams
    to one.
    """

    # --8<-- [end:hc_head]
    @classmethod
    def enabled(cls) -> bool:
        return True

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_norm_eps: float,
        hc_eps: float,
    ) -> torch.Tensor:
        hc_mult, hidden_size = hidden_states.shape[-2:]
        outer_shape = hidden_states.shape[:-2]
        hs_flat = hidden_states.view(-1, hc_mult, hidden_size)
        num_tokens = hs_flat.shape[0]

        if _use_mhc_torch_fallback():
            out = torch.empty(
                num_tokens,
                hidden_size,
                dtype=torch.bfloat16,
                device=hidden_states.device,
            )
            _hc_head_fused_kernel(
                hs_flat=hs_flat,
                fn=hc_fn,
                hc_scale=hc_scale,
                hc_base=hc_base,
                out=out,
                hidden_size=hidden_size,
                rms_eps=rms_norm_eps,
                hc_eps=hc_eps,
                hc_mult=hc_mult,
            )
            return out.view(*outer_shape, hidden_size)
        out = torch.ops.vllm.hc_head_fused_kernel_tilelang(
            hs_flat,
            hc_fn,
            hc_scale,
            hc_base,
            rms_norm_eps,
            hc_eps,
        )
        return out.view(*outer_shape, hidden_size)

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_norm_eps: float,
        hc_eps: float,
    ) -> torch.Tensor:
        hc_mult, hidden_size = hidden_states.shape[-2:]
        outer_shape = hidden_states.shape[:-2]
        hs_flat = hidden_states.view(-1, hc_mult, hidden_size)
        num_tokens = hs_flat.shape[0]

        out = torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=hidden_states.device
        )
        # Triton, not torch.ops._xpu_C: that namespace does not exist on a ROCm
        # build. It is here because forward_xpu was deleted and its body was
        # left attached to this header, so every ROCm hc_head raised. There is
        # no aiter/ROCm hc_head kernel, and triton is the one implementation
        # that runs on both.
        torch.ops.vllm.hc_head_triton(
            hs_flat,
            hc_fn,
            hc_scale,
            hc_base,
            out,
            hidden_size,
            rms_norm_eps,
            hc_eps,
            hc_mult,
        )
        return out.view(*outer_shape, hidden_size)

    def forward_native(self, *args, **kwargs):
        raise NotImplementedError("Native implementation of hc_head is not available")


# --8<-- [start:mhc_fused_post_pre]
@CustomOp.register("mhc_fused_post_pre")
class MHCFusedPostPreOp(CustomOp):
    """Fused MHC post block followed by the next MHC pre block.

    Equivalent to applying MHCPostOp and then MHCPreOp to the updated
    residual streams, returning residual_cur, post_mix_cur, comb_mix_cur,
    and layer_input_cur.
    """

    # --8<-- [end:mhc_fused_post_pre]
    @classmethod
    def enabled(cls) -> bool:
        return True

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        tile_n: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if mhc_uses_sparkinfer():
            from vllm.models.deepseek_v4.nvidia_sm12x.mhc import (
                sparkinfer_mhc_post_pre,
            )

            return sparkinfer_mhc_post_pre(
                x,
                residual,
                post_layer_mix,
                comb_res_mix,
                fn,
                hc_scale,
                hc_base,
                rms_eps=rms_eps,
                hc_eps=hc_pre_eps,
                sinkhorn_iters=sinkhorn_repeat,
                norm_weight=norm_weight,
                norm_eps=norm_eps,
            )
        if _use_mhc_torch_fallback():
            if _use_mhc_post_triton() and x.is_cuda:
                return torch.ops.vllm.mhc_fused_post_pre_triton(
                    x,
                    residual,
                    post_layer_mix,
                    comb_res_mix,
                    fn,
                    hc_scale,
                    hc_base,
                    rms_eps,
                    hc_pre_eps,
                    hc_sinkhorn_eps,
                    hc_post_mult_value,
                    sinkhorn_repeat,
                    n_splits,
                )
            residual_cur = mhc_kernels.mhc_post_torch(
                x, residual, post_layer_mix, comb_res_mix
            )
            post_mix_cur, comb_mix_cur, layer_input_cur = mhc_kernels.mhc_pre_torch(
                residual_cur,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
            )
            return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur
        return torch.ops.vllm.mhc_fused_post_pre_tilelang(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            tile_n,
        )

    def forward_hip(self, *args, **kwargs):
        raise NotImplementedError(
            "Hip implementation of mhc_fused_post_pre is not available"
        )

    def forward_native(self, *args, **kwargs):
        raise NotImplementedError(
            "Native implementation of mhc_fused_post_pre is not available"
        )
