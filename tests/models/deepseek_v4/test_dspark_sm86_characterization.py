# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Characterization test: resolved kernel/dispatch config for DSpark-on-sm86.

Snapshots the concrete dispatch DECISIONS (not behavior) that the rebase
(`git rebase --onto 8d8ec38361 e7df232288`, landing DSpark on top of upstream
vLLM) and the MHC dispatcher fix (commit fcf42a8acf) depend on:

  * On sm_8x, DeepseekV4DecoderLayer's attention factory (_select_dsv4_attn_cls)
    resolves to DeepseekV4SM86Attention -- this is what DSpark's mtp.{0,1,2}
    layers reuse via DeepseekV4DecoderLayer, with NO override in dspark.py.
  * On sm_8x, the MHC module-level flags (_MHC_TORCH_FALLBACK/_MHC_PRE_TRITON/
    _MHC_POST_TRITON/_MHC_HEAD_TRITON) all resolve True, so MHCPreOp/MHCPostOp/
    HCHeadOp.forward_cuda route to the Triton kernels, NOT bare TileLang ops.
    This is the dispatch path commit fcf42a8acf's fix (self.mhc_post/self.hc_head
    instead of mhc_post_tilelang/hc_head_fused_kernel_tilelang) now depends on
    being live -- if these flags were ever False on this hardware, the fix
    would be a no-op (both paths would go through TileLang either way) and
    this test would need to be revisited.

These values are the dispatch logic in vllm/models/deepseek_v4/nvidia/model.py
and vllm/model_executor/layers/mhc.py, pinned here so a change to either is
deliberate rather than incidental. This test is the executable form of that
claim -- it will fail loudly if a future
change (this session or later) alters which kernel path sm86 resolves to.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and current_platform.is_cuda()),
    reason="requires a CUDA device to read the real resolved capability",
)


def _is_ampere() -> bool:
    cap = current_platform.get_device_capability()
    return cap is not None and cap.major == 8


def test_sm86_attention_class_resolution_proof_string() -> None:
    """_select_dsv4_attn_cls resolves to DeepseekV4SM86Attention on sm_8x.

    DSparkDeepseekV4Model builds its layers from DeepseekV4DecoderLayer
    directly (nvidia/dspark.py), so this factory's resolution IS DSpark's
    attention-class resolution too -- there is no separate DSpark attention
    selector to test.
    """
    from vllm.models.deepseek_v4.nvidia.model import _select_dsv4_attn_cls

    if not _is_ampere():
        pytest.skip("proof string is sm_8x-specific; other capabilities differ")

    from vllm.config import VllmConfig
    from vllm.config.attention import AttentionConfig

    class _StubAttentionConfig:
        backend = None

    class _StubVllmConfig:
        attention_config = _StubAttentionConfig()

    resolved = _select_dsv4_attn_cls(_StubVllmConfig())

    proof_string = f"sm86_dsv4_attn_cls={resolved.__module__}.{resolved.__qualname__}"
    assert proof_string == (
        "sm86_dsv4_attn_cls=vllm.models.deepseek_v4.nvidia_sm86.attention."
        "DeepseekV4TritonSM86Attention"
    ), proof_string


def test_sm86_mhc_dispatch_flags_proof_string() -> None:
    """MHC module-level dispatch flags resolve to the Triton path on sm_8x.

    This is the precondition for commit fcf42a8acf's fix to matter: DSpark's
    self.mhc_post/self.hc_head (MHCPostOp/HCHeadOp) only differ in behavior
    from the old bare mhc_post_tilelang/hc_head_fused_kernel_tilelang calls
    when these flags route forward_cuda to Triton instead of falling through
    to TileLang.
    """
    from vllm.model_executor.layers import mhc

    if not _is_ampere():
        pytest.skip("proof string is sm_8x-specific; other capabilities differ")

    proof_string = (
        f"mhc_torch_fallback={mhc._MHC_TORCH_FALLBACK},"
        f"mhc_pre_triton={mhc._MHC_PRE_TRITON},"
        f"mhc_post_triton={mhc._MHC_POST_TRITON},"
        f"mhc_head_triton={mhc._MHC_HEAD_TRITON}"
    )
    assert proof_string == (
        "mhc_torch_fallback=True,mhc_pre_triton=True,"
        "mhc_post_triton=True,mhc_head_triton=True"
    ), proof_string


def test_dspark_forward_uses_mhcop_dispatchers_not_bare_tilelang() -> None:
    """Static proof that dspark.py's forward() no longer calls the bare
    TileLang wrapper functions directly (regression guard for fcf42a8acf)."""
    import inspect

    from vllm.models.deepseek_v4.nvidia import dspark as dspark_module

    source = inspect.getsource(dspark_module.DSparkDeepseekV4Model.forward)
    assert "mhc_post_tilelang(" not in source, source
    assert "hc_head_fused_kernel_tilelang(" not in source, source
    assert "self.mhc_post(" in source, source
    assert "self.hc_head(" in source, source

    # And the bare functions must not even be imported at module scope anymore
    # -- otherwise a future edit could reintroduce the direct call silently.
    assert not hasattr(dspark_module, "mhc_post_tilelang")
    assert not hasattr(dspark_module, "hc_head_fused_kernel_tilelang")
