# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.utils import deep_gemm
from vllm.model_executor.layers.quantization.dsv4_int import Dsv4IntConfig
from vllm.model_executor.layers.sparse_attn_indexer import (
    SM120_SHORT_ROW_TOPK_ALWAYS_WIDTH,
    SM120_SHORT_ROW_TOPK_MAX_WIDTH,
    _should_use_sm120_short_row_topk_decode,
)
from vllm.models.deepseek_v4.nvidia_sm86 import triton_kernels as dsv4_sm86


@pytest.mark.parametrize(
    ("topk_tokens", "logits_width", "num_rows", "is_cuda_sm120", "expected"),
    [
        (512, SM120_SHORT_ROW_TOPK_ALWAYS_WIDTH, 32, True, True),
        (512, 8192, 16, True, True),
        (512, 8192, 32, True, True),
        (512, 12288, 32, True, False),
        (512, SM120_SHORT_ROW_TOPK_MAX_WIDTH, 1, True, False),
        (512, 4096, 1, False, False),
        (2048, 4096, 1, True, False),
    ],
)
def test_sm120_short_row_topk_decode_selector(
    topk_tokens: int,
    logits_width: int,
    num_rows: int,
    is_cuda_sm120: bool,
    expected: bool,
) -> None:
    assert (
        _should_use_sm120_short_row_topk_decode(
            topk_tokens,
            logits_width,
            num_rows,
            is_cuda_sm120,
        )
        is expected
    )


def test_dsv4_int_checkpoint_auto_enables_int8_indexer_imma(monkeypatch) -> None:
    quant_config = Dsv4IntConfig.from_config(
        {
            "quant_method": "dsv4_int",
            "config_groups": {
                "experts_w4a16": {
                    "weights": {"num_bits": 4, "type": "int"},
                },
                "linears_w8a16": {
                    "weights": {"num_bits": 8, "type": "int"},
                },
            },
            "__experimental_enable_imma_from_https://github.com/appMana/forks-vllm-ampere": True,
        }
    )

    assert dsv4_sm86.indexer_cache_is_int8()
    assert dsv4_sm86.indexer_imma_enabled()
    assert quant_config.expert_input_dtype is torch.int8


def test_fp8_mqa_logits_uses_fused_imma_workspace_on_auto_int8(
    monkeypatch,
) -> None:
    monkeypatch.setattr(deep_gemm, "_lazy_init", lambda: None)
    monkeypatch.setattr(
        deep_gemm.current_platform,
        "is_device_capability_family",
        lambda family: family == 80,
    )
    monkeypatch.setattr(dsv4_sm86, "indexer_imma_enabled", lambda: True)

    def fake_workspace(q, kv, weights, ks, ke, qk_int8=False):
        assert q.dtype == torch.int8
        assert kv[0].dtype == torch.int8
        assert qk_int8 is True
        return torch.full((q.shape[0], kv[0].shape[0]), 3.0)

    def fail_torch_path(*args, **kwargs):
        raise AssertionError("IMMA prefill should use fused workspace logits")

    monkeypatch.setattr(dsv4_sm86, "mqa_logits_workspace_triton", fake_workspace)
    monkeypatch.setattr(deep_gemm, "_fp8_mqa_logits_torch", fail_torch_path)

    q = torch.ones((2, 4, 8), dtype=torch.int8)
    k = torch.ones((5, 8), dtype=torch.int8)
    scales = torch.ones((5,), dtype=torch.float32)
    weights = torch.ones((2, 4), dtype=torch.float32)
    ks = torch.zeros((2,), dtype=torch.int32)
    ke = torch.full((2,), 5, dtype=torch.int32)

    actual = deep_gemm.fp8_fp4_mqa_logits(
        (q, None),
        (k, scales),
        weights,
        ks,
        ke,
        clean_logits=False,
    )

    torch.testing.assert_close(actual, torch.full((2, 5), 3.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mqa_logits_workspace_accepts_unaligned_workspace_views() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_rows, num_heads, head_dim, seq_len = 17, 64, 512, 1032

    q_base = torch.empty(
        (num_rows + 1, num_heads, head_dim + 1),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    q = q_base[1:, :, :head_dim]
    q.copy_(
        torch.randn(
            (num_rows, num_heads, head_dim),
            device=device,
            dtype=torch.float32,
        )
        .clamp(-4, 4)
        .to(torch.float8_e4m3fn)
    )

    k_base = torch.empty((seq_len + 1, head_dim + 1), device=device, dtype=torch.int8)
    k = k_base[1:, :head_dim]
    k.copy_(torch.randint(-32, 32, (seq_len, head_dim), device=device, dtype=torch.int8))

    scale_base = torch.empty((seq_len + 1, 1), device=device, dtype=torch.float32)
    scales = scale_base[1:, 0]
    scales.copy_(torch.rand((seq_len,), device=device, dtype=torch.float32) * 0.02 + 0.001)

    weights = torch.rand((num_rows, num_heads), device=device, dtype=torch.float32)
    ks = torch.zeros((num_rows,), device=device, dtype=torch.int32)
    ke = torch.full((num_rows,), seq_len, device=device, dtype=torch.int32)

    out = dsv4_sm86.mqa_logits_workspace_triton(
        q,
        (k, scales),
        weights,
        ks,
        ke,
        qk_int8=False,
    )
    torch.cuda.synchronize()

    assert out.shape == (num_rows, seq_len)
    assert torch.isfinite(out).all()
