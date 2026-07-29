# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.utils import deep_gemm
from vllm.model_executor.layers.quantization.dsv4_int import Dsv4IntConfig
from vllm.model_executor.layers.sparse_attn_indexer import (
    SM120_SHORT_ROW_TOPK_ALWAYS_WIDTH,
    SM120_SHORT_ROW_TOPK_MAX_WIDTH,
    _reserve_prefill_gather_workspace,
    _should_use_sm120_short_row_topk_decode,
)
from vllm.models.deepseek_v4.nvidia_imma import triton_kernels as dsv4_sm86
from vllm.transformers_utils.configs.dsv4 import kernel_config


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


def test_dsv4_int_vllm_block_enables_int8_indexer_imma(monkeypatch) -> None:
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
            "vllm": {
                "kernels": [
                    kernel_config.INDEXER_CACHE_INT8_WRITER,
                    kernel_config.INDEXER_QUERY_INT8_QUANT,
                    kernel_config.DENSE_EXPERTS_INT8_ACTIVATION,
                ],
            },
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


def test_sparse_indexer_prefill_workspace_reserves_max_total_seq_len(
    monkeypatch,
) -> None:
    calls = []

    class FakeWorkspaceManager:
        def get_simultaneous(self, *specs):
            calls.append(specs)
            return tuple(torch.empty((), dtype=dtype) for _shape, dtype in specs)

    monkeypatch.setattr(
        "vllm.model_executor.layers.sparse_attn_indexer.current_workspace_manager",
        lambda: FakeWorkspaceManager(),
    )

    _reserve_prefill_gather_workspace(
        total_seq_lens=7,
        max_total_seq_len=1234,
        head_dim=128,
        fp8_dtype=torch.uint8,
        use_fp4_cache=False,
    )

    assert calls
    values_spec, scales_spec, _topk_spec = calls[-1]
    assert values_spec == ((1234, 128), torch.uint8)
    assert scales_spec == ((1234, 4), torch.uint8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("num_rows", "num_heads", "head_dim", "seq_len"),
    [
        (17, 64, 512, 1032),
        (1024, 64, 128, 1280),
        (1024, 64, 128, 1032),
    ],
)
def test_mqa_logits_workspace_accepts_unaligned_workspace_views(
    num_rows: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")

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

    k_base = torch.empty(
        (seq_len + 1, head_dim + 1),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    k = k_base[1:, :head_dim]
    k.copy_(
        torch.randn(
            (seq_len, head_dim),
            device=device,
            dtype=torch.float32,
        )
        .clamp(-4, 4)
        .to(torch.float8_e4m3fn)
    )

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sparse_attention_bf16_accepts_unaligned_runtime_views() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_tokens, num_heads, head_dim = 3, 64, 512
    kv_rows, index_width = 1032, 1792

    q_storage = torch.empty(
        num_tokens * num_heads * head_dim + 1,
        device=device,
        dtype=torch.bfloat16,
    )
    q = q_storage[1:].view(num_tokens, num_heads, head_dim)
    q.copy_(torch.randn(q.shape, device=device, dtype=torch.float32).to(torch.bfloat16))

    kv_storage = torch.empty(
        kv_rows * head_dim + 1,
        device=device,
        dtype=torch.bfloat16,
    )
    kv = kv_storage[1:].view(kv_rows, head_dim)
    kv.copy_(torch.randn(kv.shape, device=device, dtype=torch.float32).to(torch.bfloat16))

    indices_storage = torch.empty(
        num_tokens * index_width + 1,
        device=device,
        dtype=torch.int32,
    )
    indices = indices_storage[1:].view(num_tokens, index_width)
    indices.copy_(
        torch.arange(index_width, device=device, dtype=torch.int32)
        .remainder(kv_rows)
        .expand(num_tokens, -1)
    )

    lengths = torch.full((num_tokens,), 17, device=device, dtype=torch.int32)
    sink_storage = torch.empty(num_heads + 1, device=device, dtype=torch.float32)
    sink = sink_storage[1:]
    sink.zero_()

    out_storage = torch.empty(
        num_tokens * num_heads * head_dim + 1,
        device=device,
        dtype=torch.bfloat16,
    )
    out = out_storage[1:].view(num_tokens, num_heads, head_dim)

    dsv4_sm86.sparse_attention_triton(
        q,
        kv,
        indices,
        lengths,
        scale=1.0 / head_dim**0.5,
        attn_sink=sink,
        out=out,
    )
    torch.cuda.synchronize()

    assert torch.isfinite(out).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_paged_mqa_logits_accepts_unaligned_runtime_views() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch_size, next_n, num_heads, head_dim = 2, 3, 64, 512
    num_blocks, block_size = 8, 16
    scale_bytes = torch.float32.itemsize
    token_bytes = head_dim + scale_bytes

    q_storage = torch.empty(
        batch_size * next_n * num_heads * head_dim + 1,
        device=device,
        dtype=torch.uint8,
    )
    q = q_storage[1:].view(torch.float8_e4m3fn).view(
        batch_size, next_n, num_heads, head_dim
    )
    q.copy_(
        torch.randn(q.shape, device=device, dtype=torch.float32)
        .clamp(-4, 4)
        .to(torch.float8_e4m3fn)
    )

    fused = torch.empty(
        (num_blocks, block_size, token_bytes),
        device=device,
        dtype=torch.uint8,
    )
    fused_blocks = fused.view(num_blocks, -1)
    value_end = block_size * head_dim
    scale_end = value_end + block_size * scale_bytes
    fused_blocks[:, :value_end] = torch.randint(
        0,
        255,
        (num_blocks, value_end),
        device=device,
        dtype=torch.uint8,
    )
    scales = torch.full(
        (num_blocks, block_size, 1),
        0.01,
        device=device,
        dtype=torch.float32,
    )
    fused_blocks[:, value_end:scale_end] = scales.view(torch.uint8).reshape(
        num_blocks,
        -1,
    )
    kv_cache = fused

    weights_storage = torch.empty(
        batch_size * next_n * num_heads + 1,
        device=device,
        dtype=torch.float32,
    )
    weights = weights_storage[1:].view(batch_size * next_n, num_heads)
    weights.copy_(torch.rand_like(weights))

    context_storage = torch.empty(
        batch_size * next_n + 1,
        device=device,
        dtype=torch.int32,
    )
    context_lens = context_storage[1:].view(batch_size, next_n)
    context_lens.fill_(num_blocks * block_size)

    block_tables_storage = torch.empty(
        batch_size * num_blocks + 1,
        device=device,
        dtype=torch.int32,
    )
    block_tables = block_tables_storage[1:].view(batch_size, num_blocks)
    block_tables.copy_(
        torch.arange(num_blocks, device=device, dtype=torch.int32).expand(
            batch_size,
            -1,
        )
    )

    out = dsv4_sm86.fp8_paged_mqa_logits_triton(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        max_model_len=num_blocks * block_size,
        token_start=1,
        token_count=113,
    )
    torch.cuda.synchronize()

    assert out.shape == (batch_size * next_n, 113)
    assert torch.isfinite(out).all()
