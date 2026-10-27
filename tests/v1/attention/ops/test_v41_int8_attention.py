# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 shared compressed-cache addressing through native INT8 attention."""

import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.common.ops.cache_utils import (
    get_int8_ds_mla_cache_views,
    quantize_and_insert_int8_ds_mla_cache,
)
from vllm.models.deepseek_v41.nvidia.int8_attention import DeepseekV41Int8Attention


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_v41_indexer_warmup_prevents_first_long_prompt_jit(tmp_path):
    """Fresh-cache startup covers V41's 32 heads and two-level candidates."""
    script = textwrap.dedent("""
        import torch
        from vllm.model_executor.layers.sparse_attn_indexer import (
            warmup_indexer_prefill_logits_kernel)
        from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
            warmup_candidate_block_kernels, select_candidate_blocks,
            apply_candidate_mask)
        from vllm.models.deepseek_v4.nvidia_imma.triton_kernels import (
            mqa_logits_workspace_triton)
        from vllm.utils import jit_monitor
        device = torch.device('cuda:0')
        # A different model geometry must not suppress V41's warmup.
        for heads in (64, 32):
            warmup_indexer_prefill_logits_kernel(device, qk_int8=True, num_heads=heads)
        warmup_candidate_block_kernels(device, topk_blocks=2048, block_size=8)
        jit_monitor._setup_cutedsl_jit_hook = lambda: None
        jit_monitor._setup_tilelang_jit_hook = lambda: None
        jit_monitor.activate(mode='error', verbose=True)
        for m, width in ((1, 768), (17, 777), (128, 1779)):
            q = torch.zeros(m, 32, 128, dtype=torch.int8, device=device)
            k = torch.zeros(width, 128, dtype=torch.int8, device=device)
            scales = torch.ones(width, device=device)
            weights = torch.ones(m, 32, device=device)
            starts = torch.zeros(m, dtype=torch.int32, device=device)
            ends = torch.full((m,), width, dtype=torch.int32, device=device)
            logits = mqa_logits_workspace_triton(
                q, (k, scales), weights, starts, ends, qk_int8=True)
            assert torch.count_nonzero(logits) == 0
            candidates = torch.empty(m, 2048, dtype=torch.int32, device=device)
            for bounds in (None, starts):
                logits.zero_()
                select_candidate_blocks(logits, bounds, ends, 2048, 8, candidates)
                apply_candidate_mask(logits, bounds, ends, candidates, 8)
                assert torch.count_nonzero(logits) == 0
        torch.accelerator.synchronize()
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=dict(os.environ, TRITON_CACHE_DIR=str(tmp_path / "triton")),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_v41_native_int8_shared_cache_decode_prefill_replay(ratio):
    from flash_mla import sparse_mla_decode_int8, sparse_mla_prefill_int8

    torch.manual_seed(311)
    device = "cuda"
    layer = DeepseekV41Int8Attention.__new__(DeepseekV41Int8Attention)
    torch.nn.Module.__init__(layer)
    layer.compress_ratio = ratio
    layer.scale = 512**-0.5
    layer.n_local_heads = layer.padded_heads = 32
    layer.attn_sink = torch.randn(32, device=device)
    layer._decode, layer._prefill = sparse_mla_decode_int8, sparse_mla_prefill_int8
    block = 4
    cache = torch.zeros(4, block, 528, dtype=torch.uint8, device=device)
    extra_block = 8 // max(ratio, 1)
    extra = torch.zeros(4, extra_block, 528, dtype=torch.uint8, device=device)
    for target, size in ((cache, block), (extra, extra_block)):
        values = torch.randn(4 * size, 512, dtype=torch.bfloat16, device=device)
        quantize_and_insert_int8_ds_mla_cache(
            values, target, torch.arange(4 * size, device=device), size
        )
    layer.swa_cache_layer = SimpleNamespace(kv_cache=cache)
    layer.topk_indices_buffer = torch.tensor(
        [[0, 3, -1, -1], [1, 2, 5, -1], [0, 1, 4, -1], [2, 3, -1, -1], [1, 4, 5, -1]],
        dtype=torch.int32,
        device=device,
    )
    # Two decode requests and a prefill belonging to request2. Non-monotonic
    # physical block ownership exposes accidental local/global index reuse.
    block_table = torch.tensor(
        [[2, 0], [1, 3], [3, 1]], dtype=torch.int32, device=device
    )
    swa_indices = torch.tensor(
        [
            [3, 6, -1, -1],
            [8, 9, 12, -1],
            [4, -1, -1, -1],
            [4, 5, -1, -1],
            [4, 5, 7, -1],
        ],
        dtype=torch.int32,
        device=device,
    )
    swa_lengths = torch.tensor([2, 3, 1, 2, 3], dtype=torch.int32, device=device)
    swa = SimpleNamespace(
        block_size=block,
        num_decode_tokens=2,
        num_prefill_tokens=3,
        token_to_req_indices=torch.tensor(
            [0, 1, 2, 2, 2], dtype=torch.int32, device=device
        ),
        is_valid_token=torch.ones(5, dtype=torch.bool, device=device),
        decode_swa_indices=swa_indices[:2, None],
        decode_swa_lens=swa_lengths[:2],
        prefill_swa_indices=swa_indices[2:, None],
        prefill_swa_lens=swa_lengths[2:],
    )
    metadata = SimpleNamespace(block_size=8, block_table=block_table) if ratio else None
    q = torch.randn(7, 32, 512, dtype=torch.bfloat16, device=device)
    output = torch.empty_like(q)
    positions = torch.arange(5, dtype=torch.int64, device=device)

    def run():
        layer._forward_decode(
            q[:2], extra if ratio else None, swa, metadata, ratio == 0, output[:2]
        )
        layer._forward_prefill(
            q[2:],
            positions[2:],
            extra if ratio else None,
            cache,
            output[2:],
            metadata,
            swa,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for iteration in range(2):
        if iteration:
            q.neg_()
            layer.topk_indices_buffer[:, 0].fill_(2)
        output.fill_(float("nan"))
        graph.replay()
        torch.accelerator.synchronize()
        rows, scales = get_int8_ds_mla_cache_views(cache, block)
        xrows, xscales = get_int8_ds_mla_cache_views(extra, extra_block)
        dequant = (rows.float() * scales[..., None]).reshape(-1, 512)
        xdequant = (xrows.float() * xscales[..., None]).reshape(-1, 512)
        for t in range(5):
            selected = dequant[swa_indices[t, : swa_lengths[t]]]
            if ratio:
                local = layer.topk_indices_buffer[t]
                local = local[local >= 0].long()
                req = int(swa.token_to_req_indices[t])
                global_ids = (
                    block_table[req, local // extra_block].long() * extra_block
                    + local % extra_block
                )
                selected = torch.cat((selected, xdequant[global_ids]))
            logits = q[t].float() @ selected.T * layer.scale
            weights = torch.softmax(
                torch.cat((logits, layer.attn_sink[:, None]), dim=1), dim=1
            )[:, :-1]
            expected = weights @ selected
            torch.testing.assert_close(
                output[t].float(), expected, atol=0.035, rtol=0.035
            )
        assert torch.isnan(output[5:]).all()
    graph.reset()
