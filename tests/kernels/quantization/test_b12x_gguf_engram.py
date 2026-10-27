# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk-backed original Q5_K rows across layer and TP head boundaries."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.quantization.b12x_gguf_engram import (
    B12xGGUFEngramEmbedding,
    GGUFEngramSource,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_disk_layer_tp_heads_feed_stable_graph(tmp_path):
    gguf = pytest.importorskip("gguf")
    from b12x._lib.runtime_control import kernel_resolution_guard

    rng = np.random.default_rng(190)
    raw = rng.integers(0, 256, (80, 176), dtype=np.uint8)
    raw[:, :4] = rng.normal(0, 0.02, (80, 2)).astype(np.float16).view(np.uint8)
    expected = (
        torch.from_numpy(gguf.dequantize(raw, gguf.GGMLQuantizationType.Q5_K))
        .bfloat16()
        .cuda()
    )
    path = tmp_path / "engram.bin"
    offset = 192
    with path.open("wb") as f:
        f.write(bytes(offset))
        f.write(raw.tobytes())
    source = GGUFEngramSource(path, "per_layer_token_embd.weight", offset, 80, 256)
    layout = SimpleNamespace(
        head_dim=256,
        num_embeddings=(40, 40),
        primes=(((10, 10), (10, 10)), ((10, 10), (10, 10))),
    )
    owners = []
    try:
        for layer in range(2):
            for rank in range(2):
                owners.append(
                    B12xGGUFEngramEmbedding(
                        source,
                        layout,
                        layer,
                        max_tokens=5,
                        tp_rank=rank,
                        tp_size=2,
                        device="cuda",
                    )
                )
        ids = torch.tensor(
            [[0, 11, 22, 39], [9, 10, 29, 30], [3, 14, 25, 36]], device="cuda"
        )
        out = torch.empty(3, 2, 256, dtype=torch.bfloat16, device="cuda")
        downstream = torch.empty_like(out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            torch.mul(out, 2, out=downstream)
        with kernel_resolution_guard("GGUF Engram disk lookup"):
            for iteration in range(2):
                for i, owner in enumerate(owners):
                    owner.lookup(ids, out)
                    allocated = torch.accelerator.memory_stats()[
                        "allocation.all.allocated"
                    ]
                    graph.replay()
                    torch.accelerator.synchronize()
                    assert (
                        torch.accelerator.memory_stats()["allocation.all.allocated"]
                        == allocated
                    )
                    indices = ids[:, (i % 2) * 2 : (i % 2 + 1) * 2] + (i // 2) * 40
                    assert torch.equal(out, expected[indices])
                    assert torch.equal(downstream, expected[indices] * 2)
                    assert not list(owner.parameters())
                    assert owner.table.stats()["weight_cache_bytes"] == 5 * 2 * 176
                ids.copy_(
                    torch.tensor(
                        [[7, 18, 20, 31], [1, 12, 23, 34], [9, 19, 29, 39]],
                        device="cuda",
                    )
                )
        graph.reset()
    finally:
        for owner in owners:
            owner.close()
