# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

import vllm.v1.worker.gpu.spec_decode.dflash.speculator as dflash_speculator


def test_sync_debug_does_not_synchronize_during_cuda_graph_capture(monkeypatch):
    synchronize = Mock()
    monkeypatch.setattr(dflash_speculator, "_SYNC_DEBUG", True)
    monkeypatch.setattr(
        dflash_speculator.torch.cuda,
        "is_current_stream_capturing",
        lambda: True,
    )
    monkeypatch.setattr(dflash_speculator.torch.cuda, "synchronize", synchronize)

    dflash_speculator.sync_debug("capture")

    synchronize.assert_not_called()


def test_sync_debug_synchronizes_outside_cuda_graph_capture(monkeypatch):
    synchronize = Mock()
    monkeypatch.setattr(dflash_speculator, "_SYNC_DEBUG", True)
    monkeypatch.setattr(
        dflash_speculator.torch.cuda,
        "is_current_stream_capturing",
        lambda: False,
    )
    monkeypatch.setattr(dflash_speculator.torch.cuda, "synchronize", synchronize)

    dflash_speculator.sync_debug("eager")

    synchronize.assert_called_once_with()


@pytest.mark.skipif(
    not dflash_speculator.torch.cuda.is_available(), reason="CUDA is required"
)
def test_sync_debug_is_safe_in_real_cuda_graph_capture(monkeypatch):
    torch = dflash_speculator.torch
    monkeypatch.setattr(dflash_speculator, "_SYNC_DEBUG", True)
    value = torch.zeros(1, device="cuda")
    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        dflash_speculator.sync_debug("capture")
        value.add_(1)

    graph.replay()
    torch.cuda.synchronize()
    assert value.item() == 1
