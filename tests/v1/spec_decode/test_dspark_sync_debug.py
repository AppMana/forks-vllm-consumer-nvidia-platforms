# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

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
