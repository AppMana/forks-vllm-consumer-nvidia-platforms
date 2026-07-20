# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Zero-layer PP ranks (VLLM_PP_LAYER_PARTITION tail ``...,4,0``).

The DSpark draft rank (last PP rank) hosts the 3-stage MTP graft plus the LM
head next to what is today 1 main-model decoder layer. Moving that layer to
the previous rank (partition tail ``...,4,0``) frees its weights and KV on the
draft rank. The aux relay in ``DeepseekV4Model.forward`` was already designed
for any contiguous partition: boundaries below ``start_layer`` arrive relayed,
the boundary AT ``start_layer`` is reconstructed from the received MHC stream.
These tests pin down the zero-layer forward tail: with zero local layers the
layer loop never binds ``layer``, and the final MHC collapse must still run
(reusing the ``start_layer`` cut reconstruction when it exists).

CPU-only: the model instance is built by bypassing ``__init__`` (same idiom as
test_dspark_remap.py) with deterministic stand-ins for the MHC ops.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import vllm.envs
from vllm.distributed.utils import get_pp_indices
from vllm.models.deepseek_v4.nvidia import model as m
from vllm.sequence import IntermediateTensors

HC = 2
H = 8
T = 4
NUM_LAYERS = 5


class _FakePPGroup:
    def __init__(self, is_first_rank: bool, is_last_rank: bool):
        self.is_first_rank = is_first_rank
        self.is_last_rank = is_last_rank


class _FakeDecoderLayer(nn.Module):
    """Stream-preserving stand-in: shifts hidden_states, passes MHC through."""

    def forward(self, hidden_states, positions, input_ids, post_mix, res_mix, residual):
        return hidden_states + 1.0, residual, post_mix, res_mix


def _make_model(
    monkeypatch,
    *,
    start: int,
    end: int,
    aux_layers: tuple[int, ...],
    is_last_rank: bool = True,
):
    monkeypatch.setattr(
        m, "get_pp_group", lambda: _FakePPGroup(False, is_last_rank)
    )
    inst = m.DeepseekV4Model.__new__(m.DeepseekV4Model)
    nn.Module.__init__(inst)
    inst.config = SimpleNamespace(hidden_size=H)
    inst.use_mega_moe = False
    inst.hc_mult = HC
    inst.rms_norm_eps = 1e-6
    inst.hc_eps = 1e-6
    inst.start_layer = start
    inst.end_layer = end
    inst.layers = nn.ModuleList(
        [
            _FakeDecoderLayer() if start <= i < end else m.PPMissingLayer()
            for i in range(NUM_LAYERS)
        ]
    )
    inst.aux_hidden_state_layers = aux_layers

    calls = {"mhc_post": 0}

    def mhc_post(hidden_states, residual, post_mix, res_mix):
        calls["mhc_post"] += 1
        return hidden_states + residual  # deterministic collapse

    inst.mhc_post = mhc_post
    # hc_head reduces the hc copies; norm doubles so its application is visible.
    inst.hc_head = lambda h, fn, scale, base, rms_eps, hc_eps: h.mean(dim=1)
    inst.hc_head_fn = None
    inst.hc_head_scale = None
    inst.hc_head_base = None
    inst.norm = lambda h: 2.0 * h
    inst._mtp_hidden_buffer = torch.zeros(16, HC * H)
    return inst, calls


def _recv_stream(aux_boundaries: tuple[int, ...] = ()) -> IntermediateTensors:
    torch.manual_seed(0)
    tensors = {
        "hidden_states": torch.randn(T, HC, H),
        "residual": torch.randn(T, HC, H),
        "post_mix": torch.randn(T, HC, 1),
        "res_mix": torch.randn(T, HC, HC),
    }
    for j in aux_boundaries:
        tensors[f"aux_hidden_{j}"] = torch.randn(T, H)
    return IntermediateTensors(tensors)


def _forward(inst, intermediate_tensors):
    input_ids = torch.zeros(T, dtype=torch.int32)
    positions = torch.arange(T)
    return inst.forward(input_ids, positions, intermediate_tensors)


def test_partition_zero_tail_parses(monkeypatch):
    """A trailing 0 in VLLM_PP_LAYER_PARTITION yields an empty layer range."""
    monkeypatch.setattr(
        vllm.envs,
        "VLLM_PP_LAYER_PARTITION",
        "3,4,4,4,4,4,4,4,4,4,4,0",
        raising=False,
    )
    assert get_pp_indices(43, 11, 12) == (43, 43)
    assert get_pp_indices(43, 10, 12) == (39, 43)


def test_zero_layer_last_rank_dspark_aux(monkeypatch):
    """Zero-layer draft rank: relayed aux + cut reconstruction, collapse reused.

    Mirrors the production ``...,4,0`` layout scaled down: 5-layer target, aux
    boundaries (3, 4, 5); the last rank owns no layers, receives aux 3 and 4
    relayed, and reconstructs boundary 5 (== start_layer) from the MHC stream.
    """
    inst, calls = _make_model(monkeypatch, start=5, end=5, aux_layers=(3, 4, 5))
    recv = _recv_stream(aux_boundaries=(3, 4))
    out = _forward(inst, recv)

    assert isinstance(out, tuple)
    hidden_states, aux_hidden_states = out
    assert len(aux_hidden_states) == 3

    expected_recon = recv["hidden_states"] + recv["residual"]
    # Relayed boundaries pass through untouched, in boundary order.
    assert aux_hidden_states[0] is recv.tensors["aux_hidden_3"]
    assert aux_hidden_states[1] is recv.tensors["aux_hidden_4"]
    torch.testing.assert_close(aux_hidden_states[2], expected_recon.mean(dim=1))
    # Final hidden = norm(hc_head(collapse)).
    torch.testing.assert_close(hidden_states, 2.0 * expected_recon.mean(dim=1))
    # Pre-hc_head residual stashed for the MTP draft.
    torch.testing.assert_close(
        inst._mtp_hidden_buffer[:T], expected_recon.flatten(1)
    )
    # The cut reconstruction doubles as the final collapse: exactly one call.
    assert calls["mhc_post"] == 1


def test_zero_layer_last_rank_no_aux(monkeypatch):
    """Zero-layer last rank without spec decode: plain final collapse."""
    inst, calls = _make_model(monkeypatch, start=5, end=5, aux_layers=())
    recv = _recv_stream()
    out = _forward(inst, recv)

    assert isinstance(out, torch.Tensor)
    expected_recon = recv["hidden_states"] + recv["residual"]
    torch.testing.assert_close(out, 2.0 * expected_recon.mean(dim=1))
    assert calls["mhc_post"] == 1


def test_last_rank_with_layers_unchanged(monkeypatch):
    """Regression guard: the with-layers tail reuses the last aux capture."""
    inst, calls = _make_model(monkeypatch, start=3, end=5, aux_layers=(3, 4, 5))
    recv = _recv_stream()
    out = _forward(inst, recv)

    assert isinstance(out, tuple)
    hidden_states, aux_hidden_states = out
    assert len(aux_hidden_states) == 3

    h0, r = recv["hidden_states"], recv["residual"]
    torch.testing.assert_close(aux_hidden_states[0], (h0 + r).mean(dim=1))
    torch.testing.assert_close(aux_hidden_states[1], (h0 + 1.0 + r).mean(dim=1))
    torch.testing.assert_close(aux_hidden_states[2], (h0 + 2.0 + r).mean(dim=1))
    torch.testing.assert_close(hidden_states, 2.0 * (h0 + 2.0 + r).mean(dim=1))
    # Boundary 3 (cut recon) + boundaries 4 and 5 in-loop; final reused.
    assert calls["mhc_post"] == 3


def test_penultimate_rank_relays_all_but_cut_boundary(monkeypatch):
    """Sender side of the ...,4,0 layout: rank owning layers [1, 5) relays
    boundaries 3 and 4; boundary 5 (the next rank's cut) is reconstructible
    from the stream and must NOT be sent -- matching the zero-layer receiver's
    make_empty_intermediate_tensors allocation (j < start_layer == 5)."""
    inst, _ = _make_model(
        monkeypatch, start=1, end=5, aux_layers=(3, 4, 5), is_last_rank=False
    )
    out = _forward(inst, _recv_stream())
    assert isinstance(out, IntermediateTensors)
    assert set(out.tensors) == {
        "hidden_states",
        "residual",
        "post_mix",
        "res_mix",
        "aux_hidden_3",
        "aux_hidden_4",
    }


def test_zero_layer_make_empty_intermediate_tensors(monkeypatch):
    """Receiver allocation on the zero-layer rank matches the sender's keys."""
    inst, _ = _make_model(monkeypatch, start=5, end=5, aux_layers=(3, 4, 5))
    tensors = inst.make_empty_intermediate_tensors(
        batch_size=T, dtype=torch.float32, device=torch.device("cpu")
    )
    assert set(tensors.tensors) == {
        "hidden_states",
        "residual",
        "post_mix",
        "res_mix",
        "aux_hidden_3",
        "aux_hidden_4",
    }
