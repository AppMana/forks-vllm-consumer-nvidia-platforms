# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight loading on a zero-layer PP rank (VLLM_PP_LAYER_PARTITION tail
``...,4,0``).

test_pp_zero_layer_last_rank.py pins down the zero-layer *forward* tail; this
file pins down the *load* path that crashed live on probe-113 (PP=12,
partition ``3,4,4,4,4,4,4,4,4,4,4,0``): rank 11 owns zero decoder layers, the
PP weight filter strips every ``layers.N.*`` tensor before it leaves disk, and
``DeepseekV4Model.load_weights`` then calls ``get_expert_mapping`` whose
``next(iter(islice(self.layers, start, end)))`` raises StopIteration on the
empty slice. That StopIteration surfaces inside ``AutoWeightsLoader``'s
``_load_module`` generator, so PEP 479 turns it into
``RuntimeError: generator raised StopIteration``.

The loading test goes through the real chain: a synthetic sharded safetensors
checkpoint on disk -> ``safetensors_weights_iterator`` with a zero-layer
``local_layer_range`` -> the production weights mapper ->
``AutoWeightsLoader(parent)`` exactly as
``DeepseekV4ForCausalLM.load_weights`` builds it. The rank's own tensors
(final norm, lm_head) must load; ``mtp.*`` stays skipped for the target model
(the mapper maps it to None)
(the DSpark draft loads them through its own ``DeepseekV4MTP.load_weights``,
whose mtp layer dict is never empty on the draft rank).

CPU-only: the model instance is built by bypassing ``__init__`` (same idiom as
test_pp_zero_layer_last_rank.py).
"""

from types import SimpleNamespace

import torch
import torch.nn as nn
from safetensors.torch import save_file

from vllm.model_executor.model_loader.weight_utils import (
    safetensors_weights_iterator,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, PPMissingLayer
from vllm.models.deepseek_v4.nvidia import model as m

H = 8
V = 16
NUM_LAYERS = 5


class _ParamModule(nn.Module):
    def __init__(self, *shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(*shape))


def _make_zero_layer_model(monkeypatch) -> m.DeepseekV4Model:
    monkeypatch.setattr(m, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(m, "get_tensor_model_parallel_rank", lambda: 0)
    inst = m.DeepseekV4Model.__new__(m.DeepseekV4Model)
    nn.Module.__init__(inst)
    inst.config = SimpleNamespace(
        hidden_size=H,
        num_attention_heads=2,
        n_routed_experts=4,
    )
    inst.quant_config = None
    inst.parallel_config = SimpleNamespace(use_sequence_parallel_moe=False)
    inst.start_layer = NUM_LAYERS
    inst.end_layer = NUM_LAYERS
    inst.layers = nn.ModuleList([PPMissingLayer() for _ in range(NUM_LAYERS)])
    inst.embed_tokens = PPMissingLayer()
    inst.norm = _ParamModule(H)
    return inst


class _CausalStandIn(nn.Module):
    """Structural stand-in for DeepseekV4ForCausalLM: ``model`` child plus the
    last-rank ``lm_head``."""

    def __init__(self, model: m.DeepseekV4Model):
        super().__init__()
        self.model = model
        self.lm_head = _ParamModule(V, H)


def _write_checkpoint(tmp_path) -> list[str]:
    """Two-shard checkpoint in the DSV4 naming convention (no leading
    ``model.`` prefix): one backbone shard, one shared shard. On the staged
    zero-layer rank the backbone shard is a symlink whose header is still
    readable -- the PP filter must skip its tensors before get_tensor()."""
    backbone = {}
    for lid in range(NUM_LAYERS):
        backbone[f"layers.{lid}.attn.wq_a.weight"] = torch.full((H, H), float(lid))
    backbone["layers.1.ffn.experts.0.w1.weight"] = torch.ones(H, H)
    shared = {
        "embed.weight": torch.ones(V, H),
        "norm.weight": torch.arange(H, dtype=torch.float32),
        "head.weight": torch.full((V, H), 3.0),
        "mtp.0.norm.weight": torch.ones(H),
        "mtp.1.norm.weight": torch.ones(H),
        "mtp.2.norm.weight": torch.ones(H),
    }
    files = []
    for name, tensors in (
        ("model-00001-of-00002.safetensors", backbone),
        ("model-00002-of-00002.safetensors", shared),
    ):
        path = str(tmp_path / name)
        save_file(tensors, path)
        files.append(path)
    return files


def test_get_expert_mapping_zero_layers(monkeypatch):
    """No local layers -> no local experts -> empty mapping, no StopIteration."""
    inst = _make_zero_layer_model(monkeypatch)
    assert inst.get_expert_mapping() == []


def test_zero_layer_rank_loads_sharded_checkpoint(monkeypatch, tmp_path):
    """Real load path: PP-filtered safetensors iterator + AutoWeightsLoader.

    Reproduces the probe-113 rank-11 crash (RuntimeError: generator raised
    StopIteration) on a broken tree; on a fixed tree the rank's own tensors
    load and everything else is filtered or skipped.
    """
    inst = _make_zero_layer_model(monkeypatch)
    parent = _CausalStandIn(inst)
    files = _write_checkpoint(tmp_path)

    weights = safetensors_weights_iterator(
        files,
        use_tqdm_on_load=False,
        local_layer_range=(NUM_LAYERS, NUM_LAYERS),
        is_first_pipeline_rank=False,
        is_last_pipeline_rank=True,
    )
    loader = AutoWeightsLoader(parent)
    loaded = loader.load_weights(
        weights, mapper=m._make_deepseek_v4_weights_mapper("int8")
    )

    assert "model.norm.weight" in loaded
    assert "lm_head.weight" in loaded
    torch.testing.assert_close(
        inst.norm.weight.data, torch.arange(H, dtype=torch.float32)
    )
    torch.testing.assert_close(
        parent.lm_head.weight.data, torch.full((V, H), 3.0)
    )
    # Backbone and expert tensors never reach the model on this rank, and the
    # target model never loads mtp.* (the DSpark draft owns those).
    assert not any(".layers." in name or "mtp." in name for name in loaded)


def test_zero_layer_iterator_yields_no_backbone(tmp_path):
    """The PP filter starves the iterator of every layers.N.* tensor but the
    shared tensors still flow."""
    files = _write_checkpoint(tmp_path)
    names = [
        name
        for name, _ in safetensors_weights_iterator(
            files,
            use_tqdm_on_load=False,
            local_layer_range=(NUM_LAYERS, NUM_LAYERS),
            is_first_pipeline_rank=False,
            is_last_pipeline_rank=True,
        )
    ]
    assert names == sorted(names)  # safetensors key order within the shard
    assert all(not n.startswith("layers.") for n in names)
    assert set(names) == {
        "norm.weight",
        "head.weight",
        "mtp.0.norm.weight",
        "mtp.1.norm.weight",
        "mtp.2.norm.weight",
    }
