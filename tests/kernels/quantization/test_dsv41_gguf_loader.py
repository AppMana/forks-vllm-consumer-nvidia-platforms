# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real GGUF routing and fail-before-write parameter coverage checks."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.model_loader.dsv41_gguf_loader import load_plain_parameters


@pytest.mark.parametrize("mode", list(CUDAGraphMode))
def test_disk_engram_graph_mode_checked_before_model_allocation(monkeypatch, mode):
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader
    from vllm.model_executor.model_loader.dsv41_gguf_loader import Dsv41GGUFModelLoader

    loader = Dsv41GGUFModelLoader.__new__(Dsv41GGUFModelLoader)
    loader.files = ()
    config = SimpleNamespace(
        quant_config=SimpleNamespace(files=(), engram_source=object()),
        compilation_config=SimpleNamespace(cudagraph_mode=mode),
    )
    model_config = SimpleNamespace(multimodal_config=None)
    model = torch.nn.Module()
    calls = []

    def create(*args, **kwargs):
        calls.append(True)
        return model

    monkeypatch.setattr(BaseModelLoader, "create_model", create)
    if mode.has_full_cudagraphs():
        with pytest.raises(ValueError, match="eager lookup boundary"):
            loader.create_model(config, model_config)
        assert not calls
    else:
        assert loader.create_model(config, model_config) is model
        assert calls == [True]


@pytest.fixture(autouse=True)
def local_parameter_rank(monkeypatch):
    from vllm.model_executor import parameter as parameter_module

    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )


def parameter(module, name, shape, dtype=torch.float32):
    module.register_parameter(
        name,
        torch.nn.Parameter(torch.full(shape, -999.0, dtype=dtype), requires_grad=False),
    )


def fixture(tmp_path, *, missing_gate=False, unknown=False):
    import gguf
    from b12x.loader.gguf_dense import index_dense

    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.norm = torch.nn.Module()
    parameter(root.model.norm, "weight", (4,), torch.bfloat16)
    layer = torch.nn.Module()
    root.model.layers = torch.nn.ModuleList([layer])
    parameter(layer, "hc_attn_base", (3,))
    layer.attn = torch.nn.Module()
    parameter(layer.attn, "attn_sink", (3,))
    layer.attn.compressor = torch.nn.Module()
    layer.attn.compressor.fused_wkv_wgate = MergedColumnParallelLinear(
        4, [2, 2], bias=False, params_dtype=torch.bfloat16, disable_tp=True
    )
    layer.attn.compressor.fused_wkv_wgate.weight.data.fill_(-999)
    layer.engram = torch.nn.Module()
    parameter(layer.engram, "q_weight", (2, 4), torch.bfloat16)
    parameter(layer.engram, "k_weight", (2, 4), torch.bfloat16)
    root.compact = torch.nn.Module()
    path = tmp_path / "plain.gguf"
    writer = gguf.GGUFWriter(path, "deepseek41")
    data = {
        "output_norm.weight": np.arange(1, 5, dtype=np.float32),
        "blk.0.hc_attn_base.weight": np.array([0.001, 0.004, -0.7], dtype=np.float32),
        "blk.0.attn_sinks.weight": np.arange(4, dtype=np.float32),
        "blk.0.engram_gate.weight": np.full(
            (2, 4), np.float32(1.0078125) ** 2, dtype=np.float32
        ),
    }
    for name, value in data.items():
        writer.add_tensor(name, value)
    for name, value in [("kv", 2.0), ("gate", 3.0)]:
        if name == "gate" and missing_gate:
            continue
        raw = torch.full((2, 4), value, dtype=torch.bfloat16).view(torch.uint8).numpy()
        writer.add_tensor(
            f"blk.0.attn_compressor_{name}.weight",
            raw,
            raw_dtype=gguf.GGMLQuantizationType.BF16,
        )
    raw = np.zeros((8, 8 * 34), dtype=np.uint8)
    writer.add_tensor(
        "token_embd.weight", raw, raw_dtype=gguf.GGMLQuantizationType.Q8_0
    )
    if unknown:
        writer.add_tensor("mystery.weight", np.zeros(4, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    root.compact.quant_method = SimpleNamespace(
        source=index_dense([path])["token_embd.weight"]
    )
    return root, path, data


def test_load_exact_plain_parameters(tmp_path, default_vllm_config):
    model, path, data = fixture(tmp_path)
    before = {name: p.clone() for name, p in model.named_parameters()}
    preflight = load_plain_parameters(
        model, [path], tp_rank=1, tp_size=2, audit_only=True
    )
    assert not preflight["weights_loaded"]
    assert all(torch.equal(p, before[name]) for name, p in model.named_parameters())
    audit = load_plain_parameters(model, [path], tp_rank=1, tp_size=2)
    layer = model.model.layers[0]
    assert layer.engram.q_weight.dtype == torch.float32
    assert torch.equal(
        layer.engram.q_weight, torch.from_numpy(data["blk.0.engram_gate.weight"])
    )
    assert not torch.equal(
        layer.engram.q_weight, layer.engram.q_weight.bfloat16().float()
    )
    assert torch.equal(layer.engram.k_weight, torch.ones_like(layer.engram.k_weight))
    assert layer.attn.attn_sink.tolist() == [2.0, 3.0, -float("inf")]
    fused = layer.attn.compressor.fused_wkv_wgate.weight
    assert torch.equal(fused[:2], torch.full_like(fused[:2], 2))
    assert torch.equal(fused[2:], torch.full_like(fused[2:], 3))
    assert audit["compact_tensors"] == ["token_embd.weight"]
    assert set(audit["parameters"]) == set(dict(model.named_parameters()))


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"missing_gate": True}, "Incomplete merged"),
        ({"unknown": True}, "Unmapped GGUF"),
    ],
)
def test_preflight_rejects_before_writing(tmp_path, default_vllm_config, kwargs, error):
    model, path, _ = fixture(tmp_path, **kwargs)
    before = {name: p.clone() for name, p in model.named_parameters()}
    with pytest.raises(ValueError, match=error):
        load_plain_parameters(model, [path], tp_rank=0, tp_size=2)
    for name, p in model.named_parameters():
        assert torch.equal(p, before[name])
