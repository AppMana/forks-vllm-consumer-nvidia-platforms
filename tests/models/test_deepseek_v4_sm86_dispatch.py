# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import pickle
import sys
import types
from types import SimpleNamespace

import pytest
import torch


def _install_fake_flash_mla(monkeypatch):
    fake_flash_mla = types.ModuleType("flash_mla")

    def _unexpected(*args, **kwargs):
        raise AssertionError("unpatched fake flash_mla function was called")

    fake_flash_mla.flash_sparse_mla_decode = _unexpected
    fake_flash_mla.triton_sparse_int8_mla_decode = _unexpected
    monkeypatch.setitem(sys.modules, "flash_mla", fake_flash_mla)
    return fake_flash_mla


def _load_sm86_attention(monkeypatch):
    _install_fake_flash_mla(monkeypatch)
    sys.modules.pop("vllm.models.deepseek_v4.nvidia_sm86.attention", None)
    return importlib.import_module("vllm.models.deepseek_v4.nvidia_sm86.attention")


def _attention_for_decode(module, kv_cache_dtype: str):
    attn = module.DeepseekV4SM86Attention.__new__(module.DeepseekV4SM86Attention)
    attn.kv_cache_dtype = kv_cache_dtype
    attn.swa_cache_layer = SimpleNamespace(kv_cache=torch.empty(1, 1, 584))
    attn.scale = 1.0
    attn.attn_sink = None
    attn.n_local_heads = 2
    attn.compress_ratio = 4
    attn.topk_indices_buffer = None
    attn._sparse_mla_decode_fp8 = "flash_mla.flash_sparse_mla_decode"
    attn._sparse_mla_decode_int8 = "flash_mla.triton_sparse_int8_mla_decode"
    attn._sparse_mla_prefill = (
        "vllm.models.deepseek_v4.nvidia_sm86.triton_kernels.sparse_attention_triton"
    )
    return attn


def _decode_metadata():
    return SimpleNamespace(
        num_decodes=1,
        num_decode_tokens=1,
        decode_swa_indices=torch.zeros((1, 1), dtype=torch.int32),
        decode_swa_lens=torch.ones((1,), dtype=torch.int32),
        block_size=1,
    )


def test_sm86_decode_dispatch_uses_native_flashmla_for_fp8(monkeypatch):
    module = _load_sm86_attention(monkeypatch)
    called = {}

    def fake_flash_sparse_mla_decode(**kwargs):
        called["callable"] = "flash_mla.flash_sparse_mla_decode"
        return torch.ones((1, 2, 4))

    monkeypatch.setattr(module, "flash_sparse_mla_decode", fake_flash_sparse_mla_decode)
    monkeypatch.setattr(module, "triton_sparse_int8_mla_decode", None)

    attn = _attention_for_decode(module, "fp8_ds_mla")
    q = torch.zeros((1, 2, 4))
    output = torch.empty_like(q)

    attn._forward_decode(q, None, _decode_metadata(), None, True, output)

    assert called == {"callable": "flash_mla.flash_sparse_mla_decode"}
    assert torch.equal(output, torch.ones_like(output))


def test_sm86_decode_dispatch_uses_vllm_triton_for_fp8(monkeypatch):
    module = _load_sm86_attention(monkeypatch)
    called = {}

    def fake_decode_sparse_attention_triton(**kwargs):
        called["callable"] = (
            "vllm.models.deepseek_v4.nvidia_sm86.triton_kernels."
            "decode_sparse_attention_triton"
        )
        kwargs["out"].fill_(3.0)

    monkeypatch.setattr(module, "flash_sparse_mla_decode", None)
    monkeypatch.setattr(
        module, "decode_sparse_attention_triton", fake_decode_sparse_attention_triton
    )

    attn = _attention_for_decode(module, "fp8_ds_mla")
    attn._sparse_mla_decode_fp8 = (
        "vllm.models.deepseek_v4.nvidia_sm86.triton_kernels."
        "decode_sparse_attention_triton"
    )
    q = torch.zeros((1, 2, 4))
    output = torch.empty_like(q)

    attn._forward_decode(q, None, _decode_metadata(), None, True, output)

    assert called == {
        "callable": (
            "vllm.models.deepseek_v4.nvidia_sm86.triton_kernels."
            "decode_sparse_attention_triton"
        )
    }
    assert torch.equal(output, torch.full_like(output, 3.0))


def test_sm86_decode_dispatch_uses_named_triton_int8_symbol(monkeypatch):
    module = _load_sm86_attention(monkeypatch)
    called = {}

    def fake_triton_sparse_int8_mla_decode(**kwargs):
        called["callable"] = "flash_mla.triton_sparse_int8_mla_decode"
        return torch.full((1, 2, 4), 2.0)

    monkeypatch.setattr(module, "flash_sparse_mla_decode", None)
    monkeypatch.setattr(
        module, "triton_sparse_int8_mla_decode", fake_triton_sparse_int8_mla_decode
    )
    monkeypatch.setattr(
        module,
        "get_int8_ds_mla_cache_views",
        lambda cache, block_size: (torch.empty(1, 1, 4), torch.empty(1, 1)),
    )

    attn = _attention_for_decode(module, "int8_ds_mla")
    attn.swa_cache_layer = SimpleNamespace(kv_cache=torch.empty(1, 1, 516))
    q = torch.zeros((1, 2, 4))
    output = torch.empty_like(q)

    attn._forward_decode(q, None, _decode_metadata(), None, True, output)

    assert called == {"callable": "flash_mla.triton_sparse_int8_mla_decode"}
    assert torch.equal(output, torch.full_like(output, 2.0))


def test_sm86_prefill_gather_dispatch_follows_cache_dtype(monkeypatch):
    module = _load_sm86_attention(monkeypatch)

    fp8_attn = _attention_for_decode(module, "fp8_ds_mla")
    int8_attn = _attention_for_decode(module, "int8_ds_mla")

    assert fp8_attn._dequantize_gather_k_cache_impl() is module.dequantize_and_gather_k_cache
    assert (
        int8_attn._dequantize_gather_k_cache_impl()
        is module.dequantize_and_gather_int8_ds_mla_cache
    )


def test_dsv4_selector_uses_sm86_attention_class(monkeypatch):
    _install_fake_flash_mla(monkeypatch)
    model_module = importlib.import_module("vllm.models.deepseek_v4.nvidia.model")
    monkeypatch.setattr(
        model_module.current_platform,
        "get_device_capability",
        lambda: SimpleNamespace(major=8),
    )
    vllm_config = SimpleNamespace(attention_config=SimpleNamespace(backend=None))

    selected = model_module._select_dsv4_attn_cls(vllm_config)

    assert selected.__name__ == "DeepseekV4SM86Attention"


def test_deepseek_v4_config_sm86_callables_are_scalar_and_pickleable():
    from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config

    config = DeepseekV4Config(
        deepseek_v4_sm86_sparse_mla_decode_fp8=(
            "vllm.models.deepseek_v4.nvidia_sm86.triton_kernels."
            "decode_sparse_attention_triton"
        ),
        deepseek_v4_sm86_sparse_mla_decode_int8=(
            "flash_mla.triton_sparse_int8_mla_decode"
        ),
        deepseek_v4_sm86_sparse_mla_prefill=(
            "vllm.models.deepseek_v4.nvidia_sm86.triton_kernels."
            "sparse_attention_triton"
        ),
    )

    restored = pickle.loads(pickle.dumps(config))

    assert restored.deepseek_v4_sm86_sparse_mla_decode_fp8 == (
        "vllm.models.deepseek_v4.nvidia_sm86.triton_kernels."
        "decode_sparse_attention_triton"
    )
    assert (
        restored.deepseek_v4_sm86_sparse_mla_decode_int8
        == "flash_mla.triton_sparse_int8_mla_decode"
    )
    assert restored.deepseek_v4_sm86_sparse_mla_prefill == (
        "vllm.models.deepseek_v4.nvidia_sm86.triton_kernels.sparse_attention_triton"
    )


def test_sm86_rejects_mixed_sparse_mla_callables(monkeypatch):
    module = _load_sm86_attention(monkeypatch)
    attn = _attention_for_decode(module, "fp8_ds_mla")
    attn._sparse_mla_decode_fp8 = "flash_mla.triton_sparse_int8_mla_decode"

    with pytest.raises(ValueError, match="fp8 sparse MLA decode"):
        attn._validate_sparse_mla_callables()
