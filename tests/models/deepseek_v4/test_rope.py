# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DeepSeek V4 per-layer rotary embedding selection."""

import types
from typing import Any

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT, RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    DeepseekScalingRotaryEmbedding,
    DeepseekV4ScalingRotaryEmbedding,
)
from vllm.models.deepseek_v4 import attention as attention_module
from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope

NUM_HIDDEN_LAYERS = 43


@pytest.fixture(autouse=True)
def clear_rope_cache():
    _ROPE_DICT.clear()
    yield
    _ROPE_DICT.clear()


def make_config(compress_ratios: list[int]) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        compress_ratios=compress_ratios,
        hidden_size=64,
        num_attention_heads=1,
        q_lora_rank=32,
        o_lora_rank=32,
        head_dim=64,
        qk_rope_head_dim=16,
        o_groups=1,
        sliding_window=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        compress_rope_theta=1000000.0,
        max_position_embeddings=4096,
        rope_parameters={
            "rope_type": "yarn",
            "factor": 8.0,
            "original_max_position_embeddings": 512,
            "beta_fast": 32,
            "beta_slow": 1,
        },
    )


class ConcreteAttention(attention_module.DeepseekV4Attention):
    backend_cls = object

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def forward_mqa(self, *args: Any, **kwargs: Any) -> None:
        pass

    def _o_proj(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError


class DummyModule(torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()


class DummySWACache(DummyModule):
    def __init__(self, *args: Any, prefix: str, **kwargs: Any) -> None:
        super().__init__()
        self.prefix = prefix
        self.kv_cache = torch.tensor([])


def make_vllm_config(config: types.SimpleNamespace) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_config=config,
            max_model_len=config.max_position_embeddings,
        ),
        quant_config=None,
        cache_config=types.SimpleNamespace(cache_dtype="fp8", block_size=16),
        scheduler_config=types.SimpleNamespace(max_num_batched_tokens=128),
        compilation_config=types.SimpleNamespace(static_forward_context={}),
        use_v2_model_runner=True,
        kernel_config=types.SimpleNamespace(enable_jit_warmup=False),
    )


@pytest.mark.parametrize("draft_offset", [0, 1, 2])
def test_zero_draft_ratio_selects_unscaled_rope_without_zero_cache_ratio(
    draft_offset: int,
) -> None:
    config = make_config([1] * NUM_HIDDEN_LAYERS + [0, 0, 0])

    ratio, unscaled = attention_module.resolve_layer_compress_ratio(
        config, NUM_HIDDEN_LAYERS + draft_offset
    )

    assert ratio == 1
    assert unscaled is True


@pytest.mark.parametrize("draft_offset", [0, 1, 2])
def test_attention_builds_zero_ratio_draft_with_unscaled_rope_and_safe_cache_ratio(
    monkeypatch: pytest.MonkeyPatch,
    draft_offset: int,
) -> None:
    config = make_config([1] * NUM_HIDDEN_LAYERS + [0, 0, 0])
    rope_options: dict[str, Any] = {}

    def build_rope(*args: Any, **kwargs: Any) -> DummyModule:
        rope_options.update(kwargs)
        return DummyModule()

    monkeypatch.setattr(
        attention_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(attention_module, "activate_kernel_config", lambda config: None)
    monkeypatch.setattr(
        attention_module, "resolve_kernel_config_from_hf_config", lambda config: None
    )
    monkeypatch.setattr(attention_module, "MergedColumnParallelLinear", DummyModule)
    monkeypatch.setattr(attention_module, "ColumnParallelLinear", DummyModule)
    monkeypatch.setattr(attention_module, "RowParallelLinear", DummyModule)
    monkeypatch.setattr(attention_module, "RMSNorm", DummyModule)
    monkeypatch.setattr(attention_module, "DeepseekV4SWACache", DummySWACache)
    monkeypatch.setattr(attention_module, "build_deepseek_v4_rope", build_rope)
    monkeypatch.setattr(torch.cuda, "Event", DummyModule)

    layer_id = NUM_HIDDEN_LAYERS + draft_offset
    attention = ConcreteAttention(
        make_vllm_config(config),
        prefix=f"model.layers.{layer_id}.attn",
    )

    assert attention.compress_ratio == 1
    assert attention.compressor is None
    assert rope_options["compress_ratio"] == 1
    assert rope_options["use_unscaled_rope"] is True


@pytest.mark.parametrize(
    ("compress_ratios", "layer_id", "expected_ratio", "expected_unscaled"),
    [
        ([0, 4] + [1] * 41, 0, 1, False),
        ([0, 4] + [1] * 41, 1, 4, False),
        ([1] * NUM_HIDDEN_LAYERS + [4], NUM_HIDDEN_LAYERS, 4, False),
        ([1] * NUM_HIDDEN_LAYERS, NUM_HIDDEN_LAYERS, 1, False),
        ([1] * NUM_HIDDEN_LAYERS, NUM_HIDDEN_LAYERS + 1, 1, False),
    ],
)
def test_layer_compress_ratio_fallbacks(
    compress_ratios: list[int],
    layer_id: int,
    expected_ratio: int,
    expected_unscaled: bool,
) -> None:
    ratio, unscaled = attention_module.resolve_layer_compress_ratio(
        make_config(compress_ratios), layer_id
    )

    assert ratio == expected_ratio
    assert unscaled is expected_unscaled
    assert ratio >= 1


def test_unscaled_draft_rope_is_plain_fp32(default_vllm_config) -> None:
    config = make_config([1] * NUM_HIDDEN_LAYERS + [0, 0, 0])
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        rope = build_deepseek_v4_rope(
            config,
            head_dim=64,
            rope_head_dim=64,
            max_position_embeddings=config.max_position_embeddings,
            compress_ratio=1,
            use_unscaled_rope=True,
        )
    finally:
        torch.set_default_dtype(old_dtype)

    assert type(rope) is RotaryEmbedding
    assert not isinstance(rope, DeepseekScalingRotaryEmbedding)
    assert rope.cos_sin_cache.dtype == torch.float32


def test_scaled_rope_remains_deepseek_v4_yarn(default_vllm_config) -> None:
    config = make_config([1] * NUM_HIDDEN_LAYERS)

    rope = build_deepseek_v4_rope(
        config,
        head_dim=64,
        rope_head_dim=64,
        max_position_embeddings=config.max_position_embeddings,
        compress_ratio=1,
    )

    assert isinstance(rope, DeepseekV4ScalingRotaryEmbedding)


def test_layer_rope_construction_does_not_mutate_shared_config(
    default_vllm_config,
) -> None:
    config = make_config([1] * NUM_HIDDEN_LAYERS + [0, 0, 0])
    original = dict(config.rope_parameters)

    for compress_ratio, use_unscaled_rope in ((1, True), (1, False), (4, False)):
        build_deepseek_v4_rope(
            config,
            head_dim=64,
            rope_head_dim=64,
            max_position_embeddings=config.max_position_embeddings,
            compress_ratio=compress_ratio,
            use_unscaled_rope=use_unscaled_rope,
        )
        assert config.rope_parameters == original
