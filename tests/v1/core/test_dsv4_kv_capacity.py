# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request KV charge of a DeepSeek V4 layer stack on the sm_86 / sm_121 path.

The spec set is the 0731 mini (layers with compress ratios 0, 0, 4, 128) with
the int8_ds_mla cache. On trunk the C4 and C128 compressor states are paged
sliding windows whose admission cap grows with the in-flight token budget, so
one request at max_model_len reserves 4,099 C4 blocks and 2,065 C128 blocks of
the 43,552-byte pool slot. The compressor states only ever hold the open
compression group (plus the previous group for the C4 overlap and one
speculative step), so each is a fixed per-request ring.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import CacheConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import (
    _get_kv_cache_bytes_per_block,
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
    get_max_concurrency_for_kv_cache_config,
)
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

pytestmark = pytest.mark.skip_global_cleanup

HEAD_DIM = 512
INDEX_HEAD_DIM = 128
COMPRESS_RATIOS = (0, 0, 4, 128)
MAX_MODEL_LEN = 65536
KV_CACHE_MEMORY = 1 << 30
# Measured on trunk 28776059bd with the mini on one A5000.
TRUNK_SLOT_BYTES = 43552
TRUNK_NUM_BLOCKS = 24654


def _config(
    *,
    num_speculative_tokens: int,
    max_num_batched_tokens: int,
    max_concurrent_batches: int,
    enable_prefix_caching: bool,
):
    config = MagicMock()
    config.cache_config = CacheConfig(
        block_size=256,
        cache_dtype="int8_ds_mla",
        enable_prefix_caching=enable_prefix_caching,
    )
    config.cache_config.kv_cache_layout = "BLHNC"
    config.cache_config.num_gpu_blocks_override = None
    config.attention_config.hisparse_config = None
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    config.speculative_config = None
    config.num_speculative_tokens = num_speculative_tokens
    config.max_in_flight_tokens = max_concurrent_batches * max_num_batched_tokens
    config.use_v2_model_runner = True
    config.model_config.max_model_len = MAX_MODEL_LEN
    config.parallel_config.decode_context_parallel_size = 1
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.use_ubatching = False
    config.compilation_config.static_forward_context = {}
    return config


def _cuda_sm86(monkeypatch) -> None:
    """The fork's RTX 30xx platform: CUDA, compute capability family 80."""
    from vllm.models.deepseek_v4 import compressor

    monkeypatch.setattr(compressor.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(compressor.current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(compressor.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(
        compressor.current_platform,
        "is_device_capability_family",
        lambda family: family == 80,
    )


def _compressor_state_spec(
    monkeypatch, config, head_dim: int, compress_ratio: int, prefix: str
) -> KVCacheSpec:
    from vllm.models.deepseek_v4 import compressor

    monkeypatch.setattr(compressor, "get_current_vllm_config", lambda: config)
    coff = 1 + (compress_ratio == 4)
    cache = compressor.CompressorStateCache(
        state_dim=2 * coff * head_dim,
        dtype=torch.float32,
        compress_ratio=compress_ratio,
        prefix=prefix,
    )
    return cache.get_kv_cache_spec(config)


def _mini_specs(monkeypatch, config) -> dict[str, KVCacheSpec]:
    """Per-layer specs as the DeepSeek V4 layers register them (attention.py,
    sparse_swa.py, compressor.py) for the int8_ds_mla cache."""
    specs: dict[str, KVCacheSpec] = {}
    for layer, ratio in enumerate(COMPRESS_RATIOS):
        prefix = f"model.layers.{layer}.attn"
        specs[f"{prefix}.swa_cache"] = SlidingWindowMLASpec(
            block_size=64,
            num_kv_heads=1,
            head_size=HEAD_DIM,
            dtype=torch.uint8,
            sliding_window=128,
            cache_dtype_str="int8_ds_mla",
            state_content_bytes=528,
            alignment=528,
            model_version="deepseek_v4",
        )
        if ratio <= 1:
            continue
        if ratio == 4:
            specs[f"{prefix}.indexer.k_cache"] = MLAAttentionSpec(
                block_size=256,
                num_kv_heads=1,
                head_size=INDEX_HEAD_DIM + 4,
                dtype=torch.uint8,
                tokens_per_state=ratio,
                alignment=512,
            )
            specs[f"{prefix}.indexer.compressor.state_cache"] = _compressor_state_spec(
                monkeypatch,
                config,
                INDEX_HEAD_DIM,
                ratio,
                f"{prefix}.indexer.compressor.state_cache",
            )
        specs[prefix] = MLAAttentionSpec(
            block_size=256,
            num_kv_heads=1,
            head_size=HEAD_DIM,
            dtype=torch.uint8,
            tokens_per_state=ratio,
            cache_dtype_str="int8_ds_mla",
            alignment=528,
            model_version="deepseek_v4",
            state_content_bytes=528,
        )
        specs[f"{prefix}.compressor.state_cache"] = _compressor_state_spec(
            monkeypatch, config, HEAD_DIM, ratio, f"{prefix}.compressor.state_cache"
        )
    return specs


def _per_request_blocks(config, groups) -> dict[str, int]:
    """Blocks one request at max_model_len charges per group, keyed by the
    group's first layer name (the quantity summed by
    ``get_max_concurrency_for_kv_cache_config``)."""
    return {
        group.layer_names[0]: cdiv(
            group.kv_cache_spec.max_memory_usage_bytes(config),
            group.kv_cache_spec.page_size_bytes,
        )
        for group in groups
    }


def _swa_blocks(max_in_flight_tokens: int) -> int:
    return cdiv(128 - 1 + max_in_flight_tokens, 64) + 1


def _ring_blocks(window: int, rows_per_block: int, num_speculative_tokens: int):
    # The ring keeps the compression window plus one speculative step, rounded
    # up to whole windows (``_c128_ring_capacity`` for C128).
    capacity = cdiv(window + num_speculative_tokens, window) * window
    return capacity // rows_per_block


@pytest.mark.parametrize(
    (
        "num_speculative_tokens",
        "max_num_batched_tokens",
        "max_concurrent_batches",
        "enable_prefix_caching",
    ),
    [
        # The 2026-09-24 in-process measurement.
        (0, 16384, 1, False),
        # Hilton: DSpark 5, 6144 batched tokens, async scheduling (2 batches).
        (5, 6144, 2, False),
        (5, 6144, 2, True),
    ],
)
def test_compressor_states_charge_a_fixed_ring_per_request(
    monkeypatch,
    num_speculative_tokens: int,
    max_num_batched_tokens: int,
    max_concurrent_batches: int,
    enable_prefix_caching: bool,
) -> None:
    _cuda_sm86(monkeypatch)
    config = _config(
        num_speculative_tokens=num_speculative_tokens,
        max_num_batched_tokens=max_num_batched_tokens,
        max_concurrent_batches=max_concurrent_batches,
        enable_prefix_caching=enable_prefix_caching,
    )
    groups = get_kv_cache_groups(config, _mini_specs(monkeypatch, config))

    # The rings pack into the slot the attention groups already need.
    assert _get_kv_cache_bytes_per_block(groups) == TRUNK_SLOT_BYTES
    kv_cache_config = get_kv_cache_config_from_groups(
        config, groups, available_memory=KV_CACHE_MEMORY
    )
    assert kv_cache_config.num_blocks == TRUNK_NUM_BLOCKS

    charges = _per_request_blocks(config, groups)
    in_flight = max_concurrent_batches * max_num_batched_tokens
    c4_ring = _ring_blocks(8, 4, num_speculative_tokens)
    c128_ring = _ring_blocks(128, 8, num_speculative_tokens)
    swa = _swa_blocks(in_flight)
    expected = {
        **{f"model.layers.{i}.attn.swa_cache": swa for i in range(4)},
        "model.layers.2.attn.indexer.k_cache": MAX_MODEL_LEN // 256,
        "model.layers.2.attn.indexer.compressor.state_cache": c4_ring,
        "model.layers.3.attn.compressor.state_cache": c128_ring,
    }
    assert charges == expected

    concurrency = get_max_concurrency_for_kv_cache_config(config, kv_cache_config)
    assert concurrency == TRUNK_NUM_BLOCKS / sum(expected.values())


def test_ring_charge_does_not_grow_with_in_flight_tokens(monkeypatch) -> None:
    """The in-flight budget sizes the paged sliding windows only."""
    _cuda_sm86(monkeypatch)
    charges = []
    for max_concurrent_batches in (1, 12):
        config = _config(
            num_speculative_tokens=7,
            max_num_batched_tokens=16384,
            max_concurrent_batches=max_concurrent_batches,
            enable_prefix_caching=True,
        )
        groups = get_kv_cache_groups(config, _mini_specs(monkeypatch, config))
        charges.append(_per_request_blocks(config, groups))
    for name in (
        "model.layers.2.attn.indexer.compressor.state_cache",
        "model.layers.3.attn.compressor.state_cache",
    ):
        assert charges[0][name] == charges[1][name]
    assert (
        charges[1]["model.layers.0.attn.swa_cache"]
        > charges[0]["model.layers.0.attn.swa_cache"]
    )


def test_cpu_platform_keeps_paged_compressor_state(monkeypatch) -> None:
    from vllm.models.deepseek_v4 import compressor

    monkeypatch.setattr(compressor.current_platform, "is_cuda", lambda: False)
    config = _config(
        num_speculative_tokens=5,
        max_num_batched_tokens=6144,
        max_concurrent_batches=2,
        enable_prefix_caching=True,
    )
    for ratio, head_dim in ((4, HEAD_DIM), (4, INDEX_HEAD_DIM), (128, HEAD_DIM)):
        spec = _compressor_state_spec(
            monkeypatch, config, head_dim, ratio, f"c{ratio}.{head_dim}"
        )
        assert isinstance(spec, SlidingWindowMLASpec)


def test_config_fixture_matches_engine_config() -> None:
    """The mocked in-flight budget is VllmConfig.max_in_flight_tokens."""
    from vllm.config import VllmConfig

    vllm_config = SimpleNamespace(
        max_concurrent_batches=2,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=6144),
    )
    assert VllmConfig.max_in_flight_tokens.fget(vllm_config) == 2 * 6144
