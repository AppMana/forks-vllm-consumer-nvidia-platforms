# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for PP weight filtering during model loading."""

import torch

from vllm.model_executor.model_loader.pp_weight_filter import (
    classify_shards,
    parse_layer_id,
    should_skip_pp_weight,
)
from vllm.model_executor.model_loader.weight_utils import (
    safetensors_weights_iterator,
)

# ---------------------------------------------------------------------------
# Unit tests for parse_layer_id
# ---------------------------------------------------------------------------


class TestParseLayerId:
    def test_attention_weight(self):
        name = "model.layers.18.self_attn.q_proj.weight"
        assert parse_layer_id(name) == 18

    def test_expert_weight(self):
        name = "model.layers.18.ffn.experts.101.w1.weight"
        assert parse_layer_id(name) == 18

    def test_layer_zero(self):
        name = "model.layers.0.input_layernorm.weight"
        assert parse_layer_id(name) == 0

    def test_large_layer_id(self):
        name = "model.layers.127.self_attn.q_proj.weight"
        assert parse_layer_id(name) == 127

    def test_embedding_not_a_layer(self):
        name = "model.embed_tokens.weight"
        assert parse_layer_id(name) is None

    def test_final_norm_not_a_layer(self):
        name = "model.norm.weight"
        assert parse_layer_id(name) is None

    def test_lm_head_not_a_layer(self):
        name = "lm_head.weight"
        assert parse_layer_id(name) is None

    def test_mtp_not_a_layer(self):
        # DeepSeek-V4's MTP head uses a separate "mtp." prefix, not nested
        # under ".layers.N." -- must never be treated as a hidden layer.
        name = "model.mtp.0.embed_tokens.weight"
        assert parse_layer_id(name) is None

    def test_bare_layers_without_dot_not_matched(self):
        # "layers" must be followed immediately by ".N." -- an unrelated
        # identifier like "layers_norm" must not false-match.
        name = "layers_norm.weight"
        assert parse_layer_id(name) is None

    def test_no_model_prefix(self):
        # DeepSeek-V4's int4/int8 checkpoint has NO leading "model." prefix
        # -- tensor names start directly with "layers.N.". Regression test
        # for a real bug: the original regex required a leading dot before
        # "layers", which never matches at the start of a string, so this
        # checkpoint's weights were never actually filtered despite the
        # computed layer range being logged correctly.
        assert parse_layer_id("layers.18.attn.attn_sink") == 18
        assert parse_layer_id("layers.0.ffn.experts.0.w1.weight") == 0

    def test_no_model_prefix_dense_weights(self):
        # Real DeepSeek-V4 int4/int8 checkpoint dense weight names.
        assert parse_layer_id("embed.weight") is None
        assert parse_layer_id("head.weight") is None
        assert parse_layer_id("hc_head_base") is None


# ---------------------------------------------------------------------------
# Unit tests for should_skip_pp_weight
# ---------------------------------------------------------------------------


class TestShouldSkipPpWeight:
    def test_no_filter(self):
        assert not should_skip_pp_weight("anything", None)

    def test_local_layer_not_skipped(self):
        local_range = (10, 20)
        assert not should_skip_pp_weight(
            "model.layers.15.self_attn.q_proj.weight", local_range
        )

    def test_layer_before_range_skipped(self):
        local_range = (10, 20)
        assert should_skip_pp_weight(
            "model.layers.5.self_attn.q_proj.weight", local_range
        )

    def test_layer_after_range_skipped(self):
        local_range = (10, 20)
        assert should_skip_pp_weight(
            "model.layers.25.self_attn.q_proj.weight", local_range
        )

    def test_range_start_inclusive(self):
        local_range = (10, 20)
        assert not should_skip_pp_weight(
            "model.layers.10.self_attn.q_proj.weight", local_range
        )

    def test_range_end_exclusive(self):
        local_range = (10, 20)
        assert should_skip_pp_weight(
            "model.layers.20.self_attn.q_proj.weight", local_range
        )

    def test_dense_weight_never_skipped(self):
        local_range = (10, 20)
        assert not should_skip_pp_weight("model.embed_tokens.weight", local_range)
        assert not should_skip_pp_weight("model.norm.weight", local_range)
        assert not should_skip_pp_weight("lm_head.weight", local_range)

    def test_mtp_weight_never_skipped(self):
        # Even though rank owning layers [10,20) doesn't own layer 0's
        # transformer block, MTP weights aren't ".layers.N."-shaped and
        # must pass through untouched.
        local_range = (10, 20)
        assert not should_skip_pp_weight("model.mtp.0.embed_tokens.weight", local_range)

    def test_expert_weight_layer_filtered_same_as_dense(self):
        local_range = (10, 20)
        assert not should_skip_pp_weight(
            "model.layers.15.ffn.experts.42.w1.weight", local_range
        )
        assert should_skip_pp_weight(
            "model.layers.5.ffn.experts.42.w1.weight", local_range
        )


# ---------------------------------------------------------------------------
# Unit tests for classify_shards
# ---------------------------------------------------------------------------


class TestClassifyShards:
    def test_shard_with_local_layer_needs_copy(self):
        weight_map = {
            "model.layers.15.self_attn.q_proj.weight": "shard-a.safetensors",
        }
        result = classify_shards(weight_map, local_layer_range=(10, 20))
        assert result == {"shard-a.safetensors": True}

    def test_shard_with_only_nonlocal_layers_symlink_only(self):
        weight_map = {
            "model.layers.5.self_attn.q_proj.weight": "shard-a.safetensors",
            "model.layers.25.self_attn.q_proj.weight": "shard-a.safetensors",
        }
        result = classify_shards(weight_map, local_layer_range=(10, 20))
        assert result == {"shard-a.safetensors": False}

    def test_mixed_shard_needs_copy_if_any_tensor_local(self):
        # One shard holding both a local and a non-local layer's tensors --
        # must be real-copied, since the local tensor's bytes are needed.
        weight_map = {
            "model.layers.5.self_attn.q_proj.weight": "shard-a.safetensors",
            "model.layers.15.self_attn.q_proj.weight": "shard-a.safetensors",
        }
        result = classify_shards(weight_map, local_layer_range=(10, 20))
        assert result == {"shard-a.safetensors": True}

    def test_dense_weight_forces_copy_of_its_shard(self):
        weight_map = {
            "model.embed_tokens.weight": "shard-a.safetensors",
            "model.layers.5.self_attn.q_proj.weight": "shard-b.safetensors",
        }
        result = classify_shards(weight_map, local_layer_range=(10, 20))
        assert result == {"shard-a.safetensors": True, "shard-b.safetensors": False}

    def test_no_filter_all_shards_need_copy(self):
        weight_map = {
            "model.layers.5.self_attn.q_proj.weight": "shard-a.safetensors",
            "model.layers.25.self_attn.q_proj.weight": "shard-b.safetensors",
        }
        result = classify_shards(weight_map, local_layer_range=None)
        assert result == {"shard-a.safetensors": True, "shard-b.safetensors": True}

    def test_no_model_prefix_checkpoint_shape(self):
        weight_map = {
            "layers.5.attn.wq_a.weight": "shard-a.safetensors",
            "layers.15.attn.wq_a.weight": "shard-b.safetensors",
            "embed.weight": "shard-c.safetensors",
        }
        result = classify_shards(weight_map, local_layer_range=(10, 20))
        assert result == {
            "shard-a.safetensors": False,
            "shard-b.safetensors": True,
            "shard-c.safetensors": True,
        }


# ---------------------------------------------------------------------------
# Integration test: safetensors_weights_iterator with PP filtering
# ---------------------------------------------------------------------------


class TestSafetensorsWeightsIteratorWithPpFilter:
    """Create a synthetic multi-layer safetensors file and verify PP
    filtering skips non-local layers while keeping dense + local weights."""

    @staticmethod
    def _make_synthetic_files(tmp_path, num_layers: int):
        from safetensors.torch import save_file

        tensors = {}
        tensors["model.embed_tokens.weight"] = torch.randn(100, 64)
        tensors["model.norm.weight"] = torch.randn(64)
        tensors["lm_head.weight"] = torch.randn(100, 64)
        for layer_id in range(num_layers):
            tensors[f"model.layers.{layer_id}.self_attn.q_proj.weight"] = torch.randn(
                64, 64
            )
            tensors[f"model.layers.{layer_id}.input_layernorm.weight"] = torch.randn(
                64
            )

        filepath = str(tmp_path / "model-00001-of-00001.safetensors")
        save_file(tensors, filepath)
        return [filepath], tensors

    def test_no_filter_returns_all(self, tmp_path):
        files, expected = self._make_synthetic_files(tmp_path, num_layers=10)
        loaded = dict(safetensors_weights_iterator(files, False))
        assert set(loaded.keys()) == set(expected.keys())

    def test_pp_filter_keeps_only_local_layers(self, tmp_path):
        files, expected = self._make_synthetic_files(tmp_path, num_layers=10)
        # Rank owns layers [3, 6) of 10.
        local_range = (3, 6)
        loaded = dict(
            safetensors_weights_iterator(files, False, local_layer_range=local_range)
        )

        for name in loaded:
            lid = parse_layer_id(name)
            if lid is not None:
                assert 3 <= lid < 6, f"Non-local layer {lid} was loaded"

        # 3 layers x 2 weights each
        layer_names = [n for n in loaded if parse_layer_id(n) is not None]
        assert len(layer_names) == 3 * 2

        # Dense weights always present regardless of PP filtering.
        assert "model.embed_tokens.weight" in loaded
        assert "model.norm.weight" in loaded
        assert "lm_head.weight" in loaded

    def test_pp_filter_full_coverage_across_ranks(self, tmp_path):
        files, expected = self._make_synthetic_files(tmp_path, num_layers=10)
        from vllm.distributed.utils import get_pp_indices

        pp_size = 4
        all_layer_names: set[str] = set()
        for pp_rank in range(pp_size):
            local_range = get_pp_indices(10, pp_rank, pp_size)
            loaded = dict(
                safetensors_weights_iterator(
                    files, False, local_layer_range=local_range
                )
            )
            layer_names = {n for n in loaded if parse_layer_id(n) is not None}
            assert all_layer_names.isdisjoint(layer_names)
            all_layer_names |= layer_names

        expected_layer_names = {
            n for n in expected if parse_layer_id(n) is not None
        }
        assert all_layer_names == expected_layer_names

    def test_tensor_values_match(self, tmp_path):
        files, _ = self._make_synthetic_files(tmp_path, num_layers=10)
        all_weights = dict(safetensors_weights_iterator(files, False))

        local_range = (3, 6)
        filtered = dict(
            safetensors_weights_iterator(files, False, local_layer_range=local_range)
        )
        for name, tensor in filtered.items():
            assert torch.equal(tensor, all_weights[name]), f"Tensor mismatch for {name}"


class TestSafetensorsWeightsIteratorNoModelPrefix:
    """Regression coverage for the real DeepSeek-V4 int4/int8 checkpoint
    naming convention: tensor names start directly with "layers.N." with no
    "model." prefix. This exact shape previously slipped through
    should_skip_pp_weight silently (parse_layer_id always returned None),
    so the filter never actually skipped anything even though the computed
    layer range was correct and logged."""

    @staticmethod
    def _make_synthetic_files(tmp_path, num_layers: int):
        from safetensors.torch import save_file

        tensors = {}
        tensors["embed.weight"] = torch.randn(100, 64)
        tensors["head.weight"] = torch.randn(100, 64)
        for layer_id in range(num_layers):
            tensors[f"layers.{layer_id}.attn.wq_a.weight"] = torch.randn(64, 64)
            tensors[f"layers.{layer_id}.attn_norm.weight"] = torch.randn(64)

        filepath = str(tmp_path / "model-00001-of-00001.safetensors")
        save_file(tensors, filepath)
        return [filepath], tensors

    def test_pp_filter_keeps_only_local_layers(self, tmp_path):
        files, expected = self._make_synthetic_files(tmp_path, num_layers=10)
        local_range = (3, 6)
        loaded = dict(
            safetensors_weights_iterator(files, False, local_layer_range=local_range)
        )

        for name in loaded:
            lid = parse_layer_id(name)
            if lid is not None:
                assert 3 <= lid < 6, f"Non-local layer {lid} was loaded"

        layer_names = [n for n in loaded if parse_layer_id(n) is not None]
        assert len(layer_names) == 3 * 2, (
            f"expected 6 local-layer tensors, got {len(layer_names)}: "
            f"{layer_names} -- filter is not actually skipping anything"
        )
        assert "embed.weight" in loaded
        assert "head.weight" in loaded
