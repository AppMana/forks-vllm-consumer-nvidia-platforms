# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for tools/prep_pp_shards.py."""

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
_spec = importlib.util.spec_from_file_location(
    "prep_pp_shards", _TOOLS_DIR / "prep_pp_shards.py"
)
prep_pp_shards = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep_pp_shards)


def _make_checkpoint(tmp_path, name="src"):
    """A small no-'model.'-prefix checkpoint (real DeepSeek-V4 int4/int8
    naming convention), 4 layers spread across 4 shards plus a dense-only
    shard, matching the real weight_map -> shard_filename shape."""
    src = tmp_path / name
    src.mkdir()
    weight_map = {
        "embed.weight": "model-00001-of-00003.safetensors",
        "head.weight": "model-00001-of-00003.safetensors",
        "layers.0.attn.wq_a.weight": "model-00002-of-00003.safetensors",
        "layers.1.attn.wq_a.weight": "model-00002-of-00003.safetensors",
        "layers.2.attn.wq_a.weight": "model-00003-of-00003.safetensors",
        "layers.3.attn.wq_a.weight": "model-00003-of-00003.safetensors",
    }
    index = {"metadata": {"total_size": 123}, "weight_map": weight_map}
    (src / prep_pp_shards.SAFE_WEIGHTS_INDEX_NAME).write_text(json.dumps(index))
    for shard in set(weight_map.values()):
        (src / shard).write_bytes(b"fake-safetensors-bytes-" + shard.encode())
    (src / "config.json").write_text(json.dumps({"num_hidden_layers": 4}))
    return src, index


class TestComputeCacheKey:
    def test_deterministic(self, tmp_path):
        src, index = _make_checkpoint(tmp_path)
        index_bytes = json.dumps(index).encode()
        k1 = prep_pp_shards.compute_cache_key(str(src), index_bytes, 4)
        k2 = prep_pp_shards.compute_cache_key(str(src), index_bytes, 4)
        assert k1 == k2

    def test_deliberately_rank_independent(self, tmp_path):
        # Ray resolves the leader's one --model path string on every remote
        # worker's node, so a rank-dependent cache key would stage each rank
        # under a different name and break remote workers. Guard the
        # signature so re-adding rank-sensitivity fails here, not in
        # production.
        import inspect

        params = inspect.signature(prep_pp_shards.compute_cache_key).parameters
        assert "pp_rank" not in params

    def test_different_index_content_different_key(self, tmp_path):
        src, index = _make_checkpoint(tmp_path)
        k_a = prep_pp_shards.compute_cache_key(str(src), b"index-v1", 4)
        k_b = prep_pp_shards.compute_cache_key(str(src), b"index-v2", 4)
        assert k_a != k_b

    def test_different_pp_size_different_key(self, tmp_path):
        src, index = _make_checkpoint(tmp_path)
        index_bytes = json.dumps(index).encode()
        k_a = prep_pp_shards.compute_cache_key(str(src), index_bytes, 2)
        k_b = prep_pp_shards.compute_cache_key(str(src), index_bytes, 4)
        assert k_a != k_b


class TestStageShards:
    def test_copies_local_shards_symlinks_rest(self, tmp_path):
        src, index = _make_checkpoint(tmp_path)
        dest_root = tmp_path / "dest"
        # Rank owns layers [0, 2) of 4 -> shard 00002 (layers 0,1) needed,
        # shard 00003 (layers 2,3) symlink-only, shard 00001 (dense) needed.
        dest_dir = prep_pp_shards.stage_shards(
            str(src), str(dest_root), pp_size=2, local_layer_range=(0, 2)
        )

        assert os.path.isfile(
            os.path.join(dest_dir, "model-00001-of-00003.safetensors")
        )
        assert not os.path.islink(
            os.path.join(dest_dir, "model-00001-of-00003.safetensors")
        )
        assert os.path.isfile(
            os.path.join(dest_dir, "model-00002-of-00003.safetensors")
        )
        assert not os.path.islink(
            os.path.join(dest_dir, "model-00002-of-00003.safetensors")
        )

        symlinked = os.path.join(dest_dir, "model-00003-of-00003.safetensors")
        assert os.path.islink(symlinked)
        assert os.path.realpath(symlinked) == os.path.realpath(
            os.path.join(src, "model-00003-of-00003.safetensors")
        )

    def test_index_copied_unmodified(self, tmp_path):
        src, index = _make_checkpoint(tmp_path)
        dest_root = tmp_path / "dest"
        dest_dir = prep_pp_shards.stage_shards(
            str(src), str(dest_root), pp_size=2, local_layer_range=(0, 2)
        )
        with open(os.path.join(dest_dir, prep_pp_shards.SAFE_WEIGHTS_INDEX_NAME)) as f:
            staged_index = json.load(f)
        assert staged_index == index

    def test_filter_duplicate_safetensors_files_succeeds_against_staged_dir(
        self, tmp_path
    ):
        # Direct regression test for the FileNotFoundError failure mode:
        # vLLM's own file-existence check must pass against the staged dir
        # with the original, unmodified index -- every shard filename it
        # references must exist (real file or symlink).
        from vllm.model_executor.model_loader.weight_utils import (
            filter_duplicate_safetensors_files,
        )

        src, _ = _make_checkpoint(tmp_path)
        dest_root = tmp_path / "dest"
        dest_dir = prep_pp_shards.stage_shards(
            str(src), str(dest_root), pp_size=2, local_layer_range=(0, 2)
        )
        hf_weights_files = [
            os.path.join(dest_dir, f)
            for f in os.listdir(dest_dir)
            if f.endswith(".safetensors")
        ]
        result = filter_duplicate_safetensors_files(
            hf_weights_files, dest_dir, prep_pp_shards.SAFE_WEIGHTS_INDEX_NAME
        )
        assert set(result) == set(hf_weights_files)

    def test_config_and_non_safetensors_files_copied(self, tmp_path):
        src, _ = _make_checkpoint(tmp_path)
        dest_root = tmp_path / "dest"
        dest_dir = prep_pp_shards.stage_shards(
            str(src), str(dest_root), pp_size=2, local_layer_range=(0, 2)
        )
        assert os.path.isfile(os.path.join(dest_dir, "config.json"))

    def test_rerun_is_idempotent_no_recopy(self, tmp_path, monkeypatch):
        src, _ = _make_checkpoint(tmp_path)
        dest_root = tmp_path / "dest"
        dest_dir = prep_pp_shards.stage_shards(
            str(src), str(dest_root), pp_size=2, local_layer_range=(0, 2)
        )

        calls = []
        orig_copy2 = prep_pp_shards.shutil.copy2
        monkeypatch.setattr(
            prep_pp_shards.shutil,
            "copy2",
            lambda *a, **k: calls.append(a) or orig_copy2(*a, **k),
        )

        dest_dir_2 = prep_pp_shards.stage_shards(
            str(src), str(dest_root), pp_size=2, local_layer_range=(0, 2)
        )
        assert dest_dir_2 == dest_dir
        assert calls == []

    def test_different_pp_size_different_directory(self, tmp_path):
        src, _ = _make_checkpoint(tmp_path)
        dest_root = tmp_path / "dest"
        dest_a = prep_pp_shards.stage_shards(
            str(src), str(dest_root), pp_size=2, local_layer_range=(0, 2)
        )
        dest_b = prep_pp_shards.stage_shards(
            str(src), str(dest_root), pp_size=4, local_layer_range=(2, 3)
        )
        assert dest_a != dest_b

    def test_same_pp_size_different_rank_same_directory_different_content(
        self, tmp_path
    ):
        # The critical, previously-broken invariant: vllm serve invokes
        # --model once with the LEADER's own staged path, and
        # distributed_executor_backend=ray resolves that SAME path string
        # independently on every remote worker's own node (each backed by
        # its own hostPath directory) -- so every rank of one real run MUST
        # stage under the identical directory NAME, with only the
        # real-copy-vs-symlink CONTENT differing per rank's own
        # local_layer_range.
        src, _ = _make_checkpoint(tmp_path)
        dest_root = tmp_path / "dest"
        dest_rank0 = prep_pp_shards.stage_shards(
            str(src), str(dest_root), pp_size=2, local_layer_range=(0, 2)
        )
        dest_rank1 = prep_pp_shards.stage_shards(
            str(src), str(dest_root), pp_size=2, local_layer_range=(2, 4)
        )
        assert dest_rank0 == dest_rank1
        # Content reflects whichever rank staged it MOST RECENTLY (rank 1):
        # shard 00002 (layers 0,1) is no longer needed, must be a symlink.
        assert os.path.islink(
            os.path.join(dest_rank1, "model-00002-of-00003.safetensors")
        )
        assert not os.path.islink(
            os.path.join(dest_rank1, "model-00003-of-00003.safetensors")
        )

    def test_incomplete_prior_run_is_not_treated_as_complete(self, tmp_path):
        src, _ = _make_checkpoint(tmp_path)
        dest_root = tmp_path / "dest"
        index_bytes = (src / prep_pp_shards.SAFE_WEIGHTS_INDEX_NAME).read_bytes()
        cache_key = prep_pp_shards.compute_cache_key(str(src), index_bytes, 2)
        dest_dir = prep_pp_shards.destination_dir(str(dest_root), cache_key)
        os.makedirs(dest_dir)
        # No .prep-complete marker written -> must be treated as incomplete
        # and (re)staged rather than silently used as-is.
        result = prep_pp_shards.stage_shards(
            str(src), str(dest_root), pp_size=2, local_layer_range=(0, 2)
        )
        assert os.path.isfile(
            os.path.join(result, prep_pp_shards.COMPLETE_MARKER)
        )


class TestResolveLocalLayerRangeStdoutIsolation:
    """Regression test for a real deployed bug: constructing ModelConfig
    logs to stdout via vLLM's own logger, and this script's contract is
    that stdout is exactly the staged path -- callers commonly do
    `path=$(prep_pp_shards.py ...)`. Noise leaking onto stdout silently
    corrupts the captured path with embedded newlines, which broke
    `vllm serve --model "$path"` downstream with an HFValidationError."""

    def test_pp_size_one_short_circuits_without_importing_vllm(self):
        # No vllm import needed at all when PP isn't active -- confirms the
        # cheap path doesn't touch any of this machinery.
        assert prep_pp_shards._resolve_local_layer_range("/nonexistent", 0, 1) is None

    def test_noisy_model_config_construction_does_not_leak_to_stdout(
        self, capsys, monkeypatch
    ):
        fake_config_mod = types.ModuleType("vllm.config")

        class FakeModelConfig:
            def __init__(self, model, trust_remote_code):
                print("INFO [model.py:619] Resolved architecture: Fake")
                print("INFO [model.py:1772] Using max model len 12345")

            def get_total_num_hidden_layers(self):
                return 10

        fake_config_mod.ModelConfig = FakeModelConfig

        fake_dist_utils_mod = types.ModuleType("vllm.distributed.utils")
        fake_dist_utils_mod.get_pp_indices = lambda total, rank, size: (
            rank * (total // size),
            (rank + 1) * (total // size),
        )

        monkeypatch.setitem(sys.modules, "vllm.config", fake_config_mod)
        monkeypatch.setitem(sys.modules, "vllm.distributed.utils", fake_dist_utils_mod)

        result = prep_pp_shards._resolve_local_layer_range(
            "/fake/src", pp_rank=1, pp_size=2, trust_remote_code=True
        )

        assert result == (5, 10)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "Resolved architecture" in captured.err
