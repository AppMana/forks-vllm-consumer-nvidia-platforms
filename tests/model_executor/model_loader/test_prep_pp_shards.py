# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Content-addressed shard staging (tools/prep_pp_shards.py).

The staged layout is two-level:
- ``<dest_root>/store/<checkpoint_id>/<shard>``: each real-copied shard
  exists exactly once per checkpoint, shared by every config.
- ``<dest_root>/<config_key>/``: per-config directory holding metadata
  copies plus one symlink per shard -- owned shards point into the store,
  foreign shards point back at the source directory.

A config change (partition, pp_size) must therefore never re-copy a shard
that any previous config already brought into the store.
"""

import importlib.util
import json
import os
import sys

import pytest

_TOOLS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "tools", "prep_pp_shards.py"
)
_spec = importlib.util.spec_from_file_location("prep_pp_shards", _TOOLS_PATH)
prep = importlib.util.module_from_spec(_spec)
sys.modules["prep_pp_shards"] = prep
_spec.loader.exec_module(prep)


def _make_checkpoint(tmp_path, num_layers=6, shards_per_layer=1):
    src = tmp_path / "snapshot"
    src.mkdir()
    weight_map = {}
    for lid in range(num_layers):
        shard = f"model-{lid:05d}.safetensors"
        weight_map[f"layers.{lid}.mlp.weight"] = shard
        (src / shard).write_bytes(b"x" * (1000 + lid))
    shard = "model-shared.safetensors"
    weight_map["embed_tokens.weight"] = shard
    (src / shard).write_bytes(b"e" * 4096)
    index = {"weight_map": weight_map}
    (src / prep.SAFE_WEIGHTS_INDEX_NAME).write_text(json.dumps(index))
    (src / "config.json").write_text("{}")
    return str(src)


def _stage(src, dest_root, layer_range, pp_size=3, partition=None):
    return prep.stage_shards(
        src,
        str(dest_root),
        pp_size,
        layer_range,
        partition=partition,
    )


def _copied_shards(dest_dir):
    out = {}
    for name in os.listdir(dest_dir):
        if not name.endswith(".safetensors"):
            continue
        path = os.path.join(dest_dir, name)
        assert os.path.islink(path), f"{name} must be a symlink in config dirs"
        out[name] = os.path.realpath(path)
    return out


def test_owned_shards_link_into_store_foreign_into_source(tmp_path):
    src = _make_checkpoint(tmp_path)
    dest = _stage(src, tmp_path / "cache", (0, 2))
    links = _copied_shards(dest)
    store_root = os.path.join(str(tmp_path / "cache"), "store")
    for lid in range(6):
        shard = f"model-{lid:05d}.safetensors"
        if lid < 2:
            assert links[shard].startswith(store_root), shard
            assert os.path.isfile(links[shard])
        else:
            assert links[shard] == os.path.realpath(
                os.path.join(src, shard)
            ), shard
    # embedding shard is owned by every rank's dir (never skipped)
    assert links["model-shared.safetensors"].startswith(store_root)


def test_config_change_reuses_store_zero_new_copies(tmp_path):
    src = _make_checkpoint(tmp_path)
    dest_root = tmp_path / "cache"
    _stage(src, dest_root, (0, 4), partition="4,1,1")

    copies = {"n": 0}
    real_copy = prep.shutil.copy2

    def counting_copy(a, b, **kw):
        if str(a).endswith(".safetensors") and "store" in str(b):
            copies["n"] += 1
        return real_copy(a, b, **kw)

    prep.shutil.copy2 = counting_copy
    try:
        # New partition, overlapping ownership 0..2 subset of 0..4: every
        # owned shard is already in the store -> zero new shard copies.
        dest2 = _stage(src, dest_root, (0, 2), partition="2,2,2")
    finally:
        prep.shutil.copy2 = real_copy
    assert copies["n"] == 0
    links = _copied_shards(dest2)
    store_root = os.path.join(str(dest_root), "store")
    assert links["model-00000.safetensors"].startswith(store_root)


def test_partition_in_config_key(tmp_path):
    src = _make_checkpoint(tmp_path)
    dest_root = tmp_path / "cache"
    d1 = _stage(src, dest_root, (0, 2), partition="2,2,2")
    d2 = _stage(src, dest_root, (0, 2), partition="4,1,1")
    assert d1 != d2


def test_idempotent_fast_path(tmp_path):
    src = _make_checkpoint(tmp_path)
    dest_root = tmp_path / "cache"
    d1 = _stage(src, dest_root, (0, 2))
    marker = os.path.join(d1, prep.COMPLETE_MARKER)
    assert os.path.exists(marker)
    stamp = os.stat(marker).st_mtime_ns
    d2 = _stage(src, dest_root, (0, 2))
    assert d1 == d2
    assert os.stat(marker).st_mtime_ns == stamp


def test_adopts_legacy_real_files_by_hardlink(tmp_path):
    """A pre-content-store config dir holds real shard files; staging must
    harvest them into the store via hardlink instead of re-copying from
    source."""
    src = _make_checkpoint(tmp_path)
    dest_root = tmp_path / "cache"
    legacy = dest_root / "deadbeefdeadbeef"
    legacy.mkdir(parents=True)
    shard = "model-00000.safetensors"
    legacy_file = legacy / shard
    legacy_file.write_bytes((tmp_path / "snapshot" / shard).read_bytes())

    copies = {"n": 0}
    real_copy = prep.shutil.copy2

    def counting_copy(a, b, **kw):
        if str(a).endswith(shard):
            copies["n"] += 1
        return real_copy(a, b, **kw)

    prep.shutil.copy2 = counting_copy
    try:
        dest = _stage(src, dest_root, (0, 1))
    finally:
        prep.shutil.copy2 = real_copy
    assert copies["n"] == 0
    links = _copied_shards(dest)
    assert os.stat(links[shard]).st_nlink >= 2  # hardlinked, not copied


def test_gc_removes_stale_configs_and_orphaned_store_files(tmp_path):
    src = _make_checkpoint(tmp_path)
    dest_root = tmp_path / "cache"
    d_old = _stage(src, dest_root, (0, 4), partition="4,1,1")
    d_new = _stage(src, dest_root, (4, 6), partition="1,1,4")
    assert d_old != d_new

    removed = prep.gc_dest_root(str(dest_root), keep_dir=d_new)
    assert not os.path.exists(d_old)
    assert os.path.exists(d_new)
    assert removed["config_dirs"] >= 1
    # store keeps only shards referenced by the surviving config
    store_files = []
    for root, _dirs, files in os.walk(os.path.join(str(dest_root), "store")):
        store_files.extend(files)
    kept_links = set(
        os.path.basename(t)
        for t in _copied_shards(d_new).values()
        if "store" in t
    )
    assert set(store_files) == kept_links
    # surviving config still verifies complete after GC
    d_again = _stage(src, dest_root, (4, 6), partition="1,1,4")
    assert d_again == d_new


def test_index_and_metadata_are_real_copies(tmp_path):
    src = _make_checkpoint(tmp_path)
    dest = _stage(src, tmp_path / "cache", (0, 2))
    for name in (prep.SAFE_WEIGHTS_INDEX_NAME, "config.json"):
        path = os.path.join(dest, name)
        assert os.path.isfile(path) and not os.path.islink(path)
    staged_index = json.load(open(os.path.join(dest, prep.SAFE_WEIGHTS_INDEX_NAME)))
    original_index = json.load(open(os.path.join(src, prep.SAFE_WEIGHTS_INDEX_NAME)))
    assert staged_index == original_index


def test_every_index_shard_resolves(tmp_path):
    src = _make_checkpoint(tmp_path)
    dest = _stage(src, tmp_path / "cache", (2, 4))
    index = json.load(open(os.path.join(dest, prep.SAFE_WEIGHTS_INDEX_NAME)))
    for shard in set(index["weight_map"].values()):
        assert os.path.exists(os.path.join(dest, shard)), shard
