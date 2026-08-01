# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Partial local HF cache staging (tools/prep_pp_shards.py).

The node-local cache is a standard HF hub cache layout:

    <dest-root>/hub/models--<org>--<name>/
        blobs/<etag>                  # content-addressed, owned shards only
        snapshots/<rev>/<filename>    # symlinks: owned -> local blob,
                                      #           foreign -> source blob
        refs/main                     # <rev>

Staging materializes exactly one snapshot per revision. Ownership changes
(partition/pp_size) repoint symlinks in place and copy only blobs the local
blobs/ directory does not already hold -- the blob names are the source
cache's own content hashes, so dedupe across configs is inherited from the
HF layout instead of reinvented.
"""

import importlib.util
import json
import os
import sys

_TOOLS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "tools", "prep_pp_shards.py"
)
_spec = importlib.util.spec_from_file_location("prep_pp_shards", _TOOLS_PATH)
prep = importlib.util.module_from_spec(_spec)
sys.modules["prep_pp_shards"] = prep
_spec.loader.exec_module(prep)

REV = "abc123def456"


def _make_source_hf_cache(tmp_path, num_layers=6):
    """Build a source cache in real HF layout: snapshot symlinks -> blobs."""
    repo = tmp_path / "hf-cache" / "hub" / "models--appmana--dsv4-test"
    blobs = repo / "blobs"
    snap = repo / "snapshots" / REV
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(REV)

    weight_map = {}
    for lid in range(num_layers):
        shard = f"model-{lid:05d}.safetensors"
        etag = f"etag{lid:056d}"
        (blobs / etag).write_bytes(b"x" * (1000 + lid))
        os.symlink(f"../../blobs/{etag}", snap / shard)
        weight_map[f"layers.{lid}.mlp.weight"] = shard
    shard = "model-shared.safetensors"
    etag = "etagshared" + "0" * 50
    (blobs / etag).write_bytes(b"e" * 4096)
    os.symlink(f"../../blobs/{etag}", snap / shard)
    weight_map["embed_tokens.weight"] = shard

    index = {"weight_map": weight_map}
    (snap / prep.SAFE_WEIGHTS_INDEX_NAME).write_text(json.dumps(index))
    (snap / "config.json").write_text("{}")
    return str(snap)


def _stage(
    src_snap,
    dest_root,
    layer_range,
    *,
    is_first_pipeline_rank=True,
    is_last_pipeline_rank=True,
):
    return prep.stage_shards(
        src_snap,
        str(dest_root),
        layer_range,
        is_first_pipeline_rank=is_first_pipeline_rank,
        is_last_pipeline_rank=is_last_pipeline_rank,
    )


def _add_global_shard(src_snap, tensor_name, shard_name, etag):
    src_snap = os.path.abspath(src_snap)
    blobs = os.path.join(os.path.dirname(os.path.dirname(src_snap)), "blobs")
    with open(os.path.join(blobs, etag), "wb") as f:
        f.write(tensor_name.encode())
    os.symlink(f"../../blobs/{etag}", os.path.join(src_snap, shard_name))
    index_path = os.path.join(src_snap, prep.SAFE_WEIGHTS_INDEX_NAME)
    with open(index_path) as f:
        index = json.load(f)
    index["weight_map"][tensor_name] = shard_name
    with open(index_path, "w") as f:
        json.dump(index, f)


def _local_repo(dest_root):
    return os.path.join(str(dest_root), "hub", "models--appmana--dsv4-test")


def test_staged_path_is_hf_snapshot_layout(tmp_path):
    src = _make_source_hf_cache(tmp_path)
    dest = _stage(src, tmp_path / "local", (0, 2))
    assert dest == os.path.join(_local_repo(tmp_path / "local"), "snapshots", REV)
    assert os.path.isdir(dest)
    refs = os.path.join(_local_repo(tmp_path / "local"), "refs", "main")
    assert open(refs).read().strip() == REV


def test_owned_link_local_blob_foreign_link_source_blob(tmp_path):
    src = _make_source_hf_cache(tmp_path)
    dest = _stage(src, tmp_path / "local", (0, 2))
    local_blobs = os.path.join(_local_repo(tmp_path / "local"), "blobs")
    src_blobs = os.path.realpath(
        os.path.join(os.path.dirname(os.path.dirname(src)), "blobs")
    )
    for lid in range(6):
        shard = os.path.join(dest, f"model-{lid:05d}.safetensors")
        assert os.path.islink(shard)
        target = os.path.realpath(shard)
        if lid < 2:
            assert target.startswith(os.path.realpath(local_blobs))
            assert os.path.isfile(target)
        else:
            assert target.startswith(src_blobs)
    shared = os.path.realpath(os.path.join(dest, "model-shared.safetensors"))
    assert shared.startswith(os.path.realpath(local_blobs))


def test_stage_shards_places_global_tensors_on_owning_pp_rank(tmp_path):
    src = _make_source_hf_cache(tmp_path)
    _add_global_shard(src, "norm.weight", "model-norm.safetensors", "norm-etag")
    _add_global_shard(src, "head.weight", "model-head.safetensors", "head-etag")
    _add_global_shard(src, "mtp.0.norm.weight", "model-mtp.safetensors", "mtp-etag")
    dest_root = tmp_path / "local"

    middle = _stage(
        src,
        dest_root,
        (2, 4),
        is_first_pipeline_rank=False,
        is_last_pipeline_rank=False,
    )
    for shard in (
        "model-shared.safetensors",
        "model-norm.safetensors",
        "model-head.safetensors",
        "model-mtp.safetensors",
    ):
        assert "hf-cache" in os.path.realpath(os.path.join(middle, shard))

    last = _stage(
        src,
        dest_root,
        (4, 6),
        is_first_pipeline_rank=False,
        is_last_pipeline_rank=True,
    )
    assert "hf-cache" in os.path.realpath(
        os.path.join(last, "model-shared.safetensors")
    )
    local_blobs = os.path.realpath(os.path.join(_local_repo(dest_root), "blobs"))
    for shard in (
        "model-norm.safetensors",
        "model-head.safetensors",
        "model-mtp.safetensors",
    ):
        assert os.path.realpath(os.path.join(last, shard)).startswith(local_blobs)


def test_blob_names_match_source_etags(tmp_path):
    src = _make_source_hf_cache(tmp_path)
    _stage(src, tmp_path / "local", (0, 1))
    local_blobs = os.path.join(_local_repo(tmp_path / "local"), "blobs")
    assert set(os.listdir(local_blobs)) == {
        "etag" + "0" * 56,
        "etagshared" + "0" * 50,
    }


def test_ownership_change_copies_only_missing_blobs(tmp_path):
    src = _make_source_hf_cache(tmp_path)
    dest_root = tmp_path / "local"
    _stage(src, dest_root, (0, 4))

    copies = {"n": 0}
    real_copy = prep.shutil.copy2

    def counting_copy(a, b, **kw):
        if "blobs" in str(b):
            copies["n"] += 1
        return real_copy(a, b, **kw)

    prep.shutil.copy2 = counting_copy
    try:
        dest = _stage(src, dest_root, (0, 2))
    finally:
        prep.shutil.copy2 = real_copy
    assert copies["n"] == 0  # subset ownership: every needed blob already local
    # foreign shards were repointed back at source blobs
    t = os.path.realpath(os.path.join(dest, "model-00003.safetensors"))
    assert "hf-cache" in t


def test_idempotent_fast_path(tmp_path):
    src = _make_source_hf_cache(tmp_path)
    dest_root = tmp_path / "local"
    d1 = _stage(src, dest_root, (0, 2))
    marker = os.path.join(d1, prep.COMPLETE_MARKER)
    stamp = os.stat(marker).st_mtime_ns
    d2 = _stage(src, dest_root, (0, 2))
    assert d1 == d2
    assert os.stat(marker).st_mtime_ns == stamp


def test_adopts_legacy_real_files_by_hardlink(tmp_path):
    """Real shard copies left by the older staging layouts (config dirs or
    store/) must be harvested by hardlink instead of re-copied."""
    src = _make_source_hf_cache(tmp_path)
    dest_root = tmp_path / "local"
    legacy = dest_root / "deadbeefdeadbeef"
    legacy.mkdir(parents=True)
    shard = "model-00000.safetensors"
    legacy_file = legacy / shard
    legacy_file.write_bytes(b"x" * 1000)

    copies = {"n": 0}
    real_copy = prep.shutil.copy2

    def counting_copy(a, b, **kw):
        copies["n"] += 1
        return real_copy(a, b, **kw)

    prep.shutil.copy2 = counting_copy
    try:
        dest = _stage(src, dest_root, (0, 1))
    finally:
        prep.shutil.copy2 = real_copy
    target = os.path.realpath(os.path.join(dest, shard))
    assert os.stat(target).st_nlink >= 2
    assert copies["n"] > 0  # metadata still copied; but not the adopted shard


def test_gc_absorbs_legacy_layouts_and_foreign_blobs(tmp_path):
    src = _make_source_hf_cache(tmp_path)
    dest_root = tmp_path / "local"
    # legacy layouts from previous designs
    (dest_root / "deadbeefdeadbeef").mkdir(parents=True)
    (dest_root / "deadbeefdeadbeef" / "junk.safetensors").write_bytes(b"j")
    (dest_root / "store" / "cafe").mkdir(parents=True)
    (dest_root / "store" / "cafe" / "old.safetensors").write_bytes(b"o")

    d_wide = _stage(src, dest_root, (0, 4))
    d_narrow = _stage(src, dest_root, (0, 1))
    assert d_wide == d_narrow

    # a blob from some other checkpoint/revision
    local_blobs = os.path.join(_local_repo(dest_root), "blobs")
    with open(os.path.join(local_blobs, "otherckpt" + "f" * 50), "w") as f:
        f.write("stale")

    removed = prep.gc_dest_root(str(dest_root), keep_dir=d_narrow, source_dir=src)
    assert not os.path.exists(dest_root / "deadbeefdeadbeef")
    assert not os.path.exists(dest_root / "store")
    # partitions are ADDITIVE: blobs staged by the wider ownership survive
    # even though the current partition references only layer 0 + shared
    assert set(os.listdir(local_blobs)) == {
        "etag" + "0" * 56,
        "etag" + "0" * 55 + "1",
        "etag" + "0" * 55 + "2",
        "etag" + "0" * 55 + "3",
        "etagshared" + "0" * 50,
    }
    assert removed["blobs"] == 1  # only the foreign-checkpoint blob
    # surviving snapshot still verifies complete
    assert _stage(src, dest_root, (0, 1)) == d_narrow


def test_partition_flip_after_gc_copies_nothing(tmp_path):
    src = _make_source_hf_cache(tmp_path)
    dest_root = tmp_path / "local"
    d = _stage(src, dest_root, (0, 4))
    _stage(src, dest_root, (0, 1))
    prep.gc_dest_root(str(dest_root), keep_dir=d, source_dir=src)

    copies = {"n": 0}
    real_copy = prep.shutil.copy2

    def counting_copy(a, b, **kw):
        if "blobs" in str(b):
            copies["n"] += 1
        return real_copy(a, b, **kw)

    prep.shutil.copy2 = counting_copy
    try:
        _stage(src, dest_root, (0, 4))
    finally:
        prep.shutil.copy2 = real_copy
    assert copies["n"] == 0


def test_metadata_real_copies_and_index_untouched(tmp_path):
    src = _make_source_hf_cache(tmp_path)
    dest = _stage(src, tmp_path / "local", (0, 2))
    for name in (prep.SAFE_WEIGHTS_INDEX_NAME, "config.json"):
        path = os.path.join(dest, name)
        assert os.path.isfile(path) and not os.path.islink(path)
    assert json.load(open(os.path.join(dest, prep.SAFE_WEIGHTS_INDEX_NAME))) == json.load(
        open(os.path.join(src, prep.SAFE_WEIGHTS_INDEX_NAME))
    )


def test_every_index_shard_resolves(tmp_path):
    src = _make_source_hf_cache(tmp_path)
    dest = _stage(src, tmp_path / "local", (2, 4))
    index = json.load(open(os.path.join(dest, prep.SAFE_WEIGHTS_INDEX_NAME)))
    for shard in set(index["weight_map"].values()):
        assert os.path.exists(os.path.join(dest, shard)), shard
