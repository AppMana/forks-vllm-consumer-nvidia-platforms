# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage a PP-rank-local subset of a safetensors checkpoint as a partial
local HF cache.

Each PP rank only needs its own contiguous slice of hidden layers (see
vllm.model_executor.model_loader.pp_weight_filter). Reading the rest of the
checkpoint every load is wasted I/O against a shared, network-backed mount
(e.g. a SeaweedFS PVC), so the owned shards are brought onto local disk and
the rest resolve back to the shared cache -- safetensors only reads a
shard's header until get_tensor() is called for a specific name, and
should_skip_pp_weight already causes the loader to skip get_tensor() for
every tensor in a foreign shard, so its bulk data is never read over the
network.

The local cache is a standard HF hub cache, not a bespoke layout:

    <dest-root>/hub/models--<org>--<name>/
        blobs/<etag>                  # content-addressed; owned shards only
        snapshots/<rev>/<filename>    # symlinks: owned -> local blob,
                                      #           foreign -> source blob
        refs/main                     # <rev>

Blob names are the source cache's own content hashes, so "copy only what is
missing" across partition/pp_size changes is inherited from the HF layout
rather than reinvented: an ownership change repoints snapshot symlinks in
place and copies just the blobs the local blobs/ directory lacks. The staged
model path is the snapshot directory -- identical on every rank (vLLM's Ray
executor resolves one --model path string independently on each worker's
node, so a rank-dependent path breaks remote workers), with per-node
divergence living entirely in which blobs are local.

``--gc`` prunes snapshots of other revisions, local blobs that do not
belong to the current checkpoint, and directories left by the older staging
layouts (16-hex config dirs, store/) -- their real files are harvested into
blobs/ by hardlink during staging first, so reclaiming them never re-reads
the network. Blobs of the current checkpoint are kept even when the current
partition does not reference them: partitions are additive against one
checkpoint, so a partition change never re-copies a blob any earlier
partition already staged.

Usage:
    python prep_pp_shards.py --source-dir /hf-cache/.../snapshots/<rev> \
        --dest-root /shard-cache-local --pp-rank 3 --pp-size 12 [--gc]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import sys

SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
COMPLETE_MARKER = ".prep-complete"
_LEGACY_CONFIG_DIR_RE = re.compile(r"^[0-9a-f]{16}$")
_LEGACY_STORE_DIR = "store"


def _parse_source_snapshot(source_dir: str) -> tuple[str, str]:
    """Return (repo_dir_name, revision) from an HF-cache snapshot path
    ``.../hub/models--org--name/snapshots/<rev>``."""
    source_dir = os.path.abspath(source_dir)
    rev = os.path.basename(source_dir)
    snapshots = os.path.dirname(source_dir)
    repo = os.path.dirname(snapshots)
    repo_name = os.path.basename(repo)
    if os.path.basename(snapshots) != "snapshots" or not repo_name.startswith(
        "models--"
    ):
        raise ValueError(
            f"--source-dir must be an HF cache snapshot directory "
            f"(.../hub/models--*/snapshots/<rev>), got {source_dir}"
        )
    return repo_name, rev


def _local_repo_dir(dest_root: str, repo_name: str) -> str:
    return os.path.join(dest_root, "hub", repo_name)


def _source_blob(source_dir: str, filename: str) -> str:
    """Resolve a snapshot entry to its blob (or to itself if not a link)."""
    return os.path.realpath(os.path.join(source_dir, filename))


def _is_complete(
    snap_dir: str,
    source_dir: str,
    blobs_dir: str,
    needs_copy: dict[str, bool],
) -> bool:
    if not os.path.exists(os.path.join(snap_dir, COMPLETE_MARKER)):
        return False
    blobs_real = os.path.realpath(blobs_dir)
    for shard_filename, owned in needs_copy.items():
        path = os.path.join(snap_dir, shard_filename)
        if not os.path.islink(path) or not os.path.exists(path):
            return False
        target = os.path.realpath(path)
        src_blob = _source_blob(source_dir, shard_filename)
        if owned:
            if os.path.dirname(target) != blobs_real:
                return False
            if os.path.getsize(target) != os.path.getsize(src_blob):
                return False
        else:
            if target != src_blob:
                return False
    return True


def _ensure_local_blob(
    dest_root: str, blobs_dir: str, source_dir: str, shard_filename: str
) -> str:
    """Ensure the shard's blob exists locally; return its path.

    Prefer hardlinking a real copy left behind by the older staging layouts
    (16-hex config dirs, store/<ckpt>/) on the same filesystem over
    re-copying bulk data from the network-backed source.
    """
    src_blob = _source_blob(source_dir, shard_filename)
    etag = os.path.basename(src_blob)
    blob_path = os.path.join(blobs_dir, etag)
    src_size = os.path.getsize(src_blob)
    if os.path.exists(blob_path):
        if os.path.getsize(blob_path) == src_size:
            return blob_path
        os.remove(blob_path)

    os.makedirs(blobs_dir, exist_ok=True)

    legacy_candidates = []
    with contextlib.suppress(FileNotFoundError):
        for entry in sorted(os.listdir(dest_root)):
            if _LEGACY_CONFIG_DIR_RE.match(entry):
                legacy_candidates.append(
                    os.path.join(dest_root, entry, shard_filename)
                )
        store_root = os.path.join(dest_root, _LEGACY_STORE_DIR)
        if os.path.isdir(store_root):
            for ckpt in sorted(os.listdir(store_root)):
                legacy_candidates.append(
                    os.path.join(store_root, ckpt, shard_filename)
                )
    for candidate in legacy_candidates:
        if (
            os.path.isfile(candidate)
            and not os.path.islink(candidate)
            and os.path.getsize(candidate) == src_size
        ):
            try:
                os.link(candidate, blob_path)
                return blob_path
            except OSError:
                break

    tmp_path = blob_path + ".tmp"
    shutil.copy2(src_blob, tmp_path)
    os.replace(tmp_path, blob_path)
    return blob_path


def stage_shards(
    source_dir: str,
    dest_root: str,
    local_layer_range: tuple[int, int] | None,
) -> str:
    from vllm.model_executor.model_loader.pp_weight_filter import classify_shards

    source_dir = os.path.abspath(source_dir)
    index_path = os.path.join(source_dir, SAFE_WEIGHTS_INDEX_NAME)
    with open(index_path, "rb") as f:
        index_bytes = f.read()
    weight_map: dict[str, str] = json.loads(index_bytes)["weight_map"]

    needs_copy = classify_shards(weight_map, local_layer_range)

    repo_name, rev = _parse_source_snapshot(source_dir)
    repo_dir = _local_repo_dir(dest_root, repo_name)
    blobs_dir = os.path.join(repo_dir, "blobs")
    snap_dir = os.path.join(repo_dir, "snapshots", rev)

    if _is_complete(snap_dir, source_dir, blobs_dir, needs_copy):
        return snap_dir

    os.makedirs(snap_dir, exist_ok=True)
    os.makedirs(os.path.join(repo_dir, "refs"), exist_ok=True)
    with open(os.path.join(repo_dir, "refs", "main"), "w") as f:
        f.write(rev)

    for entry in os.listdir(source_dir):
        src_path = os.path.join(source_dir, entry)
        if entry.endswith(".safetensors") or not os.path.isfile(src_path):
            continue
        dst_path = os.path.join(snap_dir, entry)
        if os.path.islink(dst_path):
            os.remove(dst_path)
        shutil.copy2(src_path, dst_path)

    # Original, unmodified index -- every shard filename it references must
    # resolve in snap_dir or vLLM's filter_duplicate_safetensors_files
    # raises FileNotFoundError.
    shutil.copy2(index_path, os.path.join(snap_dir, SAFE_WEIGHTS_INDEX_NAME))

    for shard_filename, owned in needs_copy.items():
        dst_path = os.path.join(snap_dir, shard_filename)
        if os.path.islink(dst_path) or os.path.exists(dst_path):
            os.remove(dst_path)
        if owned:
            target = _ensure_local_blob(
                dest_root, blobs_dir, source_dir, shard_filename
            )
        else:
            target = _source_blob(source_dir, shard_filename)
        os.symlink(target, dst_path)

    with open(os.path.join(snap_dir, COMPLETE_MARKER), "w") as f:
        f.write("")

    return snap_dir


def gc_dest_root(
    dest_root: str, keep_dir: str, source_dir: str | None = None
) -> dict[str, int]:
    """Prune legacy layout directories, snapshots of other revisions, and
    blobs from other checkpoints. Blobs belonging to the current checkpoint
    (present in the source snapshot's etag set) are kept even when the
    current partition does not reference them -- partitions add onto each
    other against one checkpoint."""
    keep_dir = os.path.realpath(keep_dir)
    removed = {"legacy_dirs": 0, "snapshots": 0, "blobs": 0}

    for entry in sorted(os.listdir(dest_root)):
        path = os.path.join(dest_root, entry)
        if not os.path.isdir(path):
            continue
        if _LEGACY_CONFIG_DIR_RE.match(entry) or entry == _LEGACY_STORE_DIR:
            shutil.rmtree(path)
            removed["legacy_dirs"] += 1

    repo_dir = os.path.dirname(os.path.dirname(keep_dir))
    snapshots_dir = os.path.join(repo_dir, "snapshots")
    if os.path.isdir(snapshots_dir):
        for entry in sorted(os.listdir(snapshots_dir)):
            path = os.path.join(snapshots_dir, entry)
            if os.path.realpath(path) != keep_dir:
                shutil.rmtree(path)
                removed["snapshots"] += 1

    keep_etags: set[str] = set()
    if source_dir is not None:
        for name in os.listdir(source_dir):
            path = os.path.join(source_dir, name)
            if name.endswith(".safetensors") and os.path.exists(path):
                keep_etags.add(os.path.basename(os.path.realpath(path)))
    else:
        # No source to define the checkpoint: fall back to keeping what the
        # surviving snapshot references.
        for name in os.listdir(keep_dir):
            path = os.path.join(keep_dir, name)
            if os.path.islink(path):
                keep_etags.add(os.path.basename(os.path.realpath(path)))

    blobs_dir = os.path.join(repo_dir, "blobs")
    if os.path.isdir(blobs_dir):
        for name in sorted(os.listdir(blobs_dir)):
            if name not in keep_etags:
                os.remove(os.path.join(blobs_dir, name))
                removed["blobs"] += 1

    return removed


def _resolve_local_layer_range(
    source_dir: str, pp_rank: int, pp_size: int, trust_remote_code: bool = False
) -> tuple[int, int] | None:
    if pp_size <= 1:
        return None
    from vllm.config import ModelConfig
    from vllm.distributed.utils import get_pp_indices

    # Constructing ModelConfig logs (architecture resolution, tokenizer
    # mode, max model len, ...) straight to stdout via vLLM's own logger --
    # this script's contract is "stdout is exactly the staged path, nothing
    # else" (callers commonly do `path=$(prep_pp_shards.py ...)`), so divert
    # that incidental logging to stderr rather than letting it silently
    # corrupt the captured path with embedded newlines.
    with contextlib.redirect_stdout(sys.stderr):
        model_config = ModelConfig(model=source_dir, trust_remote_code=trust_remote_code)
        total_num_hidden_layers = model_config.get_total_num_hidden_layers()
    return get_pp_indices(total_num_hidden_layers, pp_rank, pp_size)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--dest-root", required=True)
    parser.add_argument("--pp-rank", type=int, required=True)
    parser.add_argument("--pp-size", type=int, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--gc",
        action="store_true",
        help="after staging, prune legacy layouts, other snapshots, and "
        "unreferenced local blobs",
    )
    args = parser.parse_args(argv)

    local_layer_range = _resolve_local_layer_range(
        args.source_dir, args.pp_rank, args.pp_size, args.trust_remote_code
    )
    snap_dir = stage_shards(args.source_dir, args.dest_root, local_layer_range)
    if args.gc:
        removed = gc_dest_root(
            args.dest_root, keep_dir=snap_dir, source_dir=args.source_dir
        )
        print(
            f"[prep-gc] removed {removed['legacy_dirs']} legacy dirs, "
            f"{removed['snapshots']} stale snapshots, "
            f"{removed['blobs']} unreferenced blobs",
            file=sys.stderr,
        )
    print(snap_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
