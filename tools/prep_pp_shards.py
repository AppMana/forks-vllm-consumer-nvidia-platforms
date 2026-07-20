# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage a PP-rank-local subset of a safetensors checkpoint onto local disk.

Each PP rank only needs its own contiguous slice of hidden layers (see
vllm.model_executor.model_loader.pp_weight_filter). Reading the rest of the
checkpoint every load is wasted I/O against a shared, network-backed mount
(e.g. a SeaweedFS PVC), so the owned shards are brought onto local disk and
the rest are symlinked back to --source-dir -- safetensors only reads a
shard's header until get_tensor() is called for a specific name, and
should_skip_pp_weight already causes the loader to skip get_tensor() for
every tensor in a foreign shard, so its bulk data is never read over the
network.

Layout (two-level, content-addressed):

- ``<dest-root>/store/<checkpoint_id>/<shard>``: every real-copied shard
  exists exactly once per checkpoint here, shared by all configs. A new
  partition/pp_size only copies shards the store does not already hold --
  for most config changes, nothing.
- ``<dest-root>/<config_key>/``: cheap per-config directory: real copies of
  the small metadata files (config.json, tokenizer, the original unmodified
  index.json), plus one symlink per shard -- owned shards resolve into the
  store, foreign shards resolve into --source-dir.

The config_key hashes the checkpoint identity plus pp_size plus the layer
partition string -- deliberately NOT pp_rank (vLLM's Ray executor resolves
one --model path string independently on every worker's own node, so every
rank must produce the identical path; the partition string is identical on
all ranks, a rank's layer range is not). Confirmed live 2026-07-16: a
rank-dependent path makes every remote worker fall through to 'Repo id must
be in the form repo_name or namespace/repo_name'.

``--gc`` prunes every other config directory and any store shard no longer
referenced by the surviving config -- run it after staging to keep the
node-local PVC from accumulating dead configs.

Usage:
    python prep_pp_shards.py --source-dir /hf-cache/.../snapshots/<rev> \
        --dest-root /shard-cache-local --pp-rank 3 --pp-size 12 [--gc]
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import sys

SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
COMPLETE_MARKER = ".prep-complete"
STORE_DIR_NAME = "store"


def compute_checkpoint_id(source_dir: str, index_bytes: bytes) -> str:
    h = hashlib.sha256()
    h.update(os.path.abspath(source_dir).encode("utf-8"))
    h.update(index_bytes)
    return h.hexdigest()[:16]


def compute_config_key(
    source_dir: str, index_bytes: bytes, pp_size: int, partition: str | None
) -> str:
    h = hashlib.sha256()
    h.update(os.path.abspath(source_dir).encode("utf-8"))
    h.update(index_bytes)
    h.update(f"pp_size{pp_size}".encode("utf-8"))
    h.update(f"partition{partition or 'default'}".encode("utf-8"))
    return h.hexdigest()[:16]


def _store_dir(dest_root: str, checkpoint_id: str) -> str:
    return os.path.join(dest_root, STORE_DIR_NAME, checkpoint_id)


def _is_complete(
    dest_dir: str,
    source_dir: str,
    store_dir: str,
    needs_copy: dict[str, bool],
) -> bool:
    if not os.path.exists(os.path.join(dest_dir, COMPLETE_MARKER)):
        return False
    for shard_filename, owned in needs_copy.items():
        path = os.path.join(dest_dir, shard_filename)
        if not os.path.islink(path) or not os.path.exists(path):
            return False
        target = os.path.realpath(path)
        if owned:
            expected = os.path.realpath(os.path.join(store_dir, shard_filename))
            src_path = os.path.join(source_dir, shard_filename)
            if target != expected:
                return False
            if os.path.getsize(target) != os.path.getsize(src_path):
                return False
        else:
            if target != os.path.realpath(os.path.join(source_dir, shard_filename)):
                return False
    return True


def _adopt_or_copy_into_store(
    dest_root: str,
    store_dir: str,
    source_dir: str,
    shard_filename: str,
) -> str:
    """Ensure the shard exists in the store; return its store path.

    Prefer hardlinking an existing real copy from any sibling directory on
    the same filesystem (legacy pre-store config dirs, or another
    checkpoint's leftovers with the same name and size) over re-copying
    bulk data from the network-backed source.
    """
    store_path = os.path.join(store_dir, shard_filename)
    src_path = os.path.join(source_dir, shard_filename)
    src_size = os.path.getsize(src_path)
    if os.path.exists(store_path):
        if os.path.getsize(store_path) == src_size:
            return store_path
        os.remove(store_path)

    os.makedirs(store_dir, exist_ok=True)

    for entry in sorted(os.listdir(dest_root)):
        if entry == STORE_DIR_NAME:
            continue
        candidate = os.path.join(dest_root, entry, shard_filename)
        if (
            os.path.isfile(candidate)
            and not os.path.islink(candidate)
            and os.path.getsize(candidate) == src_size
        ):
            try:
                os.link(candidate, store_path)
                return store_path
            except OSError:
                break

    tmp_path = store_path + ".tmp"
    shutil.copy2(src_path, tmp_path)
    os.replace(tmp_path, store_path)
    return store_path


def stage_shards(
    source_dir: str,
    dest_root: str,
    pp_size: int,
    local_layer_range: tuple[int, int] | None,
    partition: str | None = None,
) -> str:
    from vllm.model_executor.model_loader.pp_weight_filter import classify_shards

    index_path = os.path.join(source_dir, SAFE_WEIGHTS_INDEX_NAME)
    with open(index_path, "rb") as f:
        index_bytes = f.read()
    index = json.loads(index_bytes)
    weight_map: dict[str, str] = index["weight_map"]

    needs_copy = classify_shards(weight_map, local_layer_range)

    checkpoint_id = compute_checkpoint_id(source_dir, index_bytes)
    store_dir = _store_dir(dest_root, checkpoint_id)
    config_key = compute_config_key(source_dir, index_bytes, pp_size, partition)
    dest_dir = os.path.join(dest_root, config_key)

    if _is_complete(dest_dir, source_dir, store_dir, needs_copy):
        return dest_dir

    os.makedirs(dest_dir, exist_ok=True)

    for entry in os.listdir(source_dir):
        src_path = os.path.join(source_dir, entry)
        if entry.endswith(".safetensors") or not os.path.isfile(src_path):
            continue
        shutil.copy2(src_path, os.path.join(dest_dir, entry))

    # Original, unmodified index -- every shard filename it references must
    # exist in dest_dir (as a resolvable symlink) or vLLM's
    # filter_duplicate_safetensors_files raises FileNotFoundError.
    shutil.copy2(index_path, os.path.join(dest_dir, SAFE_WEIGHTS_INDEX_NAME))

    for shard_filename, owned in needs_copy.items():
        dst_path = os.path.join(dest_dir, shard_filename)
        if os.path.islink(dst_path) or os.path.exists(dst_path):
            os.remove(dst_path)
        if owned:
            target = _adopt_or_copy_into_store(
                dest_root, store_dir, source_dir, shard_filename
            )
        else:
            target = os.path.join(source_dir, shard_filename)
        os.symlink(target, dst_path)

    with open(os.path.join(dest_dir, COMPLETE_MARKER), "w") as f:
        f.write("")

    return dest_dir


def gc_dest_root(dest_root: str, keep_dir: str) -> dict[str, int]:
    """Remove every config directory except ``keep_dir`` and every store
    shard no longer referenced by it. Legacy (pre-store) config dirs full of
    real copies are removed the same way -- their contents were already
    adopted into the store by hardlink during staging, so their space is
    reclaimed here without losing the harvested bytes."""
    keep_dir = os.path.realpath(keep_dir)
    removed = {"config_dirs": 0, "store_files": 0}

    for entry in sorted(os.listdir(dest_root)):
        path = os.path.join(dest_root, entry)
        if entry == STORE_DIR_NAME or not os.path.isdir(path):
            continue
        if os.path.realpath(path) == keep_dir:
            continue
        shutil.rmtree(path)
        removed["config_dirs"] += 1

    referenced: set[str] = set()
    for name in os.listdir(keep_dir):
        path = os.path.join(keep_dir, name)
        if os.path.islink(path):
            referenced.add(os.path.realpath(path))

    store_root = os.path.join(dest_root, STORE_DIR_NAME)
    if os.path.isdir(store_root):
        for root, _dirs, files in os.walk(store_root, topdown=False):
            for name in files:
                path = os.path.join(root, name)
                if os.path.realpath(path) not in referenced:
                    os.remove(path)
                    removed["store_files"] += 1
            if root != store_root and not os.listdir(root):
                os.rmdir(root)

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
        help="after staging, prune stale config dirs and orphaned store shards",
    )
    args = parser.parse_args(argv)

    local_layer_range = _resolve_local_layer_range(
        args.source_dir, args.pp_rank, args.pp_size, args.trust_remote_code
    )
    partition = os.environ.get("VLLM_PP_LAYER_PARTITION")
    dest_dir = stage_shards(
        args.source_dir,
        args.dest_root,
        args.pp_size,
        local_layer_range,
        partition=partition,
    )
    if args.gc:
        removed = gc_dest_root(args.dest_root, keep_dir=dest_dir)
        print(
            f"[prep-gc] removed {removed['config_dirs']} config dirs, "
            f"{removed['store_files']} orphaned store shards",
            file=sys.stderr,
        )
    print(dest_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
