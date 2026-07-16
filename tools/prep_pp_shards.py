# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage a PP-rank-local subset of a safetensors checkpoint onto local disk.

Each PP rank only needs its own contiguous slice of hidden layers (see
vllm.model_executor.model_loader.pp_weight_filter). Reading the rest of the
checkpoint every load is wasted I/O against a shared, network-backed mount
(e.g. a SeaweedFS PVC). This script stages a local directory that real-copies
only the shards containing at least one locally-owned tensor, and symlinks
every other shard back to --source-dir -- safetensors only reads a shard's
header until get_tensor() is called for a specific name, and
should_skip_pp_weight already causes the loader to skip get_tensor() for
every tensor in a symlinked shard, so its bulk data is never actually read
over the network.

The destination directory name is a deterministic hash of the checkpoint's
own index.json content plus (pp_rank, pp_size) -- not of any human-assigned
job/run name -- so repeated runs of the same config reuse the staged
directory for free, and a different config never collides with or silently
reuses another config's stale shards.

Usage:
    python prep_pp_shards.py --source-dir /hf-cache/.../snapshots/<rev> \
        --dest-root /local-shard-cache --pp-rank 3 --pp-size 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
COMPLETE_MARKER = ".prep-complete"


def compute_cache_key(
    source_dir: str, index_bytes: bytes, pp_rank: int, pp_size: int
) -> str:
    h = hashlib.sha256()
    h.update(os.path.abspath(source_dir).encode("utf-8"))
    h.update(index_bytes)
    h.update(f"pp{pp_rank}of{pp_size}".encode("utf-8"))
    return h.hexdigest()[:16]


def destination_dir(dest_root: str, cache_key: str) -> str:
    return os.path.join(dest_root, cache_key)


def _is_complete(dest_dir: str, weight_map: dict[str, str], needs_copy: dict[str, bool]) -> bool:
    if not os.path.exists(os.path.join(dest_dir, COMPLETE_MARKER)):
        return False
    for shard_filename, copy in needs_copy.items():
        path = os.path.join(dest_dir, shard_filename)
        is_link = os.path.islink(path)
        if copy and is_link:
            return False
        if not copy and not is_link:
            return False
        if not os.path.exists(path):
            return False
    return True


def stage_shards(
    source_dir: str,
    dest_root: str,
    pp_rank: int,
    pp_size: int,
    local_layer_range: tuple[int, int] | None,
) -> str:
    from vllm.model_executor.model_loader.pp_weight_filter import classify_shards

    index_path = os.path.join(source_dir, SAFE_WEIGHTS_INDEX_NAME)
    with open(index_path, "rb") as f:
        index_bytes = f.read()
    index = json.loads(index_bytes)
    weight_map: dict[str, str] = index["weight_map"]

    needs_copy = classify_shards(weight_map, local_layer_range)

    cache_key = compute_cache_key(source_dir, index_bytes, pp_rank, pp_size)
    dest_dir = destination_dir(dest_root, cache_key)

    if _is_complete(dest_dir, weight_map, needs_copy):
        return dest_dir

    os.makedirs(dest_dir, exist_ok=True)

    for entry in os.listdir(source_dir):
        src_path = os.path.join(source_dir, entry)
        if entry.endswith(".safetensors") or not os.path.isfile(src_path):
            continue
        shutil.copy2(src_path, os.path.join(dest_dir, entry))

    # Original, unmodified index -- every shard filename it references must
    # exist in dest_dir (real file or symlink) or vLLM's
    # filter_duplicate_safetensors_files raises FileNotFoundError.
    shutil.copy2(index_path, os.path.join(dest_dir, SAFE_WEIGHTS_INDEX_NAME))

    for shard_filename, copy in needs_copy.items():
        src_path = os.path.join(source_dir, shard_filename)
        dst_path = os.path.join(dest_dir, shard_filename)
        if os.path.islink(dst_path) or os.path.exists(dst_path):
            os.remove(dst_path)
        if copy:
            shutil.copy2(src_path, dst_path)
        else:
            os.symlink(src_path, dst_path)

    with open(os.path.join(dest_dir, COMPLETE_MARKER), "w") as f:
        f.write("")

    return dest_dir


def _resolve_local_layer_range(
    source_dir: str, pp_rank: int, pp_size: int, trust_remote_code: bool = False
) -> tuple[int, int] | None:
    if pp_size <= 1:
        return None
    from vllm.config import ModelConfig
    from vllm.distributed.utils import get_pp_indices

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
    args = parser.parse_args(argv)

    local_layer_range = _resolve_local_layer_range(
        args.source_dir, args.pp_rank, args.pp_size, args.trust_remote_code
    )
    dest_dir = stage_shards(
        args.source_dir, args.dest_root, args.pp_rank, args.pp_size, local_layer_range
    )
    print(dest_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
