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
own index.json content plus pp_size -- deliberately NOT pp_rank (vLLM's Ray
executor resolves the same --model path string independently on every
worker's own node, so every rank must agree on the same path) and not any
human-assigned job/run name -- so repeated runs of the same config reuse the
staged directory for free, and a different config never collides with or
silently reuses another config's stale shards.

Usage:
    python prep_pp_shards.py --source-dir /hf-cache/.../snapshots/<rev> \
        --dest-root /local-shard-cache --pp-rank 3 --pp-size 10
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


def compute_cache_key(source_dir: str, index_bytes: bytes, pp_size: int) -> str:
    """Deliberately excludes pp_rank. vLLM's distributed_executor_backend=ray
    resolves a SINGLE --model path string independently on every worker's
    own node -- the leader passes its own staged path once, and every
    remote worker actor re-resolves that exact same string against its own
    local disk. Since staging lives on a per-node hostPath (see
    dsv4-benchmark-jobset-proof.yaml), the physical directory a given path
    string resolves to already differs correctly per node; the path STRING
    must be identical across all ranks or a remote worker looks for the
    leader's own hash on its own disk, finds nothing, and vLLM's model-path
    resolver falls through to treating it as a bogus HF Hub repo id.
    Confirmed live 2026-07-16: 'Repo id must be in the form repo_name or
    namespace/repo_name' on every remote worker, immediately downstream of
    exactly this. pp_size stays in the hash so a differently-sized run gets
    its own directory rather than racing a concurrent, differently-shaped
    stage under the same name."""
    h = hashlib.sha256()
    h.update(os.path.abspath(source_dir).encode("utf-8"))
    h.update(index_bytes)
    h.update(f"pp_size{pp_size}".encode("utf-8"))
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

    cache_key = compute_cache_key(source_dir, index_bytes, pp_size)
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
    args = parser.parse_args(argv)

    local_layer_range = _resolve_local_layer_range(
        args.source_dir, args.pp_rank, args.pp_size, args.trust_remote_code
    )
    dest_dir = stage_shards(
        args.source_dir, args.dest_root, args.pp_size, local_layer_range
    )
    print(dest_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
