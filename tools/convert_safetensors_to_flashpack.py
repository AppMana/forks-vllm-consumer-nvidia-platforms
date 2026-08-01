# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convert a sharded safetensors checkpoint into the manifest-driven
FlashPack layout consumed by ``--load-format flashpack`` (see
vllm.model_executor.model_loader.flashpack_loader).

One ``.flashpack`` part is written per source safetensors shard with
``flashpack.pack_to_file(..., target_dtype=None)``, preserving every
tensor's original name, shape, and dtype so custom ``load_weights``
implementations (fused/quantized models) see the same stream as the
safetensors path. The source shard-to-tensor assignment is preserved, so
``model.flashpack.index.json`` inherits the safetensors index's weight
map. Part ``kind`` is ``mtp`` when a shard holds only ``mtp.*`` tensors;
``pipeline_stage`` is derived from ``should_skip_pp_weight`` so
embedding-only and head-only shards stay off pipeline ranks that cannot
consume them. Non-weight files (config, tokenizer, ...) are copied so the
destination directory is directly usable as ``--model``.

Conversion is resumable: finished part records are checkpointed in
``.flashpack-convert-state.json`` inside the destination and reused when
the recorded size still matches the file on disk.

Usage:
    python convert_safetensors_to_flashpack.py \
        --source-dir /hf-cache/.../snapshots/<rev> \
        --dest-dir /checkpoints/model-flashpack \
        [--repo-id org/name] [--revision <rev>] [--verify]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time

SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
FLASHPACK_INDEX_NAME = "model.flashpack.index.json"
FLASHPACK_INDEX_FORMAT = "vllm_sharded_flashpack_v1"
STATE_NAME = ".flashpack-convert-state.json"
_MTP_PREFIXES = ("mtp.", "model.mtp.")
# Sentinel range covering every hidden layer, so should_skip_pp_weight
# classifies stage ownership without skipping numbered layer tensors.
_ALL_LAYERS = (0, 1 << 30)


def _infer_repo_and_revision(source_dir: str) -> tuple[str | None, str | None]:
    """Infer (repo_id, revision) from an HF cache snapshot path
    ``.../hub/models--org--name/snapshots/<rev>``, else (None, None)."""
    source_dir = os.path.abspath(source_dir)
    rev = os.path.basename(source_dir)
    snapshots = os.path.dirname(source_dir)
    repo_name = os.path.basename(os.path.dirname(snapshots))
    if os.path.basename(snapshots) != "snapshots" or not repo_name.startswith(
        "models--"
    ):
        return None, None
    return repo_name.removeprefix("models--").replace("--", "/"), rev


def _part_filename(shard_filename: str) -> str:
    stem, ext = os.path.splitext(shard_filename)
    if ext != ".safetensors":
        raise ValueError(f"Not a safetensors shard: {shard_filename!r}")
    return stem + ".flashpack"


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _classify_kind(names: list[str], shard_filename: str) -> str:
    mtp = [name for name in names if name.startswith(_MTP_PREFIXES)]
    if not mtp:
        return "model"
    if len(mtp) != len(names):
        raise ValueError(
            f"Shard {shard_filename!r} mixes MTP and non-MTP tensors; "
            "part-level kind gating requires pure shards"
        )
    return "mtp"


def _classify_stage(names: list[str]) -> str:
    from vllm.model_executor.model_loader.pp_weight_filter import (
        should_skip_pp_weight,
    )

    def only_first(name: str) -> bool:
        return should_skip_pp_weight(name, _ALL_LAYERS, is_first_pipeline_rank=False)

    def only_last(name: str) -> bool:
        return should_skip_pp_weight(name, _ALL_LAYERS, is_last_pipeline_rank=False)

    if all(only_first(name) for name in names):
        return "first"
    if all(only_last(name) for name in names):
        return "last"
    return "any"


def _verify_part(path: str, tensors: dict) -> None:
    import torch
    from flashpack.deserialization import (
        get_flashpack_file_metadata,
        iterate_from_flash_tensor,
        read_flashpack_file,
    )

    storage, metadata = read_flashpack_file(
        path, device="cpu", metadata=get_flashpack_file_metadata(path)
    )
    seen = set()
    for name, tensor in iterate_from_flash_tensor(storage, metadata):
        source = tensors[name]
        if tensor.dtype != source.dtype or tuple(tensor.shape) != tuple(source.shape):
            raise ValueError(
                f"Round-trip mismatch for {name!r}: "
                f"{tensor.dtype}{tuple(tensor.shape)} != "
                f"{source.dtype}{tuple(source.shape)}"
            )
        if not torch.equal(tensor, source):
            raise ValueError(f"Round-trip payload mismatch for {name!r}")
        seen.add(name)
    if seen != set(tensors):
        raise ValueError(f"Round-trip lost tensors: {sorted(set(tensors) - seen)}")


def _load_state(dest_dir: str) -> dict:
    try:
        with open(os.path.join(dest_dir, STATE_NAME), encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_state(dest_dir: str, state: dict) -> None:
    path = os.path.join(dest_dir, STATE_NAME)
    with open(path + ".tmp", "w", encoding="utf-8") as file:
        json.dump(state, file, indent=1)
    os.replace(path + ".tmp", path)


def convert(
    source_dir: str,
    dest_dir: str,
    repo_id: str,
    revision: str,
    *,
    verify: bool,
) -> None:
    from flashpack import pack_to_file
    from safetensors.torch import load_file

    index_path = os.path.join(source_dir, SAFE_WEIGHTS_INDEX_NAME)
    with open(index_path, encoding="utf-8") as file:
        weight_map: dict[str, str] = json.load(file)["weight_map"]

    names_by_shard: dict[str, list[str]] = {}
    for name, shard_filename in weight_map.items():
        names_by_shard.setdefault(shard_filename, []).append(name)

    os.makedirs(dest_dir, exist_ok=True)
    state = _load_state(dest_dir)
    parts: dict[str, dict] = {}
    manifest_weight_map: dict[str, str] = {}

    for shard_number, shard_filename in enumerate(sorted(names_by_shard), 1):
        part_filename = _part_filename(shard_filename)
        part_path = os.path.join(dest_dir, part_filename)
        expected_names = set(names_by_shard[shard_filename])
        cached = state.get(part_filename)
        if (
            cached
            and os.path.isfile(part_path)
            and os.path.getsize(part_path) == cached["size"]
        ):
            print(
                f"[{shard_number}/{len(names_by_shard)}] {part_filename}: "
                "already converted, skipping",
                flush=True,
            )
        else:
            start = time.perf_counter()
            tensors = load_file(os.path.join(source_dir, shard_filename))
            if set(tensors) != expected_names:
                raise ValueError(
                    f"Shard {shard_filename!r} tensors disagree with "
                    f"{SAFE_WEIGHTS_INDEX_NAME}"
                )
            empty = sorted(n for n, t in tensors.items() if t.numel() == 0)
            if empty:
                raise ValueError(
                    f"Shard {shard_filename!r} has zero-element tensors "
                    f"(unrepresentable in a FlashPack footer): {empty}"
                )
            scalars = sorted(n for n, t in tensors.items() if t.dim() == 0)
            if scalars:
                raise ValueError(
                    f"Shard {shard_filename!r} has 0-dim tensors, which "
                    f"FlashPack deserializes with shape (1,): {scalars}"
                )
            pack_to_file(tensors, part_path, target_dtype=None)
            # pack_to_file writes through mkstemp, which leaves 0600.
            os.chmod(part_path, 0o644)
            if verify:
                _verify_part(part_path, tensors)
            del tensors
            cached = {
                "sha256": _sha256_file(part_path),
                "size": os.path.getsize(part_path),
                "kind": _classify_kind(sorted(expected_names), shard_filename),
                "pipeline_stage": _classify_stage(sorted(expected_names)),
            }
            state[part_filename] = cached
            _save_state(dest_dir, state)
            print(
                f"[{shard_number}/{len(names_by_shard)}] {part_filename}: "
                f"{cached['size'] / 2**30:.2f} GiB, kind={cached['kind']}, "
                f"stage={cached['pipeline_stage']}, "
                f"{time.perf_counter() - start:.1f}s",
                flush=True,
            )
        parts[part_filename] = cached
        for name in expected_names:
            manifest_weight_map[name] = part_filename

    manifest = {
        "format": FLASHPACK_INDEX_FORMAT,
        "source": {"repo_id": repo_id, "revision": revision},
        "parts": parts,
        "weight_map": manifest_weight_map,
    }
    from vllm.model_executor.model_loader.flashpack_loader import (
        parse_flashpack_index,
    )

    parse_flashpack_index(manifest)
    manifest_path = os.path.join(dest_dir, FLASHPACK_INDEX_NAME)
    with open(manifest_path + ".tmp", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=1, sort_keys=True)
    os.replace(manifest_path + ".tmp", manifest_path)

    for entry in sorted(os.listdir(source_dir)):
        source_path = os.path.join(source_dir, entry)
        if (
            entry == SAFE_WEIGHTS_INDEX_NAME
            or entry.endswith(".safetensors")
            or not os.path.isfile(source_path)
        ):
            continue
        shutil.copyfile(source_path, os.path.join(dest_dir, entry))

    total = sum(part["size"] for part in parts.values())
    print(
        f"Wrote {len(parts)} parts ({total / 2**30:.2f} GiB) and "
        f"{FLASHPACK_INDEX_NAME} to {dest_dir}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Directory holding the safetensors checkpoint (e.g. an HF cache snapshot)",
    )
    parser.add_argument(
        "--dest-dir",
        required=True,
        help="Output directory for the FlashPack checkpoint",
    )
    parser.add_argument(
        "--repo-id",
        help="Manifest source repo id; inferred from an HF cache "
        "snapshot path when omitted",
    )
    parser.add_argument(
        "--revision",
        help="Manifest source revision; inferred from an HF cache "
        "snapshot path when omitted",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Re-read each packed part and compare every tensor "
        "against the source shard",
    )
    args = parser.parse_args()

    source_dir = os.path.abspath(args.source_dir)
    dest_dir = os.path.abspath(args.dest_dir)
    if os.path.commonpath([source_dir, dest_dir]) == source_dir:
        parser.error("--dest-dir must not be inside --source-dir")
    inferred_repo_id, inferred_revision = _infer_repo_and_revision(source_dir)
    repo_id = args.repo_id or inferred_repo_id
    revision = args.revision or inferred_revision
    if not repo_id or not revision:
        parser.error(
            "--repo-id and --revision are required when --source-dir is "
            "not an HF cache snapshot path"
        )

    convert(source_dir, dest_dir, repo_id, revision, verify=args.verify)
    return 0


if __name__ == "__main__":
    sys.exit(main())
