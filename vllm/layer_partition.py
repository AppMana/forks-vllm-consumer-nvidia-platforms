# SPDX-License-Identifier: Apache-2.0
"""Helpers for deterministic PP layer ownership and shard localization.

The shard selector must use the same uneven split policy as
``vllm.distributed.utils.get_pp_indices``. Otherwise a rank-local safetensors
view can contain a different layer range than the model instance on that rank.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _is_embedding_weight(name: str) -> bool:
    return name.startswith(("embed.", "embed_tokens", "model.embed_tokens."))


def compute_layer_counts(num_layers: int, pp_size: int) -> list[int]:
    if num_layers < 0:
        raise ValueError(f"num_layers must be non-negative, got {num_layers}")
    if pp_size <= 0:
        raise ValueError(f"pp_size must be positive, got {pp_size}")

    base, remainder = divmod(num_layers, pp_size)
    partitions = [base] * pp_size
    # Match vllm.distributed.utils.get_pp_indices: put remainder layers on the
    # middle/tail ranks, excluding the final rank because it also owns the
    # output head/norm. Examples: 7/3 -> 2,3,2; 43/10 ->
    # 4,4,4,4,4,4,5,5,5,4.
    for i in range(2, remainder + 2):
        partitions[-i] += 1
    return partitions


def compute_layer_range(num_layers: int, pp_size: int, pp_rank: int) -> tuple[int, int]:
    if not 0 <= pp_rank < pp_size:
        raise ValueError(f"pp_rank must be in [0, {pp_size}), got {pp_rank}")
    counts = compute_layer_counts(num_layers, pp_size)
    start = sum(counts[:pp_rank])
    return start, start + counts[pp_rank]


def load_num_layers(config_path: Path) -> int:
    with config_path.open() as f:
        return int(json.load(f)["num_hidden_layers"])


def rank_to_pp_rank(rank: int, tp_size: int, pp_size: int) -> int:
    if tp_size <= 0:
        raise ValueError(f"tp_size must be positive, got {tp_size}")
    pp_rank = rank // tp_size
    if not 0 <= pp_rank < pp_size:
        raise ValueError(
            f"rank {rank} maps to PP rank {pp_rank}, outside [0, {pp_size})")
    return pp_rank


def select_shards(
    index_path: Path,
    config_path: Path,
    rank: int,
    tp_size: int,
    pp_size: int,
) -> list[str]:
    pp_rank = rank_to_pp_rank(rank, tp_size, pp_size)
    start, end = compute_layer_range(load_num_layers(config_path), pp_size,
                                     pp_rank)
    with index_path.open() as f:
        weights = json.load(f)["weight_map"]

    needed: set[str] = set()
    for name, shard in weights.items():
        if name.startswith("mtp."):
            if pp_rank == pp_size - 1:
                needed.add(shard)
            continue

        match = _LAYER_RE.search(name)
        if match:
            layer = int(match.group(1))
            if start <= layer < end:
                needed.add(shard)
            continue

        if _is_embedding_weight(name):
            if pp_rank == 0:
                needed.add(shard)
            continue
        elif pp_rank == pp_size - 1:
            needed.add(shard)

    return sorted(needed)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    partition = subparsers.add_parser("partition")
    partition.add_argument("--config", type=Path, required=True)
    partition.add_argument("--pp-size",
                           type=int,
                           default=_env_int("VLLM_PIPELINE_PARALLEL_SIZE", 1))

    layers = subparsers.add_parser("layers")
    layers.add_argument("--config", type=Path, required=True)
    layers.add_argument("--rank",
                        type=int,
                        default=_env_int("APPMANA_DSV4_RANK", 0))
    layers.add_argument("--tp-size",
                        type=int,
                        default=_env_int("VLLM_TENSOR_PARALLEL_SIZE", 1))
    layers.add_argument("--pp-size",
                        type=int,
                        default=_env_int("VLLM_PIPELINE_PARALLEL_SIZE", 1))

    shards = subparsers.add_parser("shards")
    shards.add_argument("--index", type=Path, required=True)
    shards.add_argument("--config", type=Path, required=True)
    shards.add_argument("--rank",
                        type=int,
                        default=_env_int("APPMANA_DSV4_RANK", 0))
    shards.add_argument("--tp-size",
                        type=int,
                        default=_env_int("VLLM_TENSOR_PARALLEL_SIZE", 1))
    shards.add_argument("--pp-size",
                        type=int,
                        default=_env_int("VLLM_PIPELINE_PARALLEL_SIZE", 1))

    args = parser.parse_args()
    if args.command == "partition":
        counts = compute_layer_counts(load_num_layers(args.config), args.pp_size)
        print(",".join(str(count) for count in counts))
    elif args.command == "layers":
        pp_rank = rank_to_pp_rank(args.rank, args.tp_size, args.pp_size)
        start, end = compute_layer_range(load_num_layers(args.config),
                                         args.pp_size, pp_rank)
        print(f"{start}:{end}")
    elif args.command == "shards":
        for shard in select_shards(args.index, args.config, args.rank,
                                   args.tp_size, args.pp_size):
            print(shard)


if __name__ == "__main__":
    main()
