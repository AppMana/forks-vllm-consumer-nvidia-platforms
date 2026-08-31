# SPDX-License-Identifier: Apache-2.0
"""Helpers for deterministic PP layer ownership and shard localization.

The shard selector must use the same uneven split policy as
``vllm.distributed.utils.get_pp_indices``. Otherwise a rank-local safetensors
view can contain a different layer range than the model instance on that rank.

Checkpoints that graft MTP/DSpark draft stages onto the last PP rank need a
different split: that rank pays for the draft block on top of its share of
target layers. Rather than making operators know a flag, this module detects
the draft stages from the checkpoint and balances the split by COST in
layer-equivalents, where the last rank's cost includes the draft block and
rank 0's includes the input embedding.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path


_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_MTP_STAGE_RE = re.compile(r"^mtp\.(\d+)\.")

# The input embedding in layer-equivalents. On the DSV4 int4/int8 rebuild a
# decoder layer measures ~3.6 GiB against a ~0.9 GiB embedding table. It only
# ever breaks ties between otherwise equal-cost ranks; it never outweighs a
# whole layer.
DEFAULT_EMBED_COST = 0.25

# The draft block in layer-equivalents when a checkpoint says it has draft
# stages but gives no way to weigh them. The DSpark graft this module was
# written for is 3 stages / 4708 tensors against a 1571-tensor mean layer.
DEFAULT_MTP_COST = 3.0

# Granularity the weight-index estimate is snapped to. Tensor counts are a
# proxy for bytes, so the raw ratio lands near but not on the stage count
# (4708/1571.23 = 2.9964). Left raw it would put the draft rank a hair below
# its neighbours and win every tie, moving an extra layer onto the one rank
# that must stay light. 0.25 is far inside the estimate's own error.
MTP_COST_QUANTUM = 0.25


def _is_embedding_weight(name: str) -> bool:
    return name.startswith(("embed.", "embed_tokens", "model.embed_tokens."))


def detect_mtp_cost(
    config: Mapping[str, object],
    weight_map: Mapping[str, str] | None = None,
) -> float:
    """Cost of a checkpoint's MTP/draft block in layer-equivalents.

    ``num_hidden_layers``' companion ``num_nextn_predict_layers`` is the
    primary presence signal. It is not a reliable stage COUNT:
    deepseek-ai/DeepSeek-V4-Flash-0731 ships the full 3-stage DSpark subtree
    while declaring 1. When a weight map is supplied its ``mtp.<stage>.*``
    tensor groups both confirm the presence and weigh the block against the
    mean ``layers.<i>.*`` group, which is the accurate number.

    Returns 0.0 when the checkpoint carries no draft stages.
    """
    declared = int(config.get("num_nextn_predict_layers") or 0)

    stages: set[int] = set()
    mtp_tensors = 0
    layer_tensors: defaultdict[int, int] = defaultdict(int)
    if weight_map is not None:
        for name in weight_map:
            stage = _MTP_STAGE_RE.match(name)
            if stage is not None:
                stages.add(int(stage.group(1)))
                mtp_tensors += 1
                continue
            layer = _LAYER_RE.search(name)
            if layer is not None:
                layer_tensors[int(layer.group(1))] += 1

    if declared <= 0 and not stages:
        return 0.0

    if stages and layer_tensors:
        mean_layer_tensors = sum(layer_tensors.values()) / len(layer_tensors)
        if mean_layer_tensors > 0:
            estimate = mtp_tensors / mean_layer_tensors
            return round(estimate / MTP_COST_QUANTUM) * MTP_COST_QUANTUM

    return float(declared or len(stages))


def resolve_mtp_cost(
    config_path: Path,
    index_path: Path | None = None,
    draft_zero_last: bool | None = None,
) -> float:
    """Draft-block cost for a checkpoint, honouring an explicit override.

    Precedence, highest first:

    1. ``draft_zero_last=False`` forces the plain ``get_pp_indices`` split
       (cost 0.0) even for a checkpoint that carries draft stages -- the
       serving process is not loading them (no ``--speculative-config``).
    2. ``draft_zero_last=True`` forces draft-aware balancing even for a
       checkpoint with no signal, falling back to ``DEFAULT_MTP_COST``.
    3. ``draft_zero_last=None`` (the default) auto-detects from the
       checkpoint via :func:`detect_mtp_cost`.
    """
    if draft_zero_last is False:
        return 0.0

    with config_path.open() as f:
        config = json.load(f)
    weight_map = None
    if index_path is not None:
        with index_path.open() as f:
            weight_map = json.load(f)["weight_map"]

    cost = detect_mtp_cost(config, weight_map)
    if draft_zero_last and cost <= 0.0:
        return DEFAULT_MTP_COST
    return cost


def _legacy_counts(num_layers: int, pp_size: int) -> list[int]:
    base, remainder = divmod(num_layers, pp_size)
    partitions = [base] * pp_size
    # Match vllm.distributed.utils.get_pp_indices: put remainder layers on the
    # middle/tail ranks, excluding the final rank because it also owns the
    # output head/norm. Examples: 7/3 -> 2,3,2; 43/10 ->
    # 4,4,4,4,4,4,5,5,5,4.
    for i in range(2, remainder + 2):
        partitions[-i] += 1
    return partitions


def _legacy_draft_zero_last_counts(num_layers: int, pp_size: int) -> list[int]:
    """The pre-cost-balancer draft policy: zero target layers on the last rank.

    Retained as the reference the balancer is compared against; it leaves the
    draft rank idle at ~3 layer-equivalents while the ranks that absorbed its
    share run 25% hot. Example: 43/11 -> 4,4,4,4,4,4,4,5,5,5,0.
    """
    if pp_size < 2:
        raise ValueError(f"draft_zero_last requires pp_size >= 2, got {pp_size}")
    effective = pp_size - 1
    base, remainder = divmod(num_layers, effective)
    partitions = [base] * effective + [0]
    for i in range(1, remainder + 1):
        partitions[effective - i] += 1
    return partitions


def _balanced_counts(
    num_layers: int,
    pp_size: int,
    mtp_cost: float,
    embed_cost: float,
) -> list[int]:
    """Contiguous split minimising the heaviest rank's cost, then levelling.

    Every target layer costs 1.0; the last rank starts at ``mtp_cost`` and
    rank 0 at ``embed_cost``. Layers are handed one at a time to whichever
    rank is cheapest, which for unit-cost items minimises the maximum rank
    cost and, because a rank's final layer was placed while it was the global
    minimum, leaves no rank more than one layer-equivalent above the lightest.

    Ranks that start at the same cost are interchangeable, so their counts are
    finally re-laid ascending by index. That reproduces get_pp_indices' habit
    of loading the tail ranks and keeps the answer stable.
    """
    extras = [0.0] * pp_size
    extras[0] += embed_cost
    extras[-1] += mtp_cost

    counts = [0] * pp_size
    costs = list(extras)
    for _ in range(num_layers):
        cheapest = min(range(pp_size), key=lambda rank: (costs[rank], rank))
        counts[cheapest] += 1
        costs[cheapest] += 1.0

    groups: defaultdict[float, list[int]] = defaultdict(list)
    for rank, extra in enumerate(extras):
        groups[round(extra, 6)].append(rank)
    for ranks in groups.values():
        for rank, count in zip(ranks, sorted(counts[r] for r in ranks)):
            counts[rank] = count

    # A draft-only final rank is a qualitatively different PP stage: it has no
    # target-model forward to bridge the ordinary pipeline and the draft/head
    # path. When there are enough target layers to populate every rank and the
    # move is cost-neutral, retain that seam by moving only the final contiguous
    # layer boundary. Taking the layer from the nearest donor preserves every
    # earlier pipeline boundary and keeps the maximum load unchanged.
    peak_cost = max(count + extra for count, extra in zip(counts, extras))
    seam_cost = mtp_cost + 1.0
    if (num_layers >= pp_size and counts[-1] == 0
            and seam_cost <= peak_cost):
        donor = next(
            (rank for rank in range(pp_size - 2, -1, -1)
             if counts[rank] > 1),
            None,
        )
        if donor is None:
            raise AssertionError(
                f"balanced partition {counts} has no donor for its empty "
                "final rank")
        counts[donor] -= 1
        counts[-1] = 1

    return counts


def compute_layer_counts(
    num_layers: int,
    pp_size: int,
    draft_zero_last: bool = False,
    *,
    mtp_cost: float | None = None,
    embed_cost: float = DEFAULT_EMBED_COST,
) -> list[int]:
    """Target layers per PP rank.

    ``mtp_cost`` in layer-equivalents switches on the cost balancer; 0.0 (or
    ``None`` without ``draft_zero_last``) keeps the plain get_pp_indices
    split. ``draft_zero_last=True`` forces the balancer on with
    ``DEFAULT_MTP_COST`` when no cost is supplied.
    """
    if num_layers < 0:
        raise ValueError(f"num_layers must be non-negative, got {num_layers}")
    if pp_size <= 0:
        raise ValueError(f"pp_size must be positive, got {pp_size}")

    effective_cost = 0.0 if mtp_cost is None else float(mtp_cost)
    if draft_zero_last:
        if pp_size < 2:
            raise ValueError(
                f"draft_zero_last requires pp_size >= 2, got {pp_size}")
        if effective_cost <= 0.0:
            effective_cost = DEFAULT_MTP_COST

    if effective_cost <= 0.0:
        return _legacy_counts(num_layers, pp_size)

    counts = _balanced_counts(num_layers, pp_size, effective_cost, embed_cost)

    if sum(counts) != num_layers or any(count < 0 for count in counts):
        raise AssertionError(
            f"balanced partition {counts} is not a valid split of "
            f"{num_layers} layers over {pp_size} ranks")
    return counts


def compute_layer_range(
    num_layers: int,
    pp_size: int,
    pp_rank: int,
    draft_zero_last: bool = False,
    *,
    mtp_cost: float | None = None,
    embed_cost: float = DEFAULT_EMBED_COST,
) -> tuple[int, int]:
    if not 0 <= pp_rank < pp_size:
        raise ValueError(f"pp_rank must be in [0, {pp_size}), got {pp_rank}")
    counts = compute_layer_counts(num_layers,
                                  pp_size,
                                  draft_zero_last,
                                  mtp_cost=mtp_cost,
                                  embed_cost=embed_cost)
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
    draft_zero_last: bool | None = None,
) -> list[str]:
    pp_rank = rank_to_pp_rank(rank, tp_size, pp_size)
    mtp_cost = resolve_mtp_cost(config_path, index_path, draft_zero_last)
    start, end = compute_layer_range(load_num_layers(config_path),
                                     pp_size,
                                     pp_rank,
                                     mtp_cost=mtp_cost)
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


def _add_draft_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--draft-zero-last",
                        action=argparse.BooleanOptionalAction,
                        default=None,
                        help="force draft-aware balancing on (--draft-zero-last) "
                        "or off (--no-draft-zero-last); the default detects "
                        "MTP/draft stages from the checkpoint")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    partition = subparsers.add_parser("partition")
    partition.add_argument("--config", type=Path, required=True)
    partition.add_argument("--index", type=Path, default=None)
    partition.add_argument("--pp-size",
                           type=int,
                           default=_env_int("VLLM_PIPELINE_PARALLEL_SIZE", 1))
    _add_draft_flags(partition)

    layers = subparsers.add_parser("layers")
    layers.add_argument("--config", type=Path, required=True)
    layers.add_argument("--index", type=Path, default=None)
    layers.add_argument("--rank",
                        type=int,
                        default=_env_int("APPMANA_DSV4_RANK", 0))
    layers.add_argument("--tp-size",
                        type=int,
                        default=_env_int("VLLM_TENSOR_PARALLEL_SIZE", 1))
    layers.add_argument("--pp-size",
                        type=int,
                        default=_env_int("VLLM_PIPELINE_PARALLEL_SIZE", 1))
    _add_draft_flags(layers)

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
    _add_draft_flags(shards)

    args = parser.parse_args()
    if args.command == "partition":
        mtp_cost = resolve_mtp_cost(args.config, args.index,
                                    args.draft_zero_last)
        counts = compute_layer_counts(load_num_layers(args.config),
                                      args.pp_size,
                                      mtp_cost=mtp_cost)
        print(",".join(str(count) for count in counts))
    elif args.command == "layers":
        pp_rank = rank_to_pp_rank(args.rank, args.tp_size, args.pp_size)
        mtp_cost = resolve_mtp_cost(args.config, args.index,
                                    args.draft_zero_last)
        start, end = compute_layer_range(load_num_layers(args.config),
                                         args.pp_size,
                                         pp_rank,
                                         mtp_cost=mtp_cost)
        print(f"{start}:{end}")
    elif args.command == "shards":
        for shard in select_shards(args.index, args.config, args.rank,
                                   args.tp_size, args.pp_size,
                                   args.draft_zero_last):
            print(shard)


if __name__ == "__main__":
    main()
