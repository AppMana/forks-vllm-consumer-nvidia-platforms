# SPDX-License-Identifier: Apache-2.0
"""Create a symlink-only checkpoint view for DSV4 boundary diagnostics."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} SOURCE_SNAPSHOT TARGET_DIRECTORY")
    source = Path(sys.argv[1]).resolve()
    target = Path(sys.argv[2])
    target.mkdir(parents=True, exist_ok=True)

    with (source / "model.safetensors.index.json").open() as handle:
        index = json.load(handle)
    weight_map = index["weight_map"]

    boundary_weights = {
        name: shard
        for name, shard in weight_map.items()
        if name.startswith("layers.0.")
        or name
        in {
            "embed.weight",
            "embed_tokens.weight",
            "norm.weight",
            "head.weight",
        }
    }
    if not boundary_weights:
        raise RuntimeError("no boundary weights matched the checkpoint index")

    metadata = dict(index.get("metadata", {}))
    metadata["boundary_snapshot"] = "layer0+embedding+norm+head"
    with (target / "model.safetensors.index.json").open("w") as handle:
        json.dump(
            {"metadata": metadata, "weight_map": boundary_weights},
            handle,
            sort_keys=True,
        )

    artifacts = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        *boundary_weights.values(),
    }
    for name in artifacts:
        destination = target / name
        if destination.is_symlink() or destination.exists():
            destination.unlink()
        destination.symlink_to(os.path.relpath(source / name, target))

    print(
        f"created {target}: {len(boundary_weights)} tensors in "
        f"{len(set(boundary_weights.values()))} shards"
    )
    for shard in sorted(set(boundary_weights.values())):
        print(shard)


if __name__ == "__main__":
    main()
