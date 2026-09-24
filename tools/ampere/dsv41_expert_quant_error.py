#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer requantization error of DeepSeek V4.1 routed experts.

Reads the MXFP4 experts (I8-packed E2M1 + E8M0 per 32) straight from the
checkpoint shards, requantizes them to Humming uint{bits} with MSE group
scales (requantize_mxfp4_to_humming_uint, the same path the checkpoint
converter uses) and reports the round-trip SNR against the exact MXFP4
values, per layer and projection. This is weight-space error; it ranks
layers for the mixed int3/int4 checkpoint before output-error measurements
on the running model decide the final map.

Run from the repository root:
    python tools/ampere/dsv41_expert_quant_error.py <snapshot dir> \
        --formats 3:64 4:64 3:128 4:128 --expert-stride 8 --out errors.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import regex as re
import torch
from safetensors import safe_open

from vllm.model_executor.layers.quantization.dsv4_int import (
    _mxfp4_grouped_values,
    dequantize_humming_uint,
    requantize_mxfp4_to_humming_uint,
)

_EXPERT_RE = re.compile(r"^(layers|mtp)\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.weight$")


def expert_locations(snapshot: Path) -> dict[tuple[str, int], list[tuple]]:
    """(block kind, block index) -> [(expert, proj, shard path)] from headers."""
    found: dict[tuple[str, int], list[tuple]] = defaultdict(list)
    for shard in sorted(snapshot.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():  # noqa: SIM118
                match = _EXPERT_RE.match(name)
                if match is not None:
                    kind, block, expert, proj = match.groups()
                    found[(kind, int(block))].append((int(expert), proj, shard))
    return found


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("snapshot", type=Path)
    p.add_argument("--formats", nargs="+", default=["3:64", "4:64", "3:128", "4:128"])
    p.add_argument("--expert-stride", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    formats = [tuple(int(v) for v in f.split(":")) for f in args.formats]
    locations = expert_locations(args.snapshot)
    with args.out.open("a") as out:
        for (kind, block), entries in sorted(locations.items()):
            signal: dict[str, float] = defaultdict(float)
            noise: dict[tuple[int, int, str], float] = defaultdict(float)
            sampled = sorted(
                (e for e in entries if e[0] % args.expert_stride == 0),
                key=lambda e: (e[0], e[1]),
            )
            for expert, proj, shard in sampled:
                prefix = f"{kind}.{block}.ffn.experts.{expert}.{proj}"
                with safe_open(shard, framework="pt", device="cpu") as handle:
                    packed = handle.get_tensor(f"{prefix}.weight")
                    scale = handle.get_tensor(f"{prefix}.scale")
                packed = packed.to(args.device)
                scale = scale.to(args.device)
                ref = _mxfp4_grouped_values(packed, scale).reshape(packed.shape[0], -1)
                signal[proj] += float(ref.double().square().sum())
                for bits, group in formats:
                    q = requantize_mxfp4_to_humming_uint(
                        packed, scale, bits=bits, group_size=group
                    )
                    deq = dequantize_humming_uint(
                        q["codes"], q["weight_scale"], bits=bits, group_size=group
                    )
                    noise[(bits, group, proj)] += float(
                        (deq - ref).double().square().sum()
                    )
            for bits, group in formats:
                row = {
                    "block": kind,
                    "index": block,
                    "bits": bits,
                    "group": group,
                    "experts_sampled": len({e[0] for e in sampled}),
                }
                total_s = total_n = 0.0
                for proj in ("w1", "w2", "w3"):
                    n = noise[(bits, group, proj)]
                    row[f"snr_db_{proj}"] = round(
                        10 * torch.log10(torch.tensor(signal[proj] / n)).item(), 3
                    )
                    total_s += signal[proj]
                    total_n += n
                row["snr_db"] = round(
                    10 * torch.log10(torch.tensor(total_s / total_n)).item(), 3
                )
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
