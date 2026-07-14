#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a shrunk-but-real DeepSeek-V4-Flash-DSpark HF config directory.

Starts from the REAL released config.json for deepseek-ai/DeepSeek-V4-Flash-DSpark
(must already be present in the local HF cache -- this tool does not download
anything) and mechanically truncates it to a few layers, matching this fork's
"shrunk model, unchanged GEMM shapes" local-dummy-smoke convention (see
tools/ampere/dsv4_int_vllm_smoke.py): every hyperparameter stays at its real
released value EXCEPT:

  * num_hidden_layers: reduced from 43 to a small prefix (default 6). The
    prefix is taken from the START of the real compress_ratios array, which is
    [0, 0, 4, 128, 4, 128, ...] -- i.e. it naturally includes 2 SWA-only
    layers (ratio 0) and (with the default prefix length) 2 compressed layers
    (ratio 4, ratio 128), the same layer-type mix the real 43-layer model
    uses. This is an exact prefix of a real array, not an invented pattern.
  * dspark_target_layer_ids: mechanically reindexed from the real [40, 41, 42]
    (the LAST 3 of 43 main-model layers) to the last 3 of the new, smaller
    main model, preserving the same relative position.
  * n_routed_experts: reduced from 256. This is a physical memory constraint,
    not an architecture guess: 256 experts x moe_intermediate_size=2048 x
    hidden_size=4096 x num_hidden_layers does not fit a single consumer GPU
    even with dummy (random, uninitialized) weights. Kept above
    num_experts_per_tok (real value 6) so top-k routing is still meaningfully
    exercised among more candidates than are selected.

Everything else (attention head dims, rope theta/scaling, hc_mult, dspark_*
fields besides target_layer_ids, quantization_config, etc.) is copied
byte-for-byte from the real released config.

Intended for use with `--load-format dummy` (vLLM's deterministic seed=1234
random weight init, see model_loader/weight_utils.py:initialize_dummy_weights)
-- this tool does NOT touch or require the real (multi-hundred-GB) safetensors
weights, only the small config.json/tokenizer files.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path


def find_real_dspark_config() -> Path:
    """Locate the real DeepSeek-V4-Flash-DSpark config.json in the local HF
    cache. Raises if not present -- this tool never downloads or invents one.
    """
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_root / "models--deepseek-ai--DeepSeek-V4-Flash-DSpark"
    candidates = sorted(model_dir.glob("snapshots/*/config.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No local snapshot of deepseek-ai/DeepSeek-V4-Flash-DSpark found "
            f"under {model_dir}. This tool builds a shrunk config from the "
            f"real released one; it does not invent config values."
        )
    return candidates[0]


def build_shrunk_config(
    real_config_path: Path,
    num_layers: int = 6,
    n_routed_experts: int = 8,
) -> dict:
    with open(real_config_path) as f:
        real = json.load(f)

    real_target_ids = real["dspark_target_layer_ids"]
    real_num_layers = real["num_hidden_layers"]
    assert len(real_target_ids) == 3, real_target_ids
    assert real_target_ids == list(
        range(real_num_layers - 3, real_num_layers)
    ), "expected dspark_target_layer_ids to be the real model's last 3 layers"

    real_compress = real["compress_ratios"]
    assert len(real_compress) == real_num_layers + 3, (
        "expected compress_ratios to be num_hidden_layers main entries + 3 "
        "trailing entries for the mtp draft layers"
    )
    if num_layers > real_num_layers:
        raise ValueError(f"num_layers must be <= real {real_num_layers}")

    cfg = copy.deepcopy(real)
    cfg["num_hidden_layers"] = num_layers
    # exact prefix of the real per-layer array + the real trailing mtp-layer
    # entries (always 3, always [0, 0, 0] in the real config).
    cfg["compress_ratios"] = real_compress[:num_layers] + real_compress[-3:]
    cfg["dspark_target_layer_ids"] = [num_layers - 3, num_layers - 2, num_layers - 1]

    real_n_experts = real["n_routed_experts"]
    real_topk = real["num_experts_per_tok"]
    if n_routed_experts < real_topk:
        raise ValueError(
            f"n_routed_experts ({n_routed_experts}) must be >= "
            f"num_experts_per_tok ({real_topk}) or top-k routing is meaningless"
        )
    cfg["n_routed_experts"] = min(n_routed_experts, real_n_experts)

    return cfg


def write_shrunk_checkpoint_dir(out_dir: Path, cfg: dict, real_config_path: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    snap_dir = real_config_path.parent
    for fname in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        src = snap_dir / fname
        if src.exists():
            shutil.copyfile(src, out_dir / fname)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--n-routed-experts", type=int, default=8)
    args = parser.parse_args()

    real_config_path = find_real_dspark_config()
    cfg = build_shrunk_config(
        real_config_path,
        num_layers=args.num_layers,
        n_routed_experts=args.n_routed_experts,
    )
    out_dir = Path(args.out_dir)
    write_shrunk_checkpoint_dir(out_dir, cfg, real_config_path)

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "real_config": str(real_config_path),
                "num_hidden_layers": cfg["num_hidden_layers"],
                "compress_ratios": cfg["compress_ratios"],
                "dspark_target_layer_ids": cfg["dspark_target_layer_ids"],
                "n_routed_experts": cfg["n_routed_experts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
