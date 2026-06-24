#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convert DeepSeek V4 FP4/FP8 checkpoint shards to Ampere-friendly formats.

The conservative Ampere baseline is ``dsv4_int``:

* routed expert MXFP4 weights -> symmetric INT4 W4A16, group size 32
* FP8 linears -> symmetric INT8 W8A16, 128x128 blocks by default, or
  channelwise biased UINT8 for the AllSpark Ampere W8A16 kernel
* BF16/F32/etc. tensors -> passthrough

The hybrid comparison path is ``dsv4_mxfp4_int8``:

* routed expert MXFP4 weights/scales -> preserved byte-for-byte for Marlin
* FP8 linears -> INT8 as above
* BF16/F32/etc. tensors -> passthrough

The converter preserves tensor names and shard names so the original
``model.safetensors.index.json`` remains valid.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from dsv4_checkpoint_audit import classify_tensor, matched_scale_name  # noqa: E402

from vllm.model_executor.layers.quantization.dsv4_int import (  # noqa: E402
    requantize_fp8_to_allspark_uint8_w8a16,
    requantize_fp8_to_int8_w8a16,
    requantize_mxfp4_to_int4_w4a16,
)

_FP8_WEIGHT_ROLES = {
    "dense_fp8_weight",
    "indexer_qk_fp8_weight",
    "mtp_fp8_weight",
}
_LAYER_NAME_RE = re.compile(r"^layers\.(\d+)\.(.*)$")


def _log(message: str) -> None:
    print(f"[dsv4-requant] {message}", flush=True)


def _remap_tensor_name(name: str, layer_remap: dict[int, int] | None) -> str | None:
    if layer_remap is None:
        return name
    match = _LAYER_NAME_RE.match(name)
    if match is None:
        return name
    source_idx = int(match.group(1))
    if source_idx not in layer_remap:
        return None
    return f"layers.{layer_remap[source_idx]}.{match.group(2)}"


_EXPERT_NAME_RE = re.compile(r"^layers\.\d+\.ffn\.experts\.(\d+)\.")
_GATE_NAME_RE = re.compile(r"^layers\.\d+\.ffn\.gate\.(weight|bias)$")


def _subset_drop(name: str, keep_experts: int | None, drop_mtp: bool) -> bool:
    """True when a tensor is dropped by the testbed subset options."""
    if drop_mtp and name.startswith("mtp."):
        return True
    if keep_experts is not None:
        match = _EXPERT_NAME_RE.match(name)
        if match is not None and int(match.group(1)) >= keep_experts:
            return True
    return False


def _subset_slice(
    name: str, tensor: torch.Tensor, keep_experts: int | None
) -> torch.Tensor:
    """Slice the router gate down to the kept expert rows."""
    if keep_experts is not None and _GATE_NAME_RE.match(name) is not None:
        return tensor[:keep_experts].contiguous()
    if keep_experts is not None and name.endswith(".gate.tid2eid"):
        # Hash-routing table maps vocab ids to expert ids of the FULL model.
        # The topk_softplus_sqrt hash path silently skips writing topk
        # outputs whose table entry is >= n_routed_experts, leaving the
        # torch.empty routing buffers uninitialized and crashing the MoE
        # kernel downstream. Remap into the kept-expert range.
        return (tensor % keep_experts).contiguous()
    return tensor


def _discover_layer_remap(src: Path) -> dict[int, int] | None:
    cfg = json.loads((src / "config.json").read_text())
    expected_layers = int(cfg.get("num_hidden_layers", 0))
    if expected_layers <= 0:
        return None

    index_path = src / "model.safetensors.index.json"
    if not index_path.exists():
        return None
    weight_map = json.loads(index_path.read_text())["weight_map"]
    layer_ids = sorted(
        {
            int(match.group(1))
            for name in weight_map
            if (match := _LAYER_NAME_RE.match(name)) is not None
        }
    )
    if layer_ids == list(range(expected_layers)):
        return None
    if len(layer_ids) != expected_layers:
        raise ValueError(
            f"cannot auto-remap {layer_ids=} to {expected_layers=} layers"
        )
    return {source: target for target, source in enumerate(layer_ids)}


def _copy_metadata(src: Path, dst: Path) -> None:
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
    ):
        src_path = src / name
        if src_path.exists():
            shutil.copy(src_path.resolve(), dst / name)


def _write_index(
    src: Path,
    dst: Path,
    layer_remap: dict[int, int] | None,
    keep_experts: int | None = None,
    drop_mtp: bool = False,
) -> None:
    index_path = src / "model.safetensors.index.json"
    if not index_path.exists():
        return
    index = json.loads(index_path.read_text())
    remapped_weight_map = {}
    for name, shard in index["weight_map"].items():
        if _subset_drop(name, keep_experts, drop_mtp):
            continue
        remapped = _remap_tensor_name(name, layer_remap)
        if remapped is not None:
            remapped_weight_map[remapped] = shard
    index["weight_map"] = remapped_weight_map
    (dst / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )


def _find_required_tensor(
    weight_map: dict[str, str], candidates: tuple[str, ...]
) -> str:
    for name in candidates:
        if name in weight_map:
            return name
    raise KeyError(f"none of {candidates!r} found in checkpoint")


def _load_index_or_build(dst: Path) -> dict[str, object]:
    index_path = dst / "model.safetensors.index.json"
    if index_path.exists():
        return json.loads(index_path.read_text())
    weight_map: dict[str, str] = {}
    for shard in sorted(dst.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle:
                weight_map[key] = shard.name
    return {"metadata": {}, "weight_map": weight_map}


def _copy_tensor_by_name(
    checkpoint: Path, weight_map: dict[str, str], name: str
) -> torch.Tensor:
    with safe_open(
        checkpoint / weight_map[name], framework="pt", device="cpu"
    ) as handle:
        return handle.get_tensor(name)


def _ensure_mtp_shared_tensors(dst: Path) -> None:
    """Materialize MTP shared embedding/head tensors into quantized snapshots.

    DeepSeek V4 checkpoints commonly store MTP as only the NextN-specific layer
    weights plus one target-level ``embed.weight`` and ``head.weight``. vLLM's
    MTP loader can share those modules at runtime for PP=1, but PP deployments
    load the draft model separately and need the checkpoint to contain the MTP
    names that the loader maps into ``model.embed_tokens`` and
    ``shared_head.head``.
    """

    cfg = json.loads((dst / "config.json").read_text())
    num_mtp_layers = int(cfg.get("num_nextn_predict_layers", 0) or 0)
    if num_mtp_layers <= 0:
        return

    index = _load_index_or_build(dst)
    weight_map = dict(index["weight_map"])
    mtp_prefixes = {
        name.split(".", 2)[1] for name in weight_map if name.startswith("mtp.")
    }
    if not mtp_prefixes:
        return

    embed_name = _find_required_tensor(
        weight_map,
        ("embed.weight", "model.embed.weight", "model.embed_tokens.weight"),
    )
    head_name = _find_required_tensor(
        weight_map,
        ("head.weight", "lm_head.weight", "model.head.weight"),
    )

    additions: dict[str, torch.Tensor] = {}
    for mtp_idx in range(num_mtp_layers):
        embed_alias = f"mtp.{mtp_idx}.emb.tok_emb.weight"
        head_alias = f"mtp.{mtp_idx}.head.weight"
        if embed_alias not in weight_map:
            additions[embed_alias] = _copy_tensor_by_name(dst, weight_map, embed_name)
        if head_alias not in weight_map:
            additions[head_alias] = _copy_tensor_by_name(dst, weight_map, head_name)

    if not additions:
        return

    shard_name = "model-mtp-shared.safetensors"
    save_file(additions, str(dst / shard_name))
    for name in additions:
        weight_map[name] = shard_name

    total_size = sum(path.stat().st_size for path in dst.glob("*.safetensors"))
    index["weight_map"] = weight_map
    index.setdefault("metadata", {})
    index["metadata"]["total_size"] = str(total_size)
    index["metadata"]["mtp_shared_tensors"] = "materialized"
    (dst / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    _log(
        "materialized MTP shared tensors: "
        + ", ".join(sorted(additions))
        + f" -> {shard_name}"
    )


def _assign_output_shard(
    name: str,
    *,
    num_output_shards: int,
    num_hidden_layers: int,
) -> int:
    if num_output_shards <= 1:
        return 0
    match = _LAYER_NAME_RE.match(name)
    if match is not None:
        layer_id = int(match.group(1))
        if num_output_shards <= 2:
            return min(
                num_output_shards - 1,
                layer_id * num_output_shards // max(1, num_hidden_layers),
            )
        # Reserve shard 0 for embeddings / config-adjacent tensors and the
        # final shard for norm/head/MTP. Hidden layers are laid out
        # monotonically across the middle shards so PP ranks open a narrow,
        # predictable file range.
        span = num_output_shards - 2
        return 1 + min(span - 1, layer_id * span // max(1, num_hidden_layers))
    if name.startswith(("embed.", "model.embed.")):
        return 0
    if name.startswith(("norm.", "head.", "hc_head", "mtp.")):
        return num_output_shards - 1
    return 0


def _reshard_safetensors(
    dst: Path,
    *,
    num_output_shards: int,
    num_hidden_layers: int,
) -> None:
    if num_output_shards <= 0:
        raise ValueError("num_output_shards must be positive")

    index_path = dst / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map: dict[str, str] = dict(index["weight_map"])
    else:
        weight_map = {}
        for shard in sorted(dst.glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as handle:
                for key in handle:
                    weight_map[key] = shard.name
        index = {"metadata": {}, "weight_map": weight_map}

    old_shards = sorted({dst / shard for shard in weight_map.values()})
    names_by_target: dict[int, list[str]] = defaultdict(list)
    for name in sorted(weight_map):
        target = _assign_output_shard(
            name,
            num_output_shards=num_output_shards,
            num_hidden_layers=num_hidden_layers,
        )
        names_by_target[target].append(name)

    tmp_dir = dst / ".reshard-tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

    new_weight_map: dict[str, str] = {}
    total_size = 0
    try:
        for target in range(num_output_shards):
            names = names_by_target.get(target)
            if not names:
                continue
            shard_name = (
                f"model-{target + 1:05d}-of-{num_output_shards:05d}.safetensors"
            )
            out: dict[str, torch.Tensor] = {}
            by_old: dict[str, list[str]] = defaultdict(list)
            for name in names:
                by_old[weight_map[name]].append(name)
            for old_name, old_names in by_old.items():
                with safe_open(dst / old_name, framework="pt", device="cpu") as handle:
                    for name in old_names:
                        out[name] = handle.get_tensor(name)
            save_file(out, str(tmp_dir / shard_name))
            total_size += (tmp_dir / shard_name).stat().st_size
            for name in names:
                new_weight_map[name] = shard_name

        for shard in old_shards:
            shard.unlink()
        for shard in sorted(tmp_dir.glob("*.safetensors")):
            shutil.move(str(shard), dst / shard.name)

        index["weight_map"] = new_weight_map
        index.setdefault("metadata", {})
        index["metadata"]["total_size"] = str(total_size)
        index["metadata"]["num_output_shards"] = str(num_output_shards)
        index["metadata"]["sharding"] = "layer_contiguous_pp_friendly"
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def _write_config(
    src: Path,
    dst: Path,
    layer_remap: dict[int, int] | None,
    *,
    dense_int8_strategy: str,
    expert_format: str,
    expert_int4_scale_mode: str,
    keep_experts: int | None = None,
    drop_mtp: bool = False,
) -> None:
    cfg = json.loads((src / "config.json").read_text())
    if layer_remap is not None:
        cfg["num_hidden_layers"] = len(layer_remap)
    if keep_experts is not None:
        cfg["n_routed_experts"] = keep_experts
    if drop_mtp:
        cfg["num_nextn_predict_layers"] = 0
    dense_weights_cfg: dict[str, object] = {
        "num_bits": 8,
        "type": "int",
        "symmetric": True,
        "strategy": dense_int8_strategy,
    }
    if dense_int8_strategy == "block":
        dense_weights_cfg["block_size"] = [128, 128]
    expert_weights_cfg: dict[str, object] = {
        "num_bits": 4,
        "type": "float" if expert_format == "mxfp4" else "int",
        "format": "mxfp4" if expert_format == "mxfp4" else "int4",
        "symmetric": expert_format != "mxfp4",
        "group_size": 32,
        "strategy": "group",
        "scale_mode": (
            "native_e8m0" if expert_format == "mxfp4" else expert_int4_scale_mode
        ),
    }
    if expert_format == "mxfp4":
        expert_weights_cfg["scale_dtype"] = "e8m0"
    cfg["expert_dtype"] = "fp4" if expert_format == "mxfp4" else "int4"
    quant_method = "dsv4_mxfp4_int8" if expert_format == "mxfp4" else "dsv4_int"
    cfg["quantization_config"] = {
        "quant_method": quant_method,
        "format": "mxfp4_int8_packed" if expert_format == "mxfp4" else "int_packed",
        "config_groups": {
            "experts_w4a16": {
                "weights": expert_weights_cfg,
                "input_activations": {"num_bits": 16, "type": "float"},
                "targets": [
                    "*.ffn.experts.*.w1",
                    "*.ffn.experts.*.w2",
                    "*.ffn.experts.*.w3",
                ],
            },
            "linears_w8a16": {
                "weights": dense_weights_cfg,
                "input_activations": {"num_bits": 16, "type": "float"},
                "targets": [
                    "*.attn.wq_a",
                    "*.attn.wq_b",
                    "*.attn.wkv",
                    "*.attn.wo_a",
                    "*.attn.wo_b",
                    "*.attn.indexer.wq_b",
                    "*.attn.indexer.compressor.wkv",
                    "*.attn.indexer.compressor.wgate",
                    "*.ffn.shared_experts.w1",
                    "*.ffn.shared_experts.w2",
                    "*.ffn.shared_experts.w3",
                    "mtp.*.e_proj",
                    "mtp.*.h_proj",
                ],
            },
        },
        "ignore": [
            "embed",
            "head",
            "norm",
            "lm_head",
            "*norm.weight",
            "attn.attn_sink",
            "*.gate.*",
            "hc_*",
            "*.hc_attn_*",
            "*.hc_ffn_*",
        ],
    }
    (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")


def _classify_shard(
    src_shard: Path,
) -> tuple[dict[str, str], dict[str, str], set[str]]:
    roles: dict[str, str] = {}
    dtypes: dict[str, str] = {}
    missing_scales: set[str] = set()
    with safe_open(src_shard, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for name in sorted(keys):
            tensor_slice = handle.get_slice(name)
            dtype = tensor_slice.get_dtype()
            role, action = classify_tensor(name, dtype)
            if role == "unknown":
                raise ValueError(f"unknown tensor in {src_shard.name}: {name} {dtype}")
            roles[name] = role
            dtypes[name] = dtype
            scale_name = matched_scale_name(name)
            if (
                scale_name is not None
                and action != "preserve"
                and scale_name not in keys
            ):
                missing_scales.add(name)
    return roles, dtypes, missing_scales


def convert_shard(
    src_shard: Path,
    dst_shard: Path,
    *,
    device: str,
    out_scale_dtype: torch.dtype,
    layer_remap: dict[int, int] | None,
    dense_int8_strategy: str,
    expert_format: str,
    expert_int4_scale_mode: str,
    keep_experts: int | None = None,
    drop_mtp: bool = False,
) -> dict[str, int]:
    roles, _dtypes, missing_scales = _classify_shard(src_shard)
    if missing_scales:
        sample = ", ".join(sorted(missing_scales)[:8])
        raise ValueError(f"{src_shard.name} missing scales for: {sample}")

    out: dict[str, torch.Tensor] = {}
    counts = {"int4": 0, "mxfp4": 0, "int8": 0, "preserve": 0}
    paired_scales = {
        matched_scale_name(name)
        for name, role in roles.items()
        if role == "routed_expert_mxfp4_weight" or role in _FP8_WEIGHT_ROLES
    }
    paired_scales.discard(None)

    with safe_open(src_shard, framework="pt", device=device) as handle:
        for name in sorted(handle.keys()):
            if name in paired_scales:
                continue
            if _subset_drop(name, keep_experts, drop_mtp):
                continue
            role = roles[name]
            out_name = _remap_tensor_name(name, layer_remap)
            if out_name is None:
                continue
            if role == "routed_expert_mxfp4_weight":
                scale_name = matched_scale_name(name)
                assert scale_name is not None
                out_scale_name = _remap_tensor_name(scale_name, layer_remap)
                assert out_scale_name is not None
                if expert_format == "mxfp4":
                    out[out_name] = handle.get_tensor(name).cpu()
                    out[out_scale_name] = handle.get_tensor(scale_name).cpu()
                    counts["mxfp4"] += 1
                else:
                    converted = requantize_mxfp4_to_int4_w4a16(
                        handle.get_tensor(name),
                        handle.get_tensor(scale_name),
                        scale_mode=expert_int4_scale_mode,
                        out_scale_dtype=out_scale_dtype,
                    )
                    out[out_name] = converted["qweight_packed"].cpu()
                    out[out_scale_name] = converted["scales"].cpu()
                    counts["int4"] += 1
            elif role in _FP8_WEIGHT_ROLES:
                scale_name = matched_scale_name(name)
                assert scale_name is not None
                out_scale_name = _remap_tensor_name(scale_name, layer_remap)
                assert out_scale_name is not None
                if dense_int8_strategy == "channel":
                    converted = requantize_fp8_to_allspark_uint8_w8a16(
                        handle.get_tensor(name),
                        handle.get_tensor(scale_name),
                        out_scale_dtype=out_scale_dtype,
                    )
                else:
                    converted = requantize_fp8_to_int8_w8a16(
                        handle.get_tensor(name),
                        handle.get_tensor(scale_name),
                        out_scale_dtype=out_scale_dtype,
                    )
                out[out_name] = converted["qweight"].cpu()
                out[out_scale_name] = converted["scales"].cpu()
                counts["int8"] += 1
            elif role.endswith("_scale"):
                raise ValueError(f"unpaired scale tensor in {src_shard.name}: {name}")
            else:
                out[out_name] = _subset_slice(
                    name, handle.get_tensor(name), keep_experts
                ).cpu()
                counts["preserve"] += 1

    save_file(out, str(dst_shard))
    return counts


def convert_checkpoint(
    src: Path,
    dst: Path,
    *,
    device: str,
    out_scale_dtype: torch.dtype,
    overwrite: bool,
    layer_remap: dict[int, int] | None,
    dense_int8_strategy: str = "block",
    expert_format: str = "int4",
    expert_int4_scale_mode: str = "absmax7",
    num_output_shards: int | None = None,
    keep_experts: int | None = None,
    drop_mtp: bool = False,
) -> None:
    if dense_int8_strategy not in ("block", "channel"):
        raise ValueError(
            f"dense_int8_strategy must be 'block' or 'channel', got "
            f"{dense_int8_strategy!r}"
        )
    if expert_format not in ("int4", "mxfp4"):
        raise ValueError(
            f"expert_format must be 'int4' or 'mxfp4', got {expert_format!r}"
        )
    if expert_int4_scale_mode not in ("absmax7", "absmax8", "mse"):
        raise ValueError(
            "expert_int4_scale_mode must be 'absmax7', 'absmax8', or 'mse', got "
            f"{expert_int4_scale_mode!r}"
        )
    if num_output_shards is not None and num_output_shards <= 0:
        raise ValueError("num_output_shards must be positive when set")
    if dst.exists() and any(dst.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{dst} exists and is not empty; pass --overwrite")
        for child in dst.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
    dst.mkdir(parents=True, exist_ok=True)

    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards in {src}")
    if layer_remap is None:
        layer_remap = _discover_layer_remap(src)
    if layer_remap is not None:
        _log(f"layer_remap={layer_remap}")

    totals = {"int4": 0, "mxfp4": 0, "int8": 0, "preserve": 0}
    _log(f"converting {len(shards)} shards from {src} to {dst}")
    for shard in shards:
        _log(f"-> {shard.name}")
        counts = convert_shard(
            shard,
            dst / shard.name,
            device=device,
            out_scale_dtype=out_scale_dtype,
            layer_remap=layer_remap,
            dense_int8_strategy=dense_int8_strategy,
            expert_format=expert_format,
            expert_int4_scale_mode=expert_int4_scale_mode,
            keep_experts=keep_experts,
            drop_mtp=drop_mtp,
        )
        for key, value in counts.items():
            totals[key] += value
        _log(
            f"{shard.name}: int4={counts['int4']} mxfp4={counts['mxfp4']} "
            f"int8={counts['int8']} preserve={counts['preserve']}"
        )

    _copy_metadata(src, dst)
    _write_index(src, dst, layer_remap, keep_experts=keep_experts, drop_mtp=drop_mtp)
    _write_config(
        src,
        dst,
        layer_remap,
        dense_int8_strategy=dense_int8_strategy,
        expert_format=expert_format,
        expert_int4_scale_mode=expert_int4_scale_mode,
        keep_experts=keep_experts,
        drop_mtp=drop_mtp,
    )
    if not drop_mtp:
        _ensure_mtp_shared_tensors(dst)
    if num_output_shards is not None:
        cfg = json.loads((dst / "config.json").read_text())
        _log(f"resharding checkpoint to {num_output_shards} output shards")
        _reshard_safetensors(
            dst,
            num_output_shards=num_output_shards,
            num_hidden_layers=int(cfg["num_hidden_layers"]),
        )
    _log(
        f"done: int4={totals['int4']} mxfp4={totals['mxfp4']} "
        f"int8={totals['int8']} preserve={totals['preserve']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scale-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--dense-int8-strategy",
        choices=("block", "channel"),
        default="block",
        help="Use 128x128 block INT8 fallback format or channelwise AllSpark "
        "biased UINT8 format for FP8 dense linears.",
    )
    parser.add_argument(
        "--expert-format",
        choices=("int4", "mxfp4"),
        default="int4",
        help="Convert routed expert MXFP4 to signed INT4, or preserve native "
        "MXFP4 experts and emit quant_method=dsv4_mxfp4_int8.",
    )
    parser.add_argument(
        "--expert-int4-scale-mode",
        choices=("absmax7", "absmax8", "mse"),
        default="absmax7",
        help="Scale selection for MXFP4 routed experts converted to signed INT4.",
    )
    parser.add_argument(
        "--num-output-shards",
        type=int,
        help="Rewrite converted safetensors into a layer-contiguous shard layout. "
        "For PP=12, use a count such as 60 or 72 if you want evenly divisible "
        "file ranges; use 64 when matching a common HF shard count matters more.",
    )
    parser.add_argument(
        "--layer-remap",
        help="JSON mapping of source layer id to destination id. If omitted, "
        "truncated checkpoints are auto-remapped when possible.",
    )
    parser.add_argument(
        "--keep-layers",
        type=int,
        help="Testbed subsetting: keep only the first N layers (shorthand for "
        "an identity --layer-remap of layers 0..N-1).",
    )
    parser.add_argument(
        "--keep-experts",
        type=int,
        help="Testbed subsetting: keep only routed experts 0..E-1 per layer and "
        "slice the router gate to match. Per-token GEMM shapes are unchanged.",
    )
    parser.add_argument(
        "--drop-mtp",
        action="store_true",
        help="Testbed subsetting: drop mtp.* tensors and set "
        "num_nextn_predict_layers=0.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    layer_remap = None
    if args.layer_remap:
        layer_remap = {int(k): int(v) for k, v in json.loads(args.layer_remap).items()}
    if args.keep_layers is not None:
        if layer_remap is not None:
            raise SystemExit("--keep-layers and --layer-remap are mutually exclusive")
        layer_remap = {i: i for i in range(args.keep_layers)}

    convert_checkpoint(
        args.src.resolve(),
        args.dst.resolve(),
        device=args.device,
        out_scale_dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[
            args.scale_dtype
        ],
        overwrite=args.overwrite,
        layer_remap=layer_remap,
        dense_int8_strategy=args.dense_int8_strategy,
        expert_format=args.expert_format,
        expert_int4_scale_mode=args.expert_int4_scale_mode,
        num_output_shards=args.num_output_shards,
        keep_experts=args.keep_experts,
        drop_mtp=args.drop_mtp,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
