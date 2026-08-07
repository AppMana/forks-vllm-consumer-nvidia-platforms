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
from collections.abc import Callable
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from dsv4_checkpoint_audit import classify_tensor, matched_scale_name  # noqa: E402

try:
    # The requantizing targets (int4/mxfp4) need the dsv4_int kernels; the
    # nvfp4 passthrough target needs only torch+safetensors so it can run on
    # a bare spark workspace without a vLLM install.
    from vllm.model_executor.layers.quantization.dsv4_int import (  # noqa: E402
        dequantize_fp8_block_to_bf16,
        requantize_fp8_to_allspark_uint8_w8a16,
        requantize_fp8_to_int8_w8a16,
        requantize_mxfp4_to_int4_w4a16,
    )
except ImportError:  # pragma: no cover - bare-workspace nvfp4 runs
    dequantize_fp8_block_to_bf16 = None
    requantize_fp8_to_allspark_uint8_w8a16 = None
    requantize_fp8_to_int8_w8a16 = None
    requantize_mxfp4_to_int4_w4a16 = None

_FP8_WEIGHT_ROLES = {
    "dense_fp8_weight",
    "indexer_qk_fp8_weight",
    "mtp_fp8_weight",
}
_LAYER_NAME_RE = re.compile(r"^layers\.(\d+)\.(.*)$")
# WO_A is consumed by a custom inverse-RoPE BF16 einsum at runtime, never a GEMM.
# Requantizing it to INT8 (fp8 -> int8 -> bf16) is pure accuracy loss with zero
# runtime benefit: the weight is BF16 in memory either way. Dequantize the fp8
# source straight to BF16 and emit it with no scale companion.
_WO_A_WEIGHT_SUBSTR = ".attn.wo_a."


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
_MTP_STAGE_RE = re.compile(r"^mtp\.(\d+)\.")


def _mtp_stage_ids(checkpoint: Path) -> list[int]:
    """MTP/DSpark stage indices actually present in a checkpoint's tensors.

    ``config.json``'s ``num_nextn_predict_layers`` is NOT a reliable stage
    count for DSpark checkpoints: deepseek-ai/DeepSeek-V4-Flash-0731 ships the
    full 3-stage DSpark subtree (``mtp.0``/``mtp.1``/``mtp.2``, matching its
    3-entry ``dspark_target_layer_ids``) while still declaring
    ``num_nextn_predict_layers: 1`` like vanilla Flash. Counting real tensor
    prefixes is the only thing that distinguishes a vanilla single-stage MTP
    from a DSpark graft, so every stage-count decision in this tool reads
    from here.
    """
    names: list[str] = []
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.exists():
        names = list(json.loads(index_path.read_text())["weight_map"])
    else:
        for shard in sorted(checkpoint.glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as handle:
                names.extend(handle.keys())
    return sorted(
        {int(match.group(1)) for name in names if (match := _MTP_STAGE_RE.match(name))}
    )


def _is_wo_a_scale(name: str) -> bool:
    """True for the fp8 block scale companion of a wo_a weight.

    wo_a is emitted as a scale-free BF16 tensor, so its source ``.scale`` entry
    must be dropped from the rewritten index too (the shard no longer holds it).
    """
    return _WO_A_WEIGHT_SUBSTR in name and name.endswith(".scale")


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
        if _is_wo_a_scale(name):
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

    stage_ids = _mtp_stage_ids(dst)
    if not stage_ids:
        return

    cfg_path = dst / "config.json"
    cfg = json.loads(cfg_path.read_text())
    declared = int(cfg.get("num_nextn_predict_layers", 0) or 0)
    # A config declaring no MTP means the shared embed/head aliases are not
    # wanted, even if stray mtp.* tensors are present. Only override the
    # declaration when the checkpoint carries a DSpark subtree it understates
    # (0731 ships 3 stages while declaring 1).
    if declared <= 0 and len(stage_ids) <= 1:
        return
    # Reconcile the declared stage count with the stages actually present.
    # A DSpark-bearing source (0731 and later) carries 3 stages
    # while still declaring 1; leaving the declaration at 1 would strand
    # mtp.1/mtp.2 without the embed/head aliases materialized below, and
    # vllm/config/speculative.py sizes the draft model off this field.
    if int(cfg.get("num_nextn_predict_layers", 0) or 0) != len(stage_ids):
        _log(
            f"num_nextn_predict_layers: "
            f"{cfg.get('num_nextn_predict_layers')} -> {len(stage_ids)} "
            f"(stages present: {stage_ids})"
        )
        cfg["num_nextn_predict_layers"] = len(stage_ids)
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

    index = _load_index_or_build(dst)
    weight_map = dict(index["weight_map"])

    embed_name = _find_required_tensor(
        weight_map,
        ("embed.weight", "model.embed.weight", "model.embed_tokens.weight"),
    )
    head_name = _find_required_tensor(
        weight_map,
        ("head.weight", "lm_head.weight", "model.head.weight"),
    )

    additions: dict[str, torch.Tensor] = {}
    for mtp_idx in stage_ids:
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
    if expert_format == "nvfp4":
        # Native-precision target (appmana/deepseek-v4-nvfp4-fp8): the source
        # modelopt MIXED_PRECISION quantization_config stays VERBATIM — it is
        # the proven loadable description of the NVFP4-expert/fp8-backbone
        # weights being passed through. Add the fork's "vllm" kernel-config
        # block selecting the GB10/sm12x sparkinfer kernels by FQN and the
        # fp8_ds_mla KV cache. The FQN strings mirror kernel_config.py's
        # SPARSE_MLA_DECODE_FP8_SPARKINFER / SPARSE_MLA_PREFILL_SPARKINFER
        # (inlined so this tool runs without a vLLM install; resolution is
        # fail-closed at serve time, so drift cannot pass silently; and
        # --hf-overrides can replace the whole block without touching
        # weights).
        cfg["vllm"] = {
            "kernels": [
                "vllm.models.deepseek_v4.nvidia_sm12x.kernels"
                ".sparkinfer_sparse_mla_decode",
                "vllm.models.deepseek_v4.nvidia_sm12x.kernels"
                ".sparkinfer_sparse_mla_extend",
            ],
            "cache_type": "fp8_ds_mla",
        }
        (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        return
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
                    # Vanilla Flash MTP output-head projections.
                    "mtp.*.e_proj",
                    "mtp.*.h_proj",
                    # DSpark's restructured mtp.0 output projection, which
                    # replaces e_proj/h_proj for the stage that feeds the next
                    # draft stage instead of emitting its own next-token
                    # distribution. Same fp8-block source convention, so it
                    # lands in the same channelwise INT8 as the rest.
                    # (mtp.1/mtp.2 are full decoder layers; their attention,
                    # shared-expert and routed-expert tensors are already
                    # covered by the '*.attn.*' / '*.ffn.*' globs above.)
                    "mtp.*.main_proj",
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
    # The official DeepSeek checkpoints do not carry this fork-specific
    # runtime block. Requantized INT4/INT8 checkpoints must emit it explicitly:
    # engine setup intentionally leaves --kv-cache-dtype=auto untouched when
    # the block is absent, while the DSV4 attention layout requires a concrete
    # int8_ds_mla cache type. Keep this identical to the proven production
    # checkpoint; every symbol is validated fail-closed by kernel_config.py.
    # This list must stay byte-identical to the block on the SERVING revision
    # of appmana/deepseek-v4-int4-int8. Two of these roles are *toggle* roles
    # (see kernel_config.py): when an explicit block is present, a role whose
    # symbol is unlisted is OFF, not defaulted. Dropping
    # ``streaming_prefill_topk`` here would silently turn the long-context
    # streaming indexer path off on every rebuilt revision while the
    # checkpoint still looked correct -- so kernel selection is preserved
    # explicitly rather than reconstructed. ``sparse_mla_decode_fp8`` is
    # deliberately absent: this checkpoint's cache is int8_ds_mla, and the
    # fp8 decode selector falls back to its documented default.
    cfg["vllm"] = {
        "kernels": [
            "flash_mla.sparse_mla_decode_int8",
            "flash_mla.sparse_mla_prefill_int8",
            "vllm._custom_ops.indexer_k_quant_and_cache_int8",
            (
                "vllm.models.deepseek_v4.common.ops.fused_indexer_q"
                ".fused_indexer_q_rope_quant_int8"
            ),
            (
                "vllm.model_executor.layers.quantization.utils.marlin_utils"
                ".marlin_act_int8_process_scales"
            ),
            "vllm.model_executor.layers.sparse_attn_indexer.streaming_prefill_topk",
        ],
        "cache_type": "int8_ds_mla",
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
    name_filter: Callable[[str], bool] | None = None,
) -> dict[str, int]:
    roles, _dtypes, missing_scales = _classify_shard(src_shard)
    if missing_scales:
        sample = ", ".join(sorted(missing_scales)[:8])
        raise ValueError(f"{src_shard.name} missing scales for: {sample}")

    # nvfp4 target (appmana/deepseek-v4-nvfp4-fp8): the checkpoint is served
    # in its NATIVE precision — NVFP4 experts and fp8 linears byte-for-byte —
    # so conversion is a classification-validated passthrough (subset/remap/
    # drop-mtp still apply). Everything else about the pipeline (mtp splice,
    # index rewrite, config) is shared with the requantizing targets.
    passthrough = expert_format == "nvfp4"

    out: dict[str, torch.Tensor] = {}
    counts = {
        "int4": 0,
        "mxfp4": 0,
        "int8": 0,
        "wo_a_bf16": 0,
        "preserve": 0,
    }
    paired_scales: set[str | None] = set()
    if not passthrough:
        paired_scales = {
            matched_scale_name(name)
            for name, role in roles.items()
            if role
            in ("routed_expert_mxfp4_weight",)
            or role in _FP8_WEIGHT_ROLES
        }
        paired_scales.discard(None)

    with safe_open(src_shard, framework="pt", device=device) as handle:
        for name in sorted(handle.keys()):
            if name in paired_scales:
                continue
            if _subset_drop(name, keep_experts, drop_mtp):
                continue
            if name_filter is not None and not name_filter(name):
                continue
            role = roles[name]
            out_name = _remap_tensor_name(name, layer_remap)
            if out_name is None:
                continue
            if passthrough:
                out[out_name] = _subset_slice(
                    name, handle.get_tensor(name), keep_experts
                ).cpu()
                counts["preserve"] += 1
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
                if _WO_A_WEIGHT_SUBSTR in name:
                    # Dequantize fp8 (128x128 UE8M0 block scales) straight to
                    # BF16; emit no scale companion (its scale name is already in
                    # ``paired_scales`` and so is skipped).
                    out[out_name] = dequantize_fp8_block_to_bf16(
                        handle.get_tensor(name),
                        handle.get_tensor(scale_name),
                    ).cpu()
                    counts["wo_a_bf16"] += 1
                    continue
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
    expert_int4_scale_mode: str = "mse",
    num_output_shards: int | None = None,
    keep_experts: int | None = None,
    drop_mtp: bool = False,
) -> None:
    if dense_int8_strategy not in ("block", "channel"):
        raise ValueError(
            f"dense_int8_strategy must be 'block' or 'channel', got "
            f"{dense_int8_strategy!r}"
        )
    if expert_format not in ("int4", "mxfp4", "nvfp4"):
        raise ValueError(
            f"expert_format must be 'int4', 'mxfp4', or 'nvfp4', got "
            f"{expert_format!r}"
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

    totals = {
        "int4": 0,
        "mxfp4": 0,
        "int8": 0,
        "wo_a_bf16": 0,
        "preserve": 0,
    }
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
            f"int8={counts['int8']} wo_a_bf16={counts['wo_a_bf16']} "
            f"preserve={counts['preserve']}"
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
        f"int8={totals['int8']} wo_a_bf16={totals['wo_a_bf16']} "
        f"preserve={totals['preserve']}"
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
        choices=("int4", "mxfp4", "nvfp4"),
        default="int4",
        help="Convert routed expert MXFP4 to signed INT4; preserve native "
        "MXFP4 experts and emit quant_method=dsv4_mxfp4_int8; or (nvfp4) "
        "pass the whole checkpoint through in native precision — modelopt "
        "NVFP4 experts + fp8 backbone byte-for-byte, source "
        "quantization_config kept verbatim, sm12x sparkinfer FQN 'vllm' "
        "block added (the appmana/deepseek-v4-nvfp4-fp8 target).",
    )
    parser.add_argument(
        "--expert-int4-scale-mode",
        choices=("absmax7", "absmax8", "mse"),
        default="mse",
        help="Scale selection for MXFP4 routed experts converted to signed INT4. "
        "mse runs a per-group scale search and is the default: it measures "
        "+4.2 to +8.7 dB reconstruction SNR over absmax7 on real 0731 expert "
        "tensors, for build time only (same on-disk layout, same kernels, no "
        "inference cost). absmax7 divides by the group max, which forces "
        "MXFP4's 0.5 and 1.0 levels onto the same INT4 code.",
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

    out_scale_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.scale_dtype]

    # The source is deepseek-ai/DeepSeek-V4-Flash-0731, which ships mtp.0/1/2
    # natively. A single-stage subtree means a pre-0731 release, whose DSpark
    # stages could only be grafted from a different checkpoint -- weights that
    # were never co-trained with this backbone.
    if not args.drop_mtp:
        src_stages = _mtp_stage_ids(args.src.resolve())
        if len(src_stages) == 1:
            raise SystemExit(
                f"{args.src} carries a single MTP stage {src_stages}; convert "
                "deepseek-ai/DeepSeek-V4-Flash-0731 or later, which ships the "
                "three DSpark stages natively"
            )

    convert_checkpoint(
        args.src.resolve(),
        args.dst.resolve(),
        device=args.device,
        out_scale_dtype=out_scale_dtype,
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
