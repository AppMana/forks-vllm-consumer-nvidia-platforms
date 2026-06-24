#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit DeepSeek V4 checkpoint tensor roles before integer conversion.

The converter must follow the original DeepSeek precision boundary:

* routed experts are MXFP4 and are candidates for asymmetric INT4/AWQ
* FP8 linears are candidates for INT8 W8A16
* BF16/FP32 tensors are preserved

This script emits a manifest so later conversion steps cannot silently quantize
an unclassified tensor.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from safetensors import safe_open

_ROUTED_EXPERT_RE = re.compile(r"\.ffn\.experts\.\d+\.w[123]\.(weight|scale)$")
_ROUTED_EXPERT_WEIGHT_RE = re.compile(r"\.ffn\.experts\.\d+\.w[123]\.weight$")
_ROUTED_EXPERT_SCALE_RE = re.compile(r"\.ffn\.experts\.\d+\.w[123]\.scale$")

_FP8_WEIGHT_PARENTS = (
    ".attn.wq_a.",
    ".attn.wq_b.",
    ".attn.wkv.",
    ".attn.wo_a.",
    ".attn.wo_b.",
    ".attn.compressor.wkv.",
    ".attn.compressor.wgate.",
    ".attn.indexer.wq_b.",
    ".attn.indexer.compressor.wkv.",
    ".attn.indexer.compressor.wgate.",
    ".ffn.shared_experts.w1.",
    ".ffn.shared_experts.w2.",
    ".ffn.shared_experts.w3.",
)
_INDEXER_QK_PARENTS = (
    ".attn.indexer.wq_b.",
    ".attn.indexer.compressor.wkv.",
)
_MTP_FP8_PARENT_RE = re.compile(r"^mtp\.\d+\.[eh]_proj\.")

_PRESERVE_DTYPE_NAMES = {
    "torch.bfloat16",
    "torch.float32",
    "torch.float16",
    "torch.int32",
    "torch.int64",
    "BF16",
    "F32",
    "F16",
    "I32",
    "I64",
}
_FP8_WEIGHT_DTYPE_NAMES = {"torch.float8_e4m3fn", "F8_E4M3"}
_FP8_SCALE_DTYPE_NAMES = {"torch.float8_e8m0fnu", "F8_E8M0"}


@dataclass(frozen=True)
class TensorRecord:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    role: str
    action: str
    scale_name: str | None = None


def matched_scale_name(name: str) -> str | None:
    if name.endswith(".weight"):
        return f"{name[:-len('.weight')]}.scale"
    return None


def _is_fp8_parent(name: str) -> bool:
    return any(parent in name for parent in _FP8_WEIGHT_PARENTS)


def _is_mtp_fp8_parent(name: str) -> bool:
    return _MTP_FP8_PARENT_RE.search(name) is not None


def classify_tensor(name: str, dtype: str) -> tuple[str, str]:
    if _ROUTED_EXPERT_WEIGHT_RE.search(name):
        return "routed_expert_mxfp4_weight", "quantize_asym_int4_awq_candidate"
    if _ROUTED_EXPERT_SCALE_RE.search(name):
        return "routed_expert_mxfp4_scale", "quantize_asym_int4_awq_candidate"
    if dtype in _PRESERVE_DTYPE_NAMES:
        return "preserved_precision_tensor", "preserve"
    if (
        _is_mtp_fp8_parent(name)
        and name.endswith(".weight")
        and dtype in _FP8_WEIGHT_DTYPE_NAMES
    ):
        return "mtp_fp8_weight", "quantize_int8_w8a16_candidate"
    if (
        _is_mtp_fp8_parent(name)
        and name.endswith(".scale")
        and dtype in _FP8_SCALE_DTYPE_NAMES
    ):
        return "mtp_fp8_scale", "quantize_int8_w8a16_candidate"
    if _is_fp8_parent(name) and name.endswith(".weight"):
        if dtype not in _FP8_WEIGHT_DTYPE_NAMES:
            return "unknown", "manual_review"
        if any(parent in name for parent in _INDEXER_QK_PARENTS):
            return "indexer_qk_fp8_weight", "measure_recall_then_quantize"
        return "dense_fp8_weight", "quantize_int8_w8a16_candidate"
    if _is_fp8_parent(name) and name.endswith(".scale"):
        if dtype not in _FP8_SCALE_DTYPE_NAMES:
            return "unknown", "manual_review"
        if any(parent in name for parent in _INDEXER_QK_PARENTS):
            return "indexer_qk_fp8_scale", "measure_recall_then_quantize"
        return "dense_fp8_scale", "quantize_int8_w8a16_candidate"
    return "unknown", "manual_review"


def _iter_safetensors(
    checkpoint: Path,
) -> Iterable[tuple[str, Path, str, tuple[int, ...]]]:
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        by_shard: dict[str, list[str]] = collections.defaultdict(list)
        for name, shard in weight_map.items():
            by_shard[shard].append(name)
        for shard, names in sorted(by_shard.items()):
            path = checkpoint / shard
            with safe_open(path, framework="pt", device="cpu") as handle:
                for name in sorted(names):
                    tensor_slice = handle.get_slice(name)
                    yield (
                        name,
                        path,
                        tensor_slice.get_dtype(),
                        tuple(tensor_slice.get_shape()),
                    )
        return

    for path in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in sorted(handle.keys()):
                tensor_slice = handle.get_slice(name)
                yield (
                    name,
                    path,
                    tensor_slice.get_dtype(),
                    tuple(tensor_slice.get_shape()),
                )


def audit_checkpoint(checkpoint: Path) -> dict[str, object]:
    cfg_path = checkpoint / "config.json"
    num_mtp_layers = 0
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        num_mtp_layers = int(cfg.get("num_nextn_predict_layers", 0) or 0)

    records: list[TensorRecord] = []
    tensor_names: set[str] = set()
    for name, shard_path, dtype, shape in _iter_safetensors(checkpoint):
        tensor_names.add(name)
        role, action = classify_tensor(name, dtype)
        scale_name = matched_scale_name(name) if name.endswith(".weight") else None
        records.append(
            TensorRecord(
                name=name,
                shard=shard_path.name,
                dtype=dtype,
                shape=shape,
                role=role,
                action=action,
                scale_name=scale_name,
            )
        )

    missing_scales = [
        r.name
        for r in records
        if r.scale_name is not None
        and r.action != "preserve"
        and r.scale_name not in tensor_names
    ]
    missing_mtp_tensors: list[str] = []
    if num_mtp_layers > 0 and any(name.startswith("mtp.") for name in tensor_names):
        for idx in range(num_mtp_layers):
            for suffix in (
                "emb.tok_emb.weight",
                "head.weight",
                "norm.weight",
                "hc_head_fn",
                "hc_head_base",
                "hc_head_scale",
            ):
                name = f"mtp.{idx}.{suffix}"
                if name not in tensor_names:
                    missing_mtp_tensors.append(name)
    summary = {
        "total_tensors": len(records),
        "by_role": dict(collections.Counter(r.role for r in records)),
        "by_action": dict(collections.Counter(r.action for r in records)),
        "by_dtype": dict(collections.Counter(r.dtype for r in records)),
        "missing_scales": missing_scales,
        "missing_mtp_tensors": missing_mtp_tensors,
        "unknown_tensors": [r.name for r in records if r.role == "unknown"],
    }
    return {
        "checkpoint": str(checkpoint),
        "summary": summary,
        "records": [asdict(r) for r in records],
    }


def _print_summary(manifest: dict[str, object]) -> None:
    summary = manifest["summary"]
    assert isinstance(summary, dict)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fail-on-unknown", action="store_true")
    args = parser.parse_args()

    manifest = audit_checkpoint(args.checkpoint.resolve())
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    _print_summary(manifest)

    summary = manifest["summary"]
    assert isinstance(summary, dict)
    if args.fail_on_unknown and (
        summary["unknown_tensors"]
        or summary["missing_scales"]
        or summary["missing_mtp_tensors"]
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
