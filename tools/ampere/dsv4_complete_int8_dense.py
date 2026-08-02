#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Complete the INT8-dense stage on a DeepSeek V4 rebuild checkpoint.

The native-DSpark rebuild pipeline emits backbone ``layers.*.attn.wo_a.weight``
as scale-free BF16 (the "prepped for int8 dense" state). The validated serving
layout instead carries those tensors as AllSpark channelwise UINT8 W8A16:

* per-output-channel scale (dim 0), FP32 ``absmax / 127`` clamped to tiny,
  stored as BF16 with tensor name ``<parent>.scale``
* weight codes ``clamp(round(w / scale), -128, 127) + 128`` stored as UINT8
  (implicit +128 bias; see ``requantize_fp8_to_allspark_uint8_w8a16`` and its
  consumer ``dequantize_allspark_uint8_w8a16`` in ``dsv4_int.py``)

This tool applies exactly that quantization to the backbone wo_a tensors of a
finished rebuild snapshot and nothing else: MTP wo_a tensors stay BF16, every
other tensor is byte-identical passthrough, shard names and the shard split are
preserved, and unaffected shards are hardlinked (or copied) into the output.
The source snapshot is never modified.

Shards are rewritten by streaming byte ranges straight between files, so peak
memory is one wo_a tensor plus the copy buffer; the tool is safe to run in a
small CPU-only pod next to a live serving deployment. It needs only
torch + safetensors, not a vLLM install.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
from pathlib import Path

import torch
from safetensors import safe_open

_WO_A_WEIGHT_RE = re.compile(r"^layers\.\d+\.attn\.wo_a\.weight$")
_COPY_CHUNK = 64 * 1024 * 1024
_METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _log(message: str) -> None:
    print(f"[dsv4-int8-dense] {message}", flush=True)


def _wo_a_scale_name(weight_name: str) -> str:
    return weight_name[: -len(".weight")] + ".scale"


def quantize_bf16_to_allspark_uint8(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16 2D weight to AllSpark channelwise biased UINT8 W8A16.

    Mirrors the tail of ``requantize_fp8_to_allspark_uint8_w8a16``: the FP32
    per-output-channel scale is used for the division and only narrowed to
    BF16 for storage.
    """
    dequant = weight.to(torch.float32)
    channel_scale = dequant.abs().amax(dim=1)
    channel_scale = channel_scale.clamp(min=torch.finfo(torch.float32).tiny) / 127.0
    q_signed = torch.round(dequant / channel_scale.unsqueeze(1)).clamp(-128, 127)
    q_biased = (q_signed.to(torch.int16) + 128).to(torch.uint8)
    return q_biased, channel_scale.to(torch.bfloat16)


def _read_header(path: Path) -> tuple[dict[str, dict], dict[str, str] | None, int]:
    """Parse a safetensors header: (tensors, __metadata__, data section offset)."""
    with path.open("rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_len))
    metadata = header.pop("__metadata__", None)
    return header, metadata, 8 + header_len


def _tensor_bytes(tensor: torch.Tensor) -> memoryview:
    flat = tensor.contiguous().reshape(-1)
    return flat.view(torch.uint8).numpy().data


def _stream_copy(src_handle, dst_handle, src_start: int, length: int) -> None:
    src_handle.seek(src_start)
    remaining = length
    while remaining > 0:
        chunk = src_handle.read(min(_COPY_CHUNK, remaining))
        if not chunk:
            raise OSError("unexpected EOF while copying tensor bytes")
        dst_handle.write(chunk)
        remaining -= len(chunk)


def rewrite_shard(src_shard: Path, dst_shard: Path) -> list[str]:
    """Rewrite one shard, quantizing backbone wo_a weights in place.

    Returns the wo_a weight names that were quantized. Every other tensor is
    copied byte-for-byte; tensor bytes are laid out in sorted-name order with
    the JSON header serialized sort_keys, so the output is deterministic.
    """
    tensors, metadata, data_base = _read_header(src_shard)
    wo_a_names = sorted(n for n in tensors if _WO_A_WEIGHT_RE.match(n))

    quantized: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    with safe_open(src_shard, framework="pt", device="cpu") as handle:
        for name in wo_a_names:
            info = tensors[name]
            if info["dtype"] != "BF16" or len(info["shape"]) != 2:
                raise ValueError(
                    f"{src_shard.name}:{name} is {info['dtype']} {info['shape']}, "
                    "expected a BF16 2D tensor prepped for int8 dense"
                )
            scale_name = _wo_a_scale_name(name)
            if scale_name in tensors:
                raise ValueError(
                    f"{src_shard.name} already carries {scale_name}; "
                    "this shard is not in the prepped-BF16 state"
                )
            quantized[name] = quantize_bf16_to_allspark_uint8(handle.get_tensor(name))

    out_entries: dict[str, dict] = {}
    for name, info in tensors.items():
        if name in quantized:
            q_biased, scale = quantized[name]
            out_entries[name] = {"dtype": "U8", "shape": list(q_biased.shape)}
            out_entries[_wo_a_scale_name(name)] = {
                "dtype": "BF16",
                "shape": list(scale.shape),
            }
        else:
            out_entries[name] = {"dtype": info["dtype"], "shape": info["shape"]}

    dtype_size = {"BF16": 2, "U8": 1}
    offset = 0
    for name in sorted(out_entries):
        entry = out_entries[name]
        if name in tensors and name not in quantized:
            start, end = tensors[name]["data_offsets"]
            nbytes = end - start
        else:
            numel = 1
            for dim in entry["shape"]:
                numel *= dim
            nbytes = numel * dtype_size[entry["dtype"]]
        entry["data_offsets"] = [offset, offset + nbytes]
        offset += nbytes

    header: dict[str, dict] = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    header.update(out_entries)
    header_json = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    pad = (8 - (8 + len(header_json)) % 8) % 8
    header_json += b" " * pad

    dst_tmp = dst_shard.with_suffix(".safetensors.tmp")
    with src_shard.open("rb") as src_handle, dst_tmp.open("wb") as dst_handle:
        dst_handle.write(struct.pack("<Q", len(header_json)))
        dst_handle.write(header_json)
        for name in sorted(out_entries):
            if name in quantized:
                dst_handle.write(_tensor_bytes(quantized[name][0]))
            elif name not in tensors:
                weight_name = name[: -len(".scale")] + ".weight"
                dst_handle.write(_tensor_bytes(quantized[weight_name][1]))
            else:
                start, end = tensors[name]["data_offsets"]
                _stream_copy(src_handle, dst_handle, data_base + start, end - start)
    dst_tmp.replace(dst_shard)
    return wo_a_names


def verify_shard(src_shard: Path, dst_shard: Path) -> dict[str, float]:
    """Verify a rewritten shard against its source.

    Passthrough tensors must be byte-identical; quantized wo_a tensors must
    dequantize back to the source BF16 within one quantization step per
    element along the channel axis.
    """
    src_tensors, _, src_base = _read_header(src_shard)
    dst_tensors, _, dst_base = _read_header(dst_shard)
    stats = {"max_steps": 0.0, "mean_steps_sum": 0.0, "quantized": 0.0}
    with (
        src_shard.open("rb") as src_handle,
        dst_shard.open("rb") as dst_handle,
        safe_open(src_shard, framework="pt", device="cpu") as src,
        safe_open(dst_shard, framework="pt", device="cpu") as dst,
    ):
        for name, src_info in src_tensors.items():
            if _WO_A_WEIGHT_RE.match(name):
                reference = src.get_tensor(name).to(torch.float32)
                codes = dst.get_tensor(name)
                scale = dst.get_tensor(_wo_a_scale_name(name)).to(torch.float32)
                if codes.dtype != torch.uint8:
                    raise ValueError(f"{name}: expected U8, got {codes.dtype}")
                dequant = (codes.to(torch.float32) - 128.0) * scale.reshape(-1, 1)
                steps = (dequant - reference).abs() / scale.reshape(-1, 1)
                max_steps = steps.max().item()
                if max_steps > 1.0:
                    raise ValueError(
                        f"{name}: round-trip error {max_steps:.4f} steps > 1"
                    )
                stats["max_steps"] = max(stats["max_steps"], max_steps)
                stats["mean_steps_sum"] += steps.mean().item()
                stats["quantized"] += 1.0
                continue
            dst_info = dst_tensors[name]
            if (
                src_info["dtype"] != dst_info["dtype"]
                or src_info["shape"] != dst_info["shape"]
            ):
                raise ValueError(f"{name}: passthrough dtype/shape changed")
            src_start, src_end = src_info["data_offsets"]
            dst_start, dst_end = dst_info["data_offsets"]
            if src_end - src_start != dst_end - dst_start:
                raise ValueError(f"{name}: passthrough byte length changed")
            src_handle.seek(src_base + src_start)
            dst_handle.seek(dst_base + dst_start)
            remaining = src_end - src_start
            while remaining > 0:
                span = min(_COPY_CHUNK, remaining)
                if src_handle.read(span) != dst_handle.read(span):
                    raise ValueError(f"{name}: passthrough bytes changed")
                remaining -= span
    return stats


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    real_src = src.resolve()
    if mode == "hardlink":
        try:
            dst.hardlink_to(real_src)
            return
        except OSError:
            _log(f"hardlink failed for {src.name}; falling back to copy")
    shutil.copyfile(real_src, dst)


def complete_checkpoint(
    src: Path, dst: Path, *, link_mode: str, overwrite: bool
) -> None:
    index_path = src / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = dict(index["weight_map"])

    cfg = json.loads((src / "config.json").read_text())
    num_layers = int(cfg["num_hidden_layers"])
    wo_a_weights = sorted(n for n in weight_map if _WO_A_WEIGHT_RE.match(n))
    if len(wo_a_weights) != num_layers:
        raise ValueError(
            f"expected {num_layers} backbone wo_a weights, found {len(wo_a_weights)}"
        )
    already_scaled = [n for n in wo_a_weights if _wo_a_scale_name(n) in weight_map]
    if already_scaled:
        raise ValueError(
            f"{len(already_scaled)} wo_a tensors already carry scales "
            f"(e.g. {already_scaled[0]!r}); nothing to complete"
        )

    if dst.exists() and any(dst.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{dst} exists and is not empty; pass --overwrite")
        for child in dst.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    dst.mkdir(parents=True, exist_ok=True)

    affected_shards = sorted({weight_map[name] for name in wo_a_weights})
    all_shards = sorted(path.name for path in src.glob("*.safetensors"))
    untouched_shards = [name for name in all_shards if name not in affected_shards]
    _log(
        f"{len(wo_a_weights)} wo_a tensors across {len(affected_shards)} shards; "
        f"{len(untouched_shards)} shards passed through ({link_mode})"
    )

    total_quantized = 0
    max_steps = 0.0
    mean_steps_sum = 0.0
    for shard_name in affected_shards:
        rewritten = rewrite_shard(src / shard_name, dst / shard_name)
        stats = verify_shard(src / shard_name, dst / shard_name)
        total_quantized += len(rewritten)
        max_steps = max(max_steps, stats["max_steps"])
        mean_steps_sum += stats["mean_steps_sum"]
        _log(
            f"{shard_name}: quantized {len(rewritten)} wo_a "
            f"(max round-trip {stats['max_steps']:.4f} steps)"
        )
        for name in rewritten:
            weight_map[_wo_a_scale_name(name)] = shard_name

    for shard_name in untouched_shards:
        _link_or_copy(src / shard_name, dst / shard_name, link_mode)

    for name in _METADATA_FILES:
        if (src / name).exists():
            shutil.copyfile((src / name).resolve(), dst / name)

    index["weight_map"] = weight_map
    index.setdefault("metadata", {})
    index["metadata"]["total_size"] = str(
        sum(path.stat().st_size for path in dst.glob("*.safetensors"))
    )
    (dst / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    _log(
        f"done: quantized {total_quantized}/{len(wo_a_weights)} wo_a tensors, "
        f"max round-trip {max_steps:.4f} steps, "
        f"mean {mean_steps_sum / max(total_quantized, 1):.4f} steps"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How unaffected shards reach the output directory. Hardlink "
        "falls back to copy when the link cannot cross filesystems.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    complete_checkpoint(
        args.src.resolve(),
        args.dst.resolve(),
        link_mode=args.link_mode,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
