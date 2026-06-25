#!/usr/bin/env python3
"""Re-layout an appmana dsv4_int (int4-experts / int8-AllSpark-linears) checkpoint
into the FUSED/STACKED parameter names the upstream-rebased DeepseekV4 model
expects, so weights load by direct name match (no expert-params-mapping).

This is a pure tensor re-layout (concat along dim 0 + stack across experts);
the quantized values are preserved byte-for-byte — no requantization. Verified
shape map (DeepSeek-V4, 256 experts, hidden 4096, moe_intermediate 2048):

  per-expert ffn.experts.N.w1[2048,2048] + .w3[2048,2048]  (int4-in-int8)
    -> ffn.experts.routed_experts.w13_weight        (256, 4096, 2048)
  per-expert ffn.experts.N.w1.scale[2048,128] + w3.scale   (bf16)
    -> ffn.experts.routed_experts.w13_weight_scale  (256, 4096, 128)
  per-expert ffn.experts.N.w2[4096,1024] / .scale[4096,64]
    -> ffn.experts.routed_experts.w2_weight / .w2_weight_scale
  attn.wq_a[1024,4096] + attn.wkv[512,4096] (uint8 AllSpark)
    -> attn.fused_wqa_wkv.weight (1536,4096) + .weight_scale_inv
  ffn.shared_experts.w1 + .w3 -> .gate_up_proj.weight ; .w2 -> .down_proj.weight
  all other ".scale" -> ".weight_scale_inv" ; everything else preserved.

The output keeps the "layers.N..." convention (vLLM's WeightsMapper adds the
"model." prefix + leaves the fused names alone).

Usage:
  dsv4_fuse_checkpoint.py --src <int4mse-int8 dir> --dst <out dir> [--drop-mtp]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_LAYER_RE = re.compile(r"^layers\.(\d+)\.")
_EXPERT_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])(\.scale)?$")
_SHARD_BYTES = 4 * 1024**3  # ~4 GiB target shard size


def _scale_out(name: str) -> str:
    # linear/dense scale suffix expected by the model for AllSpark/fp8 linears
    return name[: -len(".scale")] + ".weight_scale_inv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--drop-mtp", action="store_true", default=True)
    ap.add_argument("--keep-mtp", dest="drop_mtp", action="store_false")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    src, dst = args.src, args.dst
    dst.mkdir(parents=True, exist_ok=True)
    index = json.load(open(src / "model.safetensors.index.json"))["weight_map"]
    cfg = json.load(open(src / "config.json"))
    n_layers = cfg["num_hidden_layers"]
    n_experts = cfg["n_routed_experts"]
    print(f"layers={n_layers} experts={n_experts} src_keys={len(index)}")

    # lazy shard handles
    handles: dict[str, object] = {}

    def get(name: str) -> torch.Tensor:
        sh = index[name]
        h = handles.get(sh)
        if h is None:
            h = handles[sh] = safe_open(str(src / sh), framework="pt", device=args.device)
        return h.get_tensor(name)

    def has(name: str) -> bool:
        return name in index

    out: dict[str, torch.Tensor] = {}

    # ---- per-layer fusion ----
    for L in range(n_layers):
        p = f"layers.{L}"
        # routed experts -> stacked
        w13_w, w13_s, w2_w, w2_s = [], [], [], []
        for e in range(n_experts):
            ep = f"{p}.ffn.experts.{e}"
            w1, w3 = get(f"{ep}.w1.weight"), get(f"{ep}.w3.weight")
            w13_w.append(torch.cat([w1, w3], dim=0))
            w13_s.append(torch.cat([get(f"{ep}.w1.scale"), get(f"{ep}.w3.scale")], dim=0))
            w2_w.append(get(f"{ep}.w2.weight"))
            w2_s.append(get(f"{ep}.w2.scale"))
        re_pfx = f"{p}.ffn.experts.routed_experts"
        out[f"{re_pfx}.w13_weight"] = torch.stack(w13_w, 0).contiguous()
        out[f"{re_pfx}.w13_weight_scale"] = torch.stack(w13_s, 0).contiguous()
        out[f"{re_pfx}.w2_weight"] = torch.stack(w2_w, 0).contiguous()
        out[f"{re_pfx}.w2_weight_scale"] = torch.stack(w2_s, 0).contiguous()

        # shared experts -> gate_up_proj / down_proj
        sp = f"{p}.ffn.shared_experts"
        out[f"{sp}.gate_up_proj.weight"] = torch.cat(
            [get(f"{sp}.w1.weight"), get(f"{sp}.w3.weight")], dim=0
        ).contiguous()
        out[f"{sp}.gate_up_proj.weight_scale_inv"] = torch.cat(
            [get(f"{sp}.w1.scale"), get(f"{sp}.w3.scale")], dim=0
        ).contiguous()
        out[f"{sp}.down_proj.weight"] = get(f"{sp}.w2.weight")
        out[f"{sp}.down_proj.weight_scale_inv"] = get(f"{sp}.w2.scale")

        # attention: fuse wq_a + wkv
        ap_ = f"{p}.attn"
        out[f"{ap_}.fused_wqa_wkv.weight"] = torch.cat(
            [get(f"{ap_}.wq_a.weight"), get(f"{ap_}.wkv.weight")], dim=0
        ).contiguous()
        out[f"{ap_}.fused_wqa_wkv.weight_scale_inv"] = torch.cat(
            [get(f"{ap_}.wq_a.scale"), get(f"{ap_}.wkv.scale")], dim=0
        ).contiguous()

    # ---- pass-through everything else (renaming .scale -> .weight_scale_inv) ----
    fused_consumed = re.compile(
        r"\.ffn\.experts\.\d+\.w[123](\.scale)?$"
        r"|\.ffn\.shared_experts\.w[123](\.scale)?$"
        r"|\.attn\.(wq_a|wkv)(\.weight|\.scale)$"
    )
    for name in index:
        if args.drop_mtp and name.startswith("mtp."):
            continue
        if fused_consumed.search(name):
            continue  # already fused above
        if name.endswith(".scale"):
            out[_scale_out(name)] = get(name)
        else:
            out[name] = get(name)

    print(f"fused -> {len(out)} output tensors; sharding + writing to {dst}")

    # ---- shard + index ----
    weight_map: dict[str, str] = {}
    shard: dict[str, torch.Tensor] = {}
    shard_bytes = 0
    shard_idx = 1
    shards_total: list[dict[str, torch.Tensor]] = []
    ordered = sorted(out.keys())
    # compute number of shards first for the of-XXXXX naming
    sizes = {n: out[n].numel() * out[n].element_size() for n in ordered}
    n_shards = max(1, sum(sizes.values()) // _SHARD_BYTES + 1)

    def shard_name(i: int) -> str:
        return f"model-{i:05d}-of-{n_shards:05d}.safetensors"

    for name in ordered:
        if shard_bytes + sizes[name] > _SHARD_BYTES and shard:
            save_file(shard, str(dst / shard_name(shard_idx)),
                      metadata={"format": "pt"})
            for k in shard:
                weight_map[k] = shard_name(shard_idx)
            shard, shard_bytes = {}, 0
            shard_idx += 1
        shard[name] = out[name]
        shard_bytes += sizes[name]
    if shard:
        save_file(shard, str(dst / shard_name(shard_idx)),
                  metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = shard_name(shard_idx)
    # the actual shard count may differ from the estimate; rename if needed
    actual = sorted({v for v in weight_map.values()})
    total_size = sum((dst / s).stat().st_size for s in actual)
    json.dump(
        {"metadata": {"total_size": total_size}, "weight_map": weight_map},
        open(dst / "model.safetensors.index.json", "w"),
        indent=2,
    )

    # ---- config: copy + update quant_config config_groups to name fused params ----
    qc = cfg.get("quantization_config", {})
    groups = qc.get("config_groups", {})
    # rewrite targets to the fused param names so the scheme is self-documenting
    if "experts_w4a16" in groups:
        groups["experts_w4a16"]["targets"] = [
            "*.ffn.experts.routed_experts.w13_weight",
            "*.ffn.experts.routed_experts.w2_weight",
        ]
    if "linears_w8a16" in groups:
        groups["linears_w8a16"]["targets"] = [
            "*.attn.fused_wqa_wkv", "*.attn.wq_b", "*.attn.wo_a", "*.attn.wo_b",
            "*.attn.indexer.wq_b", "*.attn.indexer.compressor.wkv",
            "*.ffn.shared_experts.gate_up_proj", "*.ffn.shared_experts.down_proj",
        ]
    json.dump(cfg, open(dst / "config.json", "w"), indent=2)

    # copy tokenizer/aux files
    for fn in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
               "special_tokens_map.json", "configuration_deepseek.py",
               "modeling_deepseek.py", "tokenization_deepseek.py"):
        if (src / fn).exists():
            shutil.copy(src / fn, dst / fn)

    print(f"DONE: {len(weight_map)} tensors across {len(actual)} shards, "
          f"{total_size/1e9:.1f} GB at {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
