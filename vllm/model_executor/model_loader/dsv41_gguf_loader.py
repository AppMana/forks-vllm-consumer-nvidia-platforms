# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audited plain-parameter loading alongside native compact GGUF owners.

Tensor names follow the Unsloth dsv41-all converter at
4c4de092d03c1be09101c03adde849b0d7a93a40. Engram gate vectors are
stored as their FP32 product; no inverse BF16 factorization is attempted.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import regex as re
import torch

from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


@dataclass(frozen=True)
class PlainSource:
    path: Path
    name: str
    kind: str
    offset: int
    shape: tuple[int, ...]
    nbytes: int

    def read(self):
        dtype = {"F32": torch.float32, "BF16": torch.bfloat16}[self.kind]
        expected = torch.empty((), dtype=dtype).element_size()
        for dim in self.shape:
            expected *= dim
        if expected != self.nbytes:
            raise ValueError(f"GGUF plain payload size mismatch: {self.name}")
        with self.path.open("rb") as f:
            f.seek(self.offset)
            data = bytearray(f.read(self.nbytes))
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(
                    f.fileno(), self.offset, self.nbytes, os.POSIX_FADV_DONTNEED
                )
        if len(data) != self.nbytes:
            raise ValueError(f"Truncated GGUF plain payload: {self.name}")
        return torch.frombuffer(data, dtype=dtype).reshape(self.shape)


def plain_target(name):
    """Return model suffix, optional merged shard, and load transformation."""
    if name == "output_norm.weight":
        return "model.norm.weight", None, "plain"
    match = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
    if match is None:
        raise ValueError(f"Unmapped GGUF plain tensor: {name}")
    number, tail = match.groups()
    mapping = {
        "attn_norm.weight": ("attn_norm.weight", None),
        "ffn_norm.weight": ("ffn_norm.weight", None),
        "attn_q_a_norm.weight": ("attn.q_norm.weight", None),
        "attn_kv_a_norm.weight": ("attn.kv_norm.weight", None),
        "ffn_gate_inp.weight": ("ffn.gate.weight", None),
        "exp_probs_b.bias": ("ffn.gate.e_score_correction_bias", None),
        "exp_probs_b_vl.bias": ("ffn.gate.bias_vl", None),
        "indexer.attn_q_b.weight": ("attn.indexer.wq_b.weight", None),
        "indexer.attn_k.weight": ("attn.indexer.wk.weight", None),
        "indexer.k_norm.weight": ("attn.indexer.k_norm.weight", None),
        "indexer.proj.weight": ("attn.indexer.weights_proj.weight", None),
        "attn_compressor_kv.weight": ("attn.compressor.fused_wkv_wgate.weight", 0),
        "attn_compressor_gate.weight": ("attn.compressor.fused_wkv_wgate.weight", 1),
        "attn_compressor_norm.weight": ("attn.compressor.norm.weight", None),
    }
    if tail == "engram_gate.weight":
        return f"model.layers.{number}.engram.q_weight", None, "gate_product"
    if tail == "attn_sinks.weight":
        return f"model.layers.{number}.attn.attn_sink", None, "sink"
    if re.fullmatch(r"hc_(attn|ffn)_(fn|base|scale)\.weight", tail):
        return f"model.layers.{number}.{tail.removesuffix('.weight')}", None, "plain"
    if tail not in mapping:
        raise ValueError(f"Unmapped GGUF plain tensor: {name}")
    suffix, shard = mapping[tail]
    return f"model.layers.{number}.{suffix}", shard, "plain"


def index_plain(paths):
    from gguf import GGUFReader

    plain, all_names = {}, set()
    for path in paths:
        reader = GGUFReader(path, mode="r")
        if reader.byte_order != "I":
            raise ValueError("Native GGUF requires little-endian payloads")
        for tensor in reader.tensors:
            if tensor.name in all_names:
                raise ValueError(f"Duplicate GGUF tensor: {tensor.name}")
            all_names.add(tensor.name)
            if tensor.tensor_type.name in ("F32", "BF16"):
                plain[tensor.name] = PlainSource(
                    Path(path),
                    tensor.name,
                    tensor.tensor_type.name,
                    int(tensor.data_offset),
                    tuple(reversed(tuple(map(int, tensor.shape)))),
                    int(tensor.n_bytes),
                )
        del reader
    return plain, all_names


def _compact_target(name):
    if name == "token_embd.weight":
        return "model.embed_tokens.weight"
    if name == "output.weight":
        return "lm_head.weight"
    if name == "per_layer_token_embd.weight":
        return None
    match = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
    if match is None:
        return None
    number, tail = match.groups()
    mapping = {
        "attn_q_a.weight": "attn.fused_wqa_wkv.weight",
        "attn_kv.weight": "attn.fused_wqa_wkv.weight",
        "attn_q_b.weight": "attn.wq_b.weight",
        "attn_output_a.weight": "attn.wo_a.weight",
        "attn_output_b.weight": "attn.wo_b.weight",
        "indexer.attn_q_b.weight": "attn.indexer.wq_b.weight",
        "engram_kv.weight": "engram.wkv.weight",
        "ffn_gate_shexp.weight": "ffn.shared_experts.gate_up_proj.weight",
        "ffn_up_shexp.weight": "ffn.shared_experts.gate_up_proj.weight",
        "ffn_down_shexp.weight": "ffn.shared_experts.down_proj.weight",
        "ffn_gate_exps.weight": "ffn.experts.routed_experts.weight",
        "ffn_up_exps.weight": "ffn.experts.routed_experts.weight",
        "ffn_down_exps.weight": "ffn.experts.routed_experts.weight",
    }
    if tail not in mapping:
        return None
    return f"model.layers.{number}.{mapping[tail]}"


def _inner_model(model):
    lm = getattr(model, "language_model", model)
    return getattr(lm, "model", lm)


def _is_pp_missing_compact(name, model):
    from vllm.model_executor.models.utils import is_pp_missing_parameter

    inner = _inner_model(model)
    if name == "per_layer_token_embd.weight":
        engram_batch = getattr(inner, "engram_batch", "unconfigured")
        return engram_batch is None
    target = _compact_target(name)
    if target is None:
        return False
    for candidate in (target, f"language_model.{target}"):
        if is_pp_missing_parameter(candidate, model):
            return True
    match = re.match(r"model\.layers\.(\d+)\.", target)
    if match is not None:
        layer_id = int(match.group(1))
        start_layer = getattr(inner, "start_layer", 0)
        end_layer = getattr(inner, "end_layer", layer_id + 1)
        if not (start_layer <= layer_id < end_layer):
            return True
    return False


def _is_pp_missing_plain(suffix, model, params):
    from vllm.model_executor.models.utils import is_pp_missing_parameter

    for candidate in (suffix, f"language_model.{suffix}"):
        if is_pp_missing_parameter(candidate, model):
            return True
    match = re.match(r"model\.layers\.(\d+)\.", suffix)
    if match is not None:
        layer_id = int(match.group(1))
        inner = _inner_model(model)
        start_layer = getattr(inner, "start_layer", 0)
        end_layer = getattr(inner, "end_layer", layer_id + 1)
        if not (start_layer <= layer_id < end_layer):
            return not any(
                name == suffix or name.endswith("." + suffix) for name in params
            )
    return False


def _resolve(params, suffix):
    matches = [name for name in params if name == suffix or name.endswith("." + suffix)]
    if len(matches) != 1:
        raise ValueError(
            f"GGUF parameter target must resolve exactly once: {suffix}, got {matches}"
        )
    return matches[0]


@torch.no_grad()
def load_plain_parameters(model, paths, *, tp_rank, tp_size, audit_only=False):
    """Preflight all routes and payloads, then load every non-compact parameter.

    Missing parameters, unmapped tensors and incomplete merged projections fail
    before writes. Compact owners consume their descriptors during post-loading.
    The returned audit names every consumed checkpoint tensor and parameter.
    """
    if tp_size <= 0 or not 0 <= tp_rank < tp_size:
        raise ValueError("Invalid GGUF tensor-parallel rank")
    plain, all_names = index_plain(paths)
    params = dict(model.named_parameters())
    compact: set[str] = set()
    for module in model.modules():
        method = getattr(module, "quant_method", None)
        sources = tuple(getattr(method, "sources", ()))
        source = getattr(method, "source", None)
        if source is not None:
            sources += (source,)
        sources += tuple(part.source for part in getattr(method, "slices", ()))
        compact.update(source.name for source in sources)
        # Disk-backed table modules declare their source independently of a
        # linear quant method; the Engram implementation must own this payload.
        table_source = getattr(module, "gguf_source", None)
        if table_source is not None:
            compact.add(table_source.name)
    if compact - all_names:
        raise ValueError(
            f"Compact owners reference missing tensors: {sorted(compact - all_names)}"
        )
    extra = {
        name
        for name in (all_names - set(plain) - compact)
        if not _is_pp_missing_compact(name, model)
    }
    if extra:
        raise ValueError(f"GGUF tensors have no model owner: {sorted(extra)}")
    routes, covered = [], set()
    merged: dict[str, set[Any]] = {}
    for source in plain.values():
        suffix, shard, transform = plain_target(source.name)
        if _is_pp_missing_plain(suffix, model, params):
            continue
        name = _resolve(params, suffix)
        if name in covered and shard is None:
            raise ValueError(f"Multiple GGUF tensors target {name}")
        covered.add(name)
        if shard is not None:
            merged.setdefault(name, set()).add(shard)
        if transform == "gate_product":
            other = _resolve(params, suffix.removesuffix("q_weight") + "k_weight")
            if source.kind != "F32" or tuple(params[other].shape) != source.shape:
                raise ValueError(
                    "Engram folded gate requires matching FP32 source geometry"
                )
            covered.add(other)
        else:
            other = None
        if source.path.stat().st_size < source.offset + source.nbytes:
            raise ValueError(f"Truncated GGUF plain payload: {source.name}")
        if (
            shard is None
            and transform != "sink"
            and tuple(params[name].shape) != source.shape
        ):
            raise ValueError(f"GGUF shape disagrees with {name}: {source.shape}")
        if transform == "sink" and (
            len(source.shape) != 1
            or source.shape[0] % tp_size
            or params[name].numel() < source.shape[0] // tp_size
        ):
            raise ValueError("Attention sink geometry does not match TP capacity")
        if shard is not None:
            module = model.get_submodule(name.rsplit(".", 1)[0])
            if source.shape != (module.output_sizes[shard], module.input_size):
                raise ValueError(f"Merged GGUF source shape disagrees with {name}")
        routes.append((source, name, shard, transform, other))
    if set(params) - covered:
        raise ValueError(
            f"Model parameters missing from GGUF: {sorted(set(params) - covered)}"
        )
    for name, shards in merged.items():
        module = model.get_submodule(name.rsplit(".", 1)[0])
        expected = set(range(len(module.output_sizes)))
        if shards != expected:
            raise ValueError(
                f"Incomplete merged GGUF parameter: {name}, {shards} != {expected}"
            )
    audit = {
        "plain_tensors": sorted(plain),
        "compact_tensors": sorted(compact),
        "parameters": sorted(covered),
        "weights_loaded": not audit_only,
    }
    if audit_only:
        return audit
    for source, name, shard, transform, other in routes:
        data, param = source.read(), params[name]
        if transform == "gate_product":
            if data.dtype != torch.float32:
                raise ValueError("Engram folded gate must retain FP32 precision")
            param.data = data.to(device=param.device, dtype=torch.float32)
            params[other].fill_(1)
        elif transform == "sink":
            if len(source.shape) != 1 or source.shape[0] % tp_size:
                raise ValueError("Attention sink head count does not split evenly")
            local = data.chunk(tp_size)[tp_rank]
            if param.numel() < local.numel():
                raise ValueError("Attention sink capacity is too small")
            param.fill_(-float("inf"))
            param[: local.numel()].copy_(local)
        else:
            loader: Any = getattr(param, "weight_loader", default_weight_loader)
            if shard is None:
                loader(param, data)
            else:
                loader(param, data, shard)
    return audit


class Dsv41GGUFModelLoader(BaseModelLoader):
    """Load pinned local GGUF shards, retaining compact quant-method ownership."""

    def __init__(self, load_config):
        super().__init__(load_config)
        extra = load_config.model_loader_extra_config or {}
        self.files = tuple(Path(p).resolve() for p in extra.get("gguf_files", ()))
        if not self.files:
            raise ValueError(
                "dsv41_gguf loader requires gguf_files in loader extra config"
            )

    def download_model(self, model_config):
        for path in self.files:
            if not path.is_file():
                raise FileNotFoundError(path)

    def create_model(self, vllm_config, model_config, prefix=""):
        if tuple(getattr(vllm_config.quant_config, "files", ())) != self.files:
            raise ValueError(
                "GGUF loader and quant config must use identical shard paths"
            )
        graph_mode = vllm_config.compilation_config.cudagraph_mode
        if (
            getattr(vllm_config.quant_config, "engram_source", None) is not None
            and graph_mode is not None
            and graph_mode.has_full_cudagraphs()
        ):
            raise ValueError(
                "Disk-backed GGUF Engram requires an eager lookup boundary; "
                "use PIECEWISE CUDA graphs with VLLM_USE_BREAKABLE_CUDAGRAPH=1 "
                "or disable CUDA graphs. Full CUDA graphs cannot perform disk I/O."
            )
        mm_config = model_config.multimodal_config
        if mm_config is not None and mm_config.get_limit_per_prompt("image") != 0:
            raise ValueError(
                "This GGUF checkpoint contains text weights only; set image limit to 0"
            )
        model = super().create_model(vllm_config, model_config, prefix)
        # _mark_tower_model removes disabled child modules, but its delimiter
        # vectors are direct Parameters and survive that context.
        if hasattr(model, "language_model"):
            from vllm.model_executor.models.utils import StageMissingLayer

            if not isinstance(model.vision, StageMissingLayer):
                raise ValueError(
                    "Text GGUF loading requires the vision tower to be disabled"
                )
            for name in ("image_start", "image_end", "image_newline"):
                model.register_parameter(name, None)
        return model

    def load_weights(self, model, model_config):
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        model.gguf_load_audit = load_plain_parameters(
            model,
            self.files,
            tp_rank=get_tensor_model_parallel_rank(),
            tp_size=get_tensor_model_parallel_world_size(),
        )
