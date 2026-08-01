# SPDX-License-Identifier: Apache-2.0
"""Capture deterministic DSV4 INT4/INT8 activation boundaries.

The probe builds a symlink-only view of a downloaded checkpoint containing the
embedding, one decoder layer, final mHC/norm, and LM head tensors.  It then runs
one prefill and serializes module-boundary tensors for comparison across vLLM
revisions and execution modes. Checkpoint files and Hugging Face cache metadata
are never modified.

Usage::

    python tests/models/deepseek_v4/dsv4_int_boundary_probe.py view SNAPSHOT VIEW
    python tests/models/deepseek_v4/dsv4_int_boundary_probe.py capture VIEW run.pt
    python tests/models/deepseek_v4/dsv4_int_boundary_probe.py compare left.pt right.pt

Set ``VLLM_DSV4_ALLSPARK_SM12X=1`` for the experimental SM12x AllSpark arm.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

BOUNDARY_GLOBAL_WEIGHTS = {
    "embed.weight",
    "hc_head_base",
    "hc_head_fn",
    "hc_head_scale",
    "head.weight",
    "norm.weight",
}


def create_boundary_view(source: Path, target: Path, *, layer: int) -> Path:
    source = source.resolve()
    target.mkdir(parents=True, exist_ok=True)
    with (source / "model.safetensors.index.json").open() as handle:
        index = json.load(handle)
    prefix = f"layers.{layer}."
    weights = {
        name: shard
        for name, shard in index["weight_map"].items()
        if name.startswith(prefix) or name in BOUNDARY_GLOBAL_WEIGHTS
    }
    if not any(name.startswith(prefix) for name in weights):
        raise RuntimeError(f"checkpoint has no tensors matching {prefix}")

    view_index = {
        "metadata": {
            **index.get("metadata", {}),
            "boundary_probe": f"embedding+layer{layer}+final-head",
        },
        "weight_map": weights,
    }
    (target / "model.safetensors.index.json").write_text(
        json.dumps(view_index, sort_keys=True)
    )
    artifacts = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        *weights.values(),
    }
    for name in artifacts:
        source_file = source / name
        if not source_file.exists():
            continue
        destination = target / name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(os.path.relpath(source_file, target))
    return target


def _copy_output(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, tuple):
        return tuple(_copy_output(item) for item in value)
    if isinstance(value, list):
        return [_copy_output(item) for item in value]
    if isinstance(value, dict):
        return {key: _copy_output(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


def capture_boundaries(
    checkpoint: Path,
    output: Path,
    *,
    prompt: str,
    compiled: bool,
) -> None:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_MHC_CUDA_BACKEND"] = "triton"
    # The known-good revision predates explicit backend selection. Hiding
    # TileLang puts both revisions on their shared torch/Triton eager path and
    # avoids comparing different optional package installations.
    sys.modules["tilelang"] = None
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(checkpoint),
        trust_remote_code=True,
        hf_overrides={"num_hidden_layers": 1},
        safetensors_load_strategy="lazy",
        max_model_len=128,
        max_num_seqs=1,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.5,
        kv_cache_memory_bytes=1 << 30,
        enforce_eager=not compiled,
        compilation_config=(
            {"cudagraph_capture_sizes": [1]} if compiled else None
        ),
        disable_log_stats=True,
        kernel_config={
            "enable_jit_warmup": False,
            "enable_cutedsl_warmup": False,
        },
        seed=20260801,
    )
    worker = llm.llm_engine.model_executor.driver_worker.worker
    model = worker.model_runner.model
    modules = dict(model.named_modules())
    module_names = {
        "embedding": "model.embed_tokens",
        "layer0_mhc_pre": "model.layers.0.mhc_pre",
        "layer0_attn_norm": "model.layers.0.attn_norm",
        "layer0_fused_wqa_wkv": "model.layers.0.attn.fused_wqa_wkv",
        "layer0_wq_b": "model.layers.0.attn.wq_b",
        "layer0_wo_b": "model.layers.0.attn.wo_b",
        "layer0_attention": "model.layers.0.attn",
        "layer0_mhc_post_pre": "model.layers.0.mhc_fused_post_pre",
        "layer0_ffn_norm": "model.layers.0.ffn_norm",
        "layer0_ffn": "model.layers.0.ffn",
        "decoder_0": "model.layers.0",
        "hc_head": "model.hc_head",
        "final_norm": "model.norm",
        "logits": "logits_processor",
    }
    missing = [name for name in module_names.values() if name not in modules]
    if missing:
        raise RuntimeError(f"missing probe modules {missing}; have {sorted(modules)}")

    captured: dict[str, Any] = {}
    hooks = []

    def save_first(name: str):
        def hook(_module, args, result):
            if name not in captured:
                captured[f"{name}.input"] = _copy_output(args)
                captured[name] = _copy_output(result)

        return hook

    for name, module_name in module_names.items():
        hooks.append(modules[module_name].register_forward_hook(save_first(name)))

    tokenizer = llm.get_tokenizer()
    prompt_token_ids = tokenizer.encode(prompt)
    sampling = SamplingParams(temperature=0, max_tokens=1, logprobs=20, seed=20260801)
    request = llm.generate(
        [{"prompt_token_ids": prompt_token_ids}], sampling, use_tqdm=False
    )[0]
    for hook in hooks:
        hook.remove()

    sample = request.outputs[0]
    captured["metadata"] = {
        "checkpoint": str(checkpoint.resolve()),
        "prompt": prompt,
        "prompt_token_ids": prompt_token_ids,
        "output_token_ids": list(sample.token_ids),
        "torch_version": str(torch.__version__),
        "compiled": compiled,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(captured, output)
    print(json.dumps(captured["metadata"], sort_keys=True))
    print(f"saved={output} boundaries={sorted(k for k in captured if k != 'metadata')}")


def _flatten_tensors(value: Any, prefix: str = "") -> dict[str, torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return {prefix: value}
    if isinstance(value, (tuple, list)):
        result: dict[str, torch.Tensor] = {}
        for index, item in enumerate(value):
            result.update(_flatten_tensors(item, f"{prefix}.{index}"))
        return result
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(_flatten_tensors(item, f"{prefix}.{key}"))
        return result
    return {}


def compare_captures(left_path: Path, right_path: Path) -> int:
    left = torch.load(left_path, map_location="cpu", weights_only=True)
    right = torch.load(right_path, map_location="cpu", weights_only=True)
    order = [
        "embedding",
        "layer0_mhc_pre",
        "layer0_attn_norm",
        "layer0_fused_wqa_wkv.input",
        "layer0_fused_wqa_wkv",
        "layer0_wq_b.input",
        "layer0_wq_b",
        "layer0_wo_b.input",
        "layer0_wo_b",
        "layer0_attention",
        "layer0_mhc_post_pre",
        "layer0_ffn_norm",
        "layer0_ffn",
        "decoder_0",
        "hc_head",
        "final_norm",
        "logits",
    ]
    first_divergence = None
    for boundary in order:
        left_tensors = _flatten_tensors(left[boundary], boundary)
        right_tensors = _flatten_tensors(right[boundary], boundary)
        if left_tensors.keys() != right_tensors.keys():
            raise RuntimeError(
                f"{boundary} tensor structures differ: "
                f"{left_tensors.keys()} != {right_tensors.keys()}"
            )
        boundary_equal = True
        for name in left_tensors:
            a, b = left_tensors[name], right_tensors[name]
            if a.shape != b.shape or a.dtype != b.dtype:
                print(
                    f"{name} shape_or_dtype RED {a.shape}/{a.dtype} {b.shape}/{b.dtype}"
                )
                boundary_equal = False
                continue
            exact = torch.equal(a, b)
            delta = (a.float() - b.float()).abs()
            max_abs = delta.max().item() if delta.numel() else 0.0
            mean_abs = delta.mean().item() if delta.numel() else 0.0
            print(
                f"{name} {'GREEN' if exact else 'RED'} exact={exact} "
                f"max_abs={max_abs:.9g} mean_abs={mean_abs:.9g}"
            )
            boundary_equal &= exact
        if not boundary_equal and first_divergence is None:
            first_divergence = boundary
    print(f"first_divergence={first_divergence or 'none'}")
    return int(first_divergence is not None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    view = subparsers.add_parser("view")
    view.add_argument("source", type=Path)
    view.add_argument("target", type=Path)
    view.add_argument("--layer", type=int, default=0)
    capture = subparsers.add_parser("capture")
    capture.add_argument("checkpoint", type=Path)
    capture.add_argument("output", type=Path)
    capture.add_argument("--prompt", default="The capital of France is")
    capture.add_argument(
        "--compiled",
        action="store_true",
        help="request the optimized serving path instead of enforce-eager",
    )
    compare = subparsers.add_parser("compare")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    args = parser.parse_args()
    if args.command == "view":
        create_boundary_view(args.source, args.target, layer=args.layer)
        return 0
    if args.command == "capture":
        capture_boundaries(
            args.checkpoint,
            args.output,
            prompt=args.prompt,
            compiled=args.compiled,
        )
        return 0
    return compare_captures(args.left, args.right)


if __name__ == "__main__":
    raise SystemExit(main())
