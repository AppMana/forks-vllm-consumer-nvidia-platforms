# SPDX-License-Identifier: Apache-2.0
"""Reproduce long-prefill kernel faults with a one-layer DSV4 checkpoint view."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path


def run(checkpoint: Path, prompt_tokens: list[int], max_model_len: int) -> None:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_MHC_CUDA_BACKEND"] = "triton"

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(checkpoint),
        trust_remote_code=True,
        hf_overrides={"num_hidden_layers": 1},
        safetensors_load_strategy="lazy",
        max_model_len=max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.5,
        kv_cache_memory_bytes=4 << 30,
        enforce_eager=False,
        compilation_config={"cudagraph_capture_sizes": [1]},
        disable_log_stats=True,
        kernel_config={
            "enable_jit_warmup": True,
            "enable_cutedsl_warmup": True,
        },
        seed=20260801,
    )
    start = time.perf_counter()
    output = llm.generate(
        [{"prompt_token_ids": prompt_tokens}],
        SamplingParams(temperature=0, max_tokens=1, seed=20260801),
        use_tqdm=False,
    )[0]
    elapsed = time.perf_counter() - start
    print(
        json.dumps(
            {
                "elapsed_s": round(elapsed, 6),
                "input_tokens": len(prompt_tokens),
                "output_token_ids": list(output.outputs[0].token_ids),
                "prefill_tok_s": round(len(prompt_tokens) / elapsed, 2),
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prompt-tokens", type=int, default=32004)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=500)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    prompt_token_ids = [rng.randint(10, 20000) for _ in range(args.prompt_tokens)]
    run(args.checkpoint, prompt_token_ids, args.max_model_len)


if __name__ == "__main__":
    main()
