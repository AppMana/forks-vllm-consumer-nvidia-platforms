# SPDX-License-Identifier: Apache-2.0
"""Print deterministic first-token logprobs for an NVFP4 MoE backend.

This is a cluster diagnostic rather than a pytest: run it once with
``sparkinfer`` and once with a known-good fallback such as ``marlin``.
"""

from __future__ import annotations

import json
import os
import sys

from vllm import LLM, SamplingParams


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} MODEL_PATH MOE_BACKEND")
    model_path, moe_backend = sys.argv[1:]
    num_hidden_layers = int(os.getenv("NVFP4_PROBE_NUM_LAYERS", "1"))
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        hf_overrides={"num_hidden_layers": num_hidden_layers},
        safetensors_load_strategy="lazy",
        max_model_len=512,
        max_num_seqs=1,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        moe_backend=moe_backend,
        kernel_config={
            "enable_jit_warmup": False,
            "enable_cutedsl_warmup": False,
        },
        disable_log_stats=True,
    )
    prompt = "Compare deterministic NVFP4 logits. " * 16
    params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=20,
        seed=1234,
    )
    request = llm.generate([prompt], params, use_tqdm=False)[0]
    sample = request.outputs[0]
    first_token_logprobs = sample.logprobs[0]
    payload = {
        "backend": moe_backend,
        "num_hidden_layers": num_hidden_layers,
        "prompt_token_ids": request.prompt_token_ids,
        "output_token_ids": sample.token_ids,
        "logprobs": {
            str(token_id): {
                "logprob": value.logprob,
                "rank": value.rank,
                "decoded_token": value.decoded_token,
            }
            for token_id, value in first_token_logprobs.items()
        },
    }
    print("NVFP4_LOGITS=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
