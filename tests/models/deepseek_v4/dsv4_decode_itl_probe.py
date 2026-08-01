# SPDX-License-Identifier: Apache-2.0
"""Fast decode-ITL probe for the DSV4 long-context decode cliff.

Loads a reduced DSV4 (6 layers, 256 experts, dspark layers 3-5) from the real
checkpoint on a single GPU, then measures per-step decode latency at several
context lengths in-process. The reduced model loads in ~1 minute, so an
eager/compiled/breakable A/B costs minutes instead of a full TP=2 JobSet.

Environment controls (all optional):
    DSV4_PROBE_CONTEXTS      comma list of prompt lengths (default 3000,8004)
    DSV4_PROBE_DECODE_TOKENS tokens to decode per context (default 48)
    DSV4_PROBE_SPEC          1 -> dspark speculative decode (default 1)
    DSV4_PROBE_COMPILED      1 -> compiled body (default 1)
    DSV4_PROBE_GMU           gpu_memory_utilization (default 0.35)
    VLLM_USE_BREAKABLE_CUDAGRAPH  forwarded to vLLM before import

Prints one JSON line per context:
    DSV4_ITL {"context": ..., "itl_ms_p50": ..., "itl_ms_max": ...}
"""

from __future__ import annotations

import json
import os
import random
import sys
import time

REDUCED_OVERRIDES = {
    "num_hidden_layers": 6,
    "n_routed_experts": 256,
    "compress_ratios": [0, 0, 4, 128, 4, 128, 0],
    "dspark_target_layer_ids": [3, 4, 5],
    "vllm": {
        "kernels": [
            "flash_mla.sparse_mla_decode_int8",
            "flash_mla.sparse_mla_prefill_int8",
            "vllm.models.deepseek_v4.nvidia_sm12x.mhc.sparkinfer_mhc_post_pre",
            "vllm._custom_ops.indexer_k_quant_and_cache_int8",
            "vllm.models.deepseek_v4.common.ops.fused_indexer_q."
            "fused_indexer_q_rope_quant_int8",
            "vllm.model_executor.layers.quantization.utils.marlin_utils."
            "marlin_act_int8_process_scales",
        ],
        "cache_type": "int8_ds_mla",
    },
}


def main() -> int:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_MHC_CUDA_BACKEND", "triton")
    contexts = [
        int(x) for x in os.environ.get("DSV4_PROBE_CONTEXTS", "3000,8004").split(",")
    ]
    decode_tokens = int(os.environ.get("DSV4_PROBE_DECODE_TOKENS", "48"))
    use_spec = os.environ.get("DSV4_PROBE_SPEC", "1") == "1"
    compiled = os.environ.get("DSV4_PROBE_COMPILED", "1") == "1"

    from vllm import LLM, SamplingParams

    snapshot = sys.argv[1]
    llm = LLM(
        model=snapshot,
        trust_remote_code=True,
        tokenizer_mode="deepseek_v4",
        hf_overrides=REDUCED_OVERRIDES,
        kv_cache_dtype="int8_ds_mla",
        safetensors_load_strategy="lazy",
        max_model_len=70000,
        max_num_seqs=2,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=float(os.environ.get("DSV4_PROBE_GMU", "0.35")),
        kv_cache_memory_bytes=2 << 30,
        enforce_eager=not compiled,
        compilation_config=(
            {"mode": 3, "cudagraph_capture_sizes": [1, 2]} if compiled else None
        ),
        speculative_config=(
            {
                "method": "dspark",
                "num_speculative_tokens": 5,
                "draft_sample_method": "greedy",
            }
            if use_spec
            else None
        ),
        async_scheduling=True,
        disable_log_stats=True,
        seed=20260801,
    )

    rng = random.Random(7)
    for context in contexts:
        ids = [rng.randint(10, 20000) for _ in range(context)]
        sampling = SamplingParams(
            temperature=0, max_tokens=decode_tokens, ignore_eos=True
        )
        # Streaming per-token timestamps aren't exposed in-process; measure
        # total decode wall via two calls: 1 token (prefill+first) and
        # decode_tokens, then attribute the difference to decode steps.
        t0 = time.perf_counter()
        llm.generate(
            [{"prompt_token_ids": ids}],
            SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
            use_tqdm=False,
        )
        t1 = time.perf_counter()
        llm.generate([{"prompt_token_ids": ids}], sampling, use_tqdm=False)
        t2 = time.perf_counter()
        prefill_s = t1 - t0
        # Second call re-prefills (no prefix cache hit guarantee across
        # calls with identical ids when prefix caching is on -- it will hit,
        # making the second call decode-dominated either way).
        decode_s = (t2 - t1) - min(prefill_s, t2 - t1) * 0.0
        per_step_ms = (t2 - t1) * 1000.0 / max(decode_tokens, 1)
        print(
            "DSV4_ITL "
            + json.dumps(
                {
                    "context": context,
                    "prefill_s": round(prefill_s, 3),
                    "wall_decode_call_s": round(decode_s, 3),
                    "approx_ms_per_token": round(per_step_ms, 2),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
