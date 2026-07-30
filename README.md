<!-- markdownlint-disable MD001 MD041 -->

# AppMana vLLM: DeepSeek-V4 on consumer NVIDIA GPUs

This fork serves DeepSeek-V4-Flash on consumer NVIDIA GPUs:

- RTX 3090
- DGX Spark GB10

The supported INT4/INT8 checkpoint is
[`appmana/deepseek-v4-int4-int8`](https://huggingface.co/appmana/deepseek-v4-int4-int8)
at immutable revision
`597471bc544b541771306ddcdb7089ad740bb6d3`. The experimental
`appmana/deepseek-v4-nvfp4-fp8` checkpoint is a separate SparkInfer path and
does not define the INT4/INT8 kernel choices documented here.

## Supported GPUs

| Platform |
| --- |
| RTX 3090 |
| DGX Spark GB10 |

One image serves both GPUs. Kernel selection is checkpoint-driven.

## Checkpoint

The checkpoint is built from `deepseek-ai/DeepSeek-V4-Flash`: routed experts
are converted to INT4, dense/shared-expert/attention linears to INT8, and the
three MTP stages come from `deepseek-ai/DeepSeek-V4-Flash-DSpark`.

## INT4/INT8 kernel map

RTX 3090 and DGX Spark use the same implementations unless a row says
otherwise.

| Function | Stored/accumulation format | RTX 3090 | DGX Spark GB10 | Implementation source |
| --- | --- | --- | --- | --- |
| Routed-expert MoE GEMM | INT4 weights, BF16 activations/accumulation | Marlin W4A16 MoE | Marlin W4A16 MoE | vLLM vendored Marlin plus the `dsv4_int` loader |
| Dense, shared-expert and attention linear GEMMs | biased UINT8 weights with channel scales, BF16 activations/accumulation | AllSpark W8A16 | AllSpark W8A16 compiled for GB10 | vLLM AllSpark integration and native C++/CUDA extension |
| Sparse-MLA decode over the INT8 cache | packed INT8 cache rows, scale applied in the kernel, BF16 output | `flash_mla.sparse_mla_decode_int8` | same | separately installed `flash_mla` wheel from `AppMana/forks-flash-mla-ampere-dsv4` |
| Sparse-MLA prefill over the INT8 cache | packed INT8 cache rows, dequantized in-kernel, BF16 output | `flash_mla.sparse_mla_prefill_int8` | same | same `flash_mla` wheel |
| Indexer K write/cache | BF16 K quantized into symmetric INT8 plus scale | `vllm._custom_ops.indexer_k_quant_and_cache_int8` | same | vLLM native C++/CUDA cache extension |
| Indexer Q RoPE and quantization | BF16 Q to INT8 plus scale | `fused_indexer_q_rope_quant_int8` | same | vLLM Triton |
| Indexer prefill/decode logits | INT8 Q/K products accumulated into FP32 logits | Triton IMMA | Triton IMMA | `vllm.models.deepseek_v4.nvidia_imma.triton_kernels` |
| Indexer top-k | FP32 logits to token indices | vLLM native CUDA row/persistent top-k | same; the short-row decode dispatch keeps GB10 on the authored row kernel | vLLM `_C` top-k ops selected by `sparse_attn_indexer.py` |
| KV cache | 528-byte `int8_ds_mla` token slots | packed INT8 | packed INT8 | vLLM cache manager and attention class |
| mHC mixing and HCHead | BF16 gather/mixing path | vLLM torch/Triton fallback | vLLM torch/Triton fallback in the Hilton manifest | `vllm/model_executor/layers/mhc.py` |
| DSpark speculative decoding | three MTP draft stages | vLLM DSpark model runner | same when enabled | vLLM model runner v2 |

Important boundaries:

- The `flash_mla` wheel is the AppMana
  [`forks-flash-mla-ampere-dsv4`](https://github.com/AppMana/forks-flash-mla-ampere-dsv4)
  implementation. It is not SparkInfer and it is not the FlashMLA source
  vendored inside vLLM.
- The current INT4/INT8 indexer-logits path is Triton on both architectures;
  the top-k that consumes those logits is a native vLLM CUDA op, not Triton.
  Native SparkInfer INT8 indexer work is on
  `appmana-wip/int8-sparkinfer-indexer-sm121`; it is not selected by revision
  `597471bc…` and must not be described as deployed until it passes numerical
  and end-to-end tests.
- SparkInfer is therefore not required by the current INT4/INT8 attention or
  indexer path. It remains relevant to the separate NVFP4/FP8 checkpoint and
  to explicitly selected experimental components.
- TileLang is not selected by the Hilton INT4/INT8 deployment. It sets
  `VLLM_MHC_CUDA_BACKEND=triton`; every compiler and temporary cache,
  including the Triton cache, lives on a rank-local PVC rather than `/tmp`.
- AllSpark is built for both tested GPUs.

### Why the checkpoint lists an FP8 decode symbol

The checkpoint block lists both `sparse_mla_decode_fp8` and
`sparse_mla_decode_int8`. They are separate selector roles in the shared
configuration schema. With `cache_type=int8_ds_mla`, the attention class calls
the INT8 decode and INT8 prefill functions; it does not convert this cache to
FP8. The FP8 symbol does not mean the INT4/INT8 checkpoint uses SparkInfer or
an FP8 cache.

## Checkpoint configuration

The exact `config.json` block is:

```json
"vllm": {
  "kernels": [
    "flash_mla.sparse_mla_decode_fp8",
    "flash_mla.sparse_mla_decode_int8",
    "flash_mla.sparse_mla_prefill_int8",
    "vllm._custom_ops.indexer_k_quant_and_cache_int8",
    "vllm.models.deepseek_v4.common.ops.fused_indexer_q.fused_indexer_q_rope_quant_int8",
    "vllm.model_executor.layers.quantization.utils.marlin_utils.marlin_act_int8_process_scales"
  ],
  "cache_type": "int8_ds_mla"
}
```

The resolver maps symbols to roles, rejects unknown symbols and duplicate
selectors, and logs `vllm kernels resolved: ...` once during startup.
`--hf-overrides` replaces this entire block; it does not merge nested values.
An override must repeat every required entry.

The streaming prefill top-k toggle is intentionally not in the base block. For
the 1M configuration, append
`vllm.model_executor.layers.sparse_attn_indexer.streaming_prefill_topk` while
copying the rest of the block unchanged.

Weight kernels are selected by `quantization_config`; they are not swapped by
changing the attention kernel list.

## What this fork adds

- The `dsv4_int` loader and quantization format.
- AllSpark W8A16 dense linear support and Marlin INT4 routed experts.
- The packed `int8_ds_mla` cache and INT8 indexer paths.
- Checkpoint-config-driven, fail-closed kernel resolution.
- RTX 3090 and GB10 sparse-MLA attention through the external
  `flash_mla` wheel.
- Correct pipeline partitioning and rank-local checkpoint staging.
- Model runner v2 and asynchronous scheduling support for DSV4 and DSpark.
- Streaming prefill top-k so indexer working memory stays bounded at long
  context.
- Startup kernel coverage, staged health reporting, and compiler-cache
  persistence.

## Build

```bash
bash docker/build-consumer-platforms.sh --push
```

`docker/Dockerfile` builds one image for RTX 3090 and GB10, the AppMana
NCCL fork, the external `flash_mla` wheel, SparkInfer for the separate SM12x
lane, and the native vLLM extensions. `docker/versions.json` is generated from
the Dockerfile arguments and fed back into the build; stale architecture
metadata is a build failure.

The build verifies:

- expected `flash_mla` entry points import;
- both native vLLM extension libraries exist;
- AllSpark contains the requested architecture code;
- the AppMana NCCL library and routing markers exist;
- SparkInfer entry points required by the separate NVFP4 lane resolve.

## Serving

### RTX 3090

The validated large-context configuration uses PP across a chain of 24 GiB
GPUs, rank-local shard staging, async scheduling, model runner v2, and DSpark.
The 1M-context PP=12 partition is:

```text
3,4,4,4,4,4,4,4,4,4,4,0
```

The final rank carries the three draft stages and no target layers, leaving
room for the long-context cache. Use
`APPMANA_DSV4_PP_LAYER_PARTITION`; `VLLM_PP_LAYER_PARTITION` does not express
this fork-specific layout.

### DGX Spark

The Hilton GitOps manifest is
`demos-hilton/cluster/gitops/apps/base/inference/deepseek-v4-int4-int8.yaml`.
The exact rollout under test uses:

- two GB10 nodes, TP=2, PP=1, multiprocessing executor;
- direct dual-rail RoCE;
- pinned safetensors loading;
- model runner v2 and `--async-scheduling`;
- CUDA graphs enabled;
- `VLLM_MHC_CUDA_BACKEND=triton`;
- rank-local persistent compiler/autotune/temp roots;
- an initial 65,536-token acceptance ceiling before the context ladder.

The 65,536 setting is an acceptance starting point, not a claimed hardware
limit. Raise it only with measured cache capacity and correctness at each
ladder step.

Minimal common serve arguments:

```bash
vllm serve appmana/deepseek-v4-int4-int8 \
  --revision 597471bc544b541771306ddcdb7089ad740bb6d3 \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 \
  --safetensors-load-strategy pinned \
  --trust-remote-code \
  --async-scheduling
```

Use `/v1/chat/completions`; raw completions are not the validated prompting
path.

## Acceptance and benchmarking

Verify in this order:

1. Both ranks load the exact checkpoint revision and image digest.
2. The startup log prints the expected `vllm kernels resolved:` mapping and
   AllSpark selection.
3. `/health` reaches HTTP 200 with no rank restart.
4. Chat generation, reasoning-time tool calls, and DSML/Mastra code execution
   return correct results.
5. Node logs remain free of Xid, illegal-address, OOM and JIT compilation
   during measured requests.
6. Run C=1 and C=2 at short and long prompt lengths, then the context ladder.

Do not publish throughput from a run that fails output correctness. Use
`tools/ampere/dsv4_needle_bench.py` when possible because it records
throughput and verifies a request-unique needle in the same run.

### RTX 3090 results

`vllm bench serve`, C=1, PP=12 Thunderbolt chain, async scheduling:

| Context | DSpark | Decode tok/s | Prefill tok/s |
| ---: | --- | ---: | ---: |
| 4k in / 1k out | off | 33.3 | — |
| 4k in / 1k out | on | 56.1 | — |
| 620k in | off | — | 3,230 |
| 620k in | on | — | 3,390 |
| 950k | on | 66.7 | 3,020 |

The same 1M configuration without async scheduling measured 56.8/57.5 tok/s
decode.

### DGX Spark results

Two DGX Sparks, TP=2, DSpark:

| Actual prompt tokens | C | Decode tok/s per stream | Aggregate output tok/s |
| ---: | ---: | ---: | ---: |
| 8,822 | 1 | 59.27 | 31.40 |
| 8,822 | 2 | 44.86 | 46.67 |
| 17,644 | 1 | 58.56 | 4.52 |
| 17,644 | 2 | 46.58 | 15.35 |

## Operational notes

- DSV4 and DSpark must use model runner v2.
- Benchmark serving uses async scheduling and CUDA graphs; eager mode is only
  a diagnostic arm.
- On unified-memory GB10, explicit KV-cache bytes add to the GPU-utilization
  allocation. Leave host headroom or the system swaps instead of failing
  cleanly.
- Weights remain in their per-expert, separate-projection checkpoint layout
  and are fused while loading. Do not pre-fuse them.
- `--max-num-seqs` affects activation/workspace reservation as well as
  scheduling. Treat changes as memory experiments.
- Compiler caches must be persistent and rank-local. Do not point TileLang,
  Triton, TorchInductor, DeepGEMM, FlashInfer or `TMPDIR` at `/tmp` in the
  Kubernetes deployment.

## Related repositories

- [`AppMana/forks-flash-mla-ampere-dsv4`](https://github.com/AppMana/forks-flash-mla-ampere-dsv4):
  the external native sparse-MLA wheel used by this INT4/INT8 checkpoint.
- [`AppMana/forks-sparkinfer`](https://github.com/AppMana/forks-sparkinfer):
  SparkInfer kernels for the separate SM12x/NVFP4 lane and WIP native INT8
  indexer work.
- [`AppMana/forks-nccl-rdma-routing`](https://github.com/AppMana/forks-nccl-rdma-routing):
  NCCL rail and fallback routing.
- [`dragonintel performance reference`](https://github.com/hannesholste/dragonintel/blob/main/docs/dsv4-spark-performance-references.md):
  evidence map and acceptance requirements for Hilton measurements.

---

This Apache-2.0 fork is downstream of
[vLLM](https://github.com/vllm-project/vllm).
