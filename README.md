<!-- markdownlint-disable MD001 MD041 -->

# AppMana vLLM: DeepSeek-V4 on consumer NVIDIA GPUs

This fork serves DeepSeek-V4-Flash on consumer NVIDIA GPUs:

- RTX 30xx
- DGX Spark GB10

The supported INT4/INT8 checkpoint is
[`appmana/deepseek-v4-int4-int8`](https://huggingface.co/appmana/deepseek-v4-int4-int8)
at revision `ace78a6e9b5d90a43476fa1c098bfee1eb46c1de`. The experimental
`appmana/deepseek-v4-nvfp4-fp8` checkpoint is a separate SparkInfer path and
does not define the INT4/INT8 kernel choices documented here.

## What this fork adds

- `dsv4_int`: INT4 routed experts and INT8 dense, shared-expert and attention
  linears.
- INT8 KV-cache and indexer paths.
- Sparse-MLA attention for RTX 30xx and GB10 through
  [`forks-flash-mla-int`](https://github.com/AppMana/forks-flash-mla-int).
- A simple `vllm` checkpoint config block for choosing the cache and kernels.
- PP>1 support and rank-local checkpoint staging.
- Model runner v2 and asynchronous scheduling for DSV4 and DSpark.
- Streaming prefill top-k to make contexts above 640k work.
- Kernel prewarming, staged health reporting and persistent compiler caches.

## Supported GPUs

| Platform |
| --- |
| RTX 30xx |
| DGX Spark GB10 |

One image serves both GPUs. Kernel selection is checkpoint-driven.

## Checkpoint

The checkpoint is built from `deepseek-ai/DeepSeek-V4-Flash`: routed experts
are converted to INT4, dense/shared-expert/attention linears to INT8, and the
three MTP stages come from `deepseek-ai/DeepSeek-V4-Flash-DSpark`.

## INT4/INT8 kernel map

RTX 30xx and DGX Spark use the same implementations unless a row says
otherwise.

| Function | Stored/accumulation format | RTX 30xx | DGX Spark GB10 | Source |
| --- | --- | --- | --- | --- |
| Routed-expert MoE GEMM | INT4 weights, BF16 activations/accumulation | Marlin W4A16 MoE | Marlin W4A16 MoE | community ([Marlin](https://github.com/IST-DASLab/marlin)) |
| Dense, shared-expert and attention linear GEMMs | biased UINT8 weights with channel scales, BF16 activations/accumulation | AllSpark W8A16 | AllSpark W8A16 | vLLM |
| Sparse-MLA decode over the INT8 cache | packed INT8 cache rows, scale applied in the kernel, BF16 output | `flash_mla.sparse_mla_decode_int8` | `flash_mla.sparse_mla_decode_int8` | AppMana ([forks-flash-mla-int](https://github.com/AppMana/forks-flash-mla-int)) |
| Sparse-MLA prefill over the INT8 cache | packed INT8 cache rows, dequantized in-kernel, BF16 output | `flash_mla.sparse_mla_prefill_int8` | `flash_mla.sparse_mla_prefill_int8` | AppMana ([forks-flash-mla-int](https://github.com/AppMana/forks-flash-mla-int)) |
| Indexer K write/cache | BF16 K quantized into symmetric INT8 plus scale | `indexer_k_quant_and_cache_int8` | `indexer_k_quant_and_cache_int8` | vLLM |
| Indexer Q RoPE and quantization | BF16 Q to INT8 plus scale | `fused_indexer_q_rope_quant_int8` | `fused_indexer_q_rope_quant_int8` | vLLM |
| Indexer prefill/decode logits | INT8 Q/K products accumulated into FP32 logits | Triton IMMA | Triton IMMA | vLLM |
| Indexer top-k | FP32 logits to token indices | native CUDA row/persistent top-k | native CUDA row/persistent top-k | vLLM |
| KV cache | 528-byte `int8_ds_mla` token slots | packed INT8 | packed INT8 | vLLM |
| mHC mixing and HCHead | BF16 gather/mixing path | torch/Triton | torch/Triton | vLLM |
| DSpark speculative decoding | three MTP draft stages | model runner v2 | model runner v2 | vLLM |

## Checkpoint configuration

We added a `vllm` block to our checkpoint configs so each checkpoint can state
which cache and kernels it needs:

```json
"vllm": {
  "kernels": [
    "flash_mla.sparse_mla_decode_int8",
    "flash_mla.sparse_mla_prefill_int8",
    "vllm._custom_ops.indexer_k_quant_and_cache_int8",
    "vllm.models.deepseek_v4.common.ops.fused_indexer_q.fused_indexer_q_rope_quant_int8",
    "vllm.model_executor.layers.quantization.utils.marlin_utils.marlin_act_int8_process_scales",
    "vllm.model_executor.layers.sparse_attn_indexer.streaming_prefill_topk"
  ],
  "cache_type": "int8_ds_mla"
}
```

vLLM reads this block at startup. `--hf-overrides` can replace it for testing.
Streaming top-k is enabled by default for long context.

Weight kernels are selected by `quantization_config`; they are not swapped by
changing the attention kernel list.

## Build

```bash
bash docker/build-consumer-platforms.sh --push
```

`docker/Dockerfile` builds one image for RTX 30xx and GB10, the AppMana
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

### RTX 30xx

The validated large-context configuration uses PP across 24 GiB GPUs connected
by a network with 115 microseconds of latency per hop. The layer partitions
are:

```text
PP=11: 3,4,4,4,4,4,4,4,4,4,4
PP=12: 3,4,4,4,4,4,4,4,4,4,4,0
```

The PP=12 final rank carries the three draft stages and no target layers,
leaving room for the long-context cache.

Kubernetes examples:

- [RTX 30xx PP=11](examples/deployment/dsv4-int4-int8-lws-pp11.yaml)
- [RTX 30xx PP=12](examples/deployment/dsv4-int4-int8-lws-pp12.yaml)

The examples use the LWS-provided `LWS_GROUP_SIZE`, `LWS_WORKER_INDEX`, and
`LWS_LEADER_ADDRESS` values. Provide the `dsv4-cache` PVC, the `huggingface`
Secret, and optional network settings in the `dsv4-network` ConfigMap.

## Benchmarks

### RTX 30xx

PP=12, C=1, async scheduling:

| Input tokens | Output tokens | DSpark | Decode tok/s | Prefill tok/s |
| ---: | ---: | :---: | ---: | ---: |
| 4,000 | 1,000 | off | 33.3 | — |
| 4,000 | 1,000 | on | 56.1 | — |
| 620,000 | — | off | — | 3,230 |
| 620,000 | — | on | — | 3,390 |
| 950,000 | — | on | 66.7 | 3,020 |

### DGX Spark GB10

Two GB10 systems, TP=2, DSpark:

| Input tokens | C | Decode tok/s per stream | Aggregate output tok/s |
| ---: | ---: | ---: | ---: |
| 8,822 | 1 | 59.27 | 31.40 |
| 8,822 | 2 | 44.86 | 46.67 |
| 17,644 | 1 | 58.56 | 4.52 |
| 17,644 | 2 | 46.58 | 15.35 |

## Related repositories

- [`AppMana/forks-flash-mla-int`](https://github.com/AppMana/forks-flash-mla-int):
  the external native sparse-MLA wheel used by this INT4/INT8 checkpoint.
- [`AppMana/forks-sparkinfer`](https://github.com/AppMana/forks-sparkinfer):
  SparkInfer kernels for the separate SM12x/NVFP4 lane and WIP native INT8
  indexer work.
- [`AppMana/forks-nccl-rdma-routing`](https://github.com/AppMana/forks-nccl-rdma-routing):
  NCCL rail and fallback routing.

---

This Apache-2.0 fork is downstream of
[vLLM](https://github.com/vllm-project/vllm).
