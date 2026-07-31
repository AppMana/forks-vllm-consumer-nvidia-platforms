<!-- markdownlint-disable MD001 MD041 -->

# AppMana vLLM: DeepSeek-V4 on consumer NVIDIA GPUs

This fork serves DeepSeek-V4-Flash on RTX 30xx and DGX Spark GB10 GPUs.

The supported INT4/INT8 checkpoint is
[`appmana/deepseek-v4-int4-int8`](https://huggingface.co/appmana/deepseek-v4-int4-int8)
at revision `ace78a6e9b5d90a43476fa1c098bfee1eb46c1de`. Its
`quantization_config.quant_method` is `dsv4_int`:

- routed experts are stored as signed INT4 weights;
- dense, shared-expert and attention linears are stored as INT8 weights;
- the attention cache uses the packed `int8_ds_mla` layout;
- the sparse indexer uses signed INT8 K rows with per-row scales.

The separate `appmana/deepseek-v4-nvfp4-fp8` checkpoint uses a different
quantization and kernel configuration.

## What this fork adds

- `dsv4_int` loading and dispatch for INT4 routed experts and INT8 linears.
- Packed INT8 attention and sparse-indexer cache paths.
- FlashMLA INT8 sparse attention for the checkpoint's decode and prefill
  selectors.
- SparkInfer INT8 sparse-indexer scoring on GB10.
- Checkpoint-configured kernel selection with fail-closed symbol validation.
- Pipeline parallelism greater than one, including a rank that holds only the
  DSpark draft stages.
- DeepSeek-V4 and DSpark support in model runner v2 with asynchronous
  scheduling.
- Bounded-memory streaming prefill top-k and persistent compiler caches.

One image contains the RTX 30xx and GB10 implementations. The checkpoint's
`vllm` block selects cache and kernel roles at startup; device capability then
selects the platform implementation for shared roles such as indexer scoring.

## INT4/INT8 kernel map

Sparse attention and sparse-indexer scoring are separate stages. FlashMLA
consumes the selected cache rows to compute attention. The indexer scores the
cache first and selects those rows.

| Function | RTX 30xx | DGX Spark GB10 |
| --- | --- | --- |
| Routed-expert MoE | Marlin W4A8-INT8 with INT4 weights | Marlin W4A8-INT8 with INT4 weights |
| Dense, shared-expert and attention linears | AllSpark W8A16 for supported channel-INT8 shapes; BF16 dequantization fallback | INT8 weights dequantized once to BF16, then `F.linear`; SM12x AllSpark is disabled by default |
| `wo_a` projection | BF16 weight or one-time INT8-to-BF16 dequantization for its inverse-RoPE einsum | BF16 weight or one-time INT8-to-BF16 dequantization for its inverse-RoPE einsum |
| Sparse-MLA attention decode | `flash_mla.sparse_mla_decode_int8` | `flash_mla.sparse_mla_decode_int8` |
| Sparse-MLA attention prefill | `flash_mla.sparse_mla_prefill_int8` | `flash_mla.sparse_mla_prefill_int8` |
| Indexer K cache write | vLLM INT8 quantize-and-cache kernel | vLLM INT8 quantize-and-cache kernel |
| Indexer Q RoPE and quantization | vLLM fused INT8 kernel | vLLM fused INT8 kernel |
| Indexer decode scoring over paged cache | vLLM Triton IMMA | SparkInfer native INT8 paged kernel |
| Indexer prefill scoring over contiguous cache | vLLM Triton IMMA | SparkInfer native INT8 contiguous kernel |
| Indexer long-prefill scoring | Triton IMMA per slab plus native CUDA candidate selection and merge | SparkInfer contiguous INT8 scoring per slab plus native CUDA candidate selection and merge |
| Indexer top-k | native CUDA row and persistent selectors | native CUDA row and persistent selectors |
| Attention KV cache | packed `int8_ds_mla`, 528 bytes per token | packed `int8_ds_mla`, 528 bytes per token |
| DSpark speculative decoding | three MTP draft stages in model runner v2 | three MTP draft stages in model runner v2 |

The FlashMLA INT8 kernels are provided by
[`AppMana/forks-flash-mla-int`](https://github.com/AppMana/forks-flash-mla-int).
SparkInfer provides the native GB10 paged and contiguous INT8 indexer kernels.
The streaming path calls the same contiguous scoring implementation once per
slab and merges candidates without materializing full-context logits.

## Checkpoint configuration

The checkpoint declares the kernel roles it requires:

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

Unknown symbols and conflicting implementations for one role fail at startup.
`--hf-overrides` can replace the block for controlled comparisons. Weight
formats and linear methods come from `quantization_config`; changing the
attention kernel list does not change stored weights.

Listing `streaming_prefill_topk` enables slabbed prefill selection. A prefill
that fits one slab keeps the one-shot path. A larger prefill scores and reduces
one slab at a time.

## Build

```bash
bash docker/build-consumer-platforms.sh
```

`docker/Dockerfile` builds one image with:

- native vLLM extensions for RTX 30xx and GB10;
- the AppMana NCCL fork;
- the external FlashMLA wheel;
- SparkInfer, including its GB10 INT8 indexer kernels.

`docker/versions.json` records the pinned component revisions used by the
build.

## Deployment examples

The public Kubernetes examples use LeaderWorkerSet, pipeline parallelism,
model runner v2 and asynchronous scheduling:

- [RTX 30xx PP=11](examples/deployment/dsv4-int4-int8-lws-pp11.yaml)
- [RTX 30xx PP=12](examples/deployment/dsv4-int4-int8-lws-pp12.yaml)

They derive rank information from `LWS_GROUP_SIZE`, `LWS_WORKER_INDEX` and
`LWS_LEADER_ADDRESS`. Supply the `dsv4-cache` PVC and `huggingface` Secret.
Optional network settings come from the `dsv4-network` ConfigMap.

## Related repositories

- [`AppMana/forks-flash-mla-int`](https://github.com/AppMana/forks-flash-mla-int):
  native sparse-MLA kernels for the INT8 cache.
- [`AppMana/forks-sparkinfer`](https://github.com/AppMana/forks-sparkinfer):
  GB10 FP8/NVFP4 kernels, mHC and native INT8 sparse-indexer kernels.
- [`AppMana/forks-nccl-rdma-routing`](https://github.com/AppMana/forks-nccl-rdma-routing):
  NCCL rail and fallback routing.

---

This Apache-2.0 fork is downstream of
[vLLM](https://github.com/vllm-project/vllm).
