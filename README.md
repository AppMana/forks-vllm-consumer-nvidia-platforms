<!-- markdownlint-disable MD001 MD041 -->

# AppMana vLLM: DeepSeek-V4 on consumer NVIDIA GPUs

This fork serves DeepSeek-V4-Flash and DeepSeek-V4-Flash-Vision-Exp on RTX
30xx and DGX Spark GB10 GPUs from one image. Upstream vLLM is merged
regularly (last: upstream `729ebac498`, merged 2026-09-18); `MERGING.md`
records the procedure and the invariants each merge has to preserve.

## Supported checkpoints

| Checkpoint | Revision | Source | Quantization | Serving |
| --- | --- | --- | --- | --- |
| [`appmana/deepseek-v4-int4-int8`](https://huggingface.co/appmana/deepseek-v4-int4-int8) | `3b25318eca9ccebc48130251d2c8b825341ad0ea` | `deepseek-ai/DeepSeek-V4-Flash-0731` at `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` | INT4 group-32 routed experts (MSE scales), channel-INT8 dense, shared-expert and attention linears, packed `int8_ds_mla` cache, INT8 sparse indexer, three native DSpark draft stages | PP=11 or PP=12 across RTX 3090s; TP=2 across two GB10s |
| [`appmana/deepseek-v4-flash-vision-exp-int4-int8`](https://huggingface.co/appmana/deepseek-v4-flash-vision-exp-int4-int8) | `65df4c51a2e724340c7963ccddd1f5199a2575e4` (BF16 vision tower on branch `bf16-vision-reference`, `bee7f4e9f445e49067b35d88e4ef3ef3ca9f1a56`) | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` at `6821d6ad3681a4b137b066b76094fa82ebd0a380` | The same language quantization plus a group-32 W8A8 INT8 vision tower and aligner run on integer tensor cores | PP=11 across RTX 3090s with layer partition `4,4,4,4,4,4,4,4,5,5,1`, eight images per request, eight sequences, DSpark with seven probabilistic draft tokens |

Both carry `quantization_config.quant_method = dsv4_int` and a `vllm` block
naming the kernels they require; stock vLLM cannot load either. The
separate `appmana/deepseek-v4-nvfp4-fp8` checkpoint uses a different
quantization and kernel configuration.

## What this fork adds

- `dsv4_int` loading and dispatch for INT4 routed experts and INT8 linears.
- Packed INT8 attention and sparse-indexer cache paths.
- FlashMLA INT8 sparse attention for the checkpoint's decode and prefill
  selectors.
- SparkInfer INT8 sparse-indexer scoring on GB10.
- Checkpoint-configured kernel selection with fail-closed symbol validation.
- Pipeline parallelism greater than one, including a rank that holds only the
  DSpark draft stages, and a first rank that owns the vision tower, aligner
  and image embeddings.
- DeepSeek-V4 and DSpark support in model runner v2 with asynchronous
  scheduling.
- Bounded-memory streaming prefill top-k and persistent compiler caches.
- INT8 integer-tensor-core vision projections and attention whose kernels
  never recompile for a new image size.

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
| Vision patch projection, QKV, output, MLP and aligner linears (vision checkpoint) | Triton W8A8 IMMA, group-32 INT8 weights and dynamic INT8 activations, FP32 accumulation | Triton W8A8 IMMA, group-32 INT8 weights and dynamic INT8 activations, FP32 accumulation |
| Vision attention (vision checkpoint) | Triton INT8 QK and PV products with FP32 online softmax | Triton INT8 QK and PV products with FP32 online softmax |

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

The vision checkpoint adds
`vllm.models.deepseek_v4.common.vision_int8.VisionInt8LinearMethod` and
`vllm.models.deepseek_v4.common.vision_int8.vision_attention_int8` to the
list; without them the tower runs in floating point.

Unknown symbols and conflicting implementations for one role fail at startup.
`--hf-overrides` can replace the block for controlled comparisons. Weight
formats and linear methods come from `quantization_config`; changing the
attention kernel list does not change stored weights.

Listing `streaming_prefill_topk` enables slabbed prefill selection. A prefill
that fits one slab keeps the one-shot path. A larger prefill scores and reduces
one slab at a time.

## Benchmarks

Per stream is the median request of a batch; aggregate is the whole batch
over its wall time. Verbatim: each request hides a unique 1,000-token
passage in its prompt and asks for it back; the column counts how many of
the C simultaneous requests returned it exactly. Tables come from
`tools/ampere/dsv4_needle_matrix_report.py --markdown` over a run's
`needle-bench` directory.

### `appmana/deepseek-v4-int4-int8`, 11 RTX 3090s

Eleven nodes, each a Ryzen 9 7950X with 64 GB of RAM and one RTX 3090 power
limited to 250 W, linked by Thunderbolt; PP=11, TP=1, NCCL over Thunderbolt
networking, 1,000,000-token window, 2 GiB of KV cache per rank, chunked
prefill of 1,024 tokens, thinking off. Measured 2026-09-04.

Speculative decoding off, `--max-num-seqs 8`:

| Input tokens | C | TTFT s | Prefill tok/s per stream | Prefill tok/s aggregate | Output tok/s per stream | Output tok/s aggregate | Verbatim |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,000 | 1 | 2.7 | 2,985 | 2,985 | 37.6 | 34.1 | 1/1 |
| 8,000 | 2 | 2.3 | 3,602 | 6,206 | 36.7 | 67.6 | 2/2 |
| 8,000 | 4 | 3.1 | 2,601 | 6,954 | 21.8 | 81.4 | 4/4 |
| 8,000 | 8 | 5.2 | 1,545 | 7,765 | 12.5 | 93.8 | 8/8 |
| 16,000 | 1 | 2.9 | 5,587 | 5,587 | 37.4 | 33.8 | 1/1 |
| 16,000 | 2 | 3.6 | 4,593 | 7,260 | 36.1 | 63.5 | 2/2 |
| 16,000 | 4 | 5.5 | 3,012 | 7,834 | 21.7 | 73.4 | 3/4 |
| 16,000 | 8 | 9.2 | 1,765 | 8,203 | 12.0 | 84.4 | 7/8 |
| 64,000 | 1 | 10.5 | 6,113 | 6,113 | 36.9 | 26.6 | 1/1 |
| 64,000 | 2 | 12.3 | 5,714 | 8,004 | 32.7 | 46.0 | 2/2 |
| 64,000 | 4 | 20.0 | 3,317 | 8,061 | 18.3 | 51.6 | 3/4 |
| 64,000 | 8 | 35.9 | 1,805 | 8,023 | 10.8 | 61.4 | 7/8 |

DSpark on, seven draft tokens with probabilistic draft sampling
(`--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'`),
`--max-num-seqs 32`:

| Input tokens | C | TTFT s | Prefill tok/s per stream | Prefill tok/s aggregate | Output tok/s per stream | Output tok/s aggregate | Verbatim |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,000 | 1 | 2.9 | 2,795 | 2,795 | 62.3 | 52.7 | 1/1 |
| 8,000 | 2 | 2.4 | 3,553 | 5,391 | 59.8 | 103.8 | 2/2 |
| 8,000 | 4 | 3.4 | 2,444 | 6,657 | 56.1 | 180.6 | 4/4 |
| 8,000 | 8 | 5.6 | 1,454 | 6,917 | 39.0 | 252.7 | 8/8 |
| 16,000 | 1 | 3.0 | 5,306 | 5,306 | 60.9 | 51.5 | 1/1 |
| 16,000 | 2 | 4.0 | 4,212 | 6,478 | 58.3 | 92.0 | 2/2 |
| 16,000 | 4 | 6.1 | 2,704 | 6,984 | 50.6 | 145.7 | 3/4 |
| 16,000 | 8 | 10.2 | 1,588 | 7,146 | 34.9 | 193.8 | 7/8 |
| 64,000 | 1 | 10.8 | 5,923 | 5,923 | 58.5 | 35.9 | 1/1 |
| 64,000 | 2 | 13.8 | 5,175 | 7,068 | 48.6 | 56.6 | 2/2 |
| 64,000 | 4 | 22.4 | 2,970 | 7,229 | 34.0 | 76.3 | 2/4 |
| 64,000 | 8 | 40.3 | 1,608 | 7,195 | 19.2 | 86.4 | 5/8 |
| 512,000 | 1 | 111.3 | 4,600 | 4,600 | 45.9 | 7.5 | 1/1 |
| 512,000 | 2 | 159.9 | 3,599 | 4,805 | 27.7 | 9.1 | 1/2 |
| 512,000 | 4 | 266.0 | 2,005 | 4,786 | 6.1 | 9.3 | 1/4 |
| 512,000 | 8 | 484.5 | 1,071 | 4,637 | 3.6 | 9.6 | 2/8 |
| 950,000 | 1 | 266.0 | 3,571 | 3,571 | 52.0 | 3.9 | 0/1 |
| 950,000 | 2 | 393.5 | 2,722 | 3,613 | 30.6 | 4.1 | 0/2 |
| 950,000 | 4 | 657.9 | 1,505 | 3,565 | 3.1 | 4.0 | 0/4 |

Tool calling on the same configuration, Berkeley Function Calling
Leaderboard, 2026-09-04: non-live AST 87.3%, live 77.9%, multi-turn 0.0%,
overall 21.1%.

### `appmana/deepseek-v4-flash-vision-exp-int4-int8`, 11 RTX 3090s

Same chain, PP=11 with partition `4,4,4,4,4,4,4,4,5,5,1`, eight images per
request, eight sequences, DSpark with seven probabilistic draft tokens,
1,000,000-token window. A fixed set of 100 images (30 five-character OCR
codes, 30 coloured-square positions, 20 bar charts, 20 photographs) asked
for an exact answer, four requests at a time, 2026-09-14:

| Vision tower | Correct | Median request s | p95 request s |
| --- | ---: | ---: | ---: |
| INT8 IMMA (`65df4c51`) | 100/100 | 1.50 | 3.59 |
| BF16 reference (`bee7f4e9`) | 99/100 | 1.01 | 1.89 |

The IMMA latency in that run was Triton recompiling the vision kernels for
every new image size; the kernels no longer specialise on token count and a
warmup compiles them at startup, so the table is refreshed with the next
chain run. Also validated in that run: eight concurrent requests with eight
distinct images each, all answered correctly; a 990,160-token image prompt
answered correctly and reused 989,952 tokens on repeat through the KV
connector; a different image at identical geometry reused none; at least
1.16 GiB free on every GPU throughout.

Text-only rows on the vision checkpoint follow the same table format and are
recorded after each merge.

## Build

```bash
bash docker/build-consumer-platforms.sh
```

`docker/Dockerfile` builds one image with:

- native vLLM extensions for RTX 30xx and GB10;
- the AppMana NCCL fork;
- the external FlashMLA wheel;
- SparkInfer, including its GB10 INT8 indexer kernels;
- optionally the AppMana LMCache fork (`INSTALL_KV_CONNECTORS=true`,
  `LMCACHE_GIT_REF`), with OpenTelemetry pinned back afterwards.

`docker/versions.json` records the pinned component revisions used by the
build. An editable install for development is
`uv pip install --no-build-isolation --no-deps -e .` with
`TORCH_CUDA_ARCH_LIST` set to the target; FA3 is only built when a Hopper
architecture is listed.

## Deployment examples

The public Kubernetes examples use LeaderWorkerSet, pipeline parallelism,
model runner v2 and asynchronous scheduling:

- [RTX 30xx PP=11](examples/deployment/dsv4-int4-int8-lws-pp11.yaml)
- [RTX 30xx PP=12](examples/deployment/dsv4-int4-int8-lws-pp12.yaml)

They derive rank information from `LWS_GROUP_SIZE`, `LWS_WORKER_INDEX` and
`LWS_LEADER_ADDRESS`. Supply the `dsv4-cache` PVC and `huggingface` Secret.
Optional network settings come from the `dsv4-network` ConfigMap. The vision
checkpoint uses the PP=11 example with `VLLM_PP_LAYER_PARTITION` set to the
partition above and `--limit-mm-per-prompt '{"image":8}'`.

## Related repositories

- [`AppMana/forks-flash-mla-int`](https://github.com/AppMana/forks-flash-mla-int):
  native sparse-MLA kernels for the INT8 cache.
- [`AppMana/forks-sparkinfer`](https://github.com/AppMana/forks-sparkinfer):
  GB10 FP8/NVFP4 kernels, mHC and native INT8 sparse-indexer kernels.
- [`AppMana/forks-nccl-rdma-routing`](https://github.com/AppMana/forks-nccl-rdma-routing):
  NCCL rail and fallback routing.
- [`AppMana/forks-lmcache`](https://github.com/AppMana/forks-lmcache):
  the KV connector with pipeline-local layouts and image-aware cache keys.

---

This Apache-2.0 fork is downstream of
[vLLM](https://github.com/vllm-project/vllm).
