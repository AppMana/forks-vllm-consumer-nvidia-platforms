<!-- markdownlint-disable MD001 MD041 -->

# AppMana vLLM: DeepSeek-V4 on consumer NVIDIA GPUs

This fork serves DeepSeek-V4-Flash and DeepSeek-V4-Flash-Vision-Exp on RTX
30xx and DGX Spark GB10 GPUs from one image. Upstream vLLM is merged
regularly (last: upstream `0c81a0371f`, merged 2026-10-28); `MERGING.md`
records the procedure and the invariants each merge has to preserve.

## Supported checkpoints

| Checkpoint | Revision | Source | Quantization | Serving |
| --- | --- | --- | --- | --- |
| [`unsloth/DeepSeek-V4.1-Flash-1017-GGUF`](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) (`UD-Q2_K_XL`) | `60ad3651641b5c97679c807e064ab1d7a3ca1455` | `deepseek-ai/DeepSeek-V4.1-Flash-1017` | `quant_method = dsv41_gguf`: mixed `IQ2_XXS` / `IQ3_XXS` / `MXFP4` (`Q4_0`) routed experts (`b12x` W4A16 decode on SM121, `consumer_cuda` on SM86, optional borrowed-weight native `dp4a` INT8 prefill), `Q8_0` / `Q6_K` dense, shared-expert and attention linears with live-row transient BF16 expansion and grouped `wo_a` W8A8 (`consumer_int8.gguf`), disk-backed Engram embeddings, packed `int8_ds_mla` cache, INT8 sparse indexer | TP=2/EP=2 across two DGX Spark GB10s (`max_num_batched_tokens=2048`, `max_num_seqs=8`, PIECEWISE CUDA graphs `[1,2,4,8]`); PP=10 (`4,4,4,4,4,4,4,4,4,4`) across ten RTX 3090s with cross-rank `DeepseekV4PPProxyKVSource` |
| [`appmana/deepseek-v4-int4-int8`](https://huggingface.co/appmana/deepseek-v4-int4-int8) | `3b25318eca9ccebc48130251d2c8b825341ad0ea` | `deepseek-ai/DeepSeek-V4-Flash-0731` at `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` | INT4 group-32 routed experts (MSE scales), channel-INT8 dense, shared-expert and attention linears, packed `int8_ds_mla` cache, INT8 sparse indexer, three native DSpark draft stages | PP=11 or PP=12 across RTX 3090s; TP=2 across two GB10s |
| [`appmana/deepseek-v4-flash-vision-exp-int4-int8`](https://huggingface.co/appmana/deepseek-v4-flash-vision-exp-int4-int8) | `65df4c51a2e724340c7963ccddd1f5199a2575e4` (BF16 vision tower on branch `bf16-vision-reference`, `bee7f4e9f445e49067b35d88e4ef3ef3ca9f1a56`) | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` at `6821d6ad3681a4b137b066b76094fa82ebd0a380` | The same language quantization plus a group-32 W8A8 INT8 vision tower and aligner run on integer tensor cores | PP=11 across RTX 3090s with layer partition `4,4,4,4,4,4,4,4,5,5,1`, eight images per request, eight sequences, DSpark with seven probabilistic draft tokens |

The `dsv4_int` and `dsv41_gguf` configurations declare a `vllm` block naming
the kernels and `int8_ds_mla` cache they require; stock vLLM cannot load them.
The separate `appmana/deepseek-v4-nvfp4-fp8` checkpoint uses a different
quantization and kernel configuration.

## Qwen3.8-27B INT8

The `qwen_int8` adapter loads text, vision and MTP weights encoded by the
independent [`consumer-int8-kernels`](https://github.com/AppMana/consumer-int8-kernels)
(`v0.1.0`) package, which also provides AOT SM86/SM121 CUDA kernels and benchmark harnesses.
Language/MTP linears use biased UINT8 weights with BF16 channel scales;
vision linears use signed INT8 group-32 weights with FP32 scales. Recurrent
state stays FP32, and Qwen's hybrid architecture has no sparse indexer.

The opt-in `consumer_int8_prefill` backend uses fused integer GEMMs for at
least 1,024 rows and AllSpark W8A16 for smaller requests. Runtime dispatch
inside a custom operator preserves this policy during AOT graph reuse.
Native CUDA implementations handle small-batch GDN decode and INT8 cache
insertion. The `consumer_fa2` attention backend converts active INT8 cache
rows to temporary BF16 pages for FlashAttention 2, with direct packed CUDA
attention above its 512-MiB scratch cap. Short-query decode and MTP
verification (`max_seqlen_q <= 8`) use `consumer_int8.ops.paged_decode` over
the packed INT8 cache with high-plus-residual BF16 PV accumulation and
two-CTA-per-SM occupancy (`target = 2 * SMs`, `2.58x–36.35x` faster than
Triton `unified_attention` across `q=1..8` and `Hkv=2,4` on RTX A5000). GDN
prefill retains FLA. INT8 storage does not imply integer attention arithmetic.

The package's `benchmarks/run_quality.py` checks fixed text and vision
fixtures. `benchmarks/run_matrix.py` measures separate aggregate and
per-stream prefill/decode rates for TP2/PP2 long-context retrieval.

## What this fork adds

- `dsv4_int` loading and dispatch for INT4 routed experts and INT8 linears.
- Packed INT8 attention and sparse-indexer cache paths.
- FlashMLA INT8 sparse attention for the checkpoint's decode and prefill
  selectors.
- Integer-tensor-core (IMMA) sparse-indexer scoring over the INT8 indexer
  cache, in Triton, on both RTX 30xx and GB10.
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
selects the platform implementation where one differs, such as the dense
linear path.

## INT4/INT8 kernel map

Sparse attention and sparse-indexer scoring are separate stages. FlashMLA
consumes the selected cache rows to compute attention. The indexer scores the
cache first and selects those rows.

| Function | RTX 30xx | DGX Spark GB10 |
| --- | --- | --- |
| Routed-expert MoE | Marlin W4A8-INT8 with INT4 weights | Marlin W4A8-INT8 with INT4 weights |
| Dense, shared-expert and attention linears | AllSpark W8A16 for supported channel-INT8 shapes; BF16 dequantization fallback | AllSpark W8A16 for supported channel-INT8 shapes (`VLLM_DSV4_ALLSPARK_SM12X=0` turns it off for diagnosis); BF16 dequantization fallback |
| `wo_a` projection | BF16 weight or one-time INT8-to-BF16 dequantization for its inverse-RoPE einsum | BF16 weight or one-time INT8-to-BF16 dequantization for its inverse-RoPE einsum |
| Sparse-MLA attention decode | `flash_mla.sparse_mla_decode_int8` | `flash_mla.sparse_mla_decode_int8` |
| Sparse-MLA attention prefill | `flash_mla.sparse_mla_prefill_int8` | `flash_mla.sparse_mla_prefill_int8` |
| Indexer K cache write | vLLM INT8 quantize-and-cache kernel | vLLM INT8 quantize-and-cache kernel |
| Indexer Q RoPE and quantization | vLLM fused INT8 kernel | vLLM fused INT8 kernel |
| Indexer decode scoring over paged cache | vLLM Triton IMMA (`fp8_paged_mqa_logits_triton`, INT8 query and cache) | vLLM Triton IMMA (`fp8_paged_mqa_logits_triton`, INT8 query and cache) |
| Indexer prefill scoring over contiguous cache | vLLM Triton IMMA (`mqa_logits_workspace_triton`, `qk_int8`) | vLLM Triton IMMA (`mqa_logits_workspace_triton`, `qk_int8`) |
| Indexer long-prefill scoring | Triton IMMA per slab plus native CUDA candidate selection and merge | Triton IMMA per slab plus native CUDA candidate selection and merge |
| Indexer top-k | native CUDA row and persistent selectors | native CUDA row and persistent selectors |
| Attention KV cache | packed `int8_ds_mla`, 528 bytes per token | packed `int8_ds_mla`, 528 bytes per token |
| DSpark speculative decoding | three MTP draft stages in model runner v2 | three MTP draft stages in model runner v2 |
| Vision patch projection, QKV, output, MLP and aligner linears (vision checkpoint) | Triton W8A8 IMMA, group-32 INT8 weights and dynamic INT8 activations, FP32 accumulation | Triton W8A8 IMMA, group-32 INT8 weights and dynamic INT8 activations, FP32 accumulation |
| Vision attention (vision checkpoint) | Triton INT8 QK and PV products with FP32 online softmax | Triton INT8 QK and PV products with FP32 online softmax |

The FlashMLA INT8 kernels are provided by
[`AppMana/forks-flash-mla-int`](https://github.com/AppMana/forks-flash-mla-int).
Indexer scoring is the same Triton integer-MMA code on both platforms
(`vllm/models/deepseek_v4/nvidia_imma/triton_kernels.py`). SparkInfer's
native INT8 indexer kernels are not used: its contiguous kernel faulted
(Xid 13) on long prefills and was slower at serving shapes (`e73a131c55`),
and its paged wrapper allocates per layer instead of taking caller-owned
buffers (`bf753bb3e9`). `int8_mqa_logits_sparkinfer` in
`vllm/utils/deep_gemm.py` is reached only by a test, and the routing
experiment is kept under the tag
`archive/appmana-wip/int8-sparkinfer-indexer-sm121`. The streaming path calls
the same contiguous scoring implementation once per slab and merges
candidates without materializing full-context logits.

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
the `C` simultaneous requests returned it exactly. Needle-in-haystack matrices
report `C = 1, 2, 4, 8, 32` across `8,000 / 1,000`, `242,000 / 1,000` and
`950,000 / 1,000` input/output tokens (`OOM` where bounded by device/unified
memory reservation; `N/A` where exceeding the model's `max_position_embeddings`).

### `unsloth/DeepSeek-V4.1-Flash-1017-GGUF` (`UD-Q2_K_XL`), 2 DGX Spark GB10s (`TP=2/EP=2`)

Two DGX Spark GB10 nodes (`spark-2ab3` and `spark-5867`) linked by dual-rail
200 Gb/s RoCE; `TP=2, EP=2, PP=1`, `quant_method=dsv41_gguf` (`UD-Q2_K_XL` at
revision `60ad3651641b5c97679c807e064ab1d7a3ca1455`), `int8_ds_mla` cache
(`1 GiB` per rank, `95.11 GiB` model weights per rank inside the `121 GiB`
unified memory budget), disk-backed Engram embeddings (`io_uring`),
`max_num_batched_tokens=2048`, `max_num_seqs=8`, PIECEWISE CUDA graphs at
`[1,2,4,8]`, grouped W8A8 `attn.wo_a` and live-row transient BF16 expansion
(`ghcr.io/appmana/vllm-consumer:sm86-sm121-efd38faff6`). Measured 2026-10-29
with `tools/ampere/dsv4_needle_bench.py`:

| Input / Output tokens | C | TTFT s | Prefill tok/s per stream | Prefill tok/s aggregate | Output tok/s per stream | Output tok/s aggregate | Verbatim |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,000 / 1,000 | 1 | 4.0 | 1,988 | 1,988 | 19.3 | 17.9 | 1/1 |
| 8,000 / 1,000 | 2 | 6.2 | 1,429 | 1,943 | 16.3 | 29.6 | 2/2 |
| 8,000 / 1,000 | 4 | 10.7 | 783 | 1,918 | 13.2 | 46.1 | 3/4 |
| 8,000 / 1,000 | 8 | 19.2 | 422 | 1,907 | 10.2 | 67.8 | 8/8 |
| 8,000 / 1,000 | 32 | OOM | OOM | OOM | OOM | OOM | OOM |
| 242,000 / 1,000 | 1, 2, 4, 8, 32 | OOM | OOM | OOM | OOM | OOM | OOM |
| 950,000 / 1,000 | 1, 2, 4, 8, 32 | OOM | OOM | OOM | OOM | OOM | OOM |

GSM8K 128-question diagnostic subset (5-shot, `C=8`, `max_tokens=512`,
`temperature=0`, `thinking=false`): **126/128 (98.44%)**, `0` invalid,
**67.91 tok/s** aggregate output (`19,138` output tokens in `281.81 s`).

On 10 RTX 3090s (`SM86`, 24 GB/GPU, `PP=10`, partition `4,4,4,4,4,4,4,4,4,4`),
each rank holds `15.32–20.46 GiB` of weights (`expert_backend=consumer_cuda`,
`q8_backend=consumer_cuda`, `q6_backend=consumer_cuda`) and forwards cross-rank
KV-sharing groups (`[2, 8, 14, 20]`) through `DeepseekV4PPProxyKVSource` /
`DeepseekV4PPProxyIndexer` (`JobSet` `dsv41-unsloth-pp10-needle` on Kueue queue
`tb-chain`).

### `Qwen3.8-27B` INT8 (`qwen_int8`), 2 RTX A5000s (`SM86`)

Two RTX A5000 GPUs (24 GB/GPU, PCIe, `NCCL_P2P_DISABLE=1`), `qwen_int8`
checkpoint (`consumer_int8_prefill`, `consumer_fa2`, `paged_decode` 2-CTA/SM
split-KV, `int8` KV cache, `2.0 GiB` KV reservation on `MTP=0` and `3.35 GiB`
on `MTP=7`). Measured with `benchmarks/needle_matrix.py` (1 warmup + 3 measured
cohorts per cell, cold prefix cache; `242,000 / 1,000` and `C=32` exceed the
`2.98 GiB` post-weight per-GPU VRAM ceiling on 2x 24 GB; `950,000 / 1,000`
exceeds `max_position_embeddings = 262,144`):

| Topology | Input / Output tokens | C | Prefill tok/s per stream | Prefill tok/s aggregate | Decode tok/s per stream | Decode tok/s aggregate | Verbatim |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TP=2` (`MTP=0`) | 8,000 / 1,000 | 1 | 4,280 | 4,280 | 36.7 | 36.7 | 3/3 |
| `TP=2` (`MTP=0`) | 8,000 / 1,000 | 2 | 3,496 | 4,181 | 32.8 | 63.5 | 5/6 |
| `TP=2` (`MTP=0`) | 8,000 / 1,000 | 4 | 2,078 | 4,145 | 29.4 | 108.0 | 12/12 |
| `TP=2` (`MTP=0`) | 8,000 / 1,000 | 8 | 1,827 | 3,722 | 22.7 | 153.8 | 24/24 |
| `TP=2` (`MTP=0`) | 8,000 / 1,000 | 32 | OOM | OOM | OOM | OOM | OOM |
| `TP=2` (`MTP=0`) | 242,000 / 1,000 | 1, 2, 4, 8, 32 | OOM | OOM | OOM | OOM | OOM |
| `TP=2` (`MTP=0`) | 950,000 / 1,000 | 1, 2, 4, 8, 32 | N/A | N/A | N/A | N/A | N/A |
| `TP=2` (`MTP=7`) | 8,000 / 1,000 | 1 | 4,114 | 4,114 | 142.1 | 142.1 | 3/3 |
| `TP=2` (`MTP=7`) | 8,000 / 1,000 | 2 | 3,034 | 4,048 | 122.7 | 215.7 | 5/6 |
| `TP=2` (`MTP=7`) | 8,000 / 1,000 | 4 | 2,003 | 4,000 | 89.3 | 272.6 | 12/12 |
| `TP=2` (`MTP=7`) | 8,000 / 1,000 | 8, 32 | OOM | OOM | OOM | OOM | OOM |
| `TP=2` (`MTP=7`) | 242,000 / 1,000 | 1, 2, 4, 8, 32 | OOM | OOM | OOM | OOM | OOM |
| `TP=2` (`MTP=7`) | 950,000 / 1,000 | 1, 2, 4, 8, 32 | N/A | N/A | N/A | N/A | N/A |
| `PP=2` (`MTP=0`) | 8,000 / 1,000 | 1 | 3,310 | 3,310 | 20.0 | 20.0 | 3/3 |
| `PP=2` (`MTP=0`) | 8,000 / 1,000 | 2 | 2,836 | 3,260 | 17.9 | 35.0 | 5/6 |
| `PP=2` (`MTP=0`) | 8,000 / 1,000 | 4 | 1,895 | 4,251 | 17.7 | 67.7 | 12/12 |
| `PP=2` (`MTP=0`) | 8,000 / 1,000 | 8 | 1,675 | 3,851 | 14.5 | 105.1 | 24/24 |
| `PP=2` (`MTP=0`) | 8,000 / 1,000 | 32 | OOM | OOM | OOM | OOM | OOM |
| `PP=2` (`MTP=0`) | 242,000 / 1,000 | 1, 2, 4, 8, 32 | OOM | OOM | OOM | OOM | OOM |
| `PP=2` (`MTP=0`) | 950,000 / 1,000 | 1, 2, 4, 8, 32 | N/A | N/A | N/A | N/A | N/A |

### `appmana/deepseek-v4-int4-int8`, 11 RTX 3090s (`PP=11`)

Eleven nodes, each a Ryzen 9 7950X with 64 GB of RAM and one RTX 3090 power
limited to 250 W, linked by Thunderbolt; PP=11, TP=1, NCCL over Thunderbolt
networking, 1,000,000-token window, 2 GiB of KV cache per rank, chunked
prefill of 1,024 tokens, thinking off. Measured 2026-09-04.

Speculative decoding off, `--max-num-seqs 8`:

| Input / Output tokens | C | TTFT s | Prefill tok/s per stream | Prefill tok/s aggregate | Output tok/s per stream | Output tok/s aggregate | Verbatim |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,000 / 1,000 | 1 | 2.7 | 2,985 | 2,985 | 37.6 | 34.1 | 1/1 |
| 8,000 / 1,000 | 2 | 2.3 | 3,602 | 6,206 | 36.7 | 67.6 | 2/2 |
| 8,000 / 1,000 | 4 | 3.1 | 2,601 | 6,954 | 21.8 | 81.4 | 4/4 |
| 8,000 / 1,000 | 8 | 5.2 | 1,545 | 7,765 | 12.5 | 93.8 | 8/8 |

DSpark on, seven draft tokens with probabilistic draft sampling
(`--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'`),
`--max-num-seqs 32`:

| Input / Output tokens | C | TTFT s | Prefill tok/s per stream | Prefill tok/s aggregate | Output tok/s per stream | Output tok/s aggregate | Verbatim |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,000 / 1,000 | 1 | 2.9 | 2,795 | 2,795 | 62.3 | 52.7 | 1/1 |
| 8,000 / 1,000 | 2 | 2.4 | 3,553 | 5,391 | 59.8 | 103.8 | 2/2 |
| 8,000 / 1,000 | 4 | 3.4 | 2,444 | 6,657 | 56.1 | 180.6 | 4/4 |
| 8,000 / 1,000 | 8 | 5.6 | 1,454 | 6,917 | 39.0 | 252.7 | 8/8 |
| 950,000 / 1,000 | 1 | 266.0 | 3,571 | 3,571 | 52.0 | 3.9 | 0/1 |
| 950,000 / 1,000 | 2 | 393.5 | 2,722 | 3,613 | 30.6 | 4.1 | 0/2 |
| 950,000 / 1,000 | 4 | 657.9 | 1,505 | 3,565 | 3.1 | 4.0 | 0/4 |
| 950,000 / 1,000 | 8, 32 | OOM | OOM | OOM | OOM | OOM | OOM |

Tool calling on the same configuration, Berkeley Function Calling
Leaderboard, 2026-09-04: non-live AST 87.3%, live 77.9%, multi-turn 0.0%,
overall 21.1%.

### `appmana/deepseek-v4-flash-vision-exp-int4-int8`, 11 RTX 3090s (`PP=11`)

Same chain, PP=11 with partition `4,4,4,4,4,4,4,4,5,5,1`, eight images per
request, eight sequences, DSpark with seven probabilistic draft tokens,
1,000,000-token window, FULL_DECODE_ONLY graphs at capture sizes
`[8,16,32,64]` tokens. A fixed set of 100 images (30 five-character OCR
codes, 30 coloured-square positions, 20 bar charts, 20 photographs) asked
for an exact answer, four requests at a time:

| Vision tower | Runtime | Date | Correct | Median request s | p95 request s |
| --- | --- | --- | ---: | ---: | ---: |
| INT8 IMMA (`65df4c51`) | `1d8b0eae46` | 2026-09-19 | 100/100 | 0.82 | 1.31 |
| BF16 reference (`bee7f4e9`) | `b281385271` | 2026-09-14 | 99/100 | 1.01 | 1.89 |

Cold text rows on the vision checkpoint, 8,000-token prompts, thinking off,
DSpark on, 2026-09-19:

| Input / Output tokens | C | TTFT s | Prefill tok/s per stream | Prefill tok/s aggregate | Output tok/s per stream | Output tok/s aggregate | Verbatim |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,000 / 1,000 | 1 | 1.9 | 4,155 | 4,155 | 55.0 | 49.8 | 1/1 |
| 8,000 / 1,000 | 2 | 2.7 | 3,361 | 4,508 | 53.6 | 92.9 | 2/2 |
| 8,000 / 1,000 | 4 | 3.7 | 2,139 | 5,710 | 47.8 | 161.2 | 4/4 |
| 8,000 / 1,000 | 8 | 5.6 | 1,444 | 6,860 | 33.0 | 218.4 | 8/8 |

## Build

```bash
bash docker/build-consumer-platforms.sh
```

The script keeps three caches on by default and a build is only slow when
one of them is missing: buildkit's layer cache on one fixed builder pod,
the registry layer cache at `ghcr.io/appmana/vllm-consumer:buildcache`,
and sccache for the native extensions. `USE_SCCACHE` does not change the
base layers, so toggling it is cheap; the Rust and CUDA stages run
separate sccache daemons.

The script builds one platform per run. Released images are tagged
`sm86-sm121-<commit10>-amd64` and `-arm64` and joined into
`sm86-sm121-<commit10>` with `docker/merge-consumer-platforms.sh`; the
release gate in `MERGING.md` gives the steps.

`docker/Dockerfile` builds one image with:

- native vLLM extensions for RTX 30xx and GB10;
- [`consumer-int8-kernels`](https://github.com/AppMana/consumer-int8-kernels) `v0.1.0` (`consumer-int8-kernels==0.1.0`);
- the AppMana NCCL fork (`nccl-usb4==2.32.3`, tag `v2.32.3`);
- the external FlashMLA wheel (`flash-mla==2.0.1.dev1`, tag `v2.0.1.dev1`);
- SparkInfer (`sparkinfer==1.0.2.dev1`, tag `v1.0.2.dev1`), for the FP8/NVFP4
  checkpoint's GB10 attention and MoE kernels and the selectable SparkInfer mHC,
  plus the isolated `b12x==1.3.0` (`v1.3.0` / `b12x-v1.3.0`) runtime for native
  GGUF mixed-IQ execution;
- the KV connectors: the AppMana LMCache fork (tag `appmana-v0.5.7.dev1`), NIXL and
  Mooncake, with OpenTelemetry pinned back afterwards. The Dockerfile default
  is `INSTALL_KV_CONNECTORS=false`; both build scripts pass `true`.

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

## Fork documentation

- [`MERGING.md`](MERGING.md): upstream merge rules, procedure, per-area
  policy, gates and the image release gate.
- [`docs/appmana/new-models.md`](docs/appmana/new-models.md): bringing a new
  model or checkpoint variant onto RTX 30xx and GB10: conversion, the `vllm`
  kernel block, quantization method, pipeline planning and gates.
- [`docs/appmana/triton-to-cuda.md`](docs/appmana/triton-to-cuda.md):
  translating Triton kernels to native CUDA, integer tensor-core rules, and
  compiling vLLM, FlashMLA, SparkInfer and the image.
- [`tools/ampere/DSV4_VISION.md`](tools/ampere/DSV4_VISION.md): converting and
  serving the vision checkpoint.

## Related repositories

- [`AppMana/forks-flash-mla-int`](https://github.com/AppMana/forks-flash-mla-int):
  native sparse-MLA kernels for the INT8 cache.
- [`AppMana/forks-sparkinfer`](https://github.com/AppMana/forks-sparkinfer):
  GB10 FP8/NVFP4 attention and MoE kernels and mHC.
- [`AppMana/forks-nccl-rdma-routing`](https://github.com/AppMana/forks-nccl-rdma-routing):
  NCCL rail and fallback routing.
- [`AppMana/forks-lmcache`](https://github.com/AppMana/forks-lmcache):
  the KV connector with pipeline-local layouts and image-aware cache keys.

---

This Apache-2.0 fork is downstream of
[vLLM](https://github.com/vllm-project/vllm).
