<!-- markdownlint-disable MD001 MD041 -->

# AppMana vLLM: DeepSeek-V4 on consumer NVIDIA platforms

A fork of [vLLM](https://github.com/vllm-project/vllm) that runs
**DeepSeek-V4-Flash** — a model whose official kernel support starts at
Hopper — on GPUs upstream does not support. One image serves two
architectures: **Ampere (sm_86)** via ported sparse-MLA kernels and an
INT4/INT8 checkpoint, and **GB10 / consumer Blackwell (sm_121)** via
sparkinfer's CuTe-DSL kernels and an NVFP4/FP8 checkpoint.

GPU count, pipeline size, interconnect, and context length are configuration,
not assumptions.

## Architecture support

| Arch | Gencode | Kernel source | Checkpoint | Status |
| --- | --- | --- | --- | --- |
| Ampere sm_86 (RTX 3090 / A5000, 24 GB) | `8.6` | Triton + fused native CUDA (`flash_mla`) | `appmana/deepseek-v4-int4-int8` | Validated, benchmarked |
| GB10 sm_121 (DGX Spark) | `12.1a` | sparkinfer (CuTe-DSL) + `flash_mla` | `appmana/deepseek-v4-nvfp4-fp8` | Bring-up; output not yet correct |

Both gencodes are built into one image (`docker/Dockerfile`,
`torch_cuda_arch_list='8.6 12.1a'`). Kernel selection is per-checkpoint, not
per-build — see [the `vllm` config block](#configuration-the-vllm-checkpoint-config-block).

## Checkpoints

### [`appmana/deepseek-v4-int4-int8`](https://huggingface.co/appmana/deepseek-v4-int4-int8)

INT4 W4A16 Marlin routed experts, INT8 W8A16 AllSpark dense/shared/attention
linears, `int8_ds_mla` KV cache and indexer, with three DSpark MTP draft
stages grafted in. Serves a 1,000,000-token context on twelve 24 GB Ampere
GPUs. **Validated and benchmarked.**

### [`appmana/deepseek-v4-nvfp4-fp8`](https://huggingface.co/appmana/deepseek-v4-nvfp4-fp8)

NVFP4 routed experts with FP8 dense layers and an `fp8_ds_mla` KV cache,
targeting GB10. Source is `nvidia/DeepSeek-V4-Flash-NVFP4` with the DSpark
MTP stages spliced in; the MTP blocks remain MXFP4 and are excluded from the
`quantization_config.quantized_layers` table, so they load under the base
expert format rather than NVFP4.

**Status: bring-up. This checkpoint does not yet produce correct output on
sm_121.** It serves, resolves the sparkinfer kernels, and passes the kernel
parity suite, but generation is incorrect. It has no model card and no
published benchmarks. Do not use it for evaluation.

## Sibling repositories

- [`forks-flash-mla-ampere-dsv4`](https://github.com/AppMana/forks-flash-mla-ampere-dsv4)
  — fused native CUDA sparse-MLA kernels (`sparse_mla_decode_fp8`,
  `sparse_mla_decode_int8`, `sparse_mla_prefill`), built as the `flash_mla`
  wheel for sm_86 and sm_121 on both x86_64 and aarch64. Kernel
  microbenchmarks live in its README.
- [`forks-sparkinfer`](https://github.com/AppMana/forks-sparkinfer) — sm_120/121
  CuTe-DSL kernels (compressed MLA, NSA indexer, fused NVFP4 MoE),
  downstream of [`local-inference-lab/sparkinfer`](https://github.com/local-inference-lab/sparkinfer).
  Installed with `--no-build-isolation` so its PCIe comm extensions are
  compiled ahead of time rather than JIT-compiled inside the first request.
- [`forks-nccl-rdma-routing`](https://github.com/AppMana/forks-nccl-rdma-routing)
  — NCCL with rail/fallback selection.
- [`forks-kueue-chain-adjacency`](https://github.com/AppMana/forks-kueue-chain-adjacency)
  — allocates a pipeline as one contiguous chain run.

## What this fork adds

Upstream's DeepSeek-V4 sparse-MLA path assumes Hopper-or-newer kernels
(DeepGEMM, CuTeDSL, sm90 FlashMLA) plus Triton `fp8e4nv`, none of which run on
sm_86.

- `int8_ds_mla`: an INT8 KV-cache/indexer/dense-activation path (packed
  528-byte token slots: 512 int8 + fp32 scale + pad).
- `dsv4_int` quantization: INT4 Marlin routed experts, INT8 AllSpark dense
  layers, produced by `tools/ampere/dsv4_requant_checkpoint.py`.
- Checkpoint-driven kernel selection (the `vllm` key in `config.json`) rather
  than environment variables, resolved fail-closed at startup.
- Per-module quantization resolution from `quantization_config.quantized_layers`,
  so a checkpoint whose MTP blocks differ in format from its backbone loads
  each under the format it declares.
- A pipeline-parallel fix for the 4-tensor head-compression stream, which
  upstream truncated to one tensor and ran redundantly on every rank
  (PP=N token-identical to PP=1).
- Streaming (activation-chunked) prefill top-k for the sparse-MLA indexer,
  bounding memory instead of letting it scale with context. Required above
  roughly 700k context.
- A zero-target-layer last PP rank (draft stages only), freeing the memory a
  1M-token KV cache needs.
- `VLLM_RAY_WORKER_IP_ORDER`, binding PP ranks to a chain's physical index
  order.
- Warmup coverage for sparse-prefill, MHC, and fused-decode kernels, so no
  rank JIT-compiles mid-serving.
- A fail-fast executor and an HTTP health surface that reports engine boot
  stage (`{"stage": ..., "elapsed_s": ...}` while loading, 200 when ready).

## Build

```bash
bash docker/build-consumer-platforms.sh --push
```

One Dockerfile (`docker/Dockerfile`) builds the deployed image: both gencodes,
the AppMana NCCL fork, the `flash_mla` wheel, and sparkinfer. There are no
overlay Dockerfiles — an overlay image reports the base image's version
string, which makes any benchmark attributed to it unattributable.

The build asserts, and fails if any is missing: `flash_mla` importable with
each expected kernel; `libnccl.so.2.30.7` containing the rail-routing strings;
`vllm/_C_stable_libtorch*.so` and `_moe_C_stable_libtorch*.so`; the sm12x
kernel FQNs resolvable; and sparkinfer's PCIe extensions compiled.

`docker/versions.json` is generated from the Dockerfile's ARGs
(`tools/generate_versions_json.py`) and is what `docker-bake.hcl` feeds back
as `TORCH_CUDA_ARCH_LIST`. Regenerate it when the arch list changes, or a bake
build produces an image missing a gencode.

## Deploy

### Ampere, PP=12

`appmana-cluster/.../inference/lws-vllm-deepseek-v4.yaml` — a LeaderWorkerSet
translated from the benchmark JobSet in `tools/ampere/benchmark_jobset/`: same
boot script, same `prep_pp_shards.py` rank-local shard staging, same
environment. The cluster's `tb-chain-webhook` injects `NCCL_SOCKET_IFNAME` and
`VLLM_RAY_WORKER_IP_ORDER` from the admitted chain placement.

### GB10, TP=2

`demos-hilton/cluster/gitops/apps/base/inference/deepseek-v4-nvfp4.yaml` — a
two-rank LeaderWorkerSet, one GB10 per node, RoCE between them:

```yaml
- --tensor-parallel-size
- "2"
- --distributed-executor-backend
- mp
- --nnodes
- "2"
- --node-rank
- "0"                     # "1" on the worker
- --master-addr
- "10.255.0.1"
- --tokenizer-mode
- deepseek_v4
- --max-model-len
- "8192"
- --gpu-memory-utilization
- "0.68"
- --kv-cache-memory-bytes
- "8589934592"
- --speculative-config
- '{"method": "dspark", "num_speculative_tokens": 5, "draft_sample_method": "greedy"}'
```

Environment that matters on this topology:

```yaml
- name: NCCL_IB_HCA
  value: "rocep1s0f1,roceP2p1s0f1"
- name: NCCL_IB_ROCE_VERSION_NUM      # select the GID by version, never by index
  value: "2"
- name: NCCL_SOCKET_IFNAME
  value: enp1s0f1np1
- name: VLLM_HOST_IP
  value: "10.255.0.1"                 # "10.255.0.2" on the worker
```

`NCCL_IB_ROCE_VERSION_NUM=2` rather than a fixed `NCCL_IB_GID_INDEX`: GID
tables are not necessarily symmetric across nodes, and a hardcoded index that
is valid on one node can resolve to an all-zero GID on another, which surfaces
only as `NCCL error: unhandled system error` after a successful TCP rendezvous.

On a unified-memory part the GPU allocation is host RAM, and
`--kv-cache-memory-bytes` adds to the `--gpu-memory-utilization` budget rather
than coming out of it. Both together must leave headroom for the host, or
startup degrades into swapping rather than failing.

## Running and verifying

`cluster/harness/serve_dsv4.sh <arm> <rank>` in the `demos-hilton` repo
launches a two-node serve directly (no container), for bring-up and
benchmarking. `EAGER=1` (default) passes `--enforce-eager`, skipping cudagraph
capture — capture is a throughput optimisation and does not belong in a
correctness loop. `SPEC=0` drops speculative decoding. `JIT_MONITOR=error`
turns any post-warmup Triton JIT into a hard failure, which is the direct test
that warmup key sets match what the live path dispatches.

Verify in this order:

1. **Kernel resolution.** The log prints `vllm kernels resolved: ...` once.
   Treat that line as the validity gate for any measurement: it names the
   selected decode/prefill kernels and the cache type.
2. **Health.** `curl -s localhost:8000/health` returns
   `{"stage": ..., "elapsed_s": ...}` while loading and HTTP 200 when ready.
   `/metrics` is served throughout.
3. **Generation — use the chat template.** Raw `/v1/completions` prompts
   against these checkpoints degrade. `/v1/chat/completions` with
   `--tokenizer-mode deepseek_v4` is the validated path, and is the only one
   whose output is meaningful for evaluation.
4. **Collectives, before trusting any number.** A two-node all-reduce over the
   real transport; confirm the NCCL banner reports the expected version and
   that both rails appear in `NET/IB : Using [0]... [1]...`.
5. **`dmesg` clean** of `Xid` and `NV_ERR_NO_MEMORY` across load, prefill and
   sustained decode.

## Running the benchmarks

Three harnesses, for three different questions. All of them need a server that
already answered `/health` with 200 and printed its `vllm kernels resolved:`
line — a number taken before that line is not attributable to a kernel set.

### Needle-in-a-haystack (correctness *and* throughput)

[`tools/ampere/dsv4_needle_bench.py`](tools/ampere/dsv4_needle_bench.py) is the
one to reach for by default: it measures throughput and verifies the output at
the same time, so a fast run that has stopped being correct cannot be reported
as a win.

Each request embeds a unique passage at a request-specific depth inside a
unique haystack, sized with the real tokenizer so prompt lengths hit the target,
and asks the model to reproduce it verbatim — which makes the expected output
length known and correctness an exact containment check. Every haystack starts
with a request-unique salt, so prefix caching cannot serve one request's prefill
from another's.

```bash
python tools/ampere/dsv4_needle_bench.py \
  --base-url http://localhost:8000 \
  --model appmana/deepseek-v4-int4-int8 \
  --input-tokens 8000 --needle-tokens 1000 \
  --concurrency 8 \
  --output-json needle-8k-1k.json
```

Set `--concurrency` no higher than the server's `--max-num-seqs`: a sweep above
it is silently capped by the scheduler and every arm above the cap looks
identical. `--tokenizer` defaults to `--model`.

### Parallelism arms

`cluster/harness/serve_dsv4.sh <arm> <rank>` in the `demos-hilton` repo serves
the same checkpoint under arguments that differ in exactly one dimension — `A`
TP=2 dual-rail, `B` TP=2 single rail, `C` PP=2, `D` TP=2 + expert parallel — and
`cluster/harness/llm_bench.py` (`--base-url`, `--model`, `--arm`, `--out`)
records them to one CSV. `MODEL=` and `REVISION=` select the checkpoint, so both
checkpoints run through the same launcher rather than a forked copy.

Run rank 1 first; it is `--headless` and waits for the leader.

```bash
MODEL=appmana/deepseek-v4-int4-int8 REVISION= ./serve_dsv4.sh A 1   # worker
MODEL=appmana/deepseek-v4-int4-int8 REVISION= ./serve_dsv4.sh A 0   # leader
```

`EAGER=1` (the default) skips cudagraph capture — capture buys throughput, not
correctness, and belongs in the benchmark arms rather than in a debugging loop.
Set `EAGER=0` when the captured graphs are the point. `SPEC=0` drops speculative
decoding. `JIT_MONITOR=error` turns any post-warmup Triton JIT into a hard
failure, which is the direct test that the warmup key set matches what the live
path dispatches.

### Standard vLLM serving benchmark

`vllm bench serve` for TTFT/TPOT distributions and the published decode/prefill
numbers below. Use `--async-scheduling` on the server; it is worth ~10 tok/s at
1M context and the published figures assume it.

## Serving benchmarks

`vllm bench serve`, single stream (C=1), PP=12 chain over Thunderbolt RDMA,
`appmana/deepseek-v4-int4-int8`, `--async-scheduling`:

| Context | Speculative decoding | Decode (tok/s) | Prefill (tok/s) | Notes |
| ---: | --- | ---: | ---: | --- |
| 4k in / 1k out | off | 33.3 | — | TPOT 28.05 ms |
| 4k in / 1k out | DSpark | 56.1 | — | TPOT 15.90 ms |
| 620k in | off | — | ~3,230 | TTFT 191.9 s |
| 620k in | DSpark | — | ~3,390 | TTFT 182.8 s |
| 950k needle (1M window) | DSpark | 66.7 / 65.3 warm | ~3,020 | needle retrieved verbatim in 314.5 s |

The identical 1M configuration measures 56.8/57.5 tok/s decode with
`--no-async-scheduling`. Needle retrieval is verbatim at 620k, 950k, and the
1,048,576 window edge. Decode rates are per-stream output token throughput;
prefill rates are input length over TTFT.

No benchmarks are published for `appmana/deepseek-v4-nvfp4-fp8`; its output is
not yet correct.

## Benchmarking: generating a JobSet

`tools/ampere/benchmark_jobset/` carries the harness:

- `dsv4-benchmark-jobset-proof.yaml` — the JobSet template: boot script (host
  rendezvous, shard staging, staged `/health`, sanity check, `vllm bench serve`
  matrix, Ray log preservation) plus the `prep_pp_shards.py` stager.
- `generate_dsv4_benchmark_jobset.py` — stamps the template into an
  apply-ready JobSet: sets name/image/world size, bakes the chain IP order,
  scrubs webhook-owned NCCL env, gives each run its own script ConfigMap, and
  verifies the image tag exists before a multi-node slice sits in
  ImagePullBackOff.

```bash
python tools/ampere/benchmark_jobset/generate_dsv4_benchmark_jobset.py \
  --template tools/ampere/benchmark_jobset/dsv4-benchmark-jobset-proof.yaml \
  --name dsv4-bench-001 \
  --image harbor.appmana.com/appmana/vllm-ampere:<tag> \
  --world-size 12 \
  --indexed \
  --env APPMANA_DSV4_MAX_MODEL_LEN=1000000 \
  --env APPMANA_DSV4_PP_LAYER_PARTITION=3,4,4,4,4,4,4,4,4,4,4,0 \
  --output /tmp/dsv4-bench-001.yaml
kubectl apply -f /tmp/dsv4-bench-001.yaml
```

`--indexed` is the shape Kueue admits on the chain queue. `--env NAME=VALUE`
overrides any harness knob per run; `--prewarm-only` stages shards and exits.
Results land under `/jit-shared/dsv4-jobset-results/<name>/<uid>/`.

## Configuration: the `vllm` checkpoint-config block

Kernel selection is checkpoint-config driven. The checkpoint's `config.json`
carries one block, overridable wholesale via `--hf-overrides '{"vllm": {...}}'`:

```json
"vllm": {
  "kernels": [
    "flash_mla.sparse_mla_decode_fp8",
    "flash_mla.sparse_mla_decode_int8",
    "flash_mla.sparse_mla_prefill",
    "vllm._custom_ops.indexer_k_quant_and_cache_int8",
    "vllm.models.deepseek_v4.common.ops.fused_indexer_q.fused_indexer_q_rope_quant_int8",
    "vllm.model_executor.layers.quantization.utils.marlin_utils.marlin_act_int8_process_scales"
  ],
  "cache_type": "int8_ds_mla"
}
```

The sm_121 checkpoint names the sparkinfer kernels instead:

```json
"vllm": {
  "kernels": [
    "vllm.models.deepseek_v4.nvidia_sm12x.kernels.sparkinfer_sparse_mla_decode",
    "vllm.models.deepseek_v4.nvidia_sm12x.kernels.sparkinfer_sparse_mla_extend"
  ],
  "cache_type": "fp8_ds_mla"
}
```

Each entry is the importable symbol to activate; unknown symbols, duplicate
roles, and invalid `cache_type` values fail closed at startup, and the resolved
config prints once as `vllm kernels resolved: ...`. `--hf-overrides` replaces
the whole block rather than merging it, so build overrides from the
checkpoint's own block plus whatever you are adding — a dropped kernel entry
degrades silently.

Resolution, the role table, and the legacy-FQN aliasing live in
[`kernel_config.py`](vllm/transformers_utils/configs/dsv4/kernel_config.py).

### Roles and the symbols that fill them

**Selector roles** — exactly one kernel is always active; omitting the role
selects the documented default rather than disabling it.

| Role | Symbol | Implementation |
| --- | --- | --- |
| `sparse_mla_decode_fp8` | `flash_mla.sparse_mla_decode_fp8` | [forks-flash-mla-ampere-dsv4](https://github.com/AppMana/forks-flash-mla-ampere-dsv4) |
| | `vllm.models.deepseek_v4.nvidia_sm12x.kernels.sparkinfer_sparse_mla_decode` | [`nvidia_sm12x/kernels.py`](vllm/models/deepseek_v4/nvidia_sm12x/kernels.py) → [forks-sparkinfer](https://github.com/AppMana/forks-sparkinfer) |
| | `vllm.models.deepseek_v4.nvidia_imma.triton_kernels.decode_sparse_attention_triton` | [`nvidia_imma/triton_kernels.py`](vllm/models/deepseek_v4/nvidia_imma/triton_kernels.py) |
| `sparse_mla_decode_int8` | `flash_mla.sparse_mla_decode_int8` | [forks-flash-mla-ampere-dsv4](https://github.com/AppMana/forks-flash-mla-ampere-dsv4) (fused native CUDA) |
| | `flash_mla.sparse_mla_decode_int8_triton` | same repo, portable Triton |
| `sparse_mla_prefill` | `flash_mla.sparse_mla_prefill` | [forks-flash-mla-ampere-dsv4](https://github.com/AppMana/forks-flash-mla-ampere-dsv4) — fp8 path, whole-cache bf16 dequant pre-pass |
| | `flash_mla.sparse_mla_prefill_int8` | same repo — native fused int8, dequant in-kernel, allocates nothing sized by the KV pool |
| | `vllm.models.deepseek_v4.nvidia_sm12x.kernels.sparkinfer_sparse_mla_extend` | [`nvidia_sm12x/kernels.py`](vllm/models/deepseek_v4/nvidia_sm12x/kernels.py) |
| | `vllm.models.deepseek_v4.nvidia_imma.triton_kernels.sparse_attention_triton` | [`nvidia_imma/triton_kernels.py`](vllm/models/deepseek_v4/nvidia_imma/triton_kernels.py) |

**Toggle roles** — presence turns the path on; absence *in an explicit block*
turns it off.

| Role | Symbol | Implementation |
| --- | --- | --- |
| `indexer_cache_int8` | `vllm._custom_ops.indexer_k_quant_and_cache_int8` | `csrc/cache_kernels.cu` |
| `indexer_query_int8` | `vllm.models.deepseek_v4.common.ops.fused_indexer_q.fused_indexer_q_rope_quant_int8` | [`common/ops/fused_indexer_q.py`](vllm/models/deepseek_v4/common/ops/fused_indexer_q.py) |
| `dense_experts_int8_activation` | `vllm.model_executor.layers.quantization.utils.marlin_utils.marlin_act_int8_process_scales` | [`marlin_utils.py`](vllm/model_executor/layers/quantization/utils/marlin_utils.py) |
| `indexer_streaming_topk_prefill` | `vllm.model_executor.layers.sparse_attn_indexer.streaming_prefill_topk` | [`sparse_attn_indexer.py`](vllm/model_executor/layers/sparse_attn_indexer.py) |

`indexer_streaming_topk_prefill` bounds prefill top-k memory instead of letting
it scale with context; off by default, required above roughly 700k context.

### Which class serves which cache

The attention class follows the **cache type**, with compute capability only as
a floor — see
[`_select_dsv4_attn_cls`](vllm/models/deepseek_v4/nvidia/model.py).

| `cache_type` | Bytes/token | Class | Requires |
| --- | ---: | --- | --- |
| `int8_ds_mla` | 528 | [`DeepseekV4TritonSM86Attention`](vllm/models/deepseek_v4/nvidia_imma/attention.py) | IMMA + `cp.async` → sm_80 and up, **including sm_121** |
| `fp8_ds_mla` | 584 | [`DeepseekV4SparkInferSM12xAttention`](vllm/models/deepseek_v4/nvidia_sm12x/attention.py) | sm_120/121, sparkinfer + CuTe-DSL |
| `fp8_ds_mla` | 584 | [`DeepseekV4TritonSM86Attention`](vllm/models/deepseek_v4/nvidia_imma/attention.py) | sm_80 and up |

The module is `nvidia_imma`, not `nvidia_sm86`: it is named for what it
requires rather than the architecture it was written on, because selecting it
by architecture equality made an `int8_ds_mla` checkpoint unservable on GB10.
Checkpoints that name the old `nvidia_sm86.*` FQNs still resolve — the alias is
in [`kernel_config.py`](vllm/transformers_utils/configs/dsv4/kernel_config.py)
and [`nvidia_sm86/__init__.py`](vllm/models/deepseek_v4/nvidia_sm86/__init__.py).

The one genuine incompatibility: sparkinfer's `compressed_mla` reads the
584-byte fp8 page byte-for-byte, so it cannot consume `int8_ds_mla`. Every
other pairing is a configuration, not an architecture limit.

### Blocks for A/B testing

Each is a complete `--hf-overrides` value; the block replaces the checkpoint's
own, so these are self-contained.

**sm_121, sparkinfer fp8 (the GB10 default).** What the 4-bit FMMA path costs
and delivers:

```bash
--hf-overrides '{"vllm": {"kernels": [
  "vllm.models.deepseek_v4.nvidia_sm12x.kernels.sparkinfer_sparse_mla_decode",
  "vllm.models.deepseek_v4.nvidia_sm12x.kernels.sparkinfer_sparse_mla_extend"
], "cache_type": "fp8_ds_mla"}}'
```

**sm_121, portable Triton fp8.** Same cache bytes, bf16-Q Triton kernels — the
A/B that isolates sparkinfer's contribution with no rebuild:

```bash
--hf-overrides '{"vllm": {"kernels": [
  "vllm.models.deepseek_v4.nvidia_imma.triton_kernels.decode_sparse_attention_triton",
  "vllm.models.deepseek_v4.nvidia_imma.triton_kernels.sparse_attention_triton"
], "cache_type": "fp8_ds_mla"}}'
```

**IMMA int8, any arch from sm_80 up.** The int4-int8 checkpoint's own block;
runs on GB10 as well as Ampere:

```bash
--hf-overrides '{"vllm": {"kernels": [
  "flash_mla.sparse_mla_decode_int8",
  "flash_mla.sparse_mla_prefill_int8",
  "vllm._custom_ops.indexer_k_quant_and_cache_int8",
  "vllm.models.deepseek_v4.common.ops.fused_indexer_q.fused_indexer_q_rope_quant_int8",
  "vllm.model_executor.layers.quantization.utils.marlin_utils.marlin_act_int8_process_scales"
], "cache_type": "int8_ds_mla"}}'
```

**IMMA int8, portable decode.** Swaps only the fused native CUDA decode for its
Triton equivalent, to attribute a regression to that kernel:

```bash
--hf-overrides '{"vllm": {"kernels": [
  "flash_mla.sparse_mla_decode_int8_triton",
  "flash_mla.sparse_mla_prefill_int8",
  "vllm._custom_ops.indexer_k_quant_and_cache_int8",
  "vllm.models.deepseek_v4.common.ops.fused_indexer_q.fused_indexer_q_rope_quant_int8",
  "vllm.model_executor.layers.quantization.utils.marlin_utils.marlin_act_int8_process_scales"
], "cache_type": "int8_ds_mla"}}'
```

**1M context.** Add the streaming top-k to any block above:

```json
"kernels": ["...", "vllm.model_executor.layers.sparse_attn_indexer.streaming_prefill_topk"]
```

Weight quantization is **not** part of this block — it comes from
`quantization_config` and is unaffected by these overrides. Swapping kernels
holds the weights fixed, which is what makes a kernel A/B a controlled
comparison.

## Operational caveats

- **DSV4 runs on the v2 model runner only.** `DeepseekV4ForCausalLM` and
  `DSparkDraftModel` are in
  [`DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES`](vllm/config/vllm.py) so the runner
  is a property of the model rather than of the serve flags. Without that
  entry the `not is_moe` fallback puts DSV4 on the v1 runner, and it reached
  v2 only when a speculative config incidentally forced it — which means a
  benchmark run without `--speculative-config` was silently measuring a
  different runner. Do not run DSV4 on v1.
- **`vllm/layer_partition.py` ignores `VLLM_PP_LAYER_PARTITION`** — it
  replicates only the default uneven-split branch of `get_pp_indices`. Use
  `APPMANA_DSV4_PP_LAYER_PARTITION`; setting the vLLM env var alone loads
  wrong shards and yields NaN or garbage output.
- **Raw completions degrade.** Use `/v1/chat/completions`.
- **Weights load in the checkpoint's per-expert, separate-projection layout**
  and are fused at load time. Do not pre-fuse.

---

This fork is downstream of [vLLM](https://github.com/vllm-project/vllm)
([paper](https://arxiv.org/abs/2309.06180), [docs](https://docs.vllm.ai)),
Apache-2.0. Upstream's README, docs, and contribution process apply to
upstream; issues with this fork belong on this repo.
