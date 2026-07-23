<!-- markdownlint-disable MD001 MD041 -->

# AppMana vLLM: DeepSeek-V4 on Ampere (`appmana/vllm-ampere`)

This is a fork of [vLLM](https://github.com/vllm-project/vllm) that serves
**DeepSeek-V4-Flash** — a model whose official kernel support starts at
Hopper — on twelve consumer **RTX 3090 / A5000 (sm_86, Ampere, 24 GB)**
GPUs, one per node, connected into a pipeline-parallel ring over
**Thunderbolt RDMA** (`PP=12, TP=1`), with a **1,000,000-token context
window** and **DSpark speculative decoding**. Single-stream: ~3,000 tok/s
prefill at 950k context and 66 tok/s decode.

Everything here exists to make that sentence true on hardware upstream does
not support: Ampere ports of the sparse-MLA kernel surface, an INT4/INT8
quantization of the checkpoint that fits 24 GB cards, a KV-cache layout and
indexer path that survive a million tokens, and the pipeline-parallel and
scheduling fixes the 12-node chain needed. The branch is
`appmana/vllm-ampere`, kept merged with `upstream/main`.

**Checkpoint:**
[`appmana/deepseek-v4-int4-int8`](https://huggingface.co/appmana/deepseek-v4-int4-int8)
(INT4 Marlin routed experts, INT8 AllSpark dense, `int8_ds_mla` KV cache).

### What this fork adds

Upstream's DeepSeek-V4 sparse-MLA path assumes Hopper-or-newer kernels
(DeepGEMM, CuTeDSL, sm90 FlashMLA) plus Triton `fp8e4nv`, none of which run on
sm_86. This fork re-implements that kernel surface for Ampere, twice over:
portable Triton kernels (software fp8 decode or s8 x s8 integer MMA) in this
repo, and fused native CUDA kernels in the sibling
[`forks-flash-mla-ampere-dsv4`](https://github.com/AppMana/forks-flash-mla-ampere-dsv4)
(shipped into the image as the `flash_mla` wheel — its README carries the
kernel microbenchmarks and design narratives). On top of the kernels:

- `int8_ds_mla`: a full INT8 KV-cache/indexer/dense-activation path upstream
  doesn't have (packed 528-byte token slots: 512 int8 + fp32 scale + pad).
- `dsv4_int` quantization: INT4 Marlin routed experts, INT8 AllSpark dense
  layers, produced by `tools/ampere/dsv4_requant_checkpoint.py`.
- A checkpoint-driven kernel-selection config block (`vllm` key in
  `config.json`, see below) instead of environment variables.
- A pipeline-parallel fix for DeepSeek-V4's 4-tensor head-compression stream,
  which upstream truncated to one tensor and ran redundantly on every rank
  (PP=N now token-identical to PP=1).
- Streaming (activation-chunked) prefill top-k for the sparse-MLA indexer,
  bounding memory instead of letting it scale with context — this is what
  makes the 1M window fit; required above roughly 700k.
- A zero-target-layer last PP rank (only the DSpark draft stages), freeing
  the VRAM a 1M-token KV cache needs; partition `3,4,4,4,4,4,4,4,4,4,4,0`.
- In-kernel INT8 dequantization for sparse prefill, and scheduler-derived
  indexer workspaces, replacing allocations that scaled with pool size or
  `max_model_len` regardless of actual usage.
- A fail-fast executor (a worker exception with no reply path kills the
  worker instead of silently desyncing the pipeline) and an early HTTP
  health/metrics surface reporting engine boot stage.
- `VLLM_RAY_WORKER_IP_ORDER`, binding PP ranks to the Thunderbolt chain's
  physical index order (upstream's rank assignment loaded the wrong shards
  on this topology — the production NaN root cause).
- Warmup coverage for sparse-prefill, MHC, and fused-decode kernels, so no
  rank JIT-compiles mid-serving and deadlocks the chain.

### Serving benchmarks

`vllm bench serve`, single stream (C=1), PP=12 Thunderbolt chain,
`appmana/deepseek-v4-int4-int8`, `--async-scheduling`, 2026-07-21:

| Context | Speculative decoding | Decode (tok/s) | Prefill (tok/s) | Notes |
| ---: | --- | ---: | ---: | --- |
| 4k in / 1k out | off | 33.3 | — | TPOT 28.05 ms |
| 4k in / 1k out | DSpark | 56.1 | — | TPOT 15.90 ms |
| 620k in | off | — | ~3,230 | TTFT 191.9 s |
| 620k in | DSpark | — | ~3,390 | TTFT 182.8 s |
| 950k needle (1M window) | DSpark | 66.7 / 65.3 warm | ~3,020 | needle retrieved verbatim in 314.5 s |

Async scheduling matters: the identical 1M configuration measured 56.8/57.5
tok/s decode with `--no-async-scheduling`. Needle retrieval is verbatim at
620k, 950k, and the 1,048,576 window edge. The 1M rows require the streaming
indexer top-k and the zero-layer draft rank; chunked prefill, pipeline
parallelism, and speculative decoding are upstream mechanisms — the streaming
top-k and zero-layer rank are what make them fit on 24 GB GPUs.

Use the chat template (`--tokenizer-mode deepseek_v4`); raw completions
against this checkpoint degrade.

### Build

```bash
bash tools/ampere/build_vllm_ampere_image.sh --push
# tags harbor.appmana.com/appmana/vllm-ampere:<git short-sha (9)>
```

This is the only sanctioned build for a deployed image: a full build from
`docker/Dockerfile` (sm_86 arch list, AppMana NCCL fork with rail routing,
usb4-rdma provider, `flash_mla` wheel). The `Dockerfile.ampere-*` overlay
variants (`-python-hotfix`, `-nccl-overlay`, `-rdma-overlay`) exist for
fast local debugging only and must never be in a deployed image's lineage —
an overlay image reports the BASE image's version string, so a benchmark row
attributed to an overlaid image is unattributable. The build fails if the
baked `libnccl` predates the rail-routing implementation.

### Deploy (serving)

Served via the LeaderWorkerSet at
`appmana-cluster/.../inference/lws-vllm-deepseek-v4.yaml` (GitOps/Flux). That
manifest is a direct translation of the proven benchmark JobSet in
`tools/ampere/benchmark_jobset/` — same boot script, same
`prep_pp_shards.py` rank-local shard staging (HF-cache layout on a hostPath),
same env. The cluster's `tb-chain-webhook` injects `NCCL_SOCKET_IFNAME` and
`VLLM_RAY_WORKER_IP_ORDER` from Kueue's admitted chain placement; the Kueue
fork ([`forks-kueue-chain-adjacency`](https://github.com/AppMana/forks-kueue-chain-adjacency))
allocates the group as one contiguous chain run with the leader (PP rank 0)
at the chain end.

### Benchmarking: generating a JobSet

`tools/ampere/benchmark_jobset/` carries the benchmark harness:

- `dsv4-benchmark-jobset-proof.yaml` — the proven JobSet template: boot
  script (host rendezvous, shard staging, staged /health, sanity check,
  `vllm bench serve` matrix, Ray log preservation) plus the
  `prep_pp_shards.py` stager, as one ConfigMap + JobSet.
- `generate_dsv4_benchmark_jobset.py` — stamps the template into an
  apply-ready JobSet: sets name/image/world size, bakes the chain IP order,
  scrubs webhook-owned NCCL env, gives each run its own script ConfigMap
  (two runs sharing one ConfigMap = the second run's flags mid-flight), and
  verifies the image tag exists before a 12-node slice sits in
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

`--indexed` is the shape Kueue admits on the tb-chain queue (one indexed
replicatedJob; rank = position of the pod's host IP in the chain order, so
any contiguous chain window TAS picks resolves correctly). `--env NAME=VALUE`
overrides any harness knob per run; `--prewarm-only` stages shards and exits.
Results land under `/jit-shared/dsv4-jobset-results/<name>/<uid>/` (bench
JSON, vllm-server.log, per-rank boot logs, preserved Ray logs). Benchmark
methodology and per-row results live in this repo's git log — commit bodies
are the benchmark record.

### Configuration: the `vllm` checkpoint-config block

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

Each entry is the importable symbol to activate; unknown symbols, duplicate
roles, and invalid `cache_type` values fail closed at startup, and the
resolved config prints once as `vllm kernels resolved: ...` — treat that line
as the validity gate for any benchmark row. `--hf-overrides` replaces the
whole block rather than merging it, so build overrides from the checkpoint's
own block plus whatever you're adding, not a hand-curated list — a dropped
kernel entry degrades silently.

Add `vllm.model_executor.layers.sparse_attn_indexer.streaming_prefill_topk`
to `kernels` to enable streaming indexer top-k; off by default, required
above roughly 700k context.

### Operational caveats

- **`vllm/layer_partition.py` ignores `VLLM_PP_LAYER_PARTITION`** — it
  replicates only the default uneven-split branch of `get_pp_indices`. Use
  `APPMANA_DSV4_PP_LAYER_PARTITION` (consumed by the boot script, which sets
  both consistently); setting the vLLM env var alone loads wrong shards and
  yields NaN/garbage output.
- **Overlay images report the base image's version string** — one more
  reason they are debug-only (see Build above).

---

This fork is downstream of [vLLM](https://github.com/vllm-project/vllm)
([paper](https://arxiv.org/abs/2309.06180), [docs](https://docs.vllm.ai)),
Apache-2.0. Upstream's own README, docs, and contribution process apply to
upstream; issues with this fork belong on this repo.
