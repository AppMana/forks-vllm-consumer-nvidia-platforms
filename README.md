<!-- markdownlint-disable MD001 MD041 -->

# AppMana vLLM: DeepSeek-V4 on consumer NVIDIA platforms

A fork of [vLLM](https://github.com/vllm-project/vllm) that runs
**DeepSeek-V4-Flash** — a model whose official kernel support starts at
Hopper — on GPUs upstream does not support. One image serves two
architectures: **Ampere (sm_86)** via ported sparse-MLA kernels and an
INT4/INT8 checkpoint, and **GB10 / consumer Blackwell (sm_121)** through that
same INT4/INT8 + FlashMLA/Marlin core. Its shared mHC backend is under
offline validation through a narrow SparkInfer adapter. A separate,
experimental NVFP4/FP8 checkpoint uses sparkinfer throughout its SM12x kernel
lane.

GPU count, pipeline size, interconnect, and context length are configuration,
not assumptions.

## Architecture support

| Arch | Gencode | Kernel source | Checkpoint | Status |
| --- | --- | --- | --- | --- |
| Ampere sm_86 (RTX 3090 / A5000, 24 GB) | `8.6` | Triton + fused native CUDA (`flash_mla`) | `appmana/deepseek-v4-int4-int8` | Validated, benchmarked |
| GB10 sm_121 (DGX Spark) | `12.1a` | FlashMLA + Triton INT8 indexer + Marlin; native SM121 AllSpark image in validation | `appmana/deepseek-v4-int4-int8` at `597471bc…` | Canonical checkpoint audit passes; historical PP=2 C1/C2 matrix passed through 43,086 tokens, but the repaired checkpoint and rebuilt image still require live acceptance |
| GB10 sm_121 (DGX Spark) | `12.1a` | sparkinfer (CuTe-DSL) | `appmana/deepseek-v4-nvfp4-fp8` | Bring-up; output not yet correct |

Both gencodes are built into one image (`docker/Dockerfile`,
`torch_cuda_arch_list='8.6 12.1a'`). Kernel selection is per-checkpoint, not
per-build — see [the `vllm` config block](#configuration-the-vllm-checkpoint-config-block).

## Checkpoints

### [`appmana/deepseek-v4-int4-int8`](https://huggingface.co/appmana/deepseek-v4-int4-int8)

INT4 W4A16 Marlin routed experts, INT8 W8A16 dense/shared/attention linears,
`int8_ds_mla` KV cache and indexer, with three DSpark MTP draft stages grafted
in. The canonical immutable revision is
`597471bc544b541771306ddcdb7089ad740bb6d3`: historical non-Base target
shards from `57d8af1c…`, the DSpark three-stage MTP graft, and the latest
kernel configuration. Its complete 72,320-tensor header/index audit passes;
the target and all three MTP heads match the official non-Base head hash.
The former `-instruct` repository was deleted and must not be used.

AllSpark keeps the dense linears compact on validated Ampere builds. The old
Hilton SM121 image contained AllSpark code compiled as `sm_120a`, which
produced invalid results on GB10 and forced the deployed fallback to expand
compact linears to BF16. Commits `948c69d7f8` and `17080ac8a1` preserve
`sm_121a` through CUDA 13 architecture filtering and produce an immutable
image tag. That rebuilt image is not deployment proof until its direct
AllSpark numerical probe passes on GB10.

This checkpoint serves a 1,000,000-token context on twelve 24 GB Ampere GPUs.
On two DGX Sparks, its historical PP=2 run without DSpark passed the clean actual
1,110-/8,829-/43,084-token C1/C2 matrix with the native INT8-cache FlashMLA
decode/prefill kernels and the int64-addressed Triton indexer. All 6/6 cells
completed with zero pod restarts and no CUDA, Xid, or illegal-address log.
That was a serving-stability control, not evidence for canonical revision
`597471bc…`. Revision `748621a0fe36dc4d3b7f45a318ecff610f18455e`
replaced only `head.weight` from a release donor and was an isolated experiment;
it is not the live checkpoint.

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

Two SparkInfer integration defects have been found during this bring-up.
Commit `a62dd0ab63` restores ModelOpt activation global scales; a live TP=2,
target-only test changed logits but remained incorrect. Commit `3efbd54937`
also normalizes fused gate/up ordering and swizzles the ModelOpt block scales.
Its regression suite passes offline, but it has not yet been tested end to end
on the Sparks. Neither fix is evidence that the checkpoint is correct.

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
- [`dragonintel` performance reference](https://github.com/hannesholste/dragonintel/blob/main/docs/dsv4-spark-performance-references.md)
  — the evidence map for Hilton results, NVIDIA forum numbers, exact topology
  and kernel contracts, missing provenance, and the acceptance gate for any
  future NVFP4 performance claim.

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

The current Hilton node-local image is
`docker.io/appmana/vllm-consumer:sm86-sm121` at
OCI-index digest
`sha256:8ef70c5d2699900dc3f560b6f278f8a09a165d345d65a184460db38495c03e73`;
the arm64 manifest is
`sha256:e10452754b2d3d1b96db824b0e6ccd46e1ad6d93a7426a2d014b584528112f3b`
and the config/pod image ID is
`sha256:5d516d8c727a3821eda042ed3a7f9460741dda264288b6e5b401660b1b5a87b4`.
Both nodes match all three identities.
Installed-file hashes tie it to vLLM `85f78d8827` and sparkinfer `fb71dc89`;
its OCI source/revision labels are wrong and must not be cited as provenance.
It predates both NVFP4 fixes above. A tag match alone is insufficient: record
the digest and installed-file hashes for every benchmark or rollout.

The base image predates the accepted paged-indexer, pinned-loader and tokenizer
fixes. GitOps `626a51d` supplies checked runtime overlays, including tokenizer
fix `2266aeb64c`, while the immutable instruct checkpoint carries the runtime
block emitted by converter fix `a389f65984`. The live result must therefore be
attributed to the image config digest, overlays, and pinned checkpoint together,
not to the image alone. Compiler caches live on per-rank persistent volumes.
INT4 experts and INT8 sparse attention remain Marlin and FlashMLA, while the
accepted launch forces Triton for the indexer and mHC.
Commit `8c8bcac623` adds `VLLM_MHC_CUDA_BACKEND=triton`, which forces the
existing Triton/torch mHC fallback on SM121; `auto` retains TileLang. That
selector is the accepted Hilton mHC backend; TileLang mHC remains a separate
implementation concern.

Commit `75b9aff02c` adds the narrow SparkInfer shared-mHC adapter and the `mhc`
selector role. It delegates first-layer 2D broadcast plus fused norm,
inter-layer fused post/pre plus norm, and final post to `sparkinfer.norm.mhc`.
HCHead and the standalone 3D pre operation have no matching SparkInfer API and
stay on Triton, never TileLang. Its offline suite passed 126 tests with two
hardware skips, but no image containing it has been built or live-tested on
SM121. It is not deployment proof.

## Deploy

### Ampere, PP=12

`appmana-cluster/.../inference/lws-vllm-deepseek-v4.yaml` — a LeaderWorkerSet
translated from the benchmark JobSet in `tools/ampere/benchmark_jobset/`: same
boot script, same `prep_pp_shards.py` rank-local shard staging, same
environment. The cluster's `tb-chain-webhook` injects `NCCL_SOCKET_IFNAME` and
`VLLM_RAY_WORKER_IP_ORDER` from the admitted chain placement.

### GB10, PP=2: current INT4/INT8 LWS

`demos-hilton/cluster/gitops/apps/base/inference/deepseek-v4-int4-int8.yaml`
is the current two-rank LeaderWorkerSet, one GB10 per node, with the direct
dual-rail RoCE fabric between them. It derives `--node-rank` from
`LWS_WORKER_INDEX`, reads each host's fabric address from the RoCE interface,
and adds `--headless` only on nonzero ranks. The leader remains pinned to
`spark-2ab3` because the present point-to-point `/30` makes
`10.255.0.1` the fixed rendezvous address.

The manifest is enabled at GitOps `626a51d` and serves
`appmana/deepseek-v4-int4-int8-instruct` revision
`fa1e3b4728508795a68fb88972f8cfff2f0700ab`. Two node-local
5 Gi RWO volumes persist every compiler/autotune/temp root under `/jit-cache`;
they are rank-local, not shared. The accepted lane uses PP=2, TP=1,
`max_num_batched_tokens=1024`, a 4096-token long-prefill threshold, eager
execution, no DSpark, pinned safetensors, and `--max-model-len 65536`. The
checked runtime overlays are mounted onto image config digest
`sha256:5d516d8c727a3821eda042ed3a7f9460741dda264288b6e5b401660b1b5a87b4`.
Clean actual 1,112-, 8,831-, and 43,086-token prompts passed at C1 and C2:
6/6 cells, Ready/Ready, restart count zero. PP0 loaded weights in 47.27 s and
the model in 55.714 s at 78.09 GiB; PP1 took 49.74 s and 58.668 s at
81.55 GiB. DSML and Mastra code execution passed, and the exact 43,125-token
needle returned `ORCHID7429COPPER`. The raw artifact is
`demos-hilton/cluster/harness/dsv4-int4-int8-instruct-fa1e3b-pp2-2026-07-30.csv`.
The accepted launch remains capped at 65,536; 250K context is not validated.

With DSpark enabled, the same 3 GiB cache admitted about 101K tokens; without
DSpark it admits 190K tokens (2.90 concurrent 65,536-token requests). DSpark
therefore materially increased cache/scheduler working set and caused the
earlier C2 17.6K failure. The separate pre-fix 35K fault was signed-32-bit
overflow in Triton paged-cache addressing, now fixed by `5b0285ecd3`.

```yaml
- --tensor-parallel-size
- "1"
- --pipeline-parallel-size
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
- "65536"
- --max-num-seqs
- "2"
- --max-num-batched-tokens
- "1024"
- --long-prefill-token-threshold
- "4096"
- --gpu-memory-utilization
- "0.68"
- --kv-cache-memory-bytes
- "3221225472"
```

`--max-num-seqs=2` is a memory constraint, not a throughput preference.
Activation and workspace buffers are preallocated at that ceiling; a value of
8 caused `NV_ERR_NO_MEMORY` and Xid 31 during a single 8k prefill while the
Gemma standby shared rank 1.

### GB10, TP=2: shelved NVFP4 bring-up

`demos-hilton/cluster/gitops/apps/base/inference/deepseek-v4-nvfp4.yaml` is
commented out of the kustomization. It is retained as a bring-up manifest, not
as a working deployment: both speculative and target-only generation are
incorrect, and no throughput from it is publishable.

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
  --model deepseek-v4-int4-int8 \
  --input-tokens 8000 --needle-tokens 1000 \
  --concurrency 8 \
  --output-json needle-8k-1k.json
```

Set `--concurrency` no higher than the server's `--max-num-seqs`: a sweep above
it is silently capped by the scheduler and every arm above the cap looks
identical. `--tokenizer` defaults to `--model`; the Hilton API retains
`deepseek-v4-int4-int8` as its served alias even though the backing Hub repo is
the instruct checkpoint.

### Parallelism arms

`cluster/harness/serve_dsv4.sh <arm> <rank>` in the `demos-hilton` repo serves
the same checkpoint under arguments that differ in exactly one dimension — `A`
TP=2 dual-rail, `B` TP=2 single rail, `C` PP=2, `D` TP=2 + expert parallel — and
`cluster/harness/llm_bench.py` (`--base-url`, `--model`, `--arm`, `--out`)
records them to one CSV. `MODEL=` and `REVISION=` select the checkpoint, so both
checkpoints run through the same launcher rather than a forked copy.

Run rank 1 first; it is `--headless` and waits for the leader.

```bash
MODEL=appmana/deepseek-v4-int4-int8-instruct \
REVISION=fa1e3b4728508795a68fb88972f8cfff2f0700ab \
./serve_dsv4.sh A 1   # worker
MODEL=appmana/deepseek-v4-int4-int8-instruct \
REVISION=fa1e3b4728508795a68fb88972f8cfff2f0700ab \
./serve_dsv4.sh A 0   # leader
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

On two DGX Sparks at TP=2, the native-host INT4/INT8 correctness run at
8,004 prompt tokens and 1,128 completion tokens measured 5.23 s TTFT,
approximately 1,532 input tok/s, and 38.5 decode tok/s. The embedded
1,000-token needle passed at 0.94 word-level match ratio. This is a valid local
baseline, but the exact installed source was not frozen and a Triton JIT miss
occurred during inference, so it is not a release-quality benchmark artifact.
See the
[performance reference](https://github.com/hannesholste/dragonintel/blob/main/docs/dsv4-spark-performance-references.md)
for its complete limitations.

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
  --image ghcr.io/appmana/vllm-consumer:<immutable-tag> \
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

The published INT4 checkpoint block predates the `mhc` role, so it resolves to
the documented vLLM device-auto mHC default. Selecting SparkInfer mHC requires
the complete override shown below; it does not change the INT8 attention,
indexer, or Marlin roles.

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
| `mhc` | `vllm.model_executor.layers.mhc.MHCFusedPostPreOp` | vLLM device-auto default |
| | `vllm.models.deepseek_v4.nvidia_sm12x.mhc.sparkinfer_mhc_post_pre` | [`nvidia_sm12x/mhc.py`](vllm/models/deepseek_v4/nvidia_sm12x/mhc.py) → `sparkinfer.norm.mhc`; shared mHC only |

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

**GB10 IMMA int8 with SparkInfer shared mHC.** This complete SM121 variant
keeps every native INT8 role and selects only shared mHC from SparkInfer:

```bash
--hf-overrides '{"vllm": {"kernels": [
  "vllm.models.deepseek_v4.nvidia_imma.triton_kernels.decode_sparse_attention_triton",
  "flash_mla.sparse_mla_decode_int8",
  "flash_mla.sparse_mla_prefill_int8",
  "vllm._custom_ops.indexer_k_quant_and_cache_int8",
  "vllm.models.deepseek_v4.common.ops.fused_indexer_q.fused_indexer_q_rope_quant_int8",
  "vllm.model_executor.layers.quantization.utils.marlin_utils.marlin_act_int8_process_scales",
  "vllm.models.deepseek_v4.nvidia_sm12x.mhc.sparkinfer_mhc_post_pre"
], "cache_type": "int8_ds_mla"}}'
```

This override does not select the local native SparkInfer INT8 indexer
prototype. That prototype exists only as uncommitted changes across the vLLM
and SparkInfer worktrees and has not executed on GB10.

**IMMA int8, portable decode.** Swaps only the fused native CUDA decode for its
Triton equivalent, to attribute a regression to that kernel:

```bash
--hf-overrides '{"vllm": {"kernels": [
  "vllm.models.deepseek_v4.nvidia_imma.triton_kernels.decode_sparse_attention_triton",
  "flash_mla.sparse_mla_decode_int8_triton",
  "flash_mla.sparse_mla_prefill_int8",
  "vllm._custom_ops.indexer_k_quant_and_cache_int8",
  "vllm.models.deepseek_v4.common.ops.fused_indexer_q.fused_indexer_q_rope_quant_int8",
  "vllm.model_executor.layers.quantization.utils.marlin_utils.marlin_act_int8_process_scales",
  "vllm.models.deepseek_v4.nvidia_sm12x.mhc.sparkinfer_mhc_post_pre"
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
