<!-- markdownlint-disable MD001 MD041 -->

# AppMana vLLM — DeepSeek-V4 on Ampere (`appmana/vllm-ampere`)

This is AppMana's fork of vLLM that serves **DeepSeek-V4-Flash** on **RTX 3090 / A5000
(sm_86, Ampere)** GPUs across a **12-node Thunderbolt pipeline-parallel chain** (PP=12,
1 GPU/node). `appmana/vllm-ampere` and `main` are the canonical branch; the pre-rebase
history is preserved on `*-prerebase`. It is a careful re-implementation of our Ampere/
int-quant work on top of a fresh `upstream/main` rebase (so it keeps upstream's correct
pipelined `PPHandler`, not the fork's old per-token `torch.cuda.synchronize`).

**Checkpoint (weights):** [`appmana/deepseek-v4-int4mse-int8`](https://huggingface.co/appmana/deepseek-v4-int4mse-int8)
— routed experts as **INT4 W4A16 Marlin** (group 32, MSE scales), FP8 dense linears as
**INT8 W8A16 AllSpark** (channelwise biased uint8), int8 sparse-MLA indexer K-cache;
`quant_method=dsv4_int`. Produced by `tools/ampere/dsv4_requant_checkpoint.py` from the
base FP8/MXFP4 release. The model loads it directly (per-expert + separate-projection
names; fused at load by the stacked/expert mappings — do **not** pre-fuse).

### What this fork adds

Upstream vLLM's DeepSeek-V4 sparse-MLA path assumes Hopper-or-newer kernels
(DeepGEMM, CuTeDSL gathers, the sm90 FlashMLA/FlashInfer sparse attention) plus
Triton `fp8e4nv`, none of which run on sm_86. This fork re-implements the DSV4
kernel surface for Ampere twice over: portable Triton kernels that decode fp8
e4m3 arithmetically (`common/ops/fp8e4m3_arith.py`) or run s8 x s8 integer MMA,
and fused native CUDA kernels in the sibling fork
[`AppMana/forks-flash-mla-ampere-dsv4`](https://github.com/AppMana/forks-flash-mla-ampere-dsv4),
which ship into the serving image as the `flash_mla` wheel. It also adds an
INT8 variant of the whole stack end to end (`int8_ds_mla` KV cache, INT8
indexer cache and query, W4A8 dense activations); upstream has no `int8_ds_mla`
layout at all. Which kernel serves each role is selected by the checkpoint's
`appmana` config block (see below), never by env vars.

#### Kernel microbenchmarks at production shapes

DSV4 production shapes: H=64 query heads, d=576 (512 nope + 64 rope), dv=512,
top-k 512 and 1024, T=1 decode, 1024-token prefill chunks at 16k context. All
rows below were measured on an RTX A5000 (sm_86) against true >=16k-slot paged
caches (L2-resident microbenches invert the prefill conclusions; see
"Benchmarking caveats" in the flash_mla fork README), CUDA events, best of
repeated runs, measured 2026-07-13 with the committed harnesses in the
flash_mla fork (`benchmarks/bench_sparse_mla_decode_16k.py`,
`bench_int8_sparse_mla.py`, `bench_sparse_mla_16k_matrix.py`). They are
consistent with the numbers recorded at that repo's tag
`dsv4-sm86-kernels-2026-07-09` and in its commit `eb855ae`, within the
busy-sibling-GPU jitter documented in that repo's caveats.

There is no runnable upstream baseline for sparse-MLA attention on sm_86
(upstream's sparse kernels require sm90+), so the reference row for each native
CUDA kernel is this fork's own portable Triton kernel for the same role,
labeled as such.

**Sparse-MLA decode** (T=1, us per call):

| Kernel | Repo | Lang | Cache | top-k 512 | top-k 1024 |
| --- | --- | --- | --- | ---: | ---: |
| `flash_mla.flash_sparse_mla_decode` (fused: selection dequant, heads-as-M `mma.m16n8k16`, warp-wide combine) | flash_mla | CUDA | fp8_ds_mla | **24.7** | **28.4** |
| `flash_mla.sparse_int8_mla_decode` (same fused path, int8 dequant pre-pass) | flash_mla | CUDA | int8_ds_mla | **21.9** | **28.0** |
| `flash_mla.triton_sparse_int8_mla_decode` (s8 x s8 IMMA QK, bf16 PV) | flash_mla | Triton | int8_ds_mla | 45.8 | 86.2 |
| Legacy native decode (`FLASH_MLA_DECODE_FUSED=0`, in-CTA fp8 dequant; A/B reference) | flash_mla | CUDA | fp8_ds_mla | 46.8 | 80.2 |
| `nvidia_sm86.triton_kernels.decode_sparse_attention_triton` (portable baseline) | this repo | Triton | fp8_ds_mla | 213.7 | 422.8 |

The fused CUDA decode is 8.7x / 14.9x the Triton fp8 baseline and 1.9x / 2.8x
the prior native kernel; the native int8 decode is 2.1x / 3.1x the Triton int8
decode and matches fp8 latency. Across the model's 61 sparse-MLA layers the
fused decode saves roughly 1.5 ms per output token vs the prior native kernel.

**Sparse-MLA prefill** (1024-token chunk at 16k context, us per token):

| Kernel | Repo | Lang | Cache | width 512 | width 1024 |
| --- | --- | --- | --- | ---: | ---: |
| `flash_mla.flash_sparse_mla_prefill` (fused: whole-cache dequant pass, bf16 tensor-core attention) | flash_mla | CUDA | fp8_ds_mla | **2.9** | **5.2** |
| Same entry over int8 cache (`flash_mla.sparse_int8_mla_prefill`) | flash_mla | CUDA | int8_ds_mla | **2.8** | **4.9** |
| `triton_sparse_int8_mla_decode` at prefill token counts | flash_mla | Triton | int8_ds_mla | 3.3 | 6.3 |
| `sparse_attention_triton` + Triton dequant-gather (production Triton prefill) | this repo | Triton | fp8_ds_mla | 5.2 | 10.2 |
| `sparse_attention_triton` attention body alone (pre-gathered bf16 rows) | this repo | Triton | bf16 | 3.8 | 7.4 |

The flash_mla fork's README carries the design narratives (heads-as-M decode
rewrite, whole-cache-dequant prefill), the ncu evidence, the rejected
experiments, and the benchmarking caveats; read it before trusting or
re-deriving any kernel policy. That fork also makes the upstream dense
(non-sparse) FlashMLA forward launch on sm_86 at all (single-buffer `kp1`
pipeline + dv-512 dispatch; upstream's kernel exceeds the 100 KB
shared-memory cap).

**KV-cache write / gather kernels** (parity-tested; no timing harness, so no
numbers are claimed):

| Kernel | Lang | What it does | Upstream equivalent |
| --- | --- | --- | --- |
| `_C.deepseek_v4_fp8_ds_mla_dequantize_and_gather_k_cache` | CUDA | fp8_ds_mla paged cache to bf16 prefill gather workspace, zero-fill bounds guards | CuTeDSL `dequant_gather_k` (sm90+); replaced the torch fallback on sm_8x |
| `cache_utils.dequantize_global_slots_k_cache` | Triton | fp8_ds_mla dequant by global slot index | none (new indexing mode) |
| `cache_utils.fused_qnorm_rope_kv_int8_ds_mla_insert` | Triton | fused q-norm + rope + quant + insert into the int8_ds_mla SWA cache (528 B/token: 512 int8 + fp32 scale + pad, 16 B aligned) | the native fp8 writer `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` (fp8-only) |
| `cache_utils.quantize_and_insert_int8_ds_mla_cache`, `dequantize_global_slots_int8_ds_mla_cache`, `dequantize_and_gather_int8_ds_mla_cache` | Triton | int8_ds_mla insert + gather family | none; upstream has no int8_ds_mla |
| `_C_cache_ops.indexer_k_quant_and_cache(..., "int8")` | CUDA | symmetric INT8 mode added to upstream's indexer K-cache writer; 99.48% mean top-512 recall vs the fp8 path on real tensors | the same kernel's fp8 path |

**Indexer / mHC / GEMM enablement kernels.** The upstream fast path for each of
these either cannot execute on sm_86 or does not exist (the runnable fallback
was eager torch), so there is no like-for-like microbench; they are functional
replacements validated by parity tests:

| Kernel | Lang | What it does | Upstream equivalent |
| --- | --- | --- | --- |
| `nvidia_sm86.triton_kernels.fp8_mqa_logits_triton`, `fp8_paged_mqa_logits_triton`, `fp8_paged_mqa_logits_rowwise_triton` | Triton | Lightning-indexer logits over the fp8 or int8 indexer cache (software fp8 decode; s8 x s8 IMMA when the int8 query is active) | DeepGEMM `fp8_mqa_logits` / `fp8_paged_mqa_logits` (sm90) and Triton `fp8e4nv` kernels (sm89+) |
| `nvidia_sm86.triton_kernels.mqa_logits_workspace_triton` | Triton | fused MQA logits over the prefill gather workspace (fp8 or s8 IMMA); replaces the fork's own chunked torch fallback | DeepGEMM `fp8_mqa_logits` (sm90) |
| `common/ops/fused_indexer_q.fused_indexer_q_rope_quant_int8` | Triton | INT8 output mode added to the fused indexer-Q rope + quant kernel (q scale folded into weights, feeds the IMMA logits) | the same kernel's fp8/mxfp4 modes |
| `nvidia_sm86.triton_kernels.tf32_hc_prenorm_gemm_triton` | Triton | TF32 split-K prenorm GEMM feeding the mHC pre-fuse | eager torch |
| `kernels/mhc/triton.py`: `mhc_pre_triton`, `mhc_post_triton`, `mhc_fused_post_pre_triton` | Triton | the DSV4 mHC stream ops (pre-fuse, post collapse, fused post+pre between layers) | TileLang mHC kernels (TileLang is not shipped on the Ampere image), else eager torch |
| `nvidia_sm86.triton_kernels.deepseek_v4_fp8_einsum_triton`, `common/ops/fp8_einsum.deepseek_v4_sm12x_fp8_einsum` | Triton | block-scaled fp8 `bhr,hdr->bhd` einsum for `wo_a` | DeepGEMM `fp8_einsum`; off the hot path since `wo_a` is stored BF16 for dsv4_int (`152467e1ef`) |
| `backends/mla/sparse_mla_kernels.py` | Triton | portable sparse-MLA library (accumulate / merge / finish with attention sink, fp8_ds_mla paged and gathered variants) | none; test-covered, superseded in production by the kernels above |

#### Other major additions (non-kernel)

- **`appmana` checkpoint config block** — fail-closed, checkpoint-driven kernel
  selection, `cache_type` default, and `pp_transport` toggles (details below).
- **`dsv4_int` quant method** — INT4 Marlin experts (group 32, MSE scales) +
  INT8 AllSpark dense + optional W4A8 activations, wiring upstream Marlin /
  AllSpark GEMMs; produced by `tools/ampere/dsv4_requant_checkpoint.py`.
- **`int8_ds_mla` KV-cache dtype end to end** — packed-slot alignment, SWA
  insert path, runner support, cudagraph safety.
- **MHC pipeline-parallel fix** (`models/deepseek_v4/nvidia/model.py`) —
  DeepSeek-V4 carries a 4-tensor head-compression stream `(hidden, residual,
  post_mix, res_mix)` between decoder layers. Upstream passed only
  `hidden_states` across a PP boundary and ran `mhc_post` on every rank,
  corrupting the residual stream (PP=N != PP=1). All four tensors now cross via
  the async `isend/irecv_tensor_dict` path and `mhc_post` runs only on the last
  rank, so PP=N is token-identical to PP=1.
- **PP transport** — pack the PP intermediate-tensor dict into one NCCL message
  per hop (`624d5e1263`) and cache the pickled metadata schema across steps
  (`626da25f1c`); toggled via `appmana.pp_transport`.
- **`VLLM_RAY_WORKER_IP_ORDER`** (`v1/executor/ray_utils.py`) — binds vLLM PP
  ranks to the chain-index IP order injected by the `tb-chain-webhook`. Without
  it `RayExecutorV2` ranks were scrambled vs per-rank shard materialization, so
  workers loaded the wrong layers' shards (uninitialized weights, NaN logits).
  This was the production NaN root cause.
- **`vllm/layer_partition.py`** — rank-local shard partition matching vLLM's PP
  layer split (see caveat below about `VLLM_PP_LAYER_PARTITION`).
- **DSV4 warmup coverage** — sparse-prefill, mHC, fused-decode, and async-PP
  postprocess warmup so no rank JIT-compiles Triton during serving (a single
  compiling rank deadlocks the PP chain).
- **Native clamped SiLU routing on sm_86** — `SiluAndMulWithClamp` and the
  fused-MoE clamped SiLU run as eager torch ops so Python hotfix overlays stay
  independent of the base image's compiled `_C` op schema (`0cb2020074`).
- **DeepSeek V4 reasoning parser registration** — `<think>` token
  initialization without per-request `chat_template_kwargs`.
- **Benchmark client fix** — stop stream parsing at the `[DONE]` frame
  (`26e2397621`).
- **Ampere image pipeline** — `docker/Dockerfile.ampere-*` +
  `tools/ampere/build_vllm_ampere_image.sh`, AppMana NCCL fork, usb4-rdma
  provider, flash_mla wheel pinning, and a fast python-only hotfix overlay.

### Deploy

Built via `docker/docker-bake.hcl` + `Dockerfile.ampere-*`; python-only changes ship as a
fast `Dockerfile.ampere-python-hotfix` overlay (no CUDA recompile). Served on the cluster
through the LWS at `appmana-cluster/.../inference/lws-vllm-deepseek-v4.yaml` (GitOps). The
`tb-chain-webhook` injects `NCCL_SOCKET_IFNAME` + `VLLM_RAY_WORKER_IP_ORDER`; the leader/
worker commands materialize each rank's shards by pod ordinal.

**Measured (single-user, server-side Prometheus):** ~**26.7 tok/s** decode, **~175 ms TTFT**
on the int4mse-int8 checkpoint over the 12-node chain.

### Configuration: the `appmana` checkpoint-config block

Kernel selection is checkpoint-config driven, not environment driven. The
checkpoint's `config.json` carries one block, overridable wholesale via
`--hf-overrides '{"appmana": {...}}'`:

```json
"appmana": {
  "kernels": [
    "flash_mla.flash_sparse_mla_decode",
    "flash_mla.triton_sparse_int8_mla_decode",
    "vllm.models.deepseek_v4.nvidia_sm86.triton_kernels.sparse_attention_triton",
    "vllm._custom_ops.indexer_k_quant_and_cache_int8",
    "vllm.models.deepseek_v4.common.ops.fused_indexer_q.fused_indexer_q_rope_quant_int8",
    "vllm.model_executor.layers.quantization.utils.marlin_utils.marlin_act_int8_process_scales"
  ],
  "cache_type": "fp8_ds_mla",
  "pp_transport": {"pack": true, "cache_metadata": true}
}
```

Every value is the exact importable symbol that gets activated; the role is
inferred from a registry. Unknown symbols, duplicate roles, and invalid
`cache_type` values fail closed at startup. A `kernels` list is authoritative
for toggles (unlisted = off), which is what makes the int8 indexer independent
of the dense IMMA runtime. `cache_type` sets the default KV dtype only when the
CLI passes `--kv-cache-dtype auto`; an explicit CLI value always wins. The
optional `pp_transport` sub-block toggles the PP intermediate-tensor transport
optimizations (`pack`, `cache_metadata`); unlike env vars it reliably reaches
remote Ray workers. The resolved configuration prints once at startup as
`appmana kernels resolved: ...`
— treat that line as the validity gate for any benchmark row.

Legacy role-keyed `--hf-overrides` keys
(`deepseek_v4_sm86_sparse_mla_decode_fp8`, `..._decode_int8`, `..._prefill`) and
the top-level `__experimental_enable_imma_...` flag still work with a
deprecation warning; the block wins when both are present.

`APPMANA_DSV4_*` environment variables configure the *benchmark harness*, not
kernels. `APPMANA_DSV4_INDEXER_CACHE_INT8` has had zero consumers since the
upstream rebase — setting it does nothing.

### Operational caveats

- **Overlay images report the base image's version string.** The Python hotfix
  overlay copies `vllm/` over an existing image without regenerating
  `_version.py`, so a hotfix image still logs the base commit
  (e.g. `v31.dev57+g82df3f6bb`). Verify overlay contents by image tag/digest,
  never by the reported version.
- **`VLLM_PP_ASYNC_TOKEN_COMM` and `VLLM_PP_STATIC_DECODE_INTERMEDIATE_COMM`
  are dead at HEAD** — the implementation lived on `appmana/vllm-ampere-prerebase`
  and was never re-ported. A/B results on those knobs are noise.
- **`vllm/layer_partition.py` ignores `VLLM_PP_LAYER_PARTITION`.** It replicates
  only the default uneven-split branch of `get_pp_indices`. If that env is set,
  rank-local shard materialization and the model's own partition disagree, which
  loads wrong layers and yields NaN/garbage output.

---

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
