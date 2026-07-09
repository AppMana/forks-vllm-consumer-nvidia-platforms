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

### What's in this fork (sm_86 + chain serving)

- **`nvidia_sm86/` attention backend** — capability-selected at `major == 8`;
  FlashMLA CUDA sparse-MLA decode, gathered bf16 Triton sparse-MLA prefill,
  and the AppMana INT8 IMMA indexer path. The prefill path intentionally stays
  on Triton: a direct paged-cache FlashMLA prefill experiment misread vLLM's
  real `[T, 1, window]` SWA index tensor as top-k width `1`, and after fixing
  that shape handling the current FlashMLA prefill kernel was still slower than
  the materialized bf16 Triton prefill kernel on RTX A5000.
- **`dsv4_int` quant** — INT4 Marlin experts + INT8 AllSpark dense + INT8 IMMA indexer.
- **MHC pipeline-parallel fix** (`models/deepseek_v4/nvidia/model.py`) — DeepSeek-V4 carries
  a 4-tensor head-compression stream `(hidden, residual, post_mix, res_mix)` between
  decoder layers. Upstream only passed `hidden_states` across a PP boundary and ran
  `mhc_post` (the final collapse) on every rank, corrupting the residual stream (PP=N ≠
  PP=1). Now all four cross the boundary via the existing **async** `isend/irecv_tensor_dict`
  path and `mhc_post` runs only on the last rank → PP=N is token-identical to PP=1.
- **`VLLM_RAY_WORKER_IP_ORDER`** (`v1/executor/ray_utils.py`, `envs.py`) — re-ported from
  fork commit `ccf544b3c` (dropped in the rebase). Binds vLLM PP ranks to the chain-index
  IP order injected by the `tb-chain-webhook`. Without it `RayExecutorV2` ranks were
  scrambled vs the per-rank shard materialization (pod ordinal == chain index), so every
  worker loaded the **wrong layers' shards → uninitialized weights → NaN logits**. This
  was the production NaN root cause.

### Deploy

Built via `docker/docker-bake.hcl` + `Dockerfile.ampere-*`; python-only changes ship as a
fast `Dockerfile.ampere-python-hotfix` overlay (no CUDA recompile). Served on the cluster
through the LWS at `appmana-cluster/.../inference/lws-vllm-deepseek-v4.yaml` (GitOps). The
`tb-chain-webhook` injects `NCCL_SOCKET_IFNAME` + `VLLM_RAY_WORKER_IP_ORDER`; the leader/
worker commands materialize each rank's shards by pod ordinal.

**Measured (single-user, server-side Prometheus):** ~**26.7 tok/s** decode, **~175 ms TTFT**
on the int4mse-int8 checkpoint over the 12-node chain.

### sm86 Sparse MLA Kernel Notes

Both stages run the fused tensor-core kernels from
`AppMana/forks-flash-mla-ampere-dsv4` (tag `dsv4-sm86-kernels-2026-07-09`).
At true 16k-slot footprint on an A5000: decode 22.5 us @ top-k 512 (2.10x the
prior kernel, ~9.5x Triton), prefill 3.0-3.3 us/token (1.8x Triton). Both
accept `fp8_ds_mla` and `int8_ds_mla` caches. See that repo's README for the
full tables, the ncu evidence, and the benchmarking caveats.

Earlier revisions of this file recorded "Triton beats FlashMLA prefill ~3.7x".
That was measured on an L2-resident 192-slot cache; at production footprint the
result inverts. Do not re-derive kernel policy from small-cache microbenches.

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
  "cache_type": "fp8_ds_mla"
}
```

Every value is the exact importable symbol that gets activated; the role is
inferred from a registry. Unknown symbols, duplicate roles, and invalid
`cache_type` values fail closed at startup. A `kernels` list is authoritative
for toggles (unlisted = off), which is what makes the int8 indexer independent
of the dense IMMA runtime. `cache_type` sets the default KV dtype only when the
CLI passes `--kv-cache-dtype auto`; an explicit CLI value always wins. The
resolved configuration prints once at startup as `appmana kernels resolved: ...`
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
