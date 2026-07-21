<!-- markdownlint-disable MD001 MD041 -->

# AppMana vLLM: DeepSeek-V4 on Ampere (`appmana/vllm-ampere`)

AppMana's fork of vLLM, serving DeepSeek-V4-Flash on RTX 3090 / A5000 (sm_86,
Ampere) GPUs across a 12-node Thunderbolt pipeline-parallel chain (PP=12,
1 GPU/node). Checkpoint, benchmarks, and deployment config:
[`appmana/deepseek-v4-int4-int8`](https://huggingface.co/appmana/deepseek-v4-int4-int8).

### What this fork adds

- Ampere sparse-MLA decode and prefill kernels (Triton and native CUDA, in
  the sibling repo [`forks-flash-mla-ampere-dsv4`](https://github.com/AppMana/forks-flash-mla-ampere-dsv4)),
  replacing upstream's Hopper-only DeepGEMM/CuTeDSL/FlashMLA path.
- `int8_ds_mla`, a full INT8 KV-cache/indexer/dense-activation path upstream
  doesn't have.
- `dsv4_int` quantization: INT4 Marlin routed experts, INT8 AllSpark dense
  layers, produced by `tools/ampere/dsv4_requant_checkpoint.py`.
- A checkpoint-driven kernel-selection config block (`vllm` key in
  `config.json`, see below) instead of environment variables.
- A pipeline-parallel fix for DeepSeek-V4's 4-tensor head-compression stream,
  which upstream truncated to one tensor and ran redundantly on every rank.
- Streaming (activation-chunked) prefill top-k for the sparse-MLA indexer,
  bounding memory instead of letting it scale with context.
- A zero-target-layer last PP rank (only the speculative-decoding draft
  stages), freeing the VRAM a 1M-token KV cache needs.
- In-kernel INT8 dequantization for sparse prefill, replacing a whole-KV-pool
  bf16 scratch buffer that scaled with pool size instead of the tokens
  actually read.
- Scheduler-derived indexer prefill workspace and context-bounded decode
  scratch, replacing allocations sized by `max_model_len` regardless of
  actual usage.
- A fail-fast executor: a worker exception with no reply path now kills the
  worker immediately instead of silently desyncing the pipeline.
- An early HTTP health/metrics surface that reports engine boot stage and a
  stall detector, instead of no listener at all until the engine is ready.
- `VLLM_RAY_WORKER_IP_ORDER`, binding PP ranks to the Thunderbolt chain's
  index order (upstream's rank assignment loaded the wrong shards on this
  topology).
- Warmup coverage for sparse-prefill, MHC, and fused-decode kernels, so no
  rank JIT-compiles mid-serving and deadlocks the chain.
- An Ampere image pipeline (`docker/Dockerfile.ampere-*`,
  `tools/ampere/build_vllm_ampere_image.sh`).

### Deploy

Built via `docker/docker-bake.hcl` + `Dockerfile.ampere-*`; python-only changes ship as a
fast `Dockerfile.ampere-python-hotfix` overlay (no CUDA recompile). Served on the cluster
through the LWS at `appmana-cluster/.../inference/lws-vllm-deepseek-v4.yaml` (GitOps). The
`tb-chain-webhook` injects `NCCL_SOCKET_IFNAME` + `VLLM_RAY_WORKER_IP_ORDER`; the leader/
worker commands materialize each rank's shards by pod ordinal.

Benchmarks are on the [checkpoint page](https://huggingface.co/appmana/deepseek-v4-int4-int8).

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
resolved config prints once as `vllm kernels resolved: ...`. `--hf-overrides`
replaces the whole block rather than merging it, so build overrides from the
checkpoint's own block plus whatever you're adding, not a hand-curated list.

Add `vllm.model_executor.layers.sparse_attn_indexer.streaming_prefill_topk`
to `kernels` to enable streaming indexer top-k; off by default, required
above roughly 700k context.

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
