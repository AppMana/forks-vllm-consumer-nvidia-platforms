# Bringing a new model to the consumer platforms

This fork serves models on RTX 30xx (sm_86) and DGX Spark GB10 (sm_121) from
one image. A model or a new checkpoint variant arrives as four things that
have to agree: converted weights, a `quantization_config` that names the
runtime quantization method, a `vllm` block that names every fork kernel the
checkpoint needs, and a pipeline-parallel plan that fits the target ranks.
This page gives the procedure and the rules, with the DeepSeek V4 language
model and the DeepSeek V4 vision tower as the worked examples. Commit hashes
point at the change that established each rule; the evidence is in their
bodies.

Related: [`MERGING.md`](../../MERGING.md) for upstream merges and the
release gate, [`triton-to-cuda.md`](triton-to-cuda.md) for writing and
compiling kernels, `tools/ampere/DSV4_VISION.md` for the vision conversion
command.

## Rules

- Quantization is ahead of time. The converter writes final integer weights
  and scales; the runtime only loads them, repacks them into the kernel's
  layout and runs. No runtime rotation, requantization or weight transform
  (`vllm/model_executor/layers/quantization/dsv4_int.py` module docstring).
  A tensor whose runtime consumer is floating point stays floating point: the
  DSV4 `wo_a` projection feeds a BF16 inverse-RoPE einsum, so the converter
  dequantizes its FP8 source straight to BF16 and emits no scale.
- Kernel selection lives in the checkpoint's `vllm` block, never in
  environment variables. The block travels inside the model config to every
  Ray worker and every pipeline rank by construction
  (`Dsv4IntConfig.__setstate__` re-activates it after unpickling); an
  environment variable set on the leader does not. `a08c8cc12a` replaced the
  scattered flags and env fallbacks with the block, and `8d2c80b383` removed
  the compatibility paths.
- Everything fails closed. An unknown tensor stops the conversion, an
  unknown kernel symbol stops the server, and a dispatch arm that does not
  recognise its symbol raises instead of falling through to another kernel
  (`48aa4418f7`).
- One supported conversion source per checkpoint family. The DSV4 converter
  accepts `deepseek-ai/DeepSeek-V4-Flash-0731` and refuses a source with a
  single MTP stage (`c100461c04`); a source it does not know produces an
  error, not plausible weights.
- Iterate locally on the two A5000s with a mini checkpoint before the chain
  or the Sparks see anything.

## 1. Audit the source checkpoint

`tools/ampere/dsv4_checkpoint_audit.py` classifies every tensor by name and
container dtype into a role and an action (preserve, requantize) and writes a
manifest:

```bash
.venv/bin/python tools/ampere/dsv4_checkpoint_audit.py \
  --checkpoint /path/to/source --out manifest.json --fail-on-unknown
```

Rules for the classifier when a model adds tensors:

- Classify by container dtype as well as by name. A routed-expert weight is
  MXFP4 only if it is packed `I8`, NVFP4 only if it is `U8`; any other dtype
  under that name is `unknown`. Before `c100461c04` an FP8 expert weight fell
  through to the MXFP4 role and was unpacked as INT4 nibbles, which produces
  plausible garbage rather than an error.
- Count draft stages from the tensors (`mtp.<N>.` prefixes), never from
  `num_nextn_predict_layers`; Flash-0731 declares 1 and ships 3
  (`cf27b5fba0`).
- A new tensor family gets a named role. Widening a regex until the audit is
  quiet defeats it.

The converter imports the same `classify_tensor`, so an unknown tensor stops
the conversion with `unknown tensor in <shard>: <name> <dtype>` and an FP8
weight without its scale stops it with `missing scales for`.

## 2. Convert

`tools/ampere/dsv4_requant_checkpoint.py` writes the converted shards, the
rebuilt index and the new `config.json` (quantization config plus `vllm`
block). For the DSV4 language model:

```bash
.venv/bin/python tools/ampere/dsv4_requant_checkpoint.py \
  --src /path/to/DeepSeek-V4-Flash-0731 --dst /path/to/int4-int8 \
  --device cuda:0 --expert-format int4 --expert-int4-scale-mode mse \
  --dense-int8-strategy channel
```

| Tensor family | Source | Converted | Runtime |
| --- | --- | --- | --- |
| Routed experts `w1`, `w2`, `w3` | MXFP4 (`I8` packed, E8M0 group scales) | symmetric INT4, group 32, MSE-searched scales | Marlin, W4A8 when `marlin_act_int8_process_scales` is listed |
| Attention, indexer, compressor and shared-expert linears, `mtp.*.e_proj`, `h_proj`, `main_proj` | FP8 128x128 blocks | channelwise biased UINT8 (`--dense-int8-strategy channel`) or INT8 128x128 blocks | AllSpark W8A16 where the shape is supported, one-time BF16 dequantization otherwise |
| `attn.wo_a` | FP8 | BF16, no scale | BF16 inverse-RoPE einsum |
| Vision patch projection, QKV, output, MLP, aligner | BF16 | signed INT8 weights, FP32 scales per 32 input channels (`--vision-format int8-imma`) | Triton W8A8 IMMA with dynamic group-32 activations |
| Norms, router gates, mHC parameters, embeddings, head | BF16/FP32 | unchanged | unchanged |

Rules:

- `--expert-int4-scale-mode mse` is the default: a per-group scale search
  that costs build time only, with the same on-disk layout and kernels as
  `absmax7` (`cf27b5fba0`). `absmax7` maps MXFP4's 0.5 and 1.0 levels to one
  INT4 code.
- The converter preserves tensor names and each shard's pipeline-rank
  ownership, adds scale tensors and materialized draft tensors to the index,
  and keeps shards within a 4 GiB tensor-data budget without mixing owners.
- The `vllm` block the converter writes must be byte-identical to the block
  on the serving revision of the published checkpoint. Toggle roles are off
  when unlisted, so a converter that drops `streaming_prefill_topk` silently
  turns the long-context indexer path off on every rebuilt revision.
- `--vision-format auto` picks IMMA for a vision checkpoint with INT4 experts;
  `--vision-format bf16` builds the accuracy reference with the same language
  quantization.

### Mini checkpoints

The full checkpoint does not fit on the local A5000s. The converter's
testbed options shrink the model while keeping every per-token GEMM shape:
`--keep-layers N` keeps layers `0..N-1`, `--keep-experts E` keeps experts
`0..E-1` per layer and slices the router gate to match (it refuses a count
that cannot hold every hash-routing slot), and `--drop-mtp` removes the draft
stages. Kernels, cache layouts and per-token cost stay representative; model
quality does not, so a mini is compared against its own recordings, never
against the full model's answers.

## 3. The `vllm` block and the kernel registry

The block sits at the top level of `config.json`. The DSV4 INT4/INT8
checkpoint carries:

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

`vllm/transformers_utils/configs/dsv4/kernel_config.py` resolves it
(`a08c8cc12a`, renamed from the `appmana` block in `5b00f02a35`):

- Allowed keys: `kernels`, `cache_type`, `indexer_prefill_topk_slab_rows`.
  Anything else is a startup error.
- Every `kernels` entry is the importable fully qualified name of the most
  salient callable it activates. `KERNEL_REGISTRY` maps it to a role.
- Selector roles always have exactly one implementation and fall back to
  `SELECTOR_ROLE_DEFAULTS` when unlisted: `sparse_mla_decode_fp8`,
  `sparse_mla_decode_int8`, `sparse_mla_prefill`, `mhc`.
- Toggle roles are on only when listed, whenever the block has a `kernels`
  list: `indexer_cache_int8`, `indexer_query_int8`,
  `dense_experts_int8_activation`, `indexer_streaming_topk_prefill`,
  `vision_linear_int8`, `vision_attention_int8`.
- Fail closed: an unknown symbol, two symbols for one role, an unknown key,
  a `cache_type` that is not a `CacheDType`, the INT8 indexer query without
  the INT8 indexer cache, the dense INT8 activation symbol without INT4
  expert and INT8 dense weight groups, a `vision_w8a8` weight group without
  `vision_linear_int8` (or the reverse), and a vision group size other than
  32 all stop startup.
- `cache_type` becomes the default `--kv-cache-dtype`; an explicit CLI value
  still wins.
- A checkpoint without a block (the official upstream ones) gets defaults
  from its own dtypes and the device (`blockless_role_defaults`).
- `--hf-overrides '{"vllm": {...}}'` replaces the whole block; dict overrides
  are not merged. Use it for controlled A/B runs, not for serving.

The startup log prints one line that states every active role and the cache
dtype:

```text
vllm kernels resolved: sparse_mla_decode_fp8=... sparse_mla_decode_int8=flash_mla.sparse_mla_decode_int8 ...
```

This line is the validity check for every benchmark and acceptance run. It
must name the kernel that launches, which is why the dispatch returns the
effective symbol (`48aa4418f7`) and why `flash_mla.sparse_mla_prefill_int8`
is its own registry entry (`e83e939b73`).

### Adding a kernel role or symbol

1. Add the FQN constant and its `KERNEL_REGISTRY` entry. A new role goes in
   `SELECTOR_ROLE_DEFAULTS` (selector) or `TOGGLE_ROLES` (toggle), and in
   `_PROOF_ROLE_ORDER` so the proof line prints it.
2. Add cross-role validation for any dependency between roles.
3. Dispatch on the resolved symbol with an `else` arm that raises.
4. Add the symbol to the converter's `_write_config` for the checkpoints that
   need it.
5. Extend `tests/models/test_deepseek_v4_kernel_config.py`, whose
   `test_registry_symbols_are_importable_callables` imports every registry
   symbol, and keep `tests/models/deepseek_v4/test_static_name_resolution.py`
   green.
6. A published checkpoint names kernels by FQN and cannot be edited. A module
   rename keeps the old path importable and maps the old prefix in
   `_LEGACY_SYMBOL_PREFIXES` (`e83e939b73`).

## 4. The quantization method

`dsv4_int` (and the comparison method `dsv4_mxfp4_int8`) are registered
quantization methods; `quantization_config.quant_method` selects them.
`Dsv4IntConfig.get_quant_method` routes by layer type and prefix:

- `RoutedExperts` to `Dsv4Int4MoEMethod` (INT4 repacked for Marlin);
- `LinearBase` under `vision.` or `aligner.` with a `vision_w8a8` group to
  `VisionInt8LinearMethod`;
- `LinearBase` whose prefix matches `INT8_PARENT_PATTERNS` to
  `Dsv4Int8LinearMethod`;
- every other linear to `UnquantizedLinearMethod`, attention to the KV-cache
  method.

When the model gains a linear whose checkpoint weights are INT8, its prefix
goes into `INT8_PARENT_PATTERNS`, or loading fails with
`KeyError: ...weight_scale_inv` because the scale has no parameter to land in.
Upstream's stacked DSpark `context_wkv_proj` is the example (`15497cc32b`).
The loader must also know every checkpoint-required parameter; guard tests
are `tests/quantization/test_dsv4_int.py` and
`tests/models/deepseek_v4/test_quant_config_per_module.py`.

## 5. Pipeline-parallel planning

`vllm/layer_partition.py` solves a contiguous layer split by cost
(`ef6270e519`): every decoder layer costs 1.0, rank 0 starts at the
embedding's cost, the last rank starts at the draft block's cost, and layers
go one at a time to the cheapest rank. The draft block is detected and
weighed from the checkpoint's `mtp.<stage>.*` tensors, not from
`num_nextn_predict_layers`.

```bash
.venv/bin/python -m vllm.layer_partition partition \
  --config /path/to/config.json --index /path/to/model.safetensors.index.json \
  --pp-size 11
.venv/bin/python -m vllm.layer_partition partition \
  --config /path/to/config.json --pp-size 11 --memory-profile profile.json
```

The memory profile gives `layer_bytes`, `rank_overhead_bytes` and
`max_rank_bytes`; runtime reservations (KV cache, graphs, encoder
workspaces) go into the overhead or the budget. Serve the answer through
`VLLM_PP_LAYER_PARTITION`.

Rules:

- Solve again whenever the pipeline size changes. A partition string sums to
  the layer count at any size, so an 11-rank job accepted a 12-rank answer
  with its trailing zero removed and the last rank ran out of memory in
  `load_model` (`ef6270e519`).
- Accept a partition only after measuring peak memory with the intended
  image count, concurrency, KV allocation and graph configuration.
  Checkpoint bytes alone miss runtime packing, shared draft parameters and
  encoder workspaces.
- Ownership: the first rank owns anything that runs before the first decoder
  layer (the vision tower, aligner and learned image vectors); the last rank
  owns the output head and the draft stages. Rank-local weight filtering
  (`vllm/model_executor/model_loader/pp_weight_filter.py`,
  `tools/prep_pp_shards.py`) follows the same split.
- Token ids a later rank needs survive on every rank. The DSV4 hash router
  and the vision checkpoint's image router biases read raw input ids, so raw
  image sentinel ids reach every decoder rank.

## 6. Runtime requirements

- Model runner v2 and asynchronous scheduling. DSV4 and DSpark run on model
  runner v2 (`examples/deployment/AGENTS.md`); a startup warning
  `Model Runner V2 does not yet support ...` means the engine fell back to
  v1 and the run is invalid. Benchmarks and serving use `--async-scheduling`
  and CUDA graphs; eager mode is for diagnosis.
- FULL graph padding. Under a FULL cudagraph replay every per-row tensor is
  sized to the padded row count. A per-row tensor built from the real request
  count crashes the first decode whose sequence count pads up to a capture
  size (`eb2f08f109` for the target, `1d8b0eae46` for the draft). Test
  decodes at 3, 5, 6 and 7 sequences.
- Attention whose behaviour depends on runtime metadata is an opaque custom
  op. Dynamo otherwise traces the memory-profile dummy run, which has no
  metadata, and bakes that branch into the one compiled artifact
  (`3ebdf5cac9`). DSV4 attention is `torch.ops.vllm.deepseek_v4_attention`;
  a new model's equivalent is registered the same way and listed in the
  compilation config's attention ops.
- Standard piecewise CUDA graphs. Keep the architecture out of
  `DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES` (`b40ac1b6e0`) and never force
  `CompilationMode.NONE` (`bf753bb3e9`); both turned GB10 decode into tens of
  seconds per token.
- Compile-cache identity. Every environment variable that changes traced
  code is registered in `environment_variables` in `vllm/envs.py`, whose
  getter default matches the default in the consuming code, so
  `compile_factors()` hashes it; a variable that cannot change a graph goes
  into `ignored_factors`. `tests/test_envs.py` asserts the fork's variables
  (`bf753bb3e9`, `3ebdf5cac9`). An unregistered variable lets a cached
  artifact built for one code path serve another.
- No first-request compilation. Any size that varies per request is a
  runtime argument with `do_not_specialize`, never `tl.constexpr`, and a
  rank-local warmup compiles every reachable variant on the rank that owns
  the module (`5447d1523c`, `034c3092c9`). Pipeline ranks agree before any
  rank-local warmup; a rank compiling alone while the others wait on a
  collective wedges the chain.
- Compiler caches are persistent and rank-local (Triton, TorchInductor and
  the rest), never under `/tmp` in Kubernetes
  (`examples/deployment/AGENTS.md`).

## 7. Gates

Run them in order; a later gate never stands in for an earlier one.

1. Unit and kernel tests on the A5000s: the registry and quantization-config
   tests above, the parity tests for every kernel the checkpoint lists, and
   `tests/test_envs.py`. New failures get a red test first.
2. One-layer boundary probe on the real checkpoint:
   `tests/models/deepseek_v4/dsv4_int_boundary_probe.py` builds a
   symlink view with the embedding, one decoder layer and the head, and
   captures module-boundary tensors. Eager and compiled must emit the same
   token (7685 on `appmana/deepseek-v4-int4-int8` for "The capital of France
   is"); a difference is a compile or dispatch defect, not a numerics
   question (`bf753bb3e9`, `3ebdf5cac9`).
3. Mini checkpoint served locally at PP=1 and PP=2 with CUDA graphs and
   prefix caching off (`--no-enable-prefix-caching`; a prefix hit moves the
   prefill chunk boundary and flips near-tied tokens). Two servers of the
   same tree must record identical token ids; keep the recording as the
   reference for later changes and judge any differing row by its top-3
   log-probability margins (`96df8994a7`). The startup log shows the expected
   proof line, and no kernel compiles on a second request at a new context
   length or image size.
4. The full checkpoint against its reference variant on a fixed fixture set,
   for example the INT8 vision tower against the BF16 vision tower built with
   the same language quantization.
5. The deployment: the RTX 3090 pipeline chain and the GB10 pair, in the
   order `examples/deployment/AGENTS.md` gives. No throughput number is
   published from a run that fails output correctness.
6. Publish the checkpoint revision and add its row, with the revision hash,
   to the supported-checkpoint table in `README.md`.

## Worked example: DeepSeek V4 language model

- Formats: the table in section 2. `quant_method = dsv4_int`,
  `cache_type = int8_ds_mla` (512 INT8 bytes, an FP32 row scale and padding,
  528 bytes per token).
- Kernels: FlashMLA INT8 sparse decode and prefill for attention; the INT8
  indexer K-cache writer and fused INT8 query for integer-MMA indexer
  scoring; Marlin W4A8 for routed experts; streaming prefill top-k for long
  prompts. Each is one line of the block in section 3.
- Plan: 43 layers plus three native DSpark stages. PP=11 or PP=12 over RTX
  3090s with the draft block on the last rank, TP=2 over two GB10s.
- Runtime: model runner v2, asynchronous scheduling, DSpark via
  `--speculative-config '{"method":"dspark",...}'`, piecewise CUDA graphs.

## Worked example: the vision tower

`e361bb8cfa` added the INT8 vision tower as a checkpoint variant, and it
touched every layer of this page:

- Audit: the vision linears became a named role
  (`vision.patch_embed.proj`, `vision.blocks.*.attn.{wqkv,wo}`,
  `vision.blocks.*.mlp.w{1,2}`, `aligner.w{1,2}`); everything else in the
  tower is preserved.
- Conversion: `--vision-format int8-imma` writes signed INT8 weights with
  FP32 scales per 32 input channels and adds a `vision_w8a8` group to
  `quantization_config` (dynamic symmetric group-32 activations).
- Block: two toggle symbols,
  `vllm.models.deepseek_v4.common.vision_int8.VisionInt8LinearMethod` and
  `vllm.models.deepseek_v4.common.vision_int8.vision_attention_int8`.
  `Dsv4IntConfig` refuses the weight group without the linear symbol and
  any group size other than 32.
- Quantization method: `get_quant_method` routes `vision.` and `aligner.`
  linears to `VisionInt8LinearMethod` when the group is present.
- Plan: rank 0 owns the tower, aligner and learned image vectors; raw image
  sentinel ids reach every decoder rank for the image router biases;
  `layer_partition` gained the memory-profile solver. The served partition
  for 43 layers on PP=11 is `4,4,4,4,4,4,4,4,5,5,1`.
- Runtime: the vision kernels originally took the per-image token count as
  `tl.constexpr` and compiled once per image size; `5447d1523c` made the
  counts runtime arguments with `do_not_specialize` and added a warmup that
  runs one small synthetic image on the owning rank.
- Tests: `tests/kernels/attention/test_dsv4_vision_int8.py` checks both
  kernels against BF16 references at the real shapes, asserts the compiled
  PTX contains the `s32.s8.s8.s32` integer MMA, and counts compiled variants
  across token counts.
- Gates: the fixture set compares the INT8 tower against the BF16 tower;
  cache isolation is tested with different images behind identical text,
  since full-image identities and placeholder positions must be part of the
  external KV cache key.
