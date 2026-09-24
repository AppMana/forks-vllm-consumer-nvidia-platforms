# Merging upstream into this fork

This fork serves DeepSeek V4 on consumer NVIDIA GPUs (RTX 30xx, DGX Spark
GB10) from one image. Upstream vLLM restructures the same code paths every
few weeks, so a merge is a re-port of a known set of invariants into
wherever upstream moved them, followed by build and serving gates that the
unit tests cannot stand in for. This file records the rules, the procedure,
the per-area policy learned from the previous merges, the gates, the
commits worth reading before a merge, and the release gate for images.

## Rules

- One working branch: `appmana/vllm-consumer-nvidia-platforms`. Retired
  lines become `archive/<branch>` tags, never parallel branches.
- Merge, never rebase. Replaying the fork's commits loses every earlier
  resolution and breaks the ancestry other branches depend on.
- Fetch first and merge the upstream tip, not the commit that happens to
  carry the one fix you want; the tip contains it and the sync cost is paid
  once either way.
- Tag the pre-merge tip before merging with an annotated tag named
  `pre-upstream-merge-<YYYY-MM-DD>` and push it. The tag is the merge's first
  parent (`pre-upstream-merge-2026-09-18` is `53873c6c4d^1`) and the baseline
  every gate compares against. Fixes go on top of the merge as separate
  commits with their reasoning in the body; the merge is never reverted.
- Take upstream's structure; re-port the fork's data and invariants into the
  new home. When upstream deletes a fork helper, decide whether it was
  superseded (say by what) or merely dropped (reinstate it). Never resolve a
  test file by picking a side: union the guards.
- Every non-mechanical resolution is named in the merge commit body,
  grouped by the areas below, including what superseded any dropped fork
  code.

## Git settings

Set these once in the fork's checkout. They are what make a merge of
several hundred upstream commits reviewable:

```console
git -C <fork> config merge.renames true
git -C <fork> config merge.directoryRenames true
git -C <fork> config merge.renameLimit 999999
git -C <fork> config diff.renameLimit 999999
git -C <fork> config merge.conflictStyle zdiff3
git -C <fork> config rerere.enabled true
git -C <fork> config rerere.autoUpdate true
```

`merge.directoryRenames` carries the fork's edits into a directory upstream
moved instead of leaving them as added files in the old location. The rename
limits stop Git from giving up on rename detection when upstream touches
thousands of files. `zdiff3` shows the merge base in every conflict hunk,
which is the only way to tell an upstream rewrite from a fork change.
`rerere` replays resolutions recorded in earlier merge attempts; with
`autoUpdate` it also stages them, so read `git rerere diff` before
committing. Run every command with `git -C <fork>` rather than changing
directory.

## Before the merge

Work from a clean tree on the working branch. Record the numbers below in
the merge commit body.

```console
git -C <fork> fetch upstream --prune
git -C <fork> fetch origin --prune
git -C <fork> status --short --branch
git -C <fork> log --oneline --decorate -8
git -C <fork> merge-base upstream/main HEAD
git -C <fork> rev-list --count <merge-base>..upstream/main
git -C <fork> diff --name-only upstream/main HEAD > fork-files.txt
git -C <fork> diff --name-only <merge-base> upstream/main > upstream-files.txt
comm -12 <(sort fork-files.txt) <(sort upstream-files.txt) > overlap.txt
git -C <fork> tag -a pre-upstream-merge-<YYYY-MM-DD> -m "Pre-merge tip before upstream <tip>"
git -C <fork> push origin pre-upstream-merge-<YYYY-MM-DD>
```

`overlap.txt` is the conflict-risk set: every file the fork changed that
upstream also changed since the base, conflicted or not. Read the policy
row for every area it touches before starting.

## Rename rule

A file move and an edit of the moved file never share a commit. When the
fork moves one of its own files, including following upstream to a new
location, commit the move alone first:

```console
git -C <fork> mv vllm/old/path.py vllm/new/path.py
git -C <fork> commit -m "Move path.py to vllm/new"
git -C <fork> show -M --summary HEAD   # must print "rename ... (100%)"
```

Only then edit the file in a separate commit. A 100% rename is what lets
`merge.directoryRenames` and the next upstream merge follow the file.

Two consequences specific to this fork:

- Kernel modules are named by fully qualified name in published
  checkpoints, and a published checkpoint cannot be edited. Moving or
  renaming a module that a `vllm.kernels` entry names keeps the old import
  path importable and adds the old prefix to `_LEGACY_SYMBOL_PREFIXES` in
  `vllm/transformers_utils/configs/dsv4/kernel_config.py`, as
  `e83e939b73` did for `nvidia_sm86` to `nvidia_imma`.
- When upstream moves a file the fork modifies, let the merge carry the
  fork's hunks to the new path. Never resolve a moved file by deleting it
  and re-adding the fork's old copy; that drops upstream's changes and
  breaks rename detection for every later merge.

## Merge and parity checklist

1. Complete "Before the merge" and tag the pre-merge tip.
2. Rebuild the editable venv (gate B) and record the fork test set's
   baseline on the pre-merge tip (gate C) so post-merge failures can be
   classified.
3. `git -C <fork> merge --no-commit --no-ff upstream/main`. Resolve area by
   area, reading both sides of every conflicted or both-sides-changed file
   in full, with the policy table open.
4. For every fork helper upstream deleted, decide superseded (name the
   replacement) or dropped (reinstate it). For every upstream function the
   fork calls or extends, check for added or removed parameters; Marlin
   dropping its act-order arguments broke only the fork's caller
   (`7ebcd12028`).
5. Regenerate derived files only from their sources:
   `tools/generate_versions_json.py` for `docker/versions.json`. Inspect the
   final diff for dependency moves (torch, CUDA, OpenTelemetry, LMCache,
   FlashMLA, SparkInfer pins).
6. Commit with the area-organised body, then run the gates in order. Every
   failure becomes a fix commit on top, red test first where a test can
   reach it.
7. Update the "last merged" line in `README.md` and any row of this file
   that the merge changed.
8. Pass the release gate before any deployment references the new tree.

## Policy by area

| Area | Fork invariant | How previous merges resolved collisions | Guard tests | Symptom of a silent bad merge |
| --- | --- | --- | --- | --- |
| Build system (`docker/Dockerfile`, `docker/versions.json`, `docker/docker-bake.hcl`, `CMakeLists.txt`, `cmake/external_projects/*`, `setup.py`, `pyproject.toml`, `requirements/*`) | One image for sm86 and sm121: the CUDA arch list agrees across Dockerfile, versions.json and bake; CMake keeps `12.1` in the supported archs and gates the high-end externals (deepgemm, fmha_sm100, flashmla, flashkda, qutlass, tml_fa4) with an `add_custom_target` stub for every name setup.py requests; FA3 only when 9.0a is targeted, with `cmake/patches/vllm_flash_attn_arch_gating.patch`; `VLLM_SKIP_FLASH_ATTN_BUILD`; setuptools-scm matches `v[0-9]*` tags only; Ubuntu `BUILD_BASE_IMAGE` with a `BUILD_OS` switch at every package-install site; devel final base; the patched NCCL is the only `libnccl.so.2.*` left before linking; observability pins restored after the KV-connector install; a verification stage that imports the kernel registry and builds the GenAI telemetry handler. | Upstream moved its wheel build to a dnf-based image and the merge dropped `BUILD_OS` (`dnf: not found`). A new external project was added to the CUDA block without a stub (`ninja: unknown target '_flashkda_C'`). The apt NCCL shadowed the fork's build behind the soname. LMCache's PyPI metadata pinned OpenTelemetry down and the GenAI middleware failed on every request. | `tests/test_docker_build_metadata.py`; regenerate `docker/versions.json` with `tools/generate_versions_json.py` after any ARG change. | Build dies two steps in on `dnf`; `ninja: unknown target`; the NCCL stage's own assertion; `GEN_AI_CLIENT_OPERATION_TIME_TO_FIRST_CHUNK` AttributeError at the first request. |
| Native kernels (`csrc/libtorch_stable/*`, `csrc/quantization/gptq_allspark/*`) | Additive ops registered in `ops.h`, `torch_bindings.cpp` and `vllm/_custom_ops.py`: INT8 indexer K cache write (`indexer_k_quant_and_cache` with `scale_fmt="int8"`, exposed as `indexer_k_quant_and_cache_int8`), the Ampere fp8_ds_mla gather, streaming prefill top-k (`top_k_per_row_prefill_candidates` / `merge_candidates`), canonical persistent top-k ordering after both selector paths, AllSpark split-K progress and a graph-safe workspace. | Unions so far. The fork appends its arguments last, so upstream signature changes to `topKPerRowJob` or the indexer cache kernel are the thing to check. | `tests/kernels/test_top_k_per_row.py`, `tests/kernels/attention/test_indexer_streaming_topk.py`, `tests/kernels/test_cache_kernels.py`, `tests/v1/attention/ops/test_v4_kv_cache_torch_fallback.py`, `tests/kernels/quantization/test_allspark_gemm.py`. | Declaration/implementation mismatch at compile; `torch.ops._C has no attribute ...`; `test_registry_symbols_are_importable_callables` red. |
| KV cache layout (`vllm/v1/core/kv_cache_utils.py`, `vllm/v1/kv_cache_interface.py`, `vllm/config/cache.py`, both model runners, `vllm/v1/worker/utils.py`) | `int8_ds_mla` is a cache dtype whose 528-byte page is expressed at the spec construction sites (`state_content_bytes=528, alignment=528`) in `vllm/models/deepseek_v4/attention.py` and `vllm/v1/attention/backends/mla/sparse_swa.py`; every DeepSeek V4 alignment is a multiple of 16 because the fp8 writer uses 16-byte stores; `SlidingWindowMLASpec.merge` forwards `kv_quant_mode` and `alignment`; packed tensors bucket by the group's own `layer_names` and skip groups with no local layers in both the allocator and the memory estimate; `VLLM_KV_BLOCK_ZEROER`; the zeroer is built on demand. | The 16-byte rounding moved from a packed-layout helper into spec alignment when upstream rewrote the layout; the fork's zeroer fix was superseded by upstream's per-segment kernel; upstream's spec-driven bucketing emitted tensors for layers a pipeline rank does not own (`StopIteration` in `allocate_kv_cache` on every rank of a PP=11 deploy). | `tests/v1/core/test_kv_cache_utils.py` (empty projected groups, single-group rank), `tests/v1/core/test_contiguous_kv_packing.py`, `tests/kernels/test_compressor_kv_cache.py`, `tests/v1/attention/ops/test_v4_int8_ds_mla_cache.py`, `tests/v1/core/test_dsv4_packed_swa_spec.py`, `tests/v1/worker/test_dsv4_packed_zeroer_geometry.py`, `tests/models/deepseek_v4/test_kv_block_zeroing_parity.py`, `tests/utils_/test_torch_utils.py`. | `CUDA error: misaligned address` on the first insert; `StopIteration` under PP; NaN after block reuse; `int8_ds_mla is not a valid CacheDType`. |
| DeepSeek V4 model and attention (`vllm/models/deepseek_v4/**`, `vllm/v1/attention/backends/mla/{indexer,sparse_swa}.py`, `vllm/model_executor/layers/{sparse_attn_indexer,mhc}.py`, `vllm/model_executor/kernels/mhc/*`) | Attention runs through the opaque custom op `torch.ops.vllm.deepseek_v4_attention` listed in the compilation config's attention ops and calling upstream's prepare-and-attend entry point; `_run_indexer_op` is shared by the timed path and the eager break; the attention class is chosen by fully qualified name through the checkpoint's kernel registry (`nvidia_imma` for Ampere, `nvidia_sm12x` for GB10); the SparkInfer sparse-MLA backend is in the backend registry and the warmup backend set; the mHC backend is chosen by env and cached at layer init; the indexer carries the INT8 cache, streaming and exact prefill top-k and decode width bucketing; the compressor gates CuTe on `has_cutedsl()`; the vision tower selects its INT8 kernels through the registry and the vision kernels never specialise on the per-image token count; the DSpark model subclass carries PP intermediate tensors, name remapping and the confidence head; fp8e4nv has torch fallbacks on sm86. | Upstream's prepare-and-attend split: the custom op now calls the new function; the fp4 indexer helper superseded the fork's config flag (mxfp4 indexer is no longer selectable on sm86, accepted); the deleted sparse-MLA warmup module's backend set was reinstated in `attention.py`; an early-return dispatch absorbed upstream's new platform branch; renames (`slot_mapping_for_cache`, `compress_ratio`) were adopted; XPU mHC paths were deliberately not ported. | `tests/models/deepseek_v4/test_torch_compile_registration.py`, `tests/models/test_deepseek_v4_kernel_config.py`, `tests/models/deepseek_v4/test_static_name_resolution.py` (pyflakes over the merged trees), `tests/models/deepseek_v4/test_capability_gates.py`, `tests/v1/attention/test_sparse_mla_backends.py`, `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py`, `tests/v1/attention/test_dsv4_int8_indexer_addressing.py`, `tests/kernels/test_mhc_kernels.py`, `tests/kernels/test_fused_indexer_q_rope_quant.py`, `tests/kernels/attention/test_dsv4_vision_int8.py`, `tests/models/deepseek_v4/test_dspark_remap.py`, `tests/models/deepseek_v4/test_rope.py`. | Compiled and eager logits diverge on the one-layer boundary probe (`tests/models/deepseek_v4/dsv4_int_boundary_probe.py`, both must emit token 7685); ImportError from `nvidia_imma/attention.py`; JIT compiles on the first live request; undefined names. |
| Pipeline parallel and executor (`vllm/distributed/parallel_state.py`, `vllm/v1/worker/gpu_worker.py`, `vllm/v1/worker/gpu/{pp_utils,cudagraph_utils,model_runner}.py`, `vllm/v1/executor/ray_utils.py`, `vllm/config/vllm.py`, `vllm/model_executor/model_loader/*`, `vllm/layer_partition.py`) | `isend_tensor_dict` sends metadata synchronously and returns tensor handles only, the reap gates on every handle, and the worker waits for the previous send right before `irecv_tensor_dict`; one PP broadcast carries sampled tokens, the draft block and the accepted/rejected counts, 16-byte aligned; raw input ids survive on ranks whose model requires them (hash router, image sentinels), in normal execution and during graph capture; Ray workers are ordered by `VLLM_RAY_WORKER_IP_ORDER`; DeepSeek V4 architectures stay out of the breakable-cudagraph defaults and nothing forces `CompilationMode.NONE`; the checkpoint's kernel block is applied to the config; per-rank Triton cache directories; the flashpack loader, rank-local weight filtering and partial DSpark shard sets; cost-based layer partitioning. | Upstream's async metadata send and its `handles[1:]` reap were incompatible with the synchronous contract and had to be fixed at both ends, with the comm tests rewritten; upstream's config methods superseded the fork's helper functions but the exclusion data was re-applied; the fork's old PP transport layer was removed on purpose and must not be resurrected. | `tests/distributed/test_comm_ops.py`, `tests/v1/worker/test_gpu_worker.py`, `tests/v1/worker/test_pp_utils.py`, `tests/distributed/test_pp_utils_speculative_broadcast.py`, `tests/distributed/test_multiproc_executor.py`, `tests/v1/cudagraph/test_cudagraph_manager.py`, `tests/config/test_breakable_cudagraph_config.py`, `tests/model_executor/model_loader/test_pp_weight_filter.py`, `tests/model_executor/model_loader/test_flashpack_loader.py`, `tests/test_layer_partition.py`. | A test asserting a metadata handle at index 0 reappears; PP deadlock or long-lived send kernels on deep rings; decode ITL of tens of seconds on GB10; scrambled Ray rank order; `FlashPackModelLoader` missing from the loader table. |
| Speculative decoding and scheduler (`vllm/v1/worker/gpu/spec_decode/**`, `vllm/config/speculative.py`, `vllm/v1/core/sched/*`, `vllm/v1/engine/core.py`) | PP-deferred verification: a steady verify replays the last emitted anchor as row 0 of the same forward as its draft block (`replayed_pp_anchor_req_ids`, the per-request draft combine kernel, `_expand_for_deferred`, async-scheduler bonus placeholders); the broadcast follows `propose()`; `scatter_draft_tokens` on the consume path; DSpark is exempt from the PP block; `defer_block_free` for any multi-batch deployment; per-request draft-slot and confidence budgets; draft ids are not taken while every request is mid-prefill; the survival-probability knob; no embedding sharing under PP; a fully rejected deferred verify reads no stale position. | Upstream's draft-metadata builder gained required parameters; its chunked verify coexists with the fork's deferred path (the fork's is live); upstream's own adaptive-verification field superseded the fork's; the stale-frame guard moved to upstream's output-staleness handling. | `tests/v1/core/test_scheduler.py`, `tests/v1/core/test_async_scheduler.py`, `tests/v1/worker/test_gpu_rejection_sampler_chunking.py`, `tests/v1/worker/test_gpu_model_runner_v2_pp.py`, `tests/v1/spec_decode/test_dspark_sync_debug.py`, `tests/v1/spec_decode/test_rejection_sampler_utils.py`, `tests/test_envs.py`. | Zero acceptance with drafts shifted by one row; alternating tokens; garbage after a full rejection; `TypeError` on the draft metadata builder. |
| Warmup and JIT (`vllm/model_executor/warmup/*`, `vllm/v1/worker/gpu/warmup.py`, `vllm/utils/*`) | Every specialization a served request can reach is compiled before the first request, pipeline ranks agree before any rank-local warmup, the DeepSeek V4 sparse-MLA prefill, block-table, Marlin MoE, native gather and vision warmups run on the ranks that own them, long-prefill synthetic requests are capped at the model length, sampler and streaming top-k warmups exist, mHC is warmed over every decode token count, aux-stream tensors are recorded against both consuming streams. | Upstream superseded three warmups with its own and deleted one module whose backend set had to be reinstated; upstream replaced a `do_not_specialize` decorator with a specialization class, so the guard asserts every decode length maps to a warmed class; runner fakes in tests had to grow the attributes the merged warmup reads. | `tests/v1/worker/test_gpu_warmup.py`, `tests/model_executor/warmup/test_deepseek_v4_*`, `tests/models/deepseek_v4/test_warmup_live_key_parity.py`, `tests/kernels/attention/test_dsv4_chunked_prefill_jit_coverage.py`, `tests/kernels/attention/test_dsv4_indexer_context_specialization.py`, `tests/v1/determinism/test_aux_stream_lifetime.py`. | A first-request stall on one rank (the post-warmup JIT monitor logs it); a startup hang when one rank warms alone; nondeterministic rows under concurrency. |
| Entrypoints, metrics, tracing, envs (`vllm/entrypoints/**`, `vllm/v1/engine/*`, `vllm/v1/metrics/*`, `vllm/tracing/*`, `vllm/tokenizers/deepseek_v4.py`, `vllm/envs.py`) | Startup stages recorded from the API server launcher and served as 503-with-detail by the startup probe; stall detection with `EngineStalledError` deriving from the server error base; scheduled-prefill and per-sequence throughput metrics; the GenAI trace middleware wired through `--middleware`; `reasoning_content` mirrored; the DeepSeek V4 reasoning-effort mapping and tools attached to an existing system message; fork envs registered as compile factors. | Upstream split the API server module; the probe wiring was re-ported to the new launcher and the stall error rebased onto upstream's error hierarchy. | `tests/entrypoints/serve/instrumentator/test_startup_probe.py`, `tests/entrypoints/serve/instrumentator/test_health_payloads.py`, `tests/v1/engine/test_stall_detection.py`, `tests/v1/metrics/test_live_prefill_metrics.py`, `tests/entrypoints/openai/test_genai_trace_middleware.py`, `tests/entrypoints/openai/test_reasoning_content_compat.py`, `tests/entrypoints/openai/chat_completion/test_deepseek_v4_thinking_modes.py`, `tests/tokenizers_/test_deepseek_v4.py`, `tests/test_envs.py`; `tests/utils.py` treats 503 during init as not-yet-ready. | The probe reports nothing; a high reasoning effort emits no reasoning; `reasoning_content` missing; the compile-factor test red. |
| Quantization and MoE (`vllm/model_executor/layers/quantization/*`, `vllm/model_executor/layers/fused_moe/**`, `vllm/model_executor/layers/activation.py`) | `dsv4_int` and `dsv4_mxfp4_int8` are registered quantization methods; INT4 Marlin experts, AllSpark W8A16 dense and INT8 vision live in `dsv4_int.py`; the loader audit knows checkpoint-required parameters; NVFP4 fails closed on per-expert scales; Marlin MoE canonicalises routed-token order with the decode fast path intact; shared experts record their aux-stream input and output against the consuming streams; native clamped SiLU on CUDA. | Upstream moved shared experts to an event design and the two `record_stream` calls were re-applied to it (the contract tests were red without them); `dsv4_int` was added to the compile-supported list. | `tests/quantization/test_dsv4_int.py`, `tests/models/deepseek_v4/test_quant_config_per_module.py`, `tests/kernels/moe/test_moe.py`, `tests/kernels/quantization/test_allspark_splitk_planner.py`, `tests/kernels/quantization/test_fp8_marlin_is_bmm_bypass.py`, `tests/v1/determinism/test_aux_stream_lifetime.py`, `tests/kernels/core/test_activation.py`. | `Unknown quantization method dsv4_int`; ImportError inside `dsv4_int.py`; nondeterministic MoE rows. |
| Shared test files the fork modifies | Each carries a guard from the rows above (the comm-ops contract, the 503 startup handling in `tests/utils.py`, KV packing, scheduler accounting, cudagraph router tokens, sparse-MLA dispatch, top-k order, env compile factors, tokenizer fixtures, import-time skips when optional backends are absent). | Always union; a merge that takes upstream's copy wholesale loses the guard silently. | The files themselves. | A guard that "went green" because it vanished. |
| Fork-only files | `vllm/models/deepseek_v4/nvidia_imma/*`, `nvidia_sm12x/*`, `quantization/dsv4_int.py`, `fused_moe/experts/sparkinfer_moe.py`, `model_loader/flashpack_loader.py`, `entrypoints/serve/instrumentator/startup.py`, `transformers_utils/configs/dsv4/kernel_config.py` (whose registry strings name symbols in `_custom_ops`, the indexer, mHC and both attention packages). | They never conflict; they break when an upstream symbol they import moves. | `tests/models/deepseek_v4/test_static_name_resolution.py`, `tests/models/test_deepseek_v4_kernel_config.py`. | ImportError at model construction. |

## Known traps

- `dnf: not found`: a new upstream package-install site without a
  `BUILD_OS` branch.
- `ninja: error: unknown target '_<name>_C'`: a new
  `cmake/external_projects/<name>.cmake` inside the CUDA block with no
  `add_custom_target` stub for other architectures.
- `error: can't copy '.../_vllm_fa3_C.abi3.so'`: setup.py requested an
  extension the architecture list does not build.
- The NCCL stage assertion: another `libnccl.so.2.*` present when the
  fork's build is linked.
- `GEN_AI_CLIENT_OPERATION_TIME_TO_FIRST_CHUNK` AttributeError: the
  KV-connector install moved OpenTelemetry.
- `StopIteration` from `allocate_kv_cache` on a pipeline rank: packed KV
  tensors bucketed off a group's spec instead of its local layers.
- PP hangs or a comm test expecting a metadata handle: the synchronous
  metadata send contract was lost.
- Tens of seconds per token on GB10: breakable cudagraphs re-enabled for
  DeepSeek V4 or `CompilationMode.NONE` forced.
- Empty responses on the needle benchmark: thinking is on by default; the
  benchmark passes `chat_template_kwargs` explicitly.
- `patch does not apply` during the flash-attention fetch: a previously
  patched checkout under `.deps/`. The patch step checks whether the patch
  is already applied (`git apply --reverse --check`, `262e67e0f5`), so this
  message means the patch itself no longer matches the pinned
  flash-attention tag.
- `deepseek_v4_sparse_mla_attention_warmup` imported again in
  `kernel_warmup.py`: upstream keeps that warmup; the fork removed it because
  it drives `execute_model` from a per-rank gate and deadlocks a pipeline
  chain. Drop the import and the call.
- Two draft-token relays after a merge of `pp_utils.py` and
  `model_runner.py`: upstream broadcasts drafts separately
  (`broadcast_drafts`, `draft_tokens_to_update`) while the fork packs the
  proposed block into the single payload broadcast. Keep the payload
  contract, delete upstream's second broadcast, and keep
  `get_prev_sampled_outputs()` argument-free. Upstream's version of this
  (#56956) also moved the relay after `propose()`, added draft adoption
  arguments to `post_update` and a `warmup_pp_decode_update` for them; the
  fork passes no draft arguments, so that warmup is re-ported to
  `PPHandler.warmup_sampled_outputs()` (the live payload slicing), guarded
  by `test_warmup_pp_decode_update_matches_serving_specialization`.
- `torch._dynamo.exc.Unsupported: Unsupported hasattr call` at startup with
  CUDA graphs: a Triton JIT owner resolved its kernel's argument names for
  the first time inside the model's trace. The fork compiles DeepSeek V4,
  upstream does not; `VllmTritonJitKernel.__init__` resolves them eagerly
  (`31d143c7c5`).
- AllSpark deleted upstream (#58001): the fork's `dsv4_int` dense path calls
  its ops directly, so the kernels, schemas, `_custom_ops` wrappers and
  `allspark_utils.py` are reinstated on every merge that drops them.
- `KeyError: aux_hidden_states_0` or a receive tensor nobody sends under PP:
  upstream's runner-side aux relay (`EagleModelMixin` slot keys,
  `reserve_aux_intermediate_tensor_slots`) counted slots for a model that
  relays its own `aux_hidden_{j}` boundaries in `forward`; the DeepSeek V4
  model pins its slot count to zero.
- `deep_jit/utils/exception.hpp: fatal error: format: No such file or
  directory` while building DeepGEMM for the high-end architecture list:
  the Ubuntu build stage selected a host compiler older than GCC 13.
  DeepGEMM's `deep_jit` needs the C++20 `<format>` header; the CUDA 13
  Ubuntu 24.04 base already ships GCC 13, so the Dockerfile installs the
  distro `gcc`/`g++` and never pins an older one. The sm86-only image does
  not build DeepGEMM and so cannot catch this; the documented
  `build-consumer-platforms.sh` path (`8.6 12.1a`) does.
- `AttributeError: 'DeepseekV32IndexerMetadataBuilder' object has no
  attribute ...` at the first decode step, only from the sm86+sm121 image:
  that image vendors DeepGEMM, so `has_deep_gemm()` is true on every CUDA
  device and the decode build reaches the paged-MQA schedule gate that the
  sm86-only build short-circuits. An attribute upstream renamed in the
  builder's constructor (`use_fp4_indexer_cache` became
  `indexer_uses_fp4`) stays stale in a fork-only branch and no sm86 test,
  venv or Harbor image ever evaluates it. Serve the mini from the
  multi-architecture image as well; a build with more optional extensions
  reaches more branches.
- `RuntimeError: A KV connector reported block-level load failures
  (invalid_block_ids) on a layout with multiple KV cache groups` killing
  the engine on the first failed LMCache retrieve: upstream now refuses
  block-level failure reports on multi-group layouts (DeepSeek V4 has
  several) and expects `KVConnectorTransferResults.failed_recving`. The
  LMCache fork's MP adapter records the failed request ids and its
  connector reports them per request; the fallback class in
  `lmcache_mp_connector.py` does the same. A load failure must lead to a
  recompute, never to a dead engine.
- TileLang `register_warmup` calls in `nvidia/model.py` after a merge: the
  fork's `MHC*Op` objects dispatch by platform and the fork's mHC warmup
  covers them; the TileLang registrations import kernels Ampere never runs.

## Gates, in order

A. Static: byte-compile and import every changed `.py`; then
   `tests/models/deepseek_v4/test_static_name_resolution.py`,
   `tests/models/test_deepseek_v4_kernel_config.py`,
   `tests/models/deepseek_v4/test_torch_compile_registration.py`,
   `tests/config/test_breakable_cudagraph_config.py`,
   `tests/test_docker_build_metadata.py`, `tests/test_envs.py`.
B. Native: `uv pip install -e . --no-build-isolation --no-deps` with the
   deployment architecture list, after reverting or deleting stale
   `.deps/` checkouts and `build/temp*/CMakeCache.txt`.
C. The fork test set, defined as every test file the fork adds or modifies
   (`git diff --name-only upstream/main HEAD -- tests/`), run on both A5000s
   against the recorded pre-merge baseline; the post-merge failure set must
   be a subset of it, and every new failure gets a red-first fix. Tests that
   upstream added for hardware the deployment does not have (DeepSeek V4.1
   NVFP4/MXFP8 cache kernels, TileLang mHC, anything casting to fp8e4nv in
   Triton below SM89) are recorded as not applicable with their error
   signature, not fixed and not skipped in the shared test files.
D. Serving on the mini checkpoints at PP=1 and PP=2 with CUDA graphs,
   prefix caching off (a hit moves the prefill chunk boundary and the mini
   flips near-tied tokens on a 0.1 nat shift): two servers of the same tree
   must record identical token ids for every image count; against the
   pre-merge recordings, every row that differs is checked by top-3
   log-probability margins before a kernel is suspected, since PP=2 with
   graphs and PP=1 eager legitimately sit about 0.1 nat apart. The JIT
   monitor must report no compiles on a second request at a new context
   length or image size. This gate finds defects the test set cannot reach:
   a local import that shadows a module-level name (`de305121ae`), a warmup
   registration that imports a kernel the device cannot load
   (`696fb46fa4`), and a new upstream linear that needs the fork's
   quantization method (`15497cc32b`) all fail only at model construction.
E. The full image build (`docker/Dockerfile`, target `vllm-openai`) for the
   deployment architectures with the KV connector installed, through the
   build script with its caches left on (pinned builder, registry layer
   cache, sccache); its label must name the merge commit and no overlay.
   Then the documented
   multi-architecture build (`docker/build-consumer-platforms.sh`), which
   compiles the externals the deployment image skips, and gate D's PP=1
   recording served from each image: the multi-architecture image reaches
   branches on the deployment device that the deployment image never
   evaluates.
F. Chain acceptance on the deployment: same-day baseline against the
   previous image, needle recall and latency at C=1 through C=8, the vision
   fixture set, concurrency, cache isolation, Redis recovery and the
   per-rank memory floor, as documented in the deployment repository.
   Read the leader's `cuda_graph.py` runtime statistics during the C=8
   rows: every decode forward must be a FULL replay. A `Runtime Mode NONE`
   row at 24 tokens or more means the pinned `cudagraph_capture_sizes`
   (tokens, multiples of the speculative step) no longer reach
   max-num-seqs times the step, and those batches decode at half speed.

## Informative merge and integration commits

Read the body of each before a merge; the body carries the evidence, this
table carries the rule.

| Commit | Lesson |
| --- | --- |
| `9c9d5077ef` | Merge of upstream `f9ad9dd6b4` (218 commits), from `pre-upstream-merge-2026-09-23`. AllSpark reinstated after upstream deleted it; upstream's DSpark PP draft relay folded into the single payload. Gate D then needed `31d143c7c5` (PP=1 CUDA graphs had not started on the pre-merge tree either). |
| `53873c6c4d` | Merge of upstream `729ebac498` (773 commits). The model for a merge body: the conflict count, then resolutions grouped by the policy areas, each naming what superseded any dropped fork code. Its first parent is `pre-upstream-merge-2026-09-18`. |
| `334894183a` | Merge of upstream `e35298628f` (1798 commits). When upstream replaces a subsystem, re-express the fork's invariant in the new structure (the 528-byte `int8_ds_mla` page became spec `state_content_bytes`; 16-byte alignment moved into spec alignment) and say in the body which resolution still needs hardware validation. |
| `883eea2c3c` | Merging back two lines that had each rebased the same shared history: 13 conflicts, and about 90 files that merged cleanly only because both sides replayed identical commits, to find 7 files of new content. Source of the one-branch, merge-never-rebase rules. |
| `7ebcd12028` | Gate C catches upstream signature changes in calls only the fork makes (Marlin dropped its act-order arguments) and renamed fixtures in fork tests. Upstream tests for hardware the deployment lacks are recorded as not applicable, not skipped. |
| `de305121ae`, `696fb46fa4`, `15497cc32b` | Gate D (serving the mini) is where import-time and construction-time defects appear; no unit test constructs the full model. |
| `c2d0444322` | When upstream deletes a warmup module, check whether the kernels it warmed still run. The JIT monitor names every kernel compiled on the first request. |
| `e0463fcd3e` | The multi-architecture build compiles DeepGEMM, which needs GCC 13 for `<format>`. An sm86-only build cannot catch a toolchain regression in a component it does not build. |
| `4f8a9b2f94` | An upstream attribute rename inside a `has_deep_gemm()` branch was only evaluated by the multi-architecture image. Serve the mini from that image on every merge. |
| `6e66a3c681`, `35eedd3557` | Upstream changed the KV connector failure contract for multi-group layouts. Fix the fallback connector here and the LMCache fork, then pin the fork commit in the Dockerfile and regenerate `versions.json`. |
| `eb2f08f109`, `1d8b0eae46` | Model runner v2 sizes every per-row tensor to the padded row count of a FULL graph replay. Any per-row tensor built from the real request count crashes the first padded decode; which batch sizes pad depends on the capture-size list, so test 3, 5, 6 and 7 sequence decodes. |
| `c6a462f90c` | Capture sizes are tokens, not sequences. Confirm coverage from the `cuda_graph.py` runtime statistics, not from the configuration. |
| `96df8994a7` | Mini recordings are compared with prefix caching off and judged by log-probability margins. |
| `b4dac1e335`, `074e224edd` | One sccache daemon per build stage on its own socket, one sccache version in every stage, sccache installed unconditionally so `USE_SCCACHE` never invalidates base layers. |
| `262e67e0f5` | Every patch step over `.deps/` is idempotent, because the source lives in a shared cache mount that survives interrupted builds. |
| `e46526fc9f` | Moving the LMCache pin can change its build requirements (`grpcio-tools`); gate E is where that shows. |
| `5636629b95` | Build each platform separately, then join the two digests into one manifest list; a single-platform tag on a multi-platform deployment is a defect. |

## Release gate

An image is released per commit, never per branch. `<commit10>` is the first
ten hex digits of the commit the build script resolved (it prints
`building ... at <sha>` and stamps the wheel version
`0.0.0+consumer.<commit10>`).

1. Build each platform from the same commit with its caches on:
   - `linux/amd64` on appmana's pinned `buildkitd-vllm` pod:
     `docker/build-consumer-platforms.sh --platform linux/amd64 --tag ghcr.io/appmana/vllm-consumer:sm86-sm121-<commit10>-amd64`
   - `linux/arm64` natively on hilton's BuildKit on a Spark that is not
     serving, with the variables in the script header (`USE_SCCACHE=0`,
     `CACHE_REF=ghcr.io/appmana/vllm-consumer:buildcache-arm64`), tag
     `sm86-sm121-<commit10>-arm64`.
2. Read each pushed digest (`docker buildx imagetools inspect <tag>`) and
   combine the digests, not the tags:
   `docker/merge-consumer-platforms.sh ghcr.io/appmana/vllm-consumer:sm86-sm121-<commit10> <amd64 image>@sha256:... <arm64 image>@sha256:...`.
   The script fails unless both `linux/amd64` and `linux/arm64` are present.
3. Check the labels: `org.opencontainers.image.revision` is the full commit
   and the FlashMLA, SparkInfer, LMCache and NCCL labels match
   `docker/versions.json`. An overlay image is never a release.
4. Serve gate D's PP=1 mini recording from the multi-architecture tag,
   pulled from GHCR by digest, on every merge. The token ids must match the
   recording from the editable venv.
5. The deployment change references the image by digest, and its commit
   body records the digest, the tag and the source commit.
6. Never push to a tag that already exists. A rebuild with different
   content, even from the same commit, gets a new tag name.
