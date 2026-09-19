# Merging upstream into this fork

This fork serves DeepSeek V4 on consumer NVIDIA GPUs (RTX 30xx, DGX Spark
GB10) from one image. Upstream vLLM restructures the same code paths every
few weeks, so a merge is a re-port of a known set of invariants into
wherever upstream moved them, followed by build and serving gates that the
unit tests cannot stand in for. This file records the rules, the procedure,
the per-area policy learned from the previous merges, and the gates.

## Rules

- One working branch: `appmana/vllm-consumer-nvidia-platforms`. Retired
  lines become `archive/<branch>` tags, never parallel branches.
- Merge, never rebase. Replaying the fork's commits loses every earlier
  resolution and breaks the ancestry other branches depend on.
- Fetch first and merge the upstream tip, not the commit that happens to
  carry the one fix you want; the tip contains it and the sync cost is paid
  once either way.
- Tag the pre-merge tip (`pre-upstream-merge-<date>`) before merging. Fixes
  go on top of the merge as separate commits with their reasoning in the
  body; the merge is never reverted.
- Take upstream's structure; re-port the fork's data and invariants into the
  new home. When upstream deletes a fork helper, decide whether it was
  superseded (say by what) or merely dropped (reinstate it). Never resolve a
  test file by picking a side: union the guards.
- Every non-mechanical resolution is named in the merge commit body,
  grouped by the areas below, including what superseded any dropped fork
  code.

## Procedure

1. `git fetch upstream`; record `git merge-base upstream/main HEAD`, both
   tips, and the overlap between the fork's file set
   (`git diff --name-only upstream/main HEAD`) and the files upstream
   changed since the base. The overlap is the conflict-risk set; read the
   policy rows for every area it touches before starting.
2. Rebuild the editable venv and record the fork test set's baseline on the
   pre-merge tip (gate C) so post-merge failures can be classified.
3. `git merge --no-commit upstream/main`. Resolve area by area, reading
   both sides of every conflicted or both-sides-changed file in full.
4. Commit with the area-organised body. Then run the gates in order; every
   failure becomes a fix commit on top.

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
  patched checkout under `.deps/`; reset or delete it.
- `deepseek_v4_sparse_mla_attention_warmup` imported again in
  `kernel_warmup.py`: upstream keeps that warmup; the fork removed it because
  it drives `execute_model` from a per-rank gate and deadlocks a pipeline
  chain. Drop the import and the call.
- Two draft-token relays after a merge of `pp_utils.py` and
  `model_runner.py`: upstream broadcasts drafts separately
  (`broadcast_drafts`, `draft_tokens_to_update`) while the fork packs the
  proposed block into the single payload broadcast. Keep the payload
  contract, delete upstream's second broadcast, and keep
  `get_prev_sampled_outputs()` argument-free.
- `KeyError: aux_hidden_states_0` or a receive tensor nobody sends under PP:
  upstream's runner-side aux relay (`EagleModelMixin` slot keys,
  `reserve_aux_intermediate_tensor_slots`) counted slots for a model that
  relays its own `aux_hidden_{j}` boundaries in `forward`; the DeepSeek V4
  model pins its slot count to zero.
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
   length or image size. Expect this gate to find defects the test set
   cannot: two of the 2026-09 merge's five serving defects were import-time
   shadowing and warmup registrations, visible only at model construction.
E. The full image build (`docker/Dockerfile`, target `vllm-openai`) for the
   deployment architectures with the KV connector installed; its label must
   name the merge commit and no overlay.
F. Chain acceptance on the deployment: same-day baseline against the
   previous image, needle recall and latency at C=1 through C=8, the vision
   fixture set, concurrency, cache isolation, Redis recovery and the
   per-rank memory floor, as documented in the deployment repository.
