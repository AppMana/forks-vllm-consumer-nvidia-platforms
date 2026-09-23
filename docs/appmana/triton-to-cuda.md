# Translating Triton kernels to native CUDA

Most fork kernels start as Triton: one source runs on sm_86 and sm_121,
compiles on first use and is easy to change. Some move to native CUDA,
either in this repository (`csrc/libtorch_stable/`) or in the FlashMLA fork
([`AppMana/forks-flash-mla-int`](https://github.com/AppMana/forks-flash-mla-int)).
This page is the procedure for that move, the integer tensor-core rules the
kernels follow, and how to compile everything for both platforms. Commit
hashes point at the change that established each rule.

Reference translations:

| Commit | Translation |
| --- | --- |
| `5df26d2957` | Streaming prefill top-k candidate selection and merge moved to native CUDA (`top_k_per_row_prefill_candidates`, `top_k_per_row_merge_candidates` in `csrc/libtorch_stable/sampler.cu`). |
| `9d05f8700e` | The sm_86 `fp8_ds_mla` prefill K-cache gather moved from a pure Torch fallback to a CUDA op. |
| `2c351807ff` | Sparse-MLA decode on sm_86 dispatched to the precompiled FlashMLA CUDA kernel in one launch instead of a per-row Triton loop. |
| `7ac6fa49e9` | `int8_ds_mla` with native FlashMLA decode and prefill on sm_86, parity-tested and made selectable; the Triton decode stayed the default. |
| FlashMLA `eb855ae` | Native `int8_ds_mla` sparse decode entry point through the selection-scratch dequantization pre-pass. |
| FlashMLA `5966fcc` | INT8 prefill dequantizes in-kernel and drops the whole-cache BF16 buffer. |

## Where a kernel lives

- `csrc/libtorch_stable/` for small ops tied to vLLM's own layouts: cache
  writers and gathers, top-k selection, fused quantize-and-insert.
- The FlashMLA fork for sparse-MLA attention. It ships `cp39-abi3`
  torch-stable wheels, so one wheel covers every torch the image uses.
- SparkInfer is a GB10-only CuTe DSL library. It is a consumer here, not a
  place to put translated kernels.

## Rules

1. Measure first. Translate a kernel only when a profile at serving shapes
   puts it on the critical path, or when its fallback is not a kernel at all
   (`9d05f8700e` replaced a Torch loop).
2. The Triton kernel stays. It remains the registered default for its role
   and the parity oracle for the native one until the native kernel has
   passed every gate, and it stays importable afterwards. The FlashMLA
   package keeps `sparse_mla_decode_int8_triton` next to the native entry
   points for this reason, and `SELECTOR_ROLE_DEFAULTS` still maps the INT8
   decode role to it.
3. Selection is by the checkpoint's `vllm` block, not by an environment
   variable. `2c351807ff` gated the native decode with
   `APPMANA_DSV4_FLASH_MLA_DECODE`; `a08c8cc12a` replaced that with registry
   symbols because an environment variable does not reliably reach every
   Ray worker.
4. The native kernel reads the same memory the Triton kernel reads, through
   the same views. The FlashMLA INT8 entry points take the cache and its
   scales as separate tensors addressed only by runtime strides, so the
   interleaved 528-byte `int8_ds_mla` token row works without a copy.
5. No per-request specialization in either version. A size that varies per
   request is a runtime argument; in Triton that means `do_not_specialize`
   (`034c3092c9`, `5447d1523c`). Precompiled CUDA has no JIT, which is one
   reason to translate (`2c351807ff`).

## Procedure

### 1. Add the native op

For an op in this repository:

1. Declare it in `csrc/libtorch_stable/ops.h`.
2. Implement it in a `.cu` file listed in `VLLM_STABLE_EXT_SRC` in
   `CMakeLists.txt`, or give it its own per-kernel arch list (see
   "Compiling").
3. Register the schema and implementation in
   `csrc/libtorch_stable/torch_bindings.cpp`: an `ops.def("<name>(...) ->
   ()")` with `Tensor!` on every mutated argument, and
   `ops.impl("<name>", TORCH_BOX(&<name>))`.
4. Add a Python wrapper to `vllm/_custom_ops.py` that calls
   `torch.ops._C.<name>` (or the cache-ops namespace for cache kernels).
   Caller-owned outputs are passed in, never allocated inside the op, so the
   op is CUDA-graph safe.
5. When the fork extends an upstream op, its new arguments go last. Upstream
   changes those signatures during merges, and appended arguments are the
   only ones that do not collide (`MERGING.md`, native kernels row).

For an attention kernel in the FlashMLA fork: a `csrc/*_sm80.cu` kernel, a
binding in `csrc/flash_api.cpp` registered under `torch.ops.flash_mla`, and a
Python entry point exported from `flash_mla`. Its tests compare against an
fp32 oracle (`tests/`), and the vLLM image's verification stage asserts the
entry point exists.

### 2. Expose it as a registry symbol

The symbol is the importable fully qualified name of the callable the layer
invokes, for example `flash_mla.sparse_mla_decode_int8`. Add it to
`KERNEL_REGISTRY` under the same role as the Triton symbol, dispatch on the
resolved symbol with an `else` arm that raises (`48aa4418f7`), and make sure
the startup proof line (`vllm kernels resolved: ...`) names the kernel that
actually launches. See [`new-models.md`](new-models.md) for the registry
rules.

### 3. Parity test, red first

Test the native kernel against an fp32 or exact reference and against the
Triton kernel, on the production views and shapes, including adversarial
index patterns (`-1` and out-of-range indices, ragged lengths, two index
streams). The tolerances in use:

| Kernel family | Test | Bound |
| --- | --- | --- |
| Sparse MLA decode and prefill (FlashMLA) | `tests/v1/attention/test_sm86_flash_mla_decode_parity.py` | `cos_diff < 8e-5` against the fp32 oracle on the 528-byte strided views, plus `assert_close(rtol=2e-2, atol=2e-2)`; native against Triton `cos_diff < 8e-5` |
| Sparse MLA INT8, native against Triton (FlashMLA fork) | `tests/test_sparse_mla_decode_int8_adversarial.py` | `2e-4`, because the Triton kernel also quantizes Q and each is within `8e-5` of the oracle independently (`eb855ae`) |
| Cache gather | `tests/v1/attention/ops/test_v4_kv_cache_torch_fallback.py` | bit exact, `rtol=0, atol=0`, against the Torch reference |
| Streaming prefill top-k | `tests/kernels/attention/test_indexer_streaming_topk.py` | logits `rtol=2e-5, atol=2e-3` against a Torch eager reference; selected indices exactly equal, and tie order unchanged across slab sizes |
| Vision INT8 linear and attention (Triton IMMA) | `tests/kernels/attention/test_dsv4_vision_int8.py` | relative error below 0.025 (linear) and 0.04 (attention) against BF16; the compiled PTX contains `.s32.s8.s8.s32` |

A selection or data-movement kernel is exact. An arithmetic kernel's bound
comes from the reference suite it replaces, never from what the new kernel
happens to achieve.

### 4. End-to-end token identity

Serve real weights at temperature 0 with the Triton symbol, then with the
native symbol through `--hf-overrides`, and compare token ids. They must be
identical. `7ac6fa49e9` compared 24 generated tokens on a four-layer real
checkpoint and the token ids of a 16k-token chunked prefill
(`tools/ampere/dsv4_int_vllm_smoke.py`, `tools/ampere/dsv4_int8kv_repro.py`).
Then run the one-layer boundary probe and the mini checkpoint gate from
[`new-models.md`](new-models.md).

### 5. Microbenchmark

Time both kernels in one harness, at every shape the server runs, with
parity checked in the same harness so a wrong kernel never reports a
number. The FlashMLA fork's `benchmarks/bench_int8_sparse_mla.py` times
native INT8, Triton INT8 and native fp8 and prints `cos_diff` against fp32.
The result belongs in the commit body. `7ac6fa49e9` recorded native INT8
decode at 22.8 us and 27.9 us for top-k 512 and 1024 against 48.3 us and
89.5 us for Triton at T=1 on an RTX A5000. `eb855ae` also shows why every
shape matters: at T=4 and top-k 512 the native decode was slower than
Triton (53.6 us against 48.3 us).

### 6. Flip the default in the checkpoint block

Switch the checkpoint's `vllm.kernels` entry to the native symbol, in the
converter's `_write_config` and in a new revision of the published
checkpoint. The registry's selector default changes only when every
checkpoint that relies on the default has been checked. Keep the Triton
kernel and its parity test.

## Integer tensor-core rules

- Scales factor out of the contraction. With a per-row query scale and a
  per-token key scale, the dot runs as `s8 x s8 -> s32` and one outer-product
  multiply by `q_scale[row] * k_scale[col]` after the dot converts it to
  float (FlashMLA `docs/int8_fused_sm86_design.md`, "Scale factoring";
  `tools/int8_triton_mla_decode.py`). With group scales the dot runs per
  group and each group's `s32` result is scaled into an FP32 accumulator
  (`_vision_linear_int8` in `vllm/models/deepseek_v4/common/vision_int8.py`,
  group 32).
- Accumulate in INT32 inside a dot and in FP32 across dots, groups and tiles;
  softmax, normalisation, RoPE and residuals stay floating point
  (`tools/ampere/DSV4_VISION.md`).
- Quantize with round half away from zero, not truncation; truncation
  rounds negatives the wrong way and biases the result
  (`tools/int8_triton_mla_decode.py`). Symmetric codes clamp to plus or minus
  127.
- Prove the instruction. Assert the compiled PTX contains the integer MMA
  (`.s32.s8.s8.s32`), as the vision test does; FlashMLA carries
  `debug_imma_m16n8k32_s8s8` as a fragment probe.
- Integer MMA pays where both operands are INT8 and the contraction is
  compute bound: indexer logits (INT8 query against the INT8 indexer K cache)
  and the vision projections and attention.
- It does not pay where the kernel is bound by gathering rows. Sparse-MLA
  prefill is limited by gather latency at a few percent of bandwidth; a
  tensor-core variant tied at best and ran about 2x slower when its shared
  memory cut occupancy (FlashMLA `5a9c44c`). The FlashMLA INT8 kernels
  therefore do BF16 tensor-core math and convert INT8 rows on the way into
  the MMA ring; "int8" names the cache format, not the arithmetic (FlashMLA
  `8a76699`). The INT8 cache buys memory: 528 bytes per token against 584
  for `fp8_ds_mla`, and half the random-gather bytes.
- Dequantize where the reuse is. Within a prefill chunk each selected row is
  reused many times, so the conversion amortizes; at T=1 decode there is no
  reuse, so only the selected rows are converted. A buffer sized by the KV
  pool is not acceptable: the whole-cache BF16 pre-pass needed 2 KiB per
  pool slot and ran 24 GB ranks out of memory, so the INT8 prefill
  dequantizes in-kernel (FlashMLA `5966fcc`).
- Shared memory: sm_86 and sm_121 both allow 101,376 bytes per block
  opt-in. Kernels that must run on both assert their tile budget
  (`static_assert(sizeof(SmemInt8) <= 100 * 1024)` in the FlashMLA prefill).

## Compiling

### Architecture lists

The image targets exactly two architectures, `TORCH_CUDA_ARCH_LIST='8.6
12.1a'` (`docker/Dockerfile`, `docker/versions.json`). Do not add `+PTX` to
the global list; CMake drops it when it converts global gencode flags into
per-kernel lists. A kernel that needs PTX adds `+PTX` to its own list.

In `CMakeLists.txt`:

- `CUDA_SUPPORTED_ARCHS` keeps `12.1` explicit for CUDA 13. Collapsing a
  requested `12.1a` to the `12.0` entry makes later intersections emit
  `sm_120a` cubins, which load on GB10 but do not run the AllSpark repack and
  GEMM correctly.
- Sources in `VLLM_STABLE_EXT_SRC` compile for every target architecture.
  A kernel that needs architecture-specific instructions gets its own list:
  `cuda_archs_loose_intersection(<NAME>_ARCHS "<list>" "${CUDA_ARCHS}")`,
  then `set_gencode_flags_for_srcs(SRCS ... CUDA_ARCHS "${<NAME>_ARCHS}")`.
  Existing examples are Marlin (`8.0+PTX;12.0f` on CUDA 13,
  `8.0+PTX;12.0a;12.1a` before), AllSpark (`8.0;8.6;8.7;8.9;12.0a;12.1a`),
  the SM120 FP4 kernels (`12.0f` or `12.0a;12.1a`) and Marlin MoE (same as
  Marlin). Check that the list intersects to both `8.6` and `12.1a` for a
  kernel the checkpoint needs on both platforms.
- A new `cmake/external_projects/<name>.cmake` inside the CUDA block needs an
  `add_custom_target(_<name>_C)` stub for the architectures that do not
  build it (`MERGING.md`, known traps).

### Editable development build

```bash
source .venv/bin/activate
TORCH_CUDA_ARCH_LIST="8.6" uv pip install --no-build-isolation --no-deps -e .
```

Use `8.6` for the local A5000s and `8.6 12.1a` to check that a kernel
compiles for both platforms. Delete stale `.deps/` checkouts and
`build/temp*/CMakeCache.txt` when the arch list or the external pins change.
FA3 only builds when a Hopper architecture is listed.

### FlashMLA wheel

`FLASH_MLA_CUDA_ARCHS` is a comma-separated list of `sm_XX` numbers for the
Ampere-family sources (`csrc/*_sm80.cu`); it defaults to `80`. The Hopper
sources always build for `sm_90a`.

```bash
# local development build for the A5000s
FLASH_MLA_CUDA_ARCHS=86 uv pip install --no-build-isolation .
# both deployed platforms
FLASH_MLA_CUDA_ARCHS=86,121 python setup.py bdist_wheel
```

CI (`.github/workflows/wheels.yaml`) builds the published wheels with
`FLASH_MLA_CUDA_ARCHS=80,86,89,90,100,120,121` for jammy x86_64, manylinux
x86_64 and manylinux aarch64, and publishes a PEP 503 index on `gh-pages`.
The image installs `FLASHMLA_SPEC` from that index and asserts the version
and every entry point the checkpoints name; bumping the wheel means updating
`FLASHMLA_SPEC`, the version assertions in `docker/Dockerfile` and
`docker/versions.json` together.

### SparkInfer wheel

The SparkInfer fork's `.github/workflows/wheels.yaml` builds cp312 wheels
with `--no-build-isolation` against the image's cu130 torch, so the four
PCIe extensions link against the torch the Sparks run. The x86_64 wheel
uses `TORCH_CUDA_ARCH_LIST="8.6 12.0a 12.1a"`, the aarch64 wheel `12.1a`.
`scripts/build_aot_cache.py precompile` stages ahead-of-time CuTe DSL kernel
objects into the wheel (`CUTE_DSL_ARCH` selects the target without a GPU).
The image installs `SPARKINFER_SPEC` from the fork's index and asserts that
the PCIe extensions are inside the wheel; without them the first collective
compiles with nvcc inside a live request.

### Local build

Build the native extensions on the workstation with ccache, not sccache:

```bash
VLLM_DISABLE_SCCACHE=1 uv pip install --no-build-isolation --no-deps -e .
```

`setup.py` prefers sccache whenever it is on `PATH`, and sccache keys CUDA
objects on absolute source paths, so every worktree or second checkout
recompiles from scratch. ccache shares objects across checkouts when
`~/.config/ccache/ccache.conf` sets `base_dir` to the directory holding them
(for example `~/Documents`) and `hash_dir = false`; give it a `max_size` that
holds several sm86+sm121 builds (200G here).

### Image build

```bash
docker/build-consumer-platforms.sh --platform linux/amd64 --tag <tag>
```

The script builds from a pushed commit on the fork (it resolves `--ref` with
`git ls-remote`), so commit and push before building. Three caches make a
build take minutes instead of over an hour, and all three stay on:

- the pinned `buildkitd-vllm` pod, whose local layer cache persists between
  runs (`BUILDKIT_TARGET` can name a pod to pin one node);
- the GHCR layer cache at `ghcr.io/appmana/vllm-consumer:buildcache`,
  imported and exported on every build;
- sccache against the cluster's S3 for the native extensions, one daemon per
  build stage (`b4dac1e335`, `074e224edd`).

The native arm64 variant builds on hilton's BuildKit on a Spark that is not
serving a model, with `USE_SCCACHE=0` (hilton cannot reach appmana's S3) and
the per-architecture layer cache `buildcache-arm64`; the exact variables are
in the script header. Combine the two platform digests with
`docker/merge-consumer-platforms.sh` and follow the release gate in
[`MERGING.md`](../../MERGING.md).
