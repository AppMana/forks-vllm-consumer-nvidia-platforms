"""Standalone repro for the DSV4 fp8_ds_mla native-gather IMA (image 319caef748).

Root-cause candidate: commit 82df3f6bb0 ("Use native fp8 DS MLA gather on
Ampere") replaced the pure-torch fallback with the CUDA kernel
``deepseekV4Fp8DsMlaGatherKernel``
(csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:952).

The kernel drops TWO upper-bound guards that its own torch-fallback oracle
(``_dequantize_and_gather_k_cache_torch`` in cache_utils.py) enforces:

  1. block_in_seq vs block_table width:
       oracle: valid = block_in_seq < block_table.shape[1]; clamp + zero-fill
       kernel: block_table[req*bt_s0 + block_in_seq*bt_s1]  <-- no width guard
  2. physical_block upper bound vs k_cache.shape[0]:
       oracle: valid = block_idx < k_cache.shape[0]; clamp + zero-fill
       kernel: only `if (physical_block < 0)`  <-- no upper-bound guard

Either OOB read produces an asynchronous CUDA IMA that surfaces at the *next*
kernel touching the GPU (in the failing run: fused_marlin_moe / torch.sum).

This file compiles the kernel's *addressing* in isolation (the fp8 math is
replaced by a plain cast; the OOB is pure pointer arithmetic) so
compute-sanitizer memcheck reports the fault SYNCHRONOUSLY. It builds two
variants -- `buggy` (byte-for-byte the current addressing) and `fixed` (adds
the two oracle guards) -- and drives them with a windowed-SWA geometry the
oracle treats as partly out-of-range.

Run:
  export PATH=/usr/local/cuda-13.0/bin:$PATH
  CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 \
    /usr/local/cuda-13.0/bin/compute-sanitizer --tool memcheck \
    .venv/bin/python tools/ampere/dsv4_fp8_gather_ima_repro.py
"""

import os
import sys

import torch
from torch.utils.cpp_extension import load_inline

_CUDA = r"""
#include <cuda_bf16.h>
#include <cstdint>

constexpr int kFp8Dim = 448;
constexpr int kHeadDim = 512;
constexpr int kQuantBlock = 64;
constexpr int kTokenDataBytes = 576;  // 448 fp8 + 64 bf16 * 2
constexpr int kScaleBytes = 8;

// Exact copy of the addressing in deepseekV4Fp8DsMlaGatherKernel, with the fp8
// conversion replaced by a plain cast (the IMA is in the address math, not the
// numerics). WITH_GUARDS toggles the two missing oracle bound checks.
template <bool WITH_GUARDS>
__global__ void gatherKernel(
    __nv_bfloat16* __restrict__ out, uint8_t const* __restrict__ k_cache,
    int32_t const* __restrict__ seq_lens, int32_t const* __restrict__ gather_lens,
    int32_t const* __restrict__ block_table, int num_reqs, int num_blocks,
    int block_table_width, int64_t block_table_stride0, int64_t block_table_stride1,
    int max_gather_len, int block_size, int offset, int64_t out_stride0,
    int64_t out_stride1, int64_t out_stride2, int64_t cache_block_stride) {
  int const req_id = blockIdx.x;
  int const token_id = blockIdx.y;
  if (req_id >= num_reqs) return;
  int const gather_len = gather_lens == nullptr ? seq_lens[req_id] : gather_lens[req_id];
  if (token_id >= gather_len || token_id >= max_gather_len) return;

  int const start_pos = seq_lens[req_id] - gather_len;
  int const logical_pos = start_pos + token_id;
  int const block_in_seq = logical_pos / block_size;
  int const pos_in_block = logical_pos - block_in_seq * block_size;

  if (WITH_GUARDS) {
    // FIX 1: block_in_seq beyond the (windowed) block_table -> zero-fill.
    if (block_in_seq >= block_table_width) {
      for (int dim = threadIdx.x; dim < kHeadDim; dim += blockDim.x)
        out[req_id * out_stride0 + (offset + token_id) * out_stride1 + dim * out_stride2] =
            __float2bfloat16(0.0f);
      return;
    }
  }

  int32_t const physical_block =
      block_table[req_id * block_table_stride0 + block_in_seq * block_table_stride1];

  bool oob_block = physical_block < 0;
  if (WITH_GUARDS) oob_block = oob_block || physical_block >= num_blocks;  // FIX 2
  if (oob_block) {
    for (int dim = threadIdx.x; dim < kHeadDim; dim += blockDim.x)
      out[req_id * out_stride0 + (offset + token_id) * out_stride1 + dim * out_stride2] =
          __float2bfloat16(0.0f);
    return;
  }

  uint8_t const* token_base = k_cache +
      static_cast<int64_t>(physical_block) * cache_block_stride +
      static_cast<int64_t>(pos_in_block) * kTokenDataBytes;
  uint8_t const* scale_base = k_cache +
      static_cast<int64_t>(physical_block) * cache_block_stride +
      static_cast<int64_t>(block_size) * kTokenDataBytes +
      static_cast<int64_t>(pos_in_block) * kScaleBytes;
  int64_t const out_base =
      static_cast<int64_t>(req_id) * out_stride0 +
      static_cast<int64_t>(offset + token_id) * out_stride1;

  for (int dim = threadIdx.x; dim < kFp8Dim; dim += blockDim.x) {
    int const qblock = dim / kQuantBlock;
    float const scale = exp2f(static_cast<float>(scale_base[qblock]) - 127.0f);
    out[out_base + static_cast<int64_t>(dim) * out_stride2] =
        __float2bfloat16(static_cast<float>(token_base[dim]) * scale);
  }
  for (int dim = kFp8Dim + threadIdx.x; dim < kHeadDim; dim += blockDim.x) {
    uint16_t const raw =
        reinterpret_cast<uint16_t const*>(token_base + kFp8Dim)[dim - kFp8Dim];
    out[out_base + static_cast<int64_t>(dim) * out_stride2] =
        *reinterpret_cast<__nv_bfloat16 const*>(&raw);
  }
}

void run(torch::Tensor out, torch::Tensor k_cache, torch::Tensor seq_lens,
         torch::Tensor gather_lens, torch::Tensor block_table, int64_t num_blocks,
         int64_t block_size, int64_t offset, int64_t cache_block_stride,
         bool with_guards) {
  int num_reqs = seq_lens.size(0);
  int max_gather_len = out.size(1) - offset;
  dim3 grid(num_reqs, max_gather_len);
  dim3 block(256);
  auto* gl = gather_lens.numel() ? gather_lens.data_ptr<int32_t>() : nullptr;
  auto* k = (with_guards ? gatherKernel<true> : gatherKernel<false>);
  k<<<grid, block>>>(
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      reinterpret_cast<uint8_t*>(k_cache.data_ptr()),
      seq_lens.data_ptr<int32_t>(), gl, block_table.data_ptr<int32_t>(),
      num_reqs, (int)num_blocks, (int)block_table.size(1),
      block_table.stride(0), block_table.stride(1), max_gather_len,
      (int)block_size, (int)offset, out.stride(0), out.stride(1), out.stride(2),
      cache_block_stride);
  cudaDeviceSynchronize();
}
"""

_CPP = "void run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,bool);"


def main() -> int:
    dev = "cuda"
    mod = load_inline(
        name="dsv4_gather_ima",
        cpp_sources=_CPP,
        cuda_sources=_CUDA,
        functions=["run"],
        extra_cuda_cflags=["-gencode=arch=compute_86,code=sm_86"],
        verbose=True,
    )

    # Windowed-SWA geometry (mirrors nvidia_imma/attention.py call 2 at 16k):
    # the request's absolute context is 16k tokens, but the sliding-window SWA
    # block table only has enough COLUMNS for the window. block_in_seq is
    # computed from the ABSOLUTE logical position -> it walks off the end of the
    # block table. The torch oracle guards exactly this (valid_block_in_seq).
    block_size = 64
    seq_len = 16384                       # full context
    sliding_window = 2048
    gather_len = sliding_window           # SWA gathers only the window
    block_table_width = sliding_window // block_size   # 32 windowed columns
    num_blocks = 64                       # small pool
    start_pos = seq_len - gather_len      # 14336
    # block_in_seq reaches (16384-1)//64 = 255  >>  block_table_width 32.

    cache_block_stride = block_size * (576 + 8)  # 37376, 16B aligned
    k_cache = torch.zeros(num_blocks * cache_block_stride, dtype=torch.uint8, device=dev)
    # Valid (in-window) columns point at real blocks; the kernel never reaches
    # them because block_in_seq is out of range from token 0 (start_pos high).
    block_table = torch.randint(0, num_blocks, (1, block_table_width),
                                dtype=torch.int32, device=dev)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=dev)
    gather_lens = torch.tensor([gather_len], dtype=torch.int32, device=dev)
    out = torch.empty((1, gather_len, 512), dtype=torch.bfloat16, device=dev)

    variant = os.environ.get("VARIANT", "buggy")
    print(f"[repro] variant={variant} seq_len={seq_len} sliding_window={sliding_window} "
          f"block_table_width={block_table_width} max block_in_seq={(seq_len-1)//block_size}")
    with_guards = variant == "fixed"
    mod.run(out, k_cache, seq_lens, gather_lens, block_table,
            num_blocks, block_size, 0, cache_block_stride, with_guards)
    torch.cuda.synchronize()
    if with_guards:
        # Oracle behavior: every out-of-window position is zero-filled.
        assert torch.count_nonzero(out) == 0, "fixed kernel should zero-fill OOB rows"
        print("[repro] FIXED: sanitizer-clean, output all-zero (matches torch oracle).")
    else:
        print("[repro] BUGGY: if you see this without a sanitizer error, the OOB "
              "read landed inside another allocation (still UB / IMA under load).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
