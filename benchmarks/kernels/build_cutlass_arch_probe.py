# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Build the production c2x entry points in isolation for GPU qualification.

This does not replace an installed vLLM extension. Pass the resulting library
to consumer-int8-kernels/benchmarks/consumer_arch.py --vllm-probe.
"""

import argparse
from pathlib import Path

from torch.utils.cpp_extension import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutlass", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.build_dir.mkdir(parents=True, exist_ok=True)
    binding = args.build_dir / "binding.cpp"
    binding.write_text(r"""
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
using torch::stable::Tensor;
void cutlass_scaled_mm(Tensor&, const Tensor&, const Tensor&, const Tensor&,
                       const Tensor&, const std::optional<Tensor>&);
void boxed_mm(StableIValue* stack, uint64_t, uint64_t) {
  auto out = to<Tensor>(stack[0]);
  cutlass_scaled_mm(out, to<Tensor>(stack[1]), to<Tensor>(stack[2]),
                   to<Tensor>(stack[3]), to<Tensor>(stack[4]),
                   to<std::optional<Tensor>>(stack[5]));
}
STABLE_TORCH_LIBRARY(cutlass_arch_probe, m) {
  m.def("mm(Tensor(a!) out, Tensor a, Tensor b, Tensor sa, "
        "Tensor sb, Tensor? bias) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(cutlass_arch_probe, CUDA, m) {
  m.impl("mm", &boxed_mm);
}
""")
    base = root / "csrc/libtorch_stable"
    flags = ["-DTORCH_STABLE_ONLY", "-DENABLE_SCALED_MM_C2X=1", "-DUSE_CUDA"]
    path = load(
        name="cutlass_arch_probe",
        sources=[
            str(binding),
            str(base / "cutlass_extensions/common.cpp"),
            str(base / "quantization/w8a8/cutlass/scaled_mm_c2x.cu"),
            str(base / "quantization/w8a8/cutlass/scaled_mm_entry.cu"),
        ],
        extra_include_paths=[
            str(root / "csrc"),
            str(args.cutlass / "include"),
            str(args.cutlass / "tools/util/include"),
        ],
        extra_cflags=["-O3", *flags],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", *flags],
        build_directory=str(args.build_dir),
        is_python_module=False,
        verbose=True,
    )
    print(path)


if __name__ == "__main__":
    main()
