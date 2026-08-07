"""Direct DSV4 channel-W8A16 benchmark at the TP=2 prefill shapes.

This intentionally initializes no model and loads no checkpoint.  It exercises
the installed native operators with tensors matching one TP rank at M=8192.
"""

from __future__ import annotations

import argparse
import gc

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_gptq_marlin_linear,
    marlin_make_workspace_new,
    marlin_permute_scales,
)
from vllm.scalar_type import scalar_types


TP2_SHAPES = (
    ("wkv", 256, 4096),
    ("wq_a", 512, 4096),
    ("shared_w1", 1024, 4096),
    ("shared_w2", 4096, 1024),
    ("wo_b", 4096, 4096),
    ("wq_b", 16384, 1024),
)


def _elapsed_ms(fn, iterations: int) -> float:
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / iterations


def _pack_marlin(qweight_nk: torch.Tensor, scales_n: torch.Tensor):
    n, k = qweight_nk.shape
    codes = qweight_nk.t().contiguous().reshape(k // 4, 4, n).to(torch.int32)
    packed = codes[:, 0]
    packed = packed | (codes[:, 1] << 8)
    packed = packed | (codes[:, 2] << 16)
    packed = packed | (codes[:, 3] << 24)
    empty = torch.empty(0, dtype=torch.int, device=qweight_nk.device)
    weight = ops.gptq_marlin_repack(packed, empty, k, n, 8, False)
    scales = marlin_permute_scales(scales_n.reshape(1, n), k, n, -1, False)
    workspace = marlin_make_workspace_new(qweight_nk.device)
    return weight, scales, empty, workspace


def _snr_db(reference: torch.Tensor, actual: torch.Tensor) -> float:
    signal = reference.float().square().mean()
    noise = (reference.float() - actual.float()).square().mean()
    return (10 * torch.log10(signal / noise)).item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--custom-ops-only",
        action="store_true",
        help="Time AllSpark without generic Torch CUDA kernels or references.",
    )
    args = parser.parse_args()

    torch.manual_seed(1)
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    sm_version = properties.major * 10 + properties.minor
    sm_count = properties.multi_processor_count
    print(
        f"gpu={properties.name} sm={sm_version} sms={sm_count} m={args.m}",
        flush=True,
    )

    for label, n, k in TP2_SHAPES:
        tensor_device = "cpu" if args.custom_ops_only else device
        qweight = torch.randint(0, 256, (n, k), dtype=torch.uint8, device=tensor_device)
        scales = (torch.rand(n, device=tensor_device) * 0.01 + 0.001).to(
            torch.bfloat16
        )
        activation = torch.randn(
            args.m, k, dtype=torch.bfloat16, device=tensor_device
        )
        if args.custom_ops_only:
            qweight = qweight.to(device)
            scales = scales.to(device)
            activation = activation.to(device)

        allspark_weight, allspark_scales, _ = ops.allspark_repack_weight(
            qweight.t().contiguous(), scales.reshape(1, n), None, False
        )
        bf16_weight = None
        if not args.custom_ops_only:
            bf16_weight = ((qweight.float() - 128) * scales.float().unsqueeze(1)).to(
                torch.bfloat16
            )

        def allspark_cublas():
            return ops.allspark_w8a16_gemm(
                activation,
                allspark_weight,
                allspark_scales,
                None,
                n,
                -1,
                sm_count,
                sm_version,
                0,
                False,
                True,
            )

        def allspark_native():
            return ops.allspark_w8a16_gemm(
                activation,
                allspark_weight,
                allspark_scales,
                None,
                n,
                -1,
                sm_count,
                sm_version,
                1 << 30,
                False,
                True,
            )

        def persistent_bf16():
            return torch.nn.functional.linear(activation, bf16_weight)

        timings: dict[str, float | str] = {
            "allspark_cublas_ms": _elapsed_ms(allspark_cublas, args.iterations),
            "allspark_native_ms": _elapsed_ms(allspark_native, args.iterations),
        }
        if args.custom_ops_only:
            reference = None
        else:
            timings["persistent_bf16_ms"] = _elapsed_ms(
                persistent_bf16, args.iterations
            )
            reference = persistent_bf16()
            timings["allspark_cublas_snr_db"] = _snr_db(
                reference, allspark_cublas()
            )
            timings["allspark_native_snr_db"] = _snr_db(
                reference, allspark_native()
            )

        try:
            if args.custom_ops_only:
                raise RuntimeError("skipped in custom-ops-only mode")
            marlin_weight, marlin_scales, empty, workspace = _pack_marlin(
                qweight, scales
            )

            def marlin():
                return apply_gptq_marlin_linear(
                    activation,
                    marlin_weight,
                    marlin_scales,
                    empty,
                    empty,
                    empty,
                    workspace,
                    scalar_types.uint8b128,
                    n,
                    k,
                    True,
                    input_dtype=None,
                )

            timings["marlin_ms"] = _elapsed_ms(marlin, args.iterations)
            if reference is not None:
                timings["marlin_snr_db"] = _snr_db(reference, marlin())
        except (RuntimeError, torch.AcceleratorError) as error:
            timings["marlin_error"] = str(error).splitlines()[0]

        fields = " ".join(f"{key}={value}" for key, value in timings.items())
        print(f"shape={label} n={n} k={k} {fields}", flush=True)
        del qweight, scales, activation, allspark_weight, allspark_scales, bf16_weight
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
