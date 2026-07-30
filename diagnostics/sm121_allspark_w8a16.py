"""Numerical smoke test for the compact AllSpark W8A16 path on SM121.

This intentionally calls the compiled operator directly, bypassing the
DeepSeek-V4 Python architecture gate.  It exercises both the fused small-M
kernel and the large-M cuBLAS fallback against the dequantized BF16 reference.
"""

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.utils.platform_utils import num_compute_units


def run_case(m: int, n: int = 256, k: int = 512) -> None:
    torch.manual_seed(1000 + m)
    device = torch.device("cuda")
    # Generate on CPU: this diagnostic is also used on images whose generic
    # CUDA RNG kernels do not carry an SM121 image.
    weight = torch.randn((n, k), dtype=torch.float32) * 0.05
    scale = weight.abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).tiny) / 127
    signed = torch.round(weight / scale[:, None]).clamp(-128, 127)
    biased = (signed.to(torch.int16) + 128).to(torch.uint8)
    scale_bf16 = scale.to(torch.bfloat16)

    # Establish that cuBLAS itself is usable before testing the custom repack.
    warm = torch.ones((16, 16), dtype=torch.bfloat16).to(device)
    torch.mm(warm, warm)
    torch.cuda.synchronize()

    packed, packed_scale, _ = ops.allspark_repack_weight(
        biased.t().contiguous().to(device),
        scale_bf16.reshape(1, -1).contiguous().to(device),
        None,
        False,
    )
    torch.cuda.synchronize()
    x_cpu = torch.randn((m, k), dtype=torch.bfloat16)
    x = x_cpu.to(device)
    props = torch.cuda.get_device_properties(device)
    sm_version = props.major * 10 + props.minor

    actual = ops.allspark_w8a16_gemm(
        a=x,
        b_qweight=packed,
        b_scales=packed_scale,
        b_qzeros=None,
        n=n,
        group_size=-1,
        sm_count=num_compute_units(device.index),
        sm_version=sm_version,
        CUBLAS_M_THRESHOLD=1024,
        has_zp=False,
        n32k16_reorder=True,
    )
    reference_weight = (signed * scale[:, None]).to(torch.bfloat16)
    expected = F.linear(x_cpu.float(), reference_weight.float())
    torch.cuda.synchronize()

    actual_cpu = actual.cpu().float()
    delta = (actual_cpu - expected).abs()
    print(
        f"PASS M={m} N={n} K={k} sm={sm_version} "
        f"max_abs={delta.max().item():.6g} "
        f"mean_abs={delta.mean().item():.6g}",
        flush=True,
    )
    torch.testing.assert_close(actual_cpu, expected, rtol=0.03, atol=0.08)


if __name__ == "__main__":
    for tokens in (1, 2, 64, 1024, 1025):
        run_case(tokens)
