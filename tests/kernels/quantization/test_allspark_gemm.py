# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.kernels.utils import DEFAULT_OPCHECK_TEST_UTILS, opcheck
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.allspark_utils import (
    ALLSPARK_AMPERE_K_ALIGN,
    ALLSPARK_AMPERE_M_CUBLAS_THRESHOLD,
    ALLSPARK_AMPERE_N_ALIGN,
    is_allspark_supported_device_capability,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import quantize_weights
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils.platform_utils import num_compute_units


def is_gptq_allspark_supported() -> bool:
    if not current_platform.is_cuda():
        return False

    capability = current_platform.get_device_capability()
    assert capability is not None

    return is_allspark_supported_device_capability(capability.to_int())


MNK_FACTORS = [
    (1, 4, 8),
    (13, 17, 67),
    (26, 37, 13),
    (48, 16, 24),
    (67, 13, 88),
    (257, 13, 11),
    (658, 13, 11),
    (1033, 9, 17),
]

DTYPES = [torch.float16, torch.bfloat16]
HAS_ZP_OPTS = [False, True]


def compute_max_diff(output, output_ref):
    return torch.mean(torch.abs(output - output_ref)) / torch.mean(
        torch.abs(output_ref)
    )


def rand_data(shape, dtype=torch.float16):
    return torch.randn(shape, dtype=dtype, device="cuda")


@pytest.mark.skipif(
    not is_gptq_allspark_supported(),
    reason="AllSpark W8A16 kernel is not supported on this GPU type.",
)
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
@pytest.mark.parametrize("group_size", [-1])
@pytest.mark.parametrize("has_zp", HAS_ZP_OPTS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_gptq_allspark_gemm_ampere(mnk_factors, group_size, has_zp, dtype):
    m_factor, n_factor, k_factor = mnk_factors
    m = m_factor
    n = n_factor * ALLSPARK_AMPERE_N_ALIGN
    k = k_factor * ALLSPARK_AMPERE_K_ALIGN

    input = rand_data((m, k), dtype=dtype)
    weight = rand_data((k, n), dtype=dtype)

    # Quantize (and apply act_order if provided)
    w_ref, qw, s, zp = quantize_weights(
        weight, scalar_types.uint8b128, group_size, has_zp
    )

    qw = qw.to(torch.uint8)
    if has_zp:
        zp = zp.to(dtype)
    properties = torch.cuda.get_device_properties(qw.device.index)
    sm_count = num_compute_units(qw.device.index)
    sm_version = properties.major * 10 + properties.minor

    n_32align = (n + 32 - 1) // 32 * 32

    qw_reorder, s_reorder, zp_reorder = ops.allspark_repack_weight(qw, s, zp, has_zp)
    opcheck(
        torch.ops._C.rearrange_kn_weight_as_n32k16_order,
        (qw, s, zp, has_zp, qw_reorder, s_reorder, zp_reorder, k, n, n_32align),
    )

    opcheck(
        torch.ops._C.allspark_w8a16_gemm,
        (
            input,
            qw_reorder,
            s_reorder,
            zp_reorder,
            n,
            group_size,
            sm_count,
            sm_version,
            ALLSPARK_AMPERE_M_CUBLAS_THRESHOLD,
            has_zp,
            True,
        ),
        test_utils=DEFAULT_OPCHECK_TEST_UTILS,
    )
    output = ops.allspark_w8a16_gemm(
        input,
        qw_reorder,
        s_reorder,
        zp_reorder,
        n,
        group_size,
        sm_count,
        sm_version,
        ALLSPARK_AMPERE_M_CUBLAS_THRESHOLD,
        has_zp,
        True,
    )

    output_ref = torch.matmul(input, w_ref)
    torch.accelerator.synchronize()
    max_diff = compute_max_diff(output, output_ref)

    assert max_diff < 0.04


@pytest.mark.skipif(
    not is_gptq_allspark_supported(),
    reason="AllSpark W8A16 kernel is not supported on this GPU type.",
)
def test_native_allspark_splitk_makes_progress_and_matches_reference():
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    m, n, k = 1024, 384, 4096

    generator = torch.Generator().manual_seed(42)
    weight = torch.randint(
        0,
        256,
        (k, n),
        dtype=torch.uint8,
        generator=generator,
    ).to(device)
    scales = (
        torch.rand((1, n), dtype=torch.float32, generator=generator) * 0.01
    ).to(dtype=torch.bfloat16, device=device)
    packed_weight, packed_scales, _ = ops.allspark_repack_weight(
        weight,
        scales,
        None,
        False,
    )
    activation = torch.randn(
        (m, k), dtype=torch.float32, generator=generator
    ).to(dtype=torch.bfloat16, device=device)

    for _ in range(256):
        output = ops.allspark_w8a16_gemm(
            activation,
            packed_weight,
            packed_scales,
            None,
            n,
            -1,
            num_compute_units(device.index),
            properties.major * 10 + properties.minor,
            1024,
            False,
            True,
        )
        torch.cuda.synchronize()
        assert torch.isfinite(output).all()

    reference_weight = (weight.float() - 128.0) * scales.float()
    reference = activation.float() @ reference_weight
    error = output.float() - reference
    snr_db = 10 * torch.log10(reference.square().mean() / error.square().mean())
    assert float(snr_db) > 45.0
