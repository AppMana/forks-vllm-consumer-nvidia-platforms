# SPDX-License-Identifier: Apache-2.0
"""AllSpark parity at a real DSV4 TP=2 dense-linear shape.

This test is intentionally opt-in because it reads an external checkpoint.
Set ``DSV4_INT4_INT8_SNAPSHOT`` to the exact snapshot directory.  It can also
be executed directly to print kernel timings without initializing a model.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open

from vllm import _custom_ops as ops


TENSOR_NAME = "layers.0.attn.wq_a.weight"
SCALE_NAME = "layers.0.attn.wq_a.scale"
SHARD_NAME = "model-00002-of-00046.safetensors"
SNAPSHOT_REVISION = "ace78a6e9b5d90a43476fa1c098bfee1eb46c1de"
M_VALUES = (1, 16, 1024, 8192)


def _snapshot() -> Path:
    value = os.environ.get("DSV4_INT4_INT8_SNAPSHOT")
    if not value:
        pytest.skip("DSV4_INT4_INT8_SNAPSHOT is not set")
    snapshot = Path(value)
    if snapshot.name != SNAPSHOT_REVISION:
        pytest.fail(
            f"expected snapshot {SNAPSHOT_REVISION}, got {snapshot.name}"
        )
    if not (snapshot / SHARD_NAME).is_file():
        pytest.fail(f"missing checkpoint shard: {snapshot / SHARD_NAME}")
    return snapshot


def _load_tp2_rank0(snapshot: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with safe_open(snapshot / SHARD_NAME, framework="pt", device="cpu") as shard:
        weight = shard.get_tensor(TENSOR_NAME)
        scale = shard.get_tensor(SCALE_NAME)
    assert weight.dtype is torch.uint8
    assert weight.shape == (1024, 4096)
    assert scale.dtype is torch.bfloat16
    assert scale.shape == (1024,)
    return weight[:512].cuda(), scale[:512].cuda()


def _elapsed_ms(call, iterations: int = 5) -> float:
    for _ in range(2):
        call()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / iterations


def _snr_db(reference: torch.Tensor, actual: torch.Tensor) -> float:
    signal = reference.float().square().mean()
    noise = (reference.float() - actual.float()).square().mean()
    if noise == 0:
        return float("inf")
    return float(10 * torch.log10(signal / noise))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m", M_VALUES)
def test_real_dsv4_tp2_allspark_matches_bf16_reference(m: int) -> None:
    weight, scale = _load_tp2_rank0(_snapshot())
    properties = torch.cuda.get_device_properties(weight.device)
    sm_version = properties.major * 10 + properties.minor
    repacked_weight, repacked_scale, _ = ops.allspark_repack_weight(
        weight.t().contiguous(), scale.reshape(1, -1).contiguous(), None, False
    )
    dequantized = ((weight.float() - 128.0) * scale.float().unsqueeze(1)).to(
        torch.bfloat16
    )
    generator = torch.Generator(device=weight.device).manual_seed(121 + m)
    activation = torch.randn(
        m,
        weight.shape[1],
        dtype=torch.bfloat16,
        device=weight.device,
        generator=generator,
    )

    def reference() -> torch.Tensor:
        return F.linear(activation, dequantized)

    def allspark(threshold: int) -> torch.Tensor:
        return ops.allspark_w8a16_gemm(
            activation,
            repacked_weight,
            repacked_scale,
            None,
            weight.shape[0],
            -1,
            properties.multi_processor_count,
            sm_version,
            threshold,
            False,
            True,
        )

    expected = reference()
    cublas = allspark(0)
    native = allspark(1 << 30)
    for name, actual in (("cublas", cublas), ("native", native)):
        assert torch.isfinite(actual).all(), name
        assert torch.allclose(actual, expected, rtol=0.03, atol=0.08), (
            name,
            m,
            float((actual.float() - expected.float()).abs().max()),
            _snr_db(expected, actual),
        )
    print(
        f"m={m} reference_ms={_elapsed_ms(reference):.6f} "
        f"allspark_cublas_ms={_elapsed_ms(lambda: allspark(0)):.6f} "
        f"allspark_native_ms={_elapsed_ms(lambda: allspark(1 << 30)):.6f} "
        f"cublas_snr_db={_snr_db(expected, cublas):.3f} "
        f"native_snr_db={_snr_db(expected, native):.3f}",
        flush=True,
    )


if __name__ == "__main__":
    os.environ.setdefault(
        "DSV4_INT4_INT8_SNAPSHOT",
        "/hf-cache/hub/models--appmana--deepseek-v4-int4-int8/snapshots/"
        f"{SNAPSHOT_REVISION}",
    )
    for value in M_VALUES:
        test_real_dsv4_tp2_allspark_matches_bf16_reference(value)
