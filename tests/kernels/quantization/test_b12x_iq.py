# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native IQ quant-method lifecycle against independent GGUF references."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.b12x_iq import (
    B12xIQMoEMethod,
    _iq_scratch,
)


@pytest.mark.parametrize("backend", ["b12x", "consumer_cuda"])
def test_iq_scratch_reuses_current_arena_and_isolates_lanes(monkeypatch, backend):
    from vllm.v1.worker import workspace

    device = torch.device("cpu")
    manager = workspace.WorkspaceManager(device, num_lanes=2)
    monkeypatch.setattr(workspace, "_manager", manager)
    specs = (SimpleNamespace(shape=(32,), dtype=torch.float32),)
    state = SimpleNamespace(scratch=SimpleNamespace(scratch_specs=lambda: specs))
    if backend == "consumer_cuda":
        from vllm.model_executor.layers.quantization.consumer_iq import (
            _consumer_scratch,
        )

        plan = SimpleNamespace(scratch_specs=lambda: (((32,), torch.float32),))
        get_scratch = lambda: _consumer_scratch(plan)
    else:
        get_scratch = lambda: _iq_scratch(state, device)
    first = get_scratch()[0]
    assert get_scratch()[0].data_ptr() == first.data_ptr()
    # Later layers can grow the arena; execution must fetch the new backing.
    manager.get_simultaneous(((1024,), torch.float32))
    grown = get_scratch()[0]
    assert grown.data_ptr() != first.data_ptr()
    with workspace.use_workspace_lane(1):
        other = get_scratch()[0]
        assert other.data_ptr() != grown.data_ptr()
    manager.lock()
    assert get_scratch()[0].data_ptr() == grown.data_ptr()
    with workspace.use_workspace_lane(1):
        assert get_scratch()[0].data_ptr() == other.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("shared_scratch", [False, True])
@pytest.mark.parametrize("backend", ["b12x", "consumer_cuda"])
@torch.inference_mode()
def test_stream_load_warmup_and_frozen_method_replay(
    tmp_path, monkeypatch, shared_scratch, backend
):
    import gguf
    from b12x._lib import compiler
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.loader.gguf_experts import ExpertTensor

    device = torch.device("cuda:0")
    torch.accelerator.set_device_index(device)
    from vllm.v1.worker import workspace

    manager = workspace.WorkspaceManager(device) if shared_scratch else None
    monkeypatch.setattr(workspace, "_manager", manager)
    sources, refs = [], []
    rng = np.random.default_rng(51)
    for name, kind, size in (
        ("gate", "IQ2_S", 82),
        ("up", "IQ2_S", 82),
        ("down", "IQ3_S", 110),
    ):
        raw = rng.integers(0, 256, (4, 256, 1, size), dtype=np.uint8)
        raw[..., :2] = np.array([0.001], dtype=np.float16).view(np.uint8)
        path = tmp_path / name
        path.write_bytes(raw.tobytes())
        sources.append(ExpertTensor(path, name, kind, 0, 4, 256, 256, 256, size))
        decoded = gguf.dequantize(
            raw.reshape(4 * 256, -1), getattr(gguf.GGMLQuantizationType, kind)
        )
        refs.append(
            torch.from_numpy(decoded).reshape(4, 256, 256).bfloat16().to(device)
        )
    parallel = SimpleNamespace(
        tp_size=1,
        dp_size=1,
        is_sequence_parallel=False,
        use_all2all_kernels=False,
        enable_eplb=False,
    )
    config = SimpleNamespace(
        moe_parallel_config=parallel,
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        has_bias=False,
        is_lora_enabled=False,
        device=device,
        num_experts=4,
        num_local_experts=2,
        experts_per_token=2,
        max_num_tokens=32,
        swiglu_limit=10.0,
        swiglu_alpha=None,
        swiglu_beta=None,
    )
    method_type = B12xIQMoEMethod
    if backend == "consumer_cuda":
        from vllm.model_executor.layers.quantization.consumer_iq import (
            ConsumerIQMoEMethod,
        )

        method_type = ConsumerIQMoEMethod
    method = method_type(config, gate=sources[0], up=sources[1], down=sources[2])
    layer = torch.nn.Module()
    layer.register_buffer(
        "expert_map", torch.tensor([0, -1, 1, -1], dtype=torch.int32, device=device)
    )
    layer.apply_router_weight_on_input = False
    method.create_weights(layer, 2, 256, 256, torch.bfloat16)
    assert sum(p.numel() for p in layer.parameters()) == 0
    method.process_weights_after_loading(layer)
    # A later serving-capacity declaration must prepare the smaller route
    # without replacing output storage or invalidating an existing capture.
    assert method._output is not None
    output_pointer = method._output.data_ptr()
    method.get_b12x_warmup_unit(layer, (8, 32), torch.bfloat16).compile()
    assert method._output.data_ptr() == output_pointer
    assert set(method._plans) == {1, 8, 32}
    if manager is not None:
        assert all(scratch is None for _, scratch in method._plans.values())
        manager.lock()

    def forbidden(*args, **kwargs):
        raise AssertionError("runtime compilation")

    monkeypatch.setattr(compiler, "compile", forbidden)
    with kernel_resolution_guard("vLLM IQ method"):
        for m in (1, 3, 8, 17, 32):
            x = torch.randn(m, 256, dtype=torch.bfloat16, device=device) * 3
            ids = (
                torch.tensor([0, 2], dtype=torch.int32, device=device)
                .expand(m, -1)
                .contiguous()
            )
            probs = torch.full((m, 2), 0.5, device=device)

            def oracle(m=m, x=x, ids=ids, probs=probs):
                out = torch.zeros((m, 256), device=device)
                for e in (0, 2):
                    gate = (
                        (x.float() @ refs[0][e].float().T)
                        .bfloat16()
                        .float()
                        .clamp(max=10)
                    )
                    up = (
                        (x.float() @ refs[1][e].float().T)
                        .bfloat16()
                        .float()
                        .clamp(-10, 10)
                    )
                    mid = (F.silu(gate) * up).bfloat16().float()
                    out += (mid @ refs[2][e].float().T) * torch.where(
                        ids == e, probs, 0
                    ).sum(-1)[:, None]
                return out

            out = method.apply(layer, x, probs, ids, None, None)
            expected = oracle()
            assert (out.float() - expected).norm() / expected.norm() < 0.02
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = method.apply(layer, x, probs, ids, None, None)
            ids[:, 0] = 1
            x.mul_(0.5)
            expected = oracle()
            captured.fill_(float("nan"))
            if manager is not None:
                # Another layer may overwrite every byte between launches.
                if backend == "consumer_cuda":
                    from vllm.model_executor.layers.quantization.consumer_iq import (
                        _consumer_scratch,
                    )

                    buffers = _consumer_scratch(method._plans[32][0])
                else:
                    buffers = _iq_scratch(method._plans[32][0], device)
                for scratch in buffers:
                    scratch.view(torch.uint8).fill_(0xA5)
            allocations = torch.accelerator.memory_stats(device)[
                "allocation.all.allocated"
            ]
            graph.replay()
            torch.accelerator.synchronize()
            assert (
                torch.accelerator.memory_stats(device)["allocation.all.allocated"]
                == allocations
            )
            assert (captured.float() - expected).norm() / expected.norm() < 0.02
            probs.zero_()
            graph.replay()
            torch.accelerator.synchronize()
            assert torch.count_nonzero(captured) == 0
            graph.reset()
    with pytest.raises(ValueError, match="capacity"):
        method.apply(layer, torch.empty(33, 256, device=device), probs, ids, None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_native_dense_streamed_merged_lifecycle(tmp_path, monkeypatch):
    import gguf
    from b12x._lib import compiler
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.loader.gguf_dense import DenseTensor

    from vllm.model_executor.layers.quantization.b12x_gguf_linear import (
        B12xGGUFLinearMethod,
        GGUFLinearSlice,
    )

    device = torch.device("cuda:0")
    slices, refs = [], []
    rng = np.random.default_rng(73)
    for kind, n, block, size in (("Q6_K", 136, 256, 210), ("Q8_0", 128, 32, 34)):
        raw = rng.integers(0, 256, (n, 256 // block, size), dtype=np.uint8)
        base = rng.normal(0, 0.001, (n, 256 // block, 1)).astype(np.float16)
        if kind == "Q6_K":
            raw[..., 208:] = base.view(np.uint8)
        else:
            raw[..., :2] = base.view(np.uint8)
        path = tmp_path / kind
        path.write_bytes(raw.tobytes())
        slices.append(GGUFLinearSlice(DenseTensor(path, kind, kind, 0, n, 256), n, 256))
        refs.append(
            torch.from_numpy(
                gguf.dequantize(
                    raw.reshape(n, -1),
                    getattr(gguf.GGMLQuantizationType, kind),
                )
            )
            .bfloat16()
            .to(device)
        )
    method = B12xGGUFLinearMethod(slices, max_tokens=32)
    layer = torch.nn.Module()
    with torch.device(device):
        method.create_weights(layer, 256, [136, 128], 256, 264, torch.bfloat16)
    assert sum(p.numel() for p in layer.parameters()) == 0
    layer.is_bmm = True
    with pytest.raises(NotImplementedError, match="Grouped WO-A"):
        method.process_weights_after_loading(layer)
    layer.is_bmm = False
    method.process_weights_after_loading(layer)
    method.get_b12x_warmup_unit(layer, (1, 8, 32), torch.bfloat16).compile()

    def forbidden(*args, **kwargs):
        raise AssertionError("runtime compilation")

    monkeypatch.setattr(compiler, "compile", forbidden)
    ref = torch.cat(refs).float()
    with kernel_resolution_guard("vLLM native dense method"):
        for m in (1, 3, 8, 17, 32):
            x = torch.randn(m, 256, device=device, dtype=torch.bfloat16)
            expected = x.float() @ ref.T
            result = method.apply(layer, x)
            assert (result.float() - expected).norm() / expected.norm() < 0.004
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = method.apply(layer, x)
            x.neg_()
            captured.fill_(float("nan"))
            allocated = torch.accelerator.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.accelerator.synchronize()
            assert (
                torch.accelerator.memory_stats()["allocation.all.allocated"]
                == allocated
            )
            assert (captured.float() + expected).norm() / expected.norm() < 0.004
            graph.reset()
        with pytest.raises(ValueError, match="capacity"):
            method.apply(
                layer, torch.zeros(33, 256, device=device, dtype=torch.bfloat16)
            )
