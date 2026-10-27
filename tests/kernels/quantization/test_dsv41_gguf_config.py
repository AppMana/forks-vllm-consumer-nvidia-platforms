# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real GGUF header indexing, format selection and incomplete-file rejection."""

import numpy as np
import pytest

from vllm.model_executor.layers.quantization.dsv41_gguf import Dsv41GGUFConfig


def write_shard(path, *, architecture="deepseek41", missing_down=False, clamps=(10.0,)):
    import gguf

    writer = gguf.GGUFWriter(path, architecture)
    writer.add_uint32("deepseek41.block_count", 1)
    writer.add_array("deepseek41.swiglu_clamp_exp", list(clamps))
    writer.add_array("deepseek41.swiglu_clamp_shexp", [10.0])
    for part, kind, size in (
        ("gate", "IQ2_S", 82),
        ("up", "IQ2_S", 82),
        ("down", "IQ3_S", 110),
    ):
        if missing_down and part == "down":
            continue
        blocks = np.zeros((4, 256, size), dtype=np.uint8)
        blocks[..., :2] = np.array([0.001], dtype=np.float16).view(np.uint8)
        writer.add_tensor(
            f"blk.0.ffn_{part}_exps.weight",
            blocks,
            raw_dtype=getattr(gguf.GGMLQuantizationType, kind),
        )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_real_header_selects_mixed_iq_pair(tmp_path):
    path = tmp_path / "experts.gguf"
    write_shard(path)
    config = Dsv41GGUFConfig.from_config({"gguf_files": [str(path)]})
    assert config.expert_backend == "b12x"
    assert config.q8_backend == "b12x"
    assert config.q6_backend == "b12x"
    assert not config.b12x_ephemeral_outputs
    assert Dsv41GGUFConfig.from_config(
        {"gguf_files": [str(path)], "b12x_ephemeral_outputs": True}
    ).b12x_ephemeral_outputs
    with pytest.raises(ValueError, match="boolean"):
        Dsv41GGUFConfig.from_config(
            {"gguf_files": [str(path)], "b12x_ephemeral_outputs": "true"}
        )
    native = Dsv41GGUFConfig.from_config(
        {
            "gguf_files": [str(path)],
            "expert_backend": "consumer_cuda",
            "q8_backend": "consumer_cuda",
            "q8_expand_min_tokens": 128,
            "q6_backend": "consumer_cuda",
            "q6_expand_min_tokens": 512,
            "q6_projections": ["attn.wo_a"],
        }
    )
    assert native.expert_backend == "consumer_cuda"
    assert native.q8_backend == "consumer_cuda"
    assert native.q8_expand_min_tokens == 128
    assert native.q6_backend == "consumer_cuda"
    assert native.q6_expand_min_tokens == 512
    assert native.q6_projections == {"attn.wo_a"}
    with pytest.raises(ValueError, match="backend"):
        Dsv41GGUFConfig.from_config(
            {"gguf_files": [str(path)], "expert_backend": "invalid"}
        )
    gate, up, down = config.expert_sources[0]
    assert (gate.quant_type, up.quant_type, down.quant_type) == (
        "IQ2_S",
        "IQ2_S",
        "IQ3_S",
    )
    assert (gate.experts, gate.columns, gate.rows) == (4, 256, 256)
    assert config.routed_clamps == config.shared_clamps == (10.0,)
    assert gate.read_expert(3).shape == (256, 1, 82)
    with pytest.raises(ValueError, match="unique"):
        Dsv41GGUFConfig([path, path])


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"missing_down": True}, "Incomplete"),
        ({"architecture": "llama"}, "architecture"),
        ({"clamps": (10.0, 10.0)}, "every layer"),
        ({"clamps": (-10.0,)}, "positive"),
    ],
)
def test_reject_inconsistent_source_metadata(tmp_path, kwargs, error):
    path = tmp_path / "experts.gguf"
    write_shard(path, **kwargs)
    with pytest.raises(ValueError, match=error):
        Dsv41GGUFConfig([path])


def add_shared_shard(path):
    import gguf

    writer = gguf.GGUFWriter(path, "deepseek41")
    rng = np.random.default_rng(995)
    refs = []
    for part, n, k in (("gate", 2304, 256), ("up", 2304, 256), ("down", 256, 2304)):
        raw = rng.integers(0, 256, (n, k // 256, 210), dtype=np.uint8)
        raw[..., 208:] = np.array([0.0005], dtype=np.float16).view(np.uint8)
        writer.add_tensor(
            f"blk.0.ffn_{part}_shexp.weight",
            raw.reshape(n, -1),
            raw_dtype=gguf.GGMLQuantizationType.Q6_K,
        )
        refs.append(gguf.dequantize(raw.reshape(n, -1), gguf.GGMLQuantizationType.Q6_K))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return refs


@pytest.mark.skipif(
    __import__("torch").cuda.is_available() is False, reason="CUDA required"
)
def test_real_shared_mlp_native_tp_padding(tmp_path, monkeypatch, default_vllm_config):
    import torch
    import torch.nn.functional as F
    from b12x._lib.runtime_control import kernel_resolution_guard

    from vllm.model_executor.layers import linear
    from vllm.models.deepseek_v4.nvidia import model

    expert_path, shared_path = tmp_path / "expert.gguf", tmp_path / "shared.gguf"
    write_shard(expert_path)
    refs = add_shared_shard(shared_path)
    config = Dsv41GGUFConfig([expert_path, shared_path], max_tokens=17)
    device = torch.device("cuda:0")
    refs = [torch.from_numpy(w).bfloat16().to(device) for w in refs]
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(model, "get_tensor_model_parallel_world_size", lambda: 2)
    mlps = []
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        for rank in range(2):
            monkeypatch.setattr(
                linear, "get_tensor_model_parallel_rank", lambda rank=rank: rank
            )
            with torch.device(device):
                mlp = model.DeepseekV4MLP(
                    256,
                    2304,
                    "silu",
                    swiglu_limit=10.0,
                    quant_config=config,
                    reduce_results=False,
                    prefix="model.layers.0.ffn.shared_experts",
                )
            assert mlp.down_proj.input_size == 2560
            assert mlp.down_proj.input_size_per_partition == 1280
            for layer in (mlp.gate_up_proj, mlp.down_proj):
                assert sum(p.numel() for p in layer.parameters()) == 0
                layer.quant_method.process_weights_after_loading(layer)
            mlps.append(mlp)
    finally:
        torch.set_default_dtype(old_dtype)
    with torch.inference_mode(), kernel_resolution_guard("native shared MLP TP"):
        for m in (1, 8, 17):
            x = torch.randn(m, 256, dtype=torch.bfloat16, device=device) * 3
            gate = (x.float() @ refs[0].float().T).bfloat16().clamp(max=10)
            up = (x.float() @ refs[1].float().T).bfloat16().clamp(-10, 10)
            intermediate = (F.silu(gate.float()) * up.float()).bfloat16()
            expected = intermediate.float() @ refs[2].float().T
            result = sum(mlp(x).float() for mlp in mlps)
            assert (result - expected).norm() / expected.norm() < 0.012
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                parts = [mlp(x) for mlp in mlps]
            x.zero_()
            for output in parts:
                output.fill_(float("nan"))
            allocated = torch.accelerator.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.accelerator.synchronize()
            assert (
                allocated
                == torch.accelerator.memory_stats()["allocation.all.allocated"]
            )
            assert all(torch.count_nonzero(output) == 0 for output in parts)
            graph.reset()


@pytest.mark.parametrize(
    "groups,heads,head_dim,rank", [(2, 2, 128, 128), (4, 8, 512, 1024)]
)
@pytest.mark.parametrize(
    "backend", ["b12x", "b12x_ephemeral", "consumer_cuda", "consumer_q6"]
)
def test_native_grouped_wo_a_dispatch(
    tmp_path, default_vllm_config, groups, heads, head_dim, rank, backend
):
    import gguf
    import torch
    from b12x._lib.runtime_control import kernel_resolution_guard

    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.models.deepseek_v4.nvidia.ops.o_proj import deep_gemm_fp8_o_proj

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if backend == "consumer_q6" and torch.cuda.get_device_capability()[0] < 12:
        pytest.skip("Q6 small-row b12x arm requires SM120/121")
    expert_path, attention_path = tmp_path / "expert.gguf", tmp_path / "attn.gguf"
    write_shard(expert_path)
    rng = np.random.default_rng(778)
    n, k = 2 * groups * rank, heads * head_dim
    q6 = backend == "consumer_q6"
    kind = gguf.GGMLQuantizationType.Q6_K if q6 else gguf.GGMLQuantizationType.Q8_0
    raw = rng.integers(
        0, 256, (n, k // (256 if q6 else 32), 210 if q6 else 34), dtype=np.uint8
    )
    if q6:
        raw[..., 208:] = np.array([0.001], dtype=np.float16).view(np.uint8)
    else:
        raw[..., :2] = np.array([0.001], dtype=np.float16).view(np.uint8)
    writer = gguf.GGUFWriter(attention_path, "deepseek41")
    writer.add_tensor(
        "blk.0.attn_output_a.weight",
        raw.reshape(n, -1),
        raw_dtype=kind,
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    config = Dsv41GGUFConfig(
        [expert_path, attention_path],
        max_tokens=128,
        q8_backend="b12x" if q6 or backend == "b12x_ephemeral" else backend,
        b12x_ephemeral_outputs=backend == "b12x_ephemeral",
        q6_backend="consumer_cuda" if q6 else "b12x",
        q6_expand_min_tokens=128,
        q6_projections=["attn.wo_a"],
        q8_expand_min_tokens=128,
        q8_projections=["attn.wo_a"],
    )
    device = torch.device("cuda:0")
    with torch.device(device):
        layer = ColumnParallelLinear(
            k,
            n,
            bias=False,
            quant_config=config,
            params_dtype=torch.bfloat16,
            tp_rank=1,
            tp_size=2,
            prefix="model.layers.0.attn.wo_a",
        )
    if backend == "consumer_cuda":
        from vllm.model_executor.layers.quantization.b12x_gguf_linear import (
            B12xGGUFGroupedLinearMethod,
        )
        from vllm.model_executor.layers.quantization.consumer_q8 import (
            ConsumerQ8GroupedLinearMethod,
        )

        assert isinstance(layer.quant_method, ConsumerQ8GroupedLinearMethod)
        config.q8_projections = frozenset()
        assert isinstance(
            config.get_quant_method(layer, "model.layers.0.attn.wo_a"),
            B12xGGUFGroupedLinearMethod,
        )
    if q6:
        from vllm.model_executor.layers.quantization.b12x_gguf_linear import (
            B12xGGUFGroupedLinearMethod,
        )
        from vllm.model_executor.layers.quantization.consumer_q6 import (
            ConsumerQ6GroupedLinearMethod,
        )

        assert isinstance(layer.quant_method, ConsumerQ6GroupedLinearMethod)
        config.q6_projections = frozenset()
        assert (
            type(config.get_quant_method(layer, "model.layers.0.attn.wo_a"))
            is B12xGGUFGroupedLinearMethod
        )
    layer.is_bmm, layer.bmm_batch_size = True, groups
    layer.quant_method.process_weights_after_loading(layer)
    if backend == "b12x_ephemeral":
        from vllm.model_executor.layers.quantization.b12x_gguf_linear import (
            B12xGGUFGroupedLinearMethod,
        )

        assert isinstance(layer.quant_method, B12xGGUFGroupedLinearMethod)
        assert layer.quant_method._outputs == []
        assert layer.quant_method._output is None
        assert layer.quant_method._group_inputs is None
    weight = (
        torch.from_numpy(gguf.dequantize(raw.reshape(n, -1), kind)[n // 2 :])
        .bfloat16()
        .to(device)
        .reshape(groups, rank, k)
    )
    angles = torch.randn(256, 32, device=device)
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
    with torch.inference_mode(), kernel_resolution_guard("native grouped WO-A"):
        for m in (1, 8, 128):
            o = torch.randn(
                m, groups * heads, head_dim, dtype=torch.bfloat16, device=device
            )
            positions = torch.arange(m, device=device)
            cos, sin = cache[positions].chunk(2, dim=-1)
            cos, sin = (
                cos.repeat_interleave(2, -1)[:, None],
                sin.repeat_interleave(2, -1)[:, None],
            )
            r = o[..., head_dim - 64 :].float()
            rotated = torch.stack((-r[..., 1::2], r[..., ::2]), -1).flatten(-2)
            r = (r * cos - rotated * sin).bfloat16()
            x = torch.cat((o[..., : head_dim - 64], r), -1).reshape(m, groups, k)
            expected = torch.einsum("mgk,gnk->mgn", x.float(), weight.float()).flatten(
                1
            )

            def run(o=o, positions=positions):
                return deep_gemm_fp8_o_proj(
                    o,
                    positions,
                    cache,
                    layer,
                    lambda z: z,
                    n_groups=groups,
                    heads_per_group=heads,
                    nope_dim=head_dim - 64,
                    rope_dim=64,
                    o_lora_rank=rank,
                    einsum_recipe=(1, 1, 128),
                    tma_aligned_scales=True,
                )

            output = run()
            assert (output.float() - expected).norm() / expected.norm() < 0.005
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = run()
            o.neg_()
            output.fill_(float("nan"))
            allocated = torch.accelerator.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.accelerator.synchronize()
            assert (
                allocated
                == torch.accelerator.memory_stats()["allocation.all.allocated"]
            )
            assert (output.float() + expected).norm() / expected.norm() < 0.005
            graph.reset()


@pytest.mark.parametrize("ephemeral", [False, True])
def test_native_vocab_and_lm_head_tp(
    tmp_path, monkeypatch, default_vllm_config, ephemeral
):
    import gguf
    import torch
    from b12x._lib.runtime_control import kernel_resolution_guard

    from vllm.model_executor.layers import vocab_parallel_embedding as vocab

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    expert_path, vocab_path = tmp_path / "expert.gguf", tmp_path / "vocab.gguf"
    write_shard(expert_path)
    writer = gguf.GGUFWriter(vocab_path, "deepseek41")
    refs = []
    rng = np.random.default_rng(221)
    for name, kind, block, size in [
        ("token_embd.weight", "Q8_0", 32, 34),
        ("output.weight", "Q6_K", 256, 210),
    ]:
        raw = rng.integers(0, 256, (192, 256 // block, size), dtype=np.uint8)
        if kind == "Q8_0":
            raw[..., :2] = np.array([0.001], dtype=np.float16).view(np.uint8)
        else:
            raw[..., 208:] = np.array([0.001], dtype=np.float16).view(np.uint8)
        writer.add_tensor(
            name,
            raw.reshape(192, -1),
            raw_dtype=getattr(gguf.GGMLQuantizationType, kind),
        )
        refs.append(
            torch.from_numpy(
                gguf.dequantize(
                    raw.reshape(192, -1), getattr(gguf.GGMLQuantizationType, kind)
                )
            )
            .bfloat16()
            .cuda()
        )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    config = Dsv41GGUFConfig(
        [expert_path, vocab_path], max_tokens=17, b12x_ephemeral_outputs=ephemeral
    )
    monkeypatch.setattr(vocab, "get_tensor_model_parallel_world_size", lambda: 2)
    # Exercise actual vocabulary masking, but inspect the local contributions
    # before their collective; the test sums both owners on this physical GPU.
    monkeypatch.setattr(vocab, "tensor_model_parallel_all_reduce", lambda value: value)
    embeddings, heads = [], []
    for rank in range(2):
        monkeypatch.setattr(
            vocab, "get_tensor_model_parallel_rank", lambda rank=rank: rank
        )
        with torch.device("cuda:0"):
            emb = vocab.VocabParallelEmbedding(
                192,
                256,
                params_dtype=torch.bfloat16,
                quant_config=config,
                prefix="model.embed_tokens",
            )
            head = vocab.ParallelLMHead(
                192,
                256,
                params_dtype=torch.bfloat16,
                quant_config=config,
                prefix="lm_head",
            )
        for layer in (emb, head):
            assert sum(p.numel() for p in layer.parameters()) == 0
            layer.quant_method.process_weights_after_loading(layer)
        embeddings.append(emb)
        heads.append(head)
    with torch.inference_mode(), kernel_resolution_guard("native GGUF vocabulary"):
        for m in (1, 5, 17):
            ids = (torch.arange(m, device="cuda") * 47 + 95) % 192
            actual = sum(emb(ids).float() for emb in embeddings)
            assert torch.equal(actual.bfloat16(), refs[0][ids])
            x = torch.randn(m, 256, dtype=torch.bfloat16, device="cuda")
            logits = torch.cat([head.quant_method.apply(head, x) for head in heads], -1)
            expected = x.float() @ refs[1].float().T
            assert (logits.float() - expected).norm() / expected.norm() < 0.004
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = [emb(ids) for emb in embeddings]
                head_outputs = [head.quant_method.apply(head, x) for head in heads]
            ids.add_(7).remainder_(192)
            x.zero_()
            for out in outputs + head_outputs:
                out.fill_(float("nan"))
            allocated = torch.accelerator.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.accelerator.synchronize()
            assert (
                allocated
                == torch.accelerator.memory_stats()["allocation.all.allocated"]
            )
            assert torch.equal(
                sum(out.float() for out in outputs).bfloat16(), refs[0][ids]
            )
            assert all(torch.count_nonzero(out) == 0 for out in head_outputs)
            graph.reset()
