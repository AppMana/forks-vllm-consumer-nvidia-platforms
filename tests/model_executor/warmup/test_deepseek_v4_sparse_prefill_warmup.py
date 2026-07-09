# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch

import vllm.model_executor.warmup.kernel_warmup as kernel_warmup
from vllm.model_executor.layers.quantization.dsv4_int import Dsv4Int4MoEMethod


def _worker(architectures: list[str], max_tokens: int = 256, max_seqs: int = 6):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=architectures)
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_tokens,
            max_num_seqs=max_seqs,
        ),
        model_runner=SimpleNamespace(_dummy_run=Mock(), device="cuda:0"),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )


def test_deepseek_v4_sparse_prefill_warmup_uses_direct_kernel_warmup(monkeypatch):
    worker = _worker(["DeepseekV4ForCausalLM"])
    direct_warmup = Mock()
    monkeypatch.setattr(
        kernel_warmup,
        "_deepseek_v4_sparse_mla_prefill_kernel_warmup",
        direct_warmup,
    )

    kernel_warmup._deepseek_v4_sparse_mla_prefill_warmup(worker)

    direct_warmup.assert_called_once_with(worker)
    worker.model_runner._dummy_run.assert_not_called()


def test_deepseek_v4_sparse_prefill_warmup_skips_other_models():
    worker = _worker(["LlamaForCausalLM"])

    kernel_warmup._deepseek_v4_sparse_mla_prefill_warmup(worker)

    worker.model_runner._dummy_run.assert_not_called()


def test_deepseek_v4_slot_mapping_warmup_uses_model_runner(monkeypatch):
    worker = _worker(["DeepseekV4ForCausalLM"])
    slot_mapping_warmup = Mock()
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.warmup.warmup_block_table_slot_mapping_kernel",
        slot_mapping_warmup,
    )

    kernel_warmup._deepseek_v4_block_table_slot_mapping_warmup(worker)

    slot_mapping_warmup.assert_called_once_with(worker.model_runner, "cuda:0")


def test_deepseek_v4_slot_mapping_warmup_skips_other_models(monkeypatch):
    worker = _worker(["LlamaForCausalLM"])
    slot_mapping_warmup = Mock()
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.warmup.warmup_block_table_slot_mapping_kernel",
        slot_mapping_warmup,
    )

    kernel_warmup._deepseek_v4_block_table_slot_mapping_warmup(worker)

    slot_mapping_warmup.assert_not_called()


def test_deepseek_v4_marlin_moe_warmup_uses_direct_quant_method(monkeypatch):
    calls = []

    quant_method = Dsv4Int4MoEMethod.__new__(Dsv4Int4MoEMethod)
    quant_method.hidden_size = 8
    quant_method.intermediate_size = 16
    quant_method.input_dtype = torch.int8

    module = torch.nn.Module()
    module.quant_method = quant_method
    module.hidden_size = 8
    module.intermediate_size_per_partition = 16
    module.top_k = 2
    module.local_num_experts = 4
    module.global_num_experts = 8
    module.params_dtype = torch.bfloat16
    module.w13_weight = torch.empty(4, 8, device="cpu")
    module.w2_weight = torch.empty(4, 8, device="cpu")
    module.w13_weight_scale = torch.empty(4, 8, device="cpu")
    module.w2_weight_scale = torch.empty(4, 8, device="cpu")

    def fake_apply(_self, layer, x, topk_weights, topk_ids):
        calls.append((x.shape, topk_weights.shape, topk_ids.tolist()))
        return torch.empty_like(x)

    monkeypatch.setattr(Dsv4Int4MoEMethod, "apply", fake_apply)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)

    worker = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64),
        get_model=lambda: module,
    )

    kernel_warmup._deepseek_v4_marlin_moe_warmup(worker)

    assert [shape for shape, _weights_shape, _ids in calls] == [
        torch.Size([1, 8]),
        torch.Size([16, 8]),
        torch.Size([64, 8]),
    ]
    assert all(
        weights_shape == torch.Size([tokens, 2])
        for (tokens, _), weights_shape, _ in calls
    )
    assert all(ids[0] == [0, 1] for _shape, _weights_shape, ids in calls)


def test_deepseek_v4_marlin_moe_warmup_uses_local_expert_map(monkeypatch):
    quant_method = Dsv4Int4MoEMethod.__new__(Dsv4Int4MoEMethod)
    topk_ids = []

    module = torch.nn.Module()
    module.quant_method = quant_method
    module.hidden_size = 8
    module.intermediate_size_per_partition = 16
    module.top_k = 3
    module.local_num_experts = 2
    module.params_dtype = torch.bfloat16
    module.w13_weight = torch.empty(2, 8, device="cpu")
    module.w2_weight = torch.empty(2, 8, device="cpu")
    module.w13_weight_scale = torch.empty(2, 8, device="cpu")
    module.w2_weight_scale = torch.empty(2, 8, device="cpu")
    module._expert_map = torch.tensor([-1, 0, -1, 1], dtype=torch.int32)
    module.expert_map = lambda: module._expert_map

    quant_method.hidden_size = 8
    quant_method.intermediate_size = 16
    quant_method.input_dtype = torch.int8

    def fake_apply(_self, layer, x, topk_weights, ids):
        topk_ids.append(ids[0].tolist())
        return torch.empty_like(x)

    monkeypatch.setattr(Dsv4Int4MoEMethod, "apply", fake_apply)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)

    worker = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1),
        get_model=lambda: module,
    )

    kernel_warmup._deepseek_v4_marlin_moe_warmup(worker)

    assert topk_ids == [[1, 3, 1]]


def test_deepseek_v4_fp8_ds_mla_warmup_cache_stride_matches_native_layout():
    block_size = 64

    assert kernel_warmup._DEEPSEEK_V4_FP8_DS_MLA_PAGE_TOKEN_BYTES == 584
    assert (
        block_size * kernel_warmup._DEEPSEEK_V4_FP8_DS_MLA_PAGE_TOKEN_BYTES
        == block_size * kernel_warmup._DEEPSEEK_V4_FP8_DS_MLA_TOKEN_DATA_BYTES
        + block_size * kernel_warmup._DEEPSEEK_V4_FP8_DS_MLA_SCALE_BYTES
    )
