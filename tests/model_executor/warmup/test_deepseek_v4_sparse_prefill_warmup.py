# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import vllm.model_executor.warmup.kernel_warmup as kernel_warmup


def _worker(architectures: list[str], max_tokens: int = 256, max_seqs: int = 6):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=architectures)
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_tokens,
            max_num_seqs=max_seqs,
        ),
        model_runner=SimpleNamespace(_dummy_run=Mock()),
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
