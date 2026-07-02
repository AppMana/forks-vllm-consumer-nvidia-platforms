# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

from vllm.model_executor.warmup.kernel_warmup import (
    _deepseek_v4_sparse_mla_prefill_warmup,
)


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
    )


def test_deepseek_v4_sparse_prefill_warmup_covers_prefill_shapes():
    worker = _worker(["DeepseekV4ForCausalLM"])

    _deepseek_v4_sparse_mla_prefill_warmup(worker)

    calls = worker.model_runner._dummy_run.call_args_list
    assert len(calls) == 3
    assert calls[0].kwargs == {
        "num_tokens": 256,
        "skip_eplb": True,
        "is_profile": True,
        "force_attention": True,
        "create_single_prefill": True,
    }
    assert calls[1].kwargs == {
        "num_tokens": 256,
        "skip_eplb": True,
        "is_profile": True,
        "force_attention": True,
        "create_single_prefill": True,
        "profile_seq_lens": 512,
    }
    assert calls[2].kwargs == {
        "num_tokens": 256,
        "skip_eplb": True,
        "is_profile": True,
        "force_attention": True,
    }


def test_deepseek_v4_sparse_prefill_warmup_skips_other_models():
    worker = _worker(["LlamaForCausalLM"])

    _deepseek_v4_sparse_mla_prefill_warmup(worker)

    worker.model_runner._dummy_run.assert_not_called()
