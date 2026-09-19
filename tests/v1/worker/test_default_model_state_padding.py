# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.model_states import default
from vllm.v1.worker.gpu.model_states.default import DefaultModelState


def _padded_decode_batch(num_reqs: int, num_reqs_padded: int, query_len: int):
    """A uniform decode batch of ``num_reqs`` requests padded to
    ``num_reqs_padded`` rows for a FULL cudagraph replay: query_start_loc
    and seq_lens carry the padded row count, the per-request numpy state
    only the real one, exactly as the model runner builds it."""
    num_tokens = num_reqs * query_len
    num_tokens_padded = num_reqs_padded * query_len
    qsl = np.zeros(num_reqs_padded + 1, dtype=np.int32)
    qsl[1 : num_reqs + 1] = np.arange(1, num_reqs + 1) * query_len
    qsl[num_reqs + 1 :] = num_tokens
    seq_lens = torch.zeros(num_reqs_padded, dtype=torch.int32)
    seq_lens[:num_reqs] = 8000
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        num_reqs_after_padding=num_reqs_padded,
        num_tokens_after_padding=num_tokens_padded,
        query_start_loc_np=qsl,
        query_start_loc=torch.from_numpy(qsl.copy()),
        max_query_len=query_len,
        num_scheduled_tokens=np.full(num_reqs, query_len, dtype=np.int32),
        seq_lens_cpu_upper_bound=seq_lens.clone(),
        seq_lens=seq_lens,
        is_prefilling_np=(np.arange(num_reqs) % 2 == 1),
        dcp_local_seq_lens=None,
        dcp_local_seq_lens_cpu_upper_bound=None,
        positions=torch.zeros(num_tokens_padded, dtype=torch.int64),
        prompt_lens=None,
        idx_mapping_np=np.arange(num_reqs, dtype=np.intp),
        fast_prefill=None,
        req_ids=[f"r{i}" for i in range(num_reqs)],
    )


@pytest.mark.parametrize(("num_reqs", "num_reqs_padded"), [(3, 4), (5, 6), (7, 8)])
def test_full_graph_padding_extends_is_prefilling(
    monkeypatch: pytest.MonkeyPatch, num_reqs: int, num_reqs_padded: int
) -> None:
    """A FULL decode graph pads the request rows to the captured size, so the
    is_prefilling vector handed to the attention builders must cover the
    padded rows too (padding rows are decodes). A vector of the unpadded
    length makes ``split_decodes_and_prefills`` fail with a size mismatch on
    the first three-, five- or seven-sequence decode step."""
    state = object.__new__(DefaultModelState)
    state.max_model_len = 8192
    state.supports_mm_inputs = False
    state.encoder_cache = None
    state.model_config = SimpleNamespace(is_mm_prefix_lm=False, rswa_window=None)

    input_batch = _padded_decode_batch(num_reqs, num_reqs_padded, query_len=8)
    build_attn_metadata = Mock(return_value={})
    monkeypatch.setattr(default, "build_attn_metadata", build_attn_metadata)

    state.prepare_attn(
        input_batch=input_batch,
        cudagraph_mode=CUDAGraphMode.FULL,
        block_tables=(),
        slot_mappings=torch.empty(0, dtype=torch.int64),
        attn_groups=[],
        kv_cache_config=Mock(),
    )

    kwargs = build_attn_metadata.call_args.kwargs
    assert kwargs["num_reqs"] == num_reqs_padded
    is_prefilling = kwargs["is_prefilling"]
    assert is_prefilling.shape[0] == num_reqs_padded
    assert is_prefilling[:num_reqs].numpy().tolist() == (
        input_batch.is_prefilling_np.tolist()
    )
    assert not is_prefilling[num_reqs:].any()


def test_unpadded_batch_passes_is_prefilling_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = object.__new__(DefaultModelState)
    state.max_model_len = 8192
    state.supports_mm_inputs = False
    state.encoder_cache = None
    state.model_config = SimpleNamespace(is_mm_prefix_lm=False, rswa_window=None)

    input_batch = _padded_decode_batch(3, 3, query_len=8)
    build_attn_metadata = Mock(return_value={})
    monkeypatch.setattr(default, "build_attn_metadata", build_attn_metadata)

    state.prepare_attn(
        input_batch=input_batch,
        cudagraph_mode=CUDAGraphMode.NONE,
        block_tables=(),
        slot_mappings=torch.empty(0, dtype=torch.int64),
        attn_groups=[],
        kv_cache_config=Mock(),
    )

    is_prefilling = build_attn_metadata.call_args.kwargs["is_prefilling"]
    assert is_prefilling.numpy().tolist() == input_batch.is_prefilling_np.tolist()
