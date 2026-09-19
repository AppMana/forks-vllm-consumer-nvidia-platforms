# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode import speculator as speculator_module
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


class _Speculator(DraftModelSpeculator):
    """Concrete shell: only the shared metadata builder is under test."""

    def capture(self, *args, **kwargs):
        raise NotImplementedError

    def init_cudagraph_manager(self, *args, **kwargs):
        raise NotImplementedError

    def load_draft_model(self, *args, **kwargs):
        raise NotImplementedError

    def propose(self, *args, **kwargs):
        raise NotImplementedError


def _speculator(max_num_reqs: int) -> DraftModelSpeculator:
    spec = object.__new__(_Speculator)
    spec.max_num_reqs = max_num_reqs
    spec.max_model_len = 8192
    spec.draft_max_seq_len = 8192
    spec.arange_np = np.arange(max_num_reqs + 1, dtype=np.int32)
    spec.draft_is_prefilling = torch.zeros(max_num_reqs, dtype=torch.bool)
    spec.attn_groups = []
    spec.kv_cache_config = Mock()
    spec.input_buffers = SimpleNamespace(
        query_start_loc=torch.zeros(max_num_reqs + 1, dtype=torch.int32),
        seq_lens=torch.zeros(max_num_reqs, dtype=torch.int32),
        dcp_local_seq_lens=None,
    )
    spec.block_tables = SimpleNamespace(
        input_block_tables=[torch.zeros(max_num_reqs, 4, dtype=torch.int32)],
        slot_mappings=torch.zeros(1, max_num_reqs * 8, dtype=torch.int64),
        cp_size=1,
        cp_rank=0,
        cp_interleave=1,
    )
    return spec


@pytest.mark.parametrize(("num_reqs", "num_reqs_padded"), [(3, 4), (5, 8), (7, 8)])
def test_draft_full_graph_padding_extends_is_prefilling(
    monkeypatch: pytest.MonkeyPatch, num_reqs: int, num_reqs_padded: int
) -> None:
    """The draft step's FULL graph pads its rows exactly like the target's;
    the is_prefilling vector handed to the builders must be the padded
    length, or split_decodes_and_prefills fails on the first padded draft
    batch (seen as 'size of tensor a (5) must match b (4)' at C=5)."""
    spec = _speculator(max_num_reqs=8)
    build_attn_metadata = Mock(return_value={})
    monkeypatch.setattr(speculator_module, "build_attn_metadata", build_attn_metadata)
    batch_desc = SimpleNamespace(
        num_reqs=num_reqs_padded,
        num_tokens=num_reqs_padded,
        cg_mode=CUDAGraphMode.FULL,
    )

    spec._build_uniform_attn_metadata(
        batch_desc=batch_desc,
        num_reqs=num_reqs,
        num_query_per_req=1,
        seq_lens_cpu_upper_bound=torch.full((num_reqs,), 100, dtype=torch.int32),
        step=1,
    )

    kwargs = build_attn_metadata.call_args.kwargs
    assert kwargs["num_reqs"] == num_reqs_padded
    assert kwargs["is_prefilling"].shape[0] == num_reqs_padded
    assert not kwargs["is_prefilling"].any()
    assert kwargs["seq_lens_cpu_upper_bound"].shape[0] == num_reqs_padded
