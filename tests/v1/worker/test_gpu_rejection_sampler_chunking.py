# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace
from typing import get_args

import numpy as np
import pytest
import torch

from vllm.config.model import PROCESSED_LOGPROBS_MODES, LogprobsMode
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
    RejectionSampler,
    _iter_request_chunks,
)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_real_pp_anchor_is_verified_and_caps_emission_at_k() -> None:
    device = torch.device("cuda")
    input_ids = torch.tensor(
        [99, 101, 102, 103, 104, 105], dtype=torch.int32, device=device
    )
    positions = torch.arange(2, 8, dtype=torch.int64, device=device)
    cu_num_logits_np = np.array([0, 6], dtype=np.int32)
    input_batch = SimpleNamespace(
        num_reqs=1,
        idx_mapping=torch.tensor([0], dtype=torch.int32, device=device),
        idx_mapping_np=np.array([0], dtype=np.int32),
        input_ids=input_ids,
        positions=positions,
        logits_indices=torch.arange(6, dtype=torch.int64, device=device),
        cu_num_logits=torch.tensor([0, 6], dtype=torch.int32, device=device),
        cu_num_logits_np=cu_num_logits_np,
        expanded_idx_mapping=torch.zeros(6, dtype=torch.int32, device=device),
        expanded_local_pos=torch.arange(6, dtype=torch.int32, device=device),
        seq_lens=torch.tensor([8], dtype=torch.int32, device=device),
        num_draft_tokens_per_req=np.array([5], dtype=np.int32),
        replayed_pp_anchor=torch.tensor([True], device=device),
        replayed_pp_anchor_np=np.array([True]),
    )
    rejection_sampler = object.__new__(RejectionSampler)
    rejection_sampler.num_speculative_steps = 5
    rejection_sampler.sampler = SimpleNamespace(
        compute_nans=False,
        sampling_states=SimpleNamespace(
            max_num_logprobs=lambda _idx_mapping_np: NO_LOGPROBS
        ),
        req_states=SimpleNamespace(
            total_len=SimpleNamespace(
                gpu=torch.tensor([3], dtype=torch.int32, device=device)
            ),
            prefill_len=SimpleNamespace(
                gpu=torch.tensor([2], dtype=torch.int32, device=device)
            ),
        ),
    )

    def fake_verify(
        self,
        logits,
        _draft_logits,
        draft_sampled,
        pos,
        cu_num_logits,
        *_mappings,
    ):
        assert logits.shape[0] == 6
        assert draft_sampled.tolist() == input_ids.tolist()
        assert pos.tolist() == positions.tolist()
        assert cu_num_logits.tolist() == [0, 6]
        sampled = torch.tensor(
            [[201, 202, 203, 204, 205, 206]], dtype=torch.int64, device=device
        )
        return logits, sampled, torch.tensor([6], dtype=torch.int32, device=device)

    rejection_sampler._verify = MethodType(fake_verify, rejection_sampler)
    logits = torch.zeros((6, 32), dtype=torch.float32, device=device)

    output = rejection_sampler(
        logits,
        input_batch,
        draft_logits=None,
        prev_sampled_tokens=torch.tensor([[99]], dtype=torch.int64, device=device),
    )

    assert output.sampled_token_ids[0, :5].tolist() == [201, 202, 203, 204, 205]
    assert output.num_sampled.tolist() == [5]
    assert output.num_rejected.tolist() == [0]


def test_iter_request_chunks_preserves_request_boundaries():
    cu_num_logits = np.array([0, 3, 4, 11, 13], dtype=np.int32)

    assert list(_iter_request_chunks(cu_num_logits, max_chunk_logits=5)) == [
        (0, 2),
        (2, 3),
        (3, 4),
    ]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize("logprobs_mode", get_args(LogprobsMode))
def test_chunked_scores_match_full_batch(logprobs_mode: str):
    device = torch.device("cuda")
    cu_num_logits_np = np.array([0, 3, 4, 8, 10], dtype=np.int32)
    num_logits_per_req = np.diff(cu_num_logits_np)
    idx_mapping_np = np.array([7, 2, 9, 1], dtype=np.int32)
    input_batch = SimpleNamespace(
        num_reqs=4,
        cu_num_logits_np=cu_num_logits_np,
        cu_num_logits=torch.from_numpy(cu_num_logits_np).to(device),
        idx_mapping_np=idx_mapping_np,
        idx_mapping=torch.from_numpy(idx_mapping_np).to(device),
        expanded_idx_mapping=torch.from_numpy(
            np.repeat(idx_mapping_np, num_logits_per_req)
        ).to(device),
        expanded_local_pos=torch.from_numpy(
            np.concatenate(
                [np.arange(count, dtype=np.int32) for count in num_logits_per_req]
            )
        ).to(device),
    )
    rejection_sampler = object.__new__(RejectionSampler)
    rejection_sampler.sampler = SimpleNamespace(logprobs_mode=logprobs_mode)
    rejection_sampler.num_speculative_steps = 3
    rejection_sampler.enable_adaptive_verification = False

    def fake_verify(
        self,
        logits,
        _draft_logits,
        _draft_sampled,
        _pos,
        cu_num_logits,
        idx_mapping,
        *_mappings,
    ):
        num_sampled = torch.diff(cu_num_logits).to(torch.int32)
        sampled = (
            idx_mapping.to(torch.int64).unsqueeze(1) + torch.arange(4, device=device)
        ) % logits.shape[1]
        return logits.float() + 1, sampled, num_sampled

    rejection_sampler._verify = MethodType(fake_verify, rejection_sampler)
    logits = torch.arange(170, dtype=torch.float32, device=device).view(10, 17)

    sampled, num_sampled, chunked_logprobs = rejection_sampler._verify_in_chunks(
        logits,
        input_batch,
        draft_logits=None,
        draft_sampled=torch.arange(10, device=device),
        pos=torch.arange(10, device=device),
        max_chunk_logits=5,
        max_num_logprobs=2,
    )
    score_logits = logits + 1 if logprobs_mode in PROCESSED_LOGPROBS_MODES else logits
    full_logprobs = rejection_sampler._get_logprobs_tensors(
        sampled,
        num_sampled,
        score_logits,
        input_batch.cu_num_logits,
        input_batch.cu_num_logits_np,
        max_num_logprobs=2,
    )

    assert sampled[:, 0].tolist() == idx_mapping_np.tolist()
    assert num_sampled.tolist() == num_logits_per_req.tolist()
    assert chunked_logprobs is not None
    assert full_logprobs is not None
    assert torch.equal(
        chunked_logprobs.logprob_token_ids,
        full_logprobs.logprob_token_ids,
    )
    assert torch.equal(chunked_logprobs.logprobs, full_logprobs.logprobs)
    assert torch.equal(
        chunked_logprobs.selected_token_ranks,
        full_logprobs.selected_token_ranks,
    )
    assert (
        chunked_logprobs.cu_num_generated_tokens
        == full_logprobs.cu_num_generated_tokens
    )
