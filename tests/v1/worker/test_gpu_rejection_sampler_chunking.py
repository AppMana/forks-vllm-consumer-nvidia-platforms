# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from types import MethodType, SimpleNamespace
from typing import get_args

import numpy as np
import pytest
import torch

from vllm.config.model import PROCESSED_LOGPROBS_MODES, LogprobsMode
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.sample.output import SamplingMaskTensors
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
    RejectionSampler,
    _iter_request_chunks,
)


@pytest.mark.parametrize("use_v2", [False, True])
def test_rejection_warmup_covers_absent_drafts_and_runtime_token_dtype(
    monkeypatch, use_v2
):
    import vllm.model_executor.warmup.spec_decode_rejection_warmup as warmup
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils

    # CPU tensors suffice to inspect the JIT-relevant input contract.
    def on_cpu(factory):
        def allocate(*args, **kwargs):
            kwargs["device"] = "cpu"
            return factory(*args, **kwargs)

        return allocate

    for name in ("zeros", "full", "tensor", "arange"):
        monkeypatch.setattr(torch, name, on_cpu(getattr(torch, name)))
    signatures = set()

    def record(**kwargs):
        draft = kwargs["draft_logits"]
        signatures.add(
            (kwargs["target_logits"].dtype, None if draft is None else draft.dtype)
        )
        assert kwargs["draft_sampled"].dtype == torch.int32

    monkeypatch.setattr(rejection_sampler_utils, "rejection_sample", record)
    ids = torch.empty(8, dtype=torch.int32)
    worker = SimpleNamespace(
        use_v2_model_runner=use_v2,
        model_runner=SimpleNamespace(
            input_buffers=SimpleNamespace(input_ids=ids),
            input_ids=SimpleNamespace(gpu=ids),
        ),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                get_vocab_size=lambda: 32, dtype=torch.bfloat16
            ),
            speculative_config=SimpleNamespace(num_speculative_tokens=7),
        ),
    )
    warmup.spec_decode_rejection_warmup(worker)
    assert signatures == {
        (target, draft)
        for target in (torch.bfloat16, torch.float32)
        for draft in (None, torch.bfloat16, torch.float32)
    }


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
        seq_lens_cpu_upper_bound=torch.tensor([8], dtype=torch.int32),
    )
    rejection_sampler = object.__new__(RejectionSampler)
    rejection_sampler.num_speculative_steps = 5
    rejection_sampler.sampler = SimpleNamespace(
        compute_nans=False,
        return_sampling_mask=False,
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

    # Bind to the real signature: the deferred path is a call only this fork
    # makes, so an argument upstream adds to _verify must fail here.
    verify_signature = inspect.signature(RejectionSampler._verify)

    def fake_verify(self, *args):
        bound = verify_signature.bind(self, *args).arguments
        logits = bound["logits"]
        draft_sampled = bound["draft_sampled"]
        pos = bound["pos"]
        cu_num_logits = bound["cu_num_logits"]
        assert bound["seq_lens_upper_bound_np"].tolist() == [8]
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


def _make_input_batch(
    cu_num_logits_np: np.ndarray,
    idx_mapping_np: np.ndarray,
    device: torch.device,
) -> SimpleNamespace:
    """Minimal InputBatch fields read by RejectionSampler._verify_in_chunks."""
    num_reqs = len(idx_mapping_np)
    num_logits_per_req = np.diff(cu_num_logits_np)
    return SimpleNamespace(
        num_reqs=num_reqs,
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
        seq_lens_cpu_upper_bound=torch.full((num_reqs,), 64, dtype=torch.int32),
    )


def _make_rejection_sampler(
    sampler: SimpleNamespace, num_speculative_steps: int
) -> RejectionSampler:
    """RejectionSampler for standard, fixed-boundary, unwatermarked verification."""
    rejection_sampler = object.__new__(RejectionSampler)
    rejection_sampler.sampler = sampler
    rejection_sampler.num_speculative_steps = num_speculative_steps
    rejection_sampler.enable_adaptive_verification = False
    rejection_sampler.synthetic_conditional_rates = None
    rejection_sampler.use_block_verification = False
    rejection_sampler.watermark_key = None
    return rejection_sampler


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
    input_batch = _make_input_batch(cu_num_logits_np, idx_mapping_np, device)
    rejection_sampler = _make_rejection_sampler(
        SimpleNamespace(logprobs_mode=logprobs_mode, return_sampling_mask=False),
        num_speculative_steps=3,
    )

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

    sampled, num_sampled, chunked_logprobs, sampling_mask_tensors = (
        rejection_sampler._verify_in_chunks(
            logits,
            input_batch,
            draft_logits=None,
            draft_sampled=torch.arange(10, device=device),
            pos=torch.arange(10, device=device),
            max_chunk_logits=5,
            max_num_logprobs=2,
        )
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
    assert sampling_mask_tensors is None


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_replay_on_off_preserves_rejection_sampling_and_rng(monkeypatch):
    device = torch.device("cuda")
    num_reqs = 3
    num_speculative_steps = 3
    rows_per_request = num_speculative_steps + 1
    vocab_size = 8
    cu_num_logits_np = np.arange(0, 13, rows_per_request, dtype=np.int32)
    idx_mapping_np = np.arange(num_reqs, dtype=np.int32)
    input_batch = _make_input_batch(cu_num_logits_np, idx_mapping_np, device)
    processed_logits = torch.zeros((12, vocab_size), device=device)
    processed_logits[0, 0] = -float("inf")
    processed_logits[rows_per_request + 1, 1] = -float("inf")
    draft_logits = torch.zeros(
        (num_reqs, num_speculative_steps, vocab_size), device=device
    )
    draft_sampled = torch.zeros(
        (num_reqs, rows_per_request), dtype=torch.int64, device=device
    )
    draft_sampled[:, 1:] = torch.tensor([0, 1, 2], device=device)
    draft_sampled = draft_sampled.flatten()
    sampler = SimpleNamespace(
        logprobs_mode="processed_logprobs",
        return_sampling_mask=False,
        sampling_states=SimpleNamespace(
            top_k=SimpleNamespace(np=np.full(num_reqs, vocab_size)),
            temperature=SimpleNamespace(gpu=torch.ones(num_reqs, device=device)),
            seeds=SimpleNamespace(
                gpu=torch.arange(num_reqs, dtype=torch.int64, device=device)
            ),
        ),
        use_fp64_gumbel=False,
        apply_sampling_params=lambda logits, *_: logits,
    )
    rejection_sampler = _make_rejection_sampler(sampler, num_speculative_steps)
    pos = torch.arange(12, dtype=torch.int32, device=device)

    pack_calls: list[tuple[list[int], int, int]] = []
    pack = SamplingMaskTensors.from_logits.__func__

    def track_pack(
        cls,
        logits,
        cu_num_logits,
        num_sampled_tokens,
        max_num_kept,
        rows_per_request=1,
    ):
        pack_calls.append(
            (cu_num_logits.cpu().tolist(), max_num_kept, rows_per_request)
        )
        return pack(
            cls,
            logits,
            cu_num_logits,
            num_sampled_tokens,
            max_num_kept,
            rows_per_request,
        )

    monkeypatch.setattr(
        SamplingMaskTensors,
        "from_logits",
        classmethod(track_pack),
    )

    rng_before = torch.cuda.get_rng_state()
    off_sampled, off_counts, off_logprobs, off_masks = (
        rejection_sampler._verify_in_chunks(
            processed_logits,
            input_batch,
            draft_logits,
            draft_sampled,
            pos,
            max_chunk_logits=8,
            max_num_logprobs=0,
        )
    )
    rng_after_off = torch.cuda.get_rng_state()
    assert pack_calls == []

    sampler.return_sampling_mask = True
    on_sampled, on_counts, on_logprobs, on_masks = rejection_sampler._verify_in_chunks(
        processed_logits,
        input_batch,
        draft_logits,
        draft_sampled,
        pos,
        max_chunk_logits=8,
        max_num_logprobs=0,
    )
    rng_after_on = torch.cuda.get_rng_state()

    assert pack_calls == [
        ([0, 4, 8], vocab_size, rows_per_request),
        ([0, 4], vocab_size, rows_per_request),
    ]
    assert off_masks is None and on_masks is not None
    assert on_masks.rows_per_request == rows_per_request
    assert on_masks.token_ids.shape == (num_reqs * rows_per_request, vocab_size)
    assert on_masks.packed_mask.shape[0] == num_reqs * rows_per_request
    assert off_counts.tolist() == [1, 2, 4]
    assert torch.equal(off_counts, on_counts)
    counts = off_counts.cpu().tolist()
    for req_idx, count in enumerate(counts):
        assert torch.equal(off_sampled[req_idx, :count], on_sampled[req_idx, :count])

    mapped_rows = [
        int(input_batch.cu_num_logits_np[req_idx]) + slot_idx
        for req_idx, count in enumerate(counts)
        for slot_idx in range(count)
    ]
    assert off_logprobs is not None and on_logprobs is not None
    assert torch.equal(
        off_logprobs.logprob_token_ids[mapped_rows],
        on_logprobs.logprob_token_ids[mapped_rows],
    )
    assert torch.equal(
        off_logprobs.logprobs[mapped_rows], on_logprobs.logprobs[mapped_rows]
    )
    assert torch.equal(
        off_logprobs.selected_token_ranks[mapped_rows],
        on_logprobs.selected_token_ranks[mapped_rows],
    )
    assert torch.equal(rng_before, rng_after_off)
    assert torch.equal(rng_after_off, rng_after_on)

    masks = (
        on_masks.to_cpu_nonblocking().tolists(on_counts.cpu().numpy()).to_nested_list()
    )
    expected_masks = [
        torch.isfinite(processed_logits[row]).nonzero().flatten().tolist()
        for row in mapped_rows
    ]
    emitted = [
        int(on_sampled[req_idx, slot_idx])
        for req_idx, count in enumerate(on_counts.cpu().tolist())
        for slot_idx in range(count)
    ]
    expected_logprobs = torch.stack(
        [
            torch.log_softmax(processed_logits[row], dim=-1)[token_id]
            for row, token_id in zip(mapped_rows, emitted)
        ]
    )

    assert masks == expected_masks
    assert all(token_id in mask for token_id, mask in zip(emitted, masks))
    assert torch.equal(on_logprobs.logprobs[mapped_rows, 0], expected_logprobs)
