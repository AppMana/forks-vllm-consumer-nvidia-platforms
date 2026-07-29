# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The hash router's three integer arguments must agree on element width.

``ops.topk_hash_softplus_sqrt`` dispatches on ``topk_ids``, whose dtype is the
MoE backend's ``indices_type``, and then indexes ``hash_indices_table``
(``[vocab_size, 6]``) with ``input_tokens`` at that width. ``hash_indices_dtype``
is int64 for mega-MoE and int32 otherwise, so the two can legitimately differ --
and when they do, the table is read at the wrong stride and the hash layers (the
front of the DSv4 stack) return wrong expert ids while the model stays fluent.

The upstream merge replaced the baseline's cast-to-``indices_type`` with a cast
of ``input_tokens`` to ``hash_indices_table.dtype``, which makes the two
arguments agree with each other but not with what the kernel dispatches on
(restored in c5d7410721).

The kernel itself is a CUDA op, so it is replaced with a recorder: the cast
logic under test is pure Python and runs anywhere.
"""

import pytest
import torch

VOCAB_SIZE = 64
HASH_TOPK = 6
NUM_EXPERTS = 128
NUM_TOKENS = 4
TOPK = 8


@pytest.fixture
def record_kernel_dtypes(monkeypatch):
    """Replace the CUDA hash-topk op with a recorder of its argument dtypes."""
    import vllm._custom_ops as ops
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.layers.fused_moe.router import fused_topk_bias_router

    recorded: dict[str, torch.dtype] = {}

    def fake_topk_hash_softplus_sqrt(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        routed_scaling_factor,
        e_score_correction_bias,
        input_tokens,
        hash_indices_table,
        is_padding=None,
    ):
        recorded["topk_indices"] = topk_indices.dtype
        recorded["input_tokens"] = input_tokens.dtype
        recorded["hash_indices_table"] = hash_indices_table.dtype

    monkeypatch.setattr(ops, "topk_hash_softplus_sqrt", fake_topk_hash_softplus_sqrt)
    # Keep the routing decision identical on every machine: the fused DSv4
    # topk path is CUDA-only and bypasses the hash table entirely.
    monkeypatch.setattr(
        fused_topk_bias_router, "can_use_dsv4_topk", lambda *args, **kwargs: False
    )
    monkeypatch.setattr(
        rocm_aiter_ops, "is_fused_moe_enabled", lambda *args, **kwargs: False
    )
    return recorded


@pytest.mark.parametrize(
    ("indices_type", "hash_table_dtype", "input_tokens_dtype"),
    [
        # mega-MoE: int64 table, int32 backend indices -- the disagreeing case.
        (torch.int32, torch.int64, torch.int32),
        (torch.int32, torch.int64, torch.int64),
        # int64 backend indices against an int32 table.
        (torch.int64, torch.int32, torch.int32),
        # Already in agreement: the casts must be no-ops, not flips.
        (torch.int32, torch.int32, torch.int32),
        (torch.int64, torch.int64, torch.int64),
    ],
)
def test_hash_router_casts_both_arguments_to_indices_type(
    record_kernel_dtypes,
    indices_type: torch.dtype,
    hash_table_dtype: torch.dtype,
    input_tokens_dtype: torch.dtype,
) -> None:
    """Both ``input_tokens`` and ``hash_indices_table`` follow ``indices_type``.

    Would have caught the merge's fused_topk_bias regression (c5d7410721):
    casting ``input_tokens`` to the table's dtype leaves the table itself at a
    width the kernel does not dispatch on.
    """
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
        fused_topk_bias,
    )

    fused_topk_bias(
        hidden_states=torch.zeros(NUM_TOKENS, 16, dtype=torch.bfloat16),
        gating_output=torch.zeros(NUM_TOKENS, NUM_EXPERTS, dtype=torch.float32),
        scoring_func="sqrtsoftplus",
        e_score_correction_bias=torch.zeros(NUM_EXPERTS, dtype=torch.float32),
        topk=TOPK,
        renormalize=True,
        indices_type=indices_type,
        input_tokens=torch.zeros(NUM_TOKENS, dtype=input_tokens_dtype),
        hash_indices_table=torch.zeros(VOCAB_SIZE, HASH_TOPK, dtype=hash_table_dtype),
    )

    assert record_kernel_dtypes == {
        "topk_indices": indices_type,
        "input_tokens": indices_type,
        "hash_indices_table": indices_type,
    }, record_kernel_dtypes
