# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The indexer prefill workspace must size itself from config that exists.

``get_max_prefill_buffer_size`` runs at worker init, on the line after the
kernel resolution, for every DSV4 and DSV3.2 serve. After V1 unified chunked
prefill and removed ``SchedulerConfig.max_num_partial_prefills`` it raised
AttributeError there and took every worker down, so the sparse indexer never
ran on this branch at all (7e4c1e1559).

The companion static audit lives in ``test_static_name_resolution.py``; this is
the executable half.
"""

from types import SimpleNamespace

import pytest

MAX_MODEL_LEN = 8192


def make_vllm_config(max_model_len: int = MAX_MODEL_LEN) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        # Deliberately a SchedulerConfig-free namespace: reading anything off
        # it here is the bug under test.
        scheduler_config=SimpleNamespace(),
    )


@pytest.mark.parametrize("compress_ratio", [1, 4, 128])
def test_prefill_buffer_size_needs_no_scheduler_fields(compress_ratio: int) -> None:
    """Would have caught 7e4c1e1559: the AttributeError at worker init."""
    from vllm.utils.math_utils import cdiv
    from vllm.v1.attention.backends.mla.indexer import (
        MAX_CONCURRENT_PREFILL_CONTEXTS,
        get_max_prefill_buffer_size,
    )

    size = get_max_prefill_buffer_size(make_vllm_config(), compress_ratio)

    assert size == MAX_CONCURRENT_PREFILL_CONTEXTS * cdiv(MAX_MODEL_LEN, compress_ratio)
    assert size > 0


def test_prefill_buffer_size_stays_inside_the_upstream_envelope() -> None:
    """The compress_ratio divisor may only ever shrink the budget.

    Upstream sizes the same workspace as ``40 * max_model_len``, the figure
    that makes 132-byte indexer rows fit the flashmla_sparse workspace. The
    fork's divisor must not push it past that.
    """
    from vllm.v1.attention.backends.mla.indexer import (
        MAX_CONCURRENT_PREFILL_CONTEXTS,
        get_max_prefill_buffer_size,
    )

    assert MAX_CONCURRENT_PREFILL_CONTEXTS == 40
    envelope = MAX_CONCURRENT_PREFILL_CONTEXTS * MAX_MODEL_LEN
    for compress_ratio in (1, 4, 128):
        assert (
            get_max_prefill_buffer_size(make_vllm_config(), compress_ratio) <= envelope
        )
