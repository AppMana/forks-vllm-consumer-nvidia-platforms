# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The LMCache MP connector reports KV load failures per request.

The scheduler refuses block-level failure reports (``invalid_block_ids``) on
layouts with more than one KV cache group, because block ids are only unique
within a group, and raises a fatal engine error instead. DeepSeek V4 has
several groups, so the connector must report failed requests through
``KVConnectorTransferResults.failed_recving`` and keep the block ids for
single-group layouts only.
"""

from types import SimpleNamespace

import pytest

lmcache_mp_connector = pytest.importorskip(
    "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector"
)


class _FakeWorkerAdapter:
    def __init__(self):
        self.request_errors = {"req-fail"}
        self.block_errors = {7, 9}

    def get_finished(self, finished_req_ids):
        return set(), {"req-ok"}

    def get_request_ids_with_load_errors(self):
        errors, self.request_errors = self.request_errors, set()
        return errors

    def get_block_ids_with_load_errors(self):
        errors, self.block_errors = self.block_errors, set()
        return errors


def _connector(num_groups: int):
    connector = object.__new__(lmcache_mp_connector.LMCacheMPConnector)
    connector._kv_cache_config = SimpleNamespace(
        kv_cache_groups=[object()] * num_groups
    )
    connector.worker_adapter = _FakeWorkerAdapter()
    connector.lazy_offload = False
    connector._can_store = True
    return connector


def test_failed_loads_are_reported_per_request_on_multi_group_layouts():
    connector = _connector(num_groups=3)

    results = connector.get_transfer_results(set())

    assert results.failed_recving == {"req-fail"}
    assert results.finished_recving == {"req-ok", "req-fail"}
    assert connector.get_block_ids_with_load_errors() == set()


def test_single_group_layouts_keep_block_level_reports():
    connector = _connector(num_groups=1)

    results = connector.get_transfer_results(set())

    assert results.failed_recving == {"req-fail"}
    assert connector.get_block_ids_with_load_errors() == {7, 9}
