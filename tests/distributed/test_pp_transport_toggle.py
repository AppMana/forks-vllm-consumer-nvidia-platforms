# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PP transport toggle precedence: appmana stash > env var > built-in default.

The PP tensor-dict metadata-caching toggle historically read the
``VLLM_PP_CACHE_TENSOR_DICT_METADATA`` env var.
Env vars do NOT reliably propagate to remote Ray workers (vLLM forwards only an
allowlist), so the toggles now also honor a value resolved from the checkpoint's
``appmana.pp_transport`` block, which rides ``VllmConfig`` to every worker.

These are pure-Python unit tests (no GPU / no distributed init / no ray): the
gate is exercised by calling the unbound ``GroupCoordinator`` method against a
minimal stub ``self``, matching how a live coordinator caches the value once.
"""

from types import SimpleNamespace

import vllm.envs as envs
import vllm.transformers_utils.configs.deepseek_v4_appmana as appmana_mod
from vllm.distributed.parallel_state import GroupCoordinator


def _pp_metadata_gate(stub) -> bool:
    return GroupCoordinator._pp_metadata_cache_enabled(stub)


def test_pp_metadata_gate_uses_env_when_no_appmana_override(monkeypatch):
    monkeypatch.setattr(appmana_mod, "_PP_CACHE_METADATA_OVERRIDE", None)
    monkeypatch.setattr(envs, "VLLM_PP_CACHE_TENSOR_DICT_METADATA", False)
    assert _pp_metadata_gate(SimpleNamespace()) is False

    monkeypatch.setattr(envs, "VLLM_PP_CACHE_TENSOR_DICT_METADATA", True)
    assert _pp_metadata_gate(SimpleNamespace()) is True


def test_pp_metadata_gate_appmana_override_wins_over_env(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_PP_CACHE_TENSOR_DICT_METADATA", True)
    monkeypatch.setattr(appmana_mod, "_PP_CACHE_METADATA_OVERRIDE", False)
    assert _pp_metadata_gate(SimpleNamespace()) is False

    monkeypatch.setattr(envs, "VLLM_PP_CACHE_TENSOR_DICT_METADATA", False)
    monkeypatch.setattr(appmana_mod, "_PP_CACHE_METADATA_OVERRIDE", True)
    assert _pp_metadata_gate(SimpleNamespace()) is True


def test_pp_gate_caches_first_read(monkeypatch):
    """Once resolved, the coordinator caches the value on self (set-once)."""
    monkeypatch.setattr(appmana_mod, "_PP_CACHE_METADATA_OVERRIDE", None)
    monkeypatch.setattr(envs, "VLLM_PP_CACHE_TENSOR_DICT_METADATA", True)
    stub = SimpleNamespace()
    assert _pp_metadata_gate(stub) is True
    # Flip both sources; the cached attr must pin the first-read value.
    monkeypatch.setattr(envs, "VLLM_PP_CACHE_TENSOR_DICT_METADATA", False)
    monkeypatch.setattr(appmana_mod, "_PP_CACHE_METADATA_OVERRIDE", False)
    assert _pp_metadata_gate(stub) is True
    assert stub._pp_cache_tensor_dict_metadata is True
