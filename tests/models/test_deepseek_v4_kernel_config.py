# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified checkpoint-config-driven kernel configuration for AppMana DSV4.

Covers the ``"vllm"`` config.json block: symbol-list resolution to roles,
fail-closed validation, blockless defaults (selector roles get their
documented defaults, all toggle roles off), indexer int8 independence from
the dense runtime, cache_type defaulting, and Ray unpickle gate propagation.
"""

import pickle
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.quantization.dsv4_int as dsv4_int_module
import vllm.transformers_utils.configs.dsv4.kernel_config as kernel_config
from vllm.model_executor.layers.quantization.dsv4_int import Dsv4IntConfig
from vllm.transformers_utils.configs.dsv4.kernel_config import (
    VLLM_CONFIG_KEY,
    DENSE_EXPERTS_INT8_ACTIVATION,
    INDEXER_CACHE_INT8_WRITER,
    INDEXER_QUERY_INT8_QUANT,
    INDEXER_STREAMING_TOPK_PREFILL,
    KERNEL_REGISTRY,
    ROLE_DENSE_EXPERTS_INT8_ACTIVATION,
    ROLE_INDEXER_CACHE_INT8,
    ROLE_INDEXER_QUERY_INT8,
    ROLE_INDEXER_STREAMING_TOPK_PREFILL,
    ROLE_SPARSE_MLA_DECODE_FP8,
    ROLE_SPARSE_MLA_DECODE_INT8,
    ROLE_SPARSE_MLA_PREFILL,
    SPARSE_MLA_DECODE_FP8_FLASH,
    SPARSE_MLA_DECODE_FP8_SPARKINFER,
    SPARSE_MLA_DECODE_FP8_TRITON,
    SPARSE_MLA_DECODE_INT8_FLASH,
    SPARSE_MLA_DECODE_INT8_TRITON,
    SPARSE_MLA_PREFILL_FLASH,
    SPARSE_MLA_PREFILL_SPARKINFER,
    SPARSE_MLA_PREFILL_TRITON,
    activate_kernel_config,
    apply_checkpoint_config,
    dense_experts_int8_activation_enabled,
    indexer_cache_int8_enabled,
    indexer_prefill_topk_slab_rows_override,
    indexer_query_int8_enabled,
    indexer_streaming_topk_prefill_enabled,
    resolve_kernel_config,
    resolve_kernel_config_from_hf_config,
    resolved_proof_line,
)

_INT_GROUPS = {
    "experts_w4a16": {"weights": {"num_bits": 4, "type": "int"}},
    "linears_w8a16": {
        "weights": {
            "num_bits": 8,
            "type": "int",
            "symmetric": True,
            "strategy": "channel",
        }
    },
}


@pytest.fixture(autouse=True)
def _reset_kernel_config_globals(monkeypatch):
    monkeypatch.setattr(kernel_config, "_ACTIVE_CONFIG", None)
    monkeypatch.setattr(
        dsv4_int_module, "_DSV4_INT4_EXPERTS_INT8_DENSE_ACTIVE", False
    )


# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------


def test_registry_symbols_are_importable_callables():
    """Every registry value is the exact FQN of the activated callable."""
    import importlib

    for symbol in KERNEL_REGISTRY:
        module_name, _, attr = symbol.rpartition(".")
        module = importlib.import_module(module_name)
        obj = getattr(module, attr)
        assert callable(obj), symbol


def test_registry_covers_all_roles_with_expected_symbols():
    assert KERNEL_REGISTRY[SPARSE_MLA_DECODE_FP8_FLASH] == ROLE_SPARSE_MLA_DECODE_FP8
    assert KERNEL_REGISTRY[SPARSE_MLA_DECODE_FP8_TRITON] == ROLE_SPARSE_MLA_DECODE_FP8
    assert (
        KERNEL_REGISTRY[SPARSE_MLA_DECODE_INT8_TRITON] == ROLE_SPARSE_MLA_DECODE_INT8
    )
    assert (
        KERNEL_REGISTRY[SPARSE_MLA_DECODE_INT8_FLASH] == ROLE_SPARSE_MLA_DECODE_INT8
    )
    assert SPARSE_MLA_DECODE_INT8_FLASH == "flash_mla.sparse_mla_decode_int8"
    assert KERNEL_REGISTRY[SPARSE_MLA_PREFILL_TRITON] == ROLE_SPARSE_MLA_PREFILL
    assert KERNEL_REGISTRY[SPARSE_MLA_PREFILL_FLASH] == ROLE_SPARSE_MLA_PREFILL
    assert (
        KERNEL_REGISTRY[SPARSE_MLA_DECODE_FP8_SPARKINFER]
        == ROLE_SPARSE_MLA_DECODE_FP8
    )
    assert KERNEL_REGISTRY[SPARSE_MLA_PREFILL_SPARKINFER] == ROLE_SPARSE_MLA_PREFILL
    assert KERNEL_REGISTRY[INDEXER_CACHE_INT8_WRITER] == ROLE_INDEXER_CACHE_INT8
    assert KERNEL_REGISTRY[INDEXER_QUERY_INT8_QUANT] == ROLE_INDEXER_QUERY_INT8
    assert (
        KERNEL_REGISTRY[DENSE_EXPERTS_INT8_ACTIVATION]
        == ROLE_DENSE_EXPERTS_INT8_ACTIVATION
    )


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_symbol_list_resolves_roles():
    resolved = resolve_kernel_config(
        {
            "kernels": [
                SPARSE_MLA_DECODE_FP8_FLASH,
                SPARSE_MLA_PREFILL_TRITON,
                INDEXER_CACHE_INT8_WRITER,
                INDEXER_QUERY_INT8_QUANT,
                DENSE_EXPERTS_INT8_ACTIVATION,
            ],
            "cache_type": "fp8_ds_mla",
        }
    )
    assert resolved.explicit
    assert resolved.roles[ROLE_SPARSE_MLA_DECODE_FP8] == SPARSE_MLA_DECODE_FP8_FLASH
    assert resolved.roles[ROLE_SPARSE_MLA_PREFILL] == SPARSE_MLA_PREFILL_TRITON
    # Unlisted selector role falls back to its documented default.
    assert (
        resolved.roles[ROLE_SPARSE_MLA_DECODE_INT8] == SPARSE_MLA_DECODE_INT8_TRITON
    )
    assert resolved.roles[ROLE_INDEXER_CACHE_INT8] == INDEXER_CACHE_INT8_WRITER
    assert resolved.roles[ROLE_INDEXER_QUERY_INT8] == INDEXER_QUERY_INT8_QUANT
    assert (
        resolved.roles[ROLE_DENSE_EXPERTS_INT8_ACTIVATION]
        == DENSE_EXPERTS_INT8_ACTIVATION
    )
    assert resolved.cache_type == "fp8_ds_mla"


def test_sm12x_sparkinfer_symbols_resolve_roles():
    """The GB10 checkpoint block shape: sparkinfer decode+extend selected by
    FQN, fp8_ds_mla cache."""
    resolved = resolve_kernel_config(
        {
            "kernels": [
                SPARSE_MLA_DECODE_FP8_SPARKINFER,
                SPARSE_MLA_PREFILL_SPARKINFER,
            ],
            "cache_type": "fp8_ds_mla",
        }
    )
    assert resolved.explicit
    assert (
        resolved.roles[ROLE_SPARSE_MLA_DECODE_FP8] == SPARSE_MLA_DECODE_FP8_SPARKINFER
    )
    assert resolved.roles[ROLE_SPARSE_MLA_PREFILL] == SPARSE_MLA_PREFILL_SPARKINFER
    assert resolved.cache_type == "fp8_ds_mla"


def test_sparkinfer_and_flash_decode_conflict_is_hard_error():
    with pytest.raises(ValueError, match="same role"):
        resolve_kernel_config(
            {
                "kernels": [
                    SPARSE_MLA_DECODE_FP8_SPARKINFER,
                    SPARSE_MLA_DECODE_FP8_FLASH,
                ]
            }
        )


def test_unknown_symbol_is_hard_error():
    with pytest.raises(ValueError, match="[Uu]nknown"):
        resolve_kernel_config(
            {"kernels": ["flash_mla.no_such_kernel"], "cache_type": "fp8_ds_mla"}
        )


def test_duplicate_role_is_hard_error():
    with pytest.raises(ValueError, match="same role"):
        resolve_kernel_config(
            {
                "kernels": [
                    SPARSE_MLA_DECODE_FP8_FLASH,
                    SPARSE_MLA_DECODE_FP8_TRITON,
                ]
            }
        )


def test_unknown_block_key_is_hard_error():
    with pytest.raises(ValueError, match="key"):
        resolve_kernel_config({"kernels": [], "cachetype": "fp8_ds_mla"})


def test_invalid_cache_type_is_hard_error():
    with pytest.raises(ValueError, match="cache_type"):
        resolve_kernel_config({"kernels": [], "cache_type": "fp7_ds_mla"})


def test_indexer_query_int8_requires_cache_int8():
    with pytest.raises(ValueError, match="indexer_cache_int8"):
        resolve_kernel_config({"kernels": [INDEXER_QUERY_INT8_QUANT]})


def test_defaults_when_absent():
    resolved = resolve_kernel_config(None)
    assert not resolved.explicit
    assert resolved.roles[ROLE_SPARSE_MLA_DECODE_FP8] == SPARSE_MLA_DECODE_FP8_FLASH
    assert (
        resolved.roles[ROLE_SPARSE_MLA_DECODE_INT8] == SPARSE_MLA_DECODE_INT8_TRITON
    )
    assert resolved.roles[ROLE_SPARSE_MLA_PREFILL] == SPARSE_MLA_PREFILL_TRITON
    # Blockless resolution never activates a toggle role.
    assert ROLE_INDEXER_CACHE_INT8 not in resolved.roles
    assert ROLE_INDEXER_QUERY_INT8 not in resolved.roles
    assert ROLE_DENSE_EXPERTS_INT8_ACTIVATION not in resolved.roles
    assert ROLE_INDEXER_STREAMING_TOPK_PREFILL not in resolved.roles
    assert resolved.cache_type is None


def test_blockless_active_config_leaves_all_toggles_off():
    """Blockless resolution: selector defaults, every toggle gate False."""
    activate_kernel_config(resolve_kernel_config(None))
    assert not indexer_cache_int8_enabled()
    assert not indexer_query_int8_enabled()
    assert not dense_experts_int8_activation_enabled()
    assert not indexer_streaming_topk_prefill_enabled()


def test_toggle_gates_off_with_no_active_config():
    assert not indexer_cache_int8_enabled()
    assert not indexer_query_int8_enabled()
    assert not dense_experts_int8_activation_enabled()
    assert not indexer_streaming_topk_prefill_enabled()


def test_resolve_from_hf_config_reads_block():
    hf_config = SimpleNamespace(
        vllm={"kernels": [INDEXER_CACHE_INT8_WRITER], "cache_type": "fp8_ds_mla"},
        quantization_config={"quant_method": "dsv4_int"},
    )
    resolved = resolve_kernel_config_from_hf_config(hf_config)
    assert resolved.explicit
    assert resolved.roles[ROLE_INDEXER_CACHE_INT8] == INDEXER_CACHE_INT8_WRITER
    assert resolved.cache_type == "fp8_ds_mla"


def test_resolve_from_hf_config_without_block_uses_defaults():
    resolved = resolve_kernel_config_from_hf_config(SimpleNamespace())
    assert not resolved.explicit
    assert resolved.roles[ROLE_SPARSE_MLA_DECODE_FP8] == SPARSE_MLA_DECODE_FP8_FLASH
    assert (
        resolved.roles[ROLE_SPARSE_MLA_DECODE_INT8] == SPARSE_MLA_DECODE_INT8_TRITON
    )
    assert resolved.roles[ROLE_SPARSE_MLA_PREFILL] == SPARSE_MLA_PREFILL_TRITON
    assert ROLE_INDEXER_CACHE_INT8 not in resolved.roles
    assert ROLE_DENSE_EXPERTS_INT8_ACTIVATION not in resolved.roles


# ---------------------------------------------------------------------------
# Gates: indexer int8 independent of the dense runtime
# ---------------------------------------------------------------------------


def test_blockless_dsv4_int_config_leaves_dense_and_indexer_off():
    """Without a "vllm" block, the dense W4A8 runtime and both indexer int8
    paths stay OFF, even on a checkpoint carrying INT4/INT8 weight groups."""
    from vllm.models.deepseek_v4.nvidia_sm86 import triton_kernels as dsv4_sm86

    cfg = Dsv4IntConfig.from_config(
        {
            "quant_method": "dsv4_int",
            "config_groups": _INT_GROUPS,
        }
    )
    assert not cfg.experimental_int8_runtime
    assert cfg.expert_input_dtype is None
    assert not dsv4_int_module.dsv4_int4_experts_int8_dense_active()
    assert not indexer_cache_int8_enabled()
    assert not indexer_query_int8_enabled()
    assert not dense_experts_int8_activation_enabled()
    assert not dsv4_sm86.indexer_cache_is_int8()
    assert not dsv4_sm86.indexer_imma_enabled()


def test_vllm_block_enables_dense_runtime():
    cfg = Dsv4IntConfig.from_config(
        {
            "quant_method": "dsv4_int",
            "config_groups": _INT_GROUPS,
            VLLM_CONFIG_KEY: {"kernels": [DENSE_EXPERTS_INT8_ACTIVATION]},
        }
    )
    assert cfg.experimental_int8_runtime
    assert cfg.expert_input_dtype is torch.int8
    assert dsv4_int_module.dsv4_int4_experts_int8_dense_active()
    assert dense_experts_int8_activation_enabled()


def test_vllm_block_enables_indexer_int8_without_dense_runtime():
    """Indexer int8 is activatable independently of the dense runtime."""
    from vllm.models.deepseek_v4.nvidia_sm86 import triton_kernels as dsv4_sm86

    cfg = Dsv4IntConfig.from_config(
        {
            "quant_method": "dsv4_int",
            "config_groups": _INT_GROUPS,
            VLLM_CONFIG_KEY: {
                "kernels": [INDEXER_CACHE_INT8_WRITER, INDEXER_QUERY_INT8_QUANT],
            },
        }
    )
    # Dense/W4A8 runtime stays OFF: the block did not list its symbol.
    assert not cfg.experimental_int8_runtime
    assert cfg.expert_input_dtype is None
    assert not dsv4_int_module.dsv4_int4_experts_int8_dense_active()
    # Indexer int8 is ON, independent of the dense flag.
    assert indexer_cache_int8_enabled()
    assert indexer_query_int8_enabled()
    assert dsv4_sm86.indexer_cache_is_int8()
    assert dsv4_sm86.indexer_imma_enabled()


def test_vllm_block_dense_runtime_does_not_ride_indexer_int8():
    """Listing only the dense symbol keeps both indexer int8 paths OFF: the
    indexer never rides the dense runtime."""
    from vllm.models.deepseek_v4.nvidia_sm86 import triton_kernels as dsv4_sm86

    cfg = Dsv4IntConfig.from_config(
        {
            "quant_method": "dsv4_int",
            "config_groups": _INT_GROUPS,
            VLLM_CONFIG_KEY: {
                "kernels": [DENSE_EXPERTS_INT8_ACTIVATION],
            },
        }
    )
    # Dense runtime is on (listed), but the indexer does not ride it.
    assert cfg.experimental_int8_runtime
    assert cfg.expert_input_dtype is torch.int8
    assert not indexer_cache_int8_enabled()
    assert not indexer_query_int8_enabled()
    assert not dsv4_sm86.indexer_cache_is_int8()
    assert not dsv4_sm86.indexer_imma_enabled()


def test_dense_symbol_requires_int_weight_groups():
    with pytest.raises(ValueError, match="dense_experts_int8_activation"):
        Dsv4IntConfig.from_config(
            {
                "quant_method": "dsv4_int",
                VLLM_CONFIG_KEY: {"kernels": [DENSE_EXPERTS_INT8_ACTIVATION]},
            }
        )


def test_vanilla_checkpoint_block_enables_indexer_int8_without_quant_config():
    """The kernels list works on vanilla-weights deployments (no dsv4_int)."""
    from vllm.models.deepseek_v4.nvidia_sm86 import triton_kernels as dsv4_sm86

    resolved = resolve_kernel_config(
        {"kernels": [INDEXER_CACHE_INT8_WRITER, INDEXER_QUERY_INT8_QUANT]}
    )
    activate_kernel_config(resolved)
    assert dsv4_sm86.indexer_cache_is_int8()
    assert dsv4_sm86.indexer_imma_enabled()


# ---------------------------------------------------------------------------
# Marlin input dtype: vllm block > VLLM_MARLIN_INPUT_DTYPE env > default
# ---------------------------------------------------------------------------


def test_marlin_input_dtype_from_block_with_env_unset(monkeypatch):
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_MARLIN_INPUT_DTYPE", None)
    cfg = Dsv4IntConfig.from_config(
        {
            "quant_method": "dsv4_int",
            "config_groups": _INT_GROUPS,
            VLLM_CONFIG_KEY: {"kernels": [DENSE_EXPERTS_INT8_ACTIVATION]},
        }
    )
    assert cfg.resolve_marlin_input_dtype() is torch.int8


def test_marlin_input_dtype_env_still_works_without_block(monkeypatch):
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_MARLIN_INPUT_DTYPE", "int8")
    cfg = Dsv4IntConfig.from_config(
        {"quant_method": "dsv4_int", "config_groups": _INT_GROUPS}
    )
    assert cfg.expert_input_dtype is None
    assert cfg.resolve_marlin_input_dtype() is torch.int8


def test_marlin_input_dtype_block_overrides_env(monkeypatch):
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_MARLIN_INPUT_DTYPE", "int8")
    cfg = Dsv4IntConfig.from_config(
        {
            "quant_method": "dsv4_int",
            "config_groups": _INT_GROUPS,
            # Explicit block WITHOUT the dense symbol: W4A16, env ignored.
            VLLM_CONFIG_KEY: {"kernels": []},
        }
    )
    assert cfg.resolve_marlin_input_dtype() is None


# ---------------------------------------------------------------------------
# Ray unpickle gate propagation
# ---------------------------------------------------------------------------


def test_dsv4_int_pickle_restores_kernel_block_gates(monkeypatch):
    cfg = Dsv4IntConfig.from_config(
        {
            "quant_method": "dsv4_int",
            "config_groups": _INT_GROUPS,
            VLLM_CONFIG_KEY: {
                "kernels": [INDEXER_CACHE_INT8_WRITER, INDEXER_QUERY_INT8_QUANT],
            },
        }
    )
    assert indexer_cache_int8_enabled()
    payload = pickle.dumps(cfg)

    monkeypatch.setattr(kernel_config, "_ACTIVE_CONFIG", None)
    monkeypatch.setattr(
        dsv4_int_module, "_DSV4_INT4_EXPERTS_INT8_DENSE_ACTIVE", False
    )
    assert not indexer_cache_int8_enabled()

    restored = pickle.loads(payload)

    assert not restored.experimental_int8_runtime
    assert indexer_cache_int8_enabled()
    assert indexer_query_int8_enabled()


def test_dsv4_int_pickle_restores_dense_runtime_gate(monkeypatch):
    """The dense W4A8 runtime gate survives Ray unpickle via the block."""
    cfg = Dsv4IntConfig.from_config(
        {
            "quant_method": "dsv4_int",
            "config_groups": _INT_GROUPS,
            VLLM_CONFIG_KEY: {"kernels": [DENSE_EXPERTS_INT8_ACTIVATION]},
        }
    )
    assert cfg.experimental_int8_runtime
    payload = pickle.dumps(cfg)
    monkeypatch.setattr(
        dsv4_int_module, "_DSV4_INT4_EXPERTS_INT8_DENSE_ACTIVE", False
    )

    restored = pickle.loads(payload)

    assert restored.experimental_int8_runtime
    assert dsv4_int_module.dsv4_int4_experts_int8_dense_active()


# ---------------------------------------------------------------------------
# cache_type: default vs explicit CLI precedence
# ---------------------------------------------------------------------------


def _fake_model_config(vllm_block, quant_method="dsv4_int"):
    hf_config = SimpleNamespace(
        vllm=vllm_block,
        quantization_config={"quant_method": quant_method},
    )
    return SimpleNamespace(hf_config=hf_config, quantization=quant_method)


def test_cache_type_sets_default_when_cli_is_auto():
    model_config = _fake_model_config({"kernels": [], "cache_type": "fp8_ds_mla"})
    cache_config = SimpleNamespace(cache_dtype="auto")
    apply_checkpoint_config(model_config, cache_config)
    assert cache_config.cache_dtype == "fp8_ds_mla"


def test_explicit_cli_kv_cache_dtype_wins_over_cache_type():
    """Docs rule: if the user explicitly asks for fp8, it must stay fp8."""
    model_config = _fake_model_config({"kernels": [], "cache_type": "int8_ds_mla"})
    cache_config = SimpleNamespace(cache_dtype="fp8")
    apply_checkpoint_config(model_config, cache_config)
    assert cache_config.cache_dtype == "fp8"


def test_apply_is_a_noop_without_block():
    model_config = _fake_model_config(None)
    cache_config = SimpleNamespace(cache_dtype="auto")
    apply_checkpoint_config(model_config, cache_config)
    assert cache_config.cache_dtype == "auto"


def test_apply_fails_closed_on_dense_symbol_with_vanilla_weights():
    hf_config = SimpleNamespace(
        vllm={"kernels": [DENSE_EXPERTS_INT8_ACTIVATION]},
        quantization_config=None,
    )
    model_config = SimpleNamespace(hf_config=hf_config, quantization=None)
    cache_config = SimpleNamespace(cache_dtype="auto")
    with pytest.raises(ValueError, match="dense_experts_int8_activation"):
        apply_checkpoint_config(model_config, cache_config)


def test_apply_fails_closed_on_bad_block_at_startup():
    model_config = _fake_model_config({"kernels": ["not.a.symbol"]})
    cache_config = SimpleNamespace(cache_dtype="auto")
    with pytest.raises(ValueError, match="[Uu]nknown"):
        apply_checkpoint_config(model_config, cache_config)


# ---------------------------------------------------------------------------
# Proof line
# ---------------------------------------------------------------------------


def test_resolved_proof_line_is_single_stable_line():
    resolved = resolve_kernel_config(
        {
            "kernels": [
                SPARSE_MLA_DECODE_FP8_FLASH,
                SPARSE_MLA_PREFILL_FLASH,
                INDEXER_CACHE_INT8_WRITER,
                INDEXER_QUERY_INT8_QUANT,
                DENSE_EXPERTS_INT8_ACTIVATION,
            ],
            "cache_type": "fp8_ds_mla",
        }
    )
    activate_kernel_config(resolved)
    line = resolved_proof_line(resolved, kv_cache_dtype="fp8_ds_mla")
    assert "\n" not in line
    assert line == (
        "vllm kernels resolved:"
        f" sparse_mla_decode_fp8={SPARSE_MLA_DECODE_FP8_FLASH}"
        f" sparse_mla_decode_int8={SPARSE_MLA_DECODE_INT8_TRITON}"
        f" sparse_mla_prefill={SPARSE_MLA_PREFILL_FLASH}"
        f" indexer_cache_int8={INDEXER_CACHE_INT8_WRITER}"
        f" indexer_query_int8={INDEXER_QUERY_INT8_QUANT}"
        f" dense_experts_int8_activation={DENSE_EXPERTS_INT8_ACTIVATION}"
        " indexer_streaming_topk_prefill=off"
        " cache_type=fp8_ds_mla"
    )


def test_resolved_proof_line_marks_inactive_toggles_off():
    resolved = resolve_kernel_config(None)
    activate_kernel_config(resolved)
    line = resolved_proof_line(resolved, kv_cache_dtype="fp8_ds_mla")
    assert "indexer_cache_int8=off" in line
    assert "indexer_query_int8=off" in line
    assert "dense_experts_int8_activation=off" in line
    assert "indexer_streaming_topk_prefill=off" in line


# ---------------------------------------------------------------------------
# SM86 dispatch wiring (source-level; GPU parity lives in
# tests/v1/attention/test_sm86_flash_mla_decode_parity.py)
# ---------------------------------------------------------------------------


def test_sm86_attention_dispatches_all_registered_prefill_symbols():
    import inspect

    from vllm.models.deepseek_v4.nvidia_sm86.attention import (
        DeepseekV4SM86Attention,
    )

    source = inspect.getsource(DeepseekV4SM86Attention._forward_prefill)
    assert "sparse_attention_triton" in source
    assert "_forward_prefill_flash" in source
    native = inspect.getsource(DeepseekV4SM86Attention._forward_prefill_flash)
    assert "sparse_mla_prefill(" in native
    # int8_ds_mla caches dispatch to the fused int8 variant of the SAME native
    # prefill (528B strided views), still selected by SPARSE_MLA_PREFILL_FLASH.
    assert "sparse_mla_prefill_int8(" in native
    decode = inspect.getsource(DeepseekV4SM86Attention._forward_decode)
    assert "SPARSE_MLA_DECODE_INT8_FLASH" in decode
    assert "sparse_mla_decode_int8(" in decode
    assert "sparse_mla_decode_int8_triton(" in decode


# ---------------------------------------------------------------------------
# PP transport toggles (pp_transport block key)
# ---------------------------------------------------------------------------


def test_pp_transport_absent_leaves_override_none():
    """No pp_transport key -> override None (fall back to env/default)."""
    resolved = resolve_kernel_config({"kernels": []})
    assert resolved.pp_cache_metadata is None
    resolved = resolve_kernel_config(None)
    assert resolved.pp_cache_metadata is None


def test_pp_transport_round_trips_booleans():
    resolved = resolve_kernel_config(
        {"kernels": [], "pp_transport": {"cache_metadata": True}}
    )
    assert resolved.pp_cache_metadata is True

    resolved = resolve_kernel_config(
        {"kernels": [], "pp_transport": {"cache_metadata": False}}
    )
    assert resolved.pp_cache_metadata is False


def test_pp_transport_empty_block_leaves_override_none():
    resolved = resolve_kernel_config(
        {"kernels": [], "pp_transport": {}}
    )
    assert resolved.pp_cache_metadata is None


def test_pp_transport_unknown_subkey_is_hard_error():
    # "pack" was removed with the packed wire format; it must now be
    # rejected like any other unknown key.
    with pytest.raises(ValueError, match="pp_transport"):
        resolve_kernel_config(
            {"kernels": [], "pp_transport": {"pack": True}}
        )


def test_pp_transport_non_bool_value_is_hard_error():
    with pytest.raises(ValueError, match="pp_transport"):
        resolve_kernel_config(
            {"kernels": [], "pp_transport": {"cache_metadata": 1}}
        )


def test_pp_transport_non_dict_is_hard_error():
    with pytest.raises(ValueError, match="pp_transport"):
        resolve_kernel_config({"kernels": [], "pp_transport": [1, 2]})


def test_pp_transport_resolves_from_hf_config():
    hf_config = SimpleNamespace(
        vllm={"kernels": [], "pp_transport": {"cache_metadata": False}},
        quantization_config={"quant_method": "dsv4_int"},
    )
    resolved = resolve_kernel_config_from_hf_config(hf_config)
    assert resolved.pp_cache_metadata is False


def test_pp_transport_stash_and_precedence_over_env(monkeypatch):
    """The stashed override wins over the env var; None falls back."""
    import vllm.transformers_utils.configs.dsv4.kernel_config as kernel_config_mod

    monkeypatch.setattr(kernel_config_mod, "_PP_CACHE_METADATA_OVERRIDE", None)

    # Nothing stashed -> the override reports None.
    assert kernel_config_mod.pp_cache_metadata_override() is None

    resolved = resolve_kernel_config(
        {"kernels": [], "pp_transport": {"cache_metadata": True}}
    )
    kernel_config_mod.stash_pp_transport_overrides(resolved.pp_cache_metadata)
    assert kernel_config_mod.pp_cache_metadata_override() is True


def test_activate_stashes_pp_transport_overrides(monkeypatch):
    """activate_kernel_config propagates pp_transport to the stash
    (covers the model-build and Ray-unpickle activation paths)."""
    import vllm.transformers_utils.configs.dsv4.kernel_config as kernel_config_mod

    monkeypatch.setattr(kernel_config_mod, "_PP_CACHE_METADATA_OVERRIDE", None)
    resolved = resolve_kernel_config(
        {"kernels": [], "pp_transport": {"cache_metadata": False}}
    )
    activate_kernel_config(resolved)
    assert kernel_config_mod.pp_cache_metadata_override() is False


def test_sm86_native_prefill_supports_fp8_and_int8_caches():
    """The fused native prefill consumes BOTH fp8_ds_mla and int8_ds_mla paged
    caches (flash_mla 93bbf4e: int8 whole-cache dequant pass, runtime row
    stride), so selecting it must validate under either cache dtype."""
    from vllm.models.deepseek_v4.nvidia_sm86.attention import (
        validate_sm86_kernel_selection,
    )

    resolved = resolve_kernel_config(
        {"kernels": [SPARSE_MLA_PREFILL_FLASH]}
    )
    validate_sm86_kernel_selection(resolved, kv_cache_dtype="fp8_ds_mla")
    validate_sm86_kernel_selection(resolved, kv_cache_dtype="int8_ds_mla")
    # non-ds_mla caches still fail closed.
    with pytest.raises(ValueError, match="ds_mla"):
        validate_sm86_kernel_selection(resolved, kv_cache_dtype="fp8")


def test_sm86_int8_decode_native_selectable_triton_default():
    """The native flash_mla int8 decode is SELECTABLE for the int8 decode role;
    the Triton int8 decode remains the documented default."""
    from vllm.transformers_utils.configs.dsv4.kernel_config import (
        SELECTOR_ROLE_DEFAULTS,
    )

    assert (
        SELECTOR_ROLE_DEFAULTS[ROLE_SPARSE_MLA_DECODE_INT8]
        == SPARSE_MLA_DECODE_INT8_TRITON
    )
    resolved = resolve_kernel_config(
        {"kernels": [SPARSE_MLA_DECODE_INT8_FLASH]}
    )
    assert resolved.roles[ROLE_SPARSE_MLA_DECODE_INT8] == SPARSE_MLA_DECODE_INT8_FLASH
    # Both int8 decode symbols claim one role: listing both is a hard error.
    with pytest.raises(ValueError, match="same role"):
        resolve_kernel_config(
            {
                "kernels": [
                    SPARSE_MLA_DECODE_INT8_FLASH,
                    SPARSE_MLA_DECODE_INT8_TRITON,
                ]
            }
        )


# ---------------------------------------------------------------------------
# Indexer streaming top-k prefill (toggle role + slab-rows tuning key)
# ---------------------------------------------------------------------------


def test_streaming_topk_prefill_symbol_registered():
    assert INDEXER_STREAMING_TOPK_PREFILL == (
        "vllm.model_executor.layers.sparse_attn_indexer.streaming_prefill_topk"
    )
    assert (
        KERNEL_REGISTRY[INDEXER_STREAMING_TOPK_PREFILL]
        == ROLE_INDEXER_STREAMING_TOPK_PREFILL
    )


def test_streaming_topk_prefill_on_when_listed():
    resolved = resolve_kernel_config(
        {"kernels": [INDEXER_STREAMING_TOPK_PREFILL]}
    )
    activate_kernel_config(resolved)
    assert (
        resolved.roles[ROLE_INDEXER_STREAMING_TOPK_PREFILL]
        == INDEXER_STREAMING_TOPK_PREFILL
    )
    assert indexer_streaming_topk_prefill_enabled()
    line = resolved_proof_line(resolved, kv_cache_dtype="fp8_ds_mla")
    assert f"indexer_streaming_topk_prefill={INDEXER_STREAMING_TOPK_PREFILL}" in line


def test_streaming_topk_prefill_off_when_unlisted():
    resolved = resolve_kernel_config({"kernels": []})
    activate_kernel_config(resolved)
    assert not indexer_streaming_topk_prefill_enabled()


def test_streaming_topk_prefill_off_without_block():
    """Default OFF: an absent block preserves the one-shot prefill path
    bit-for-bit."""
    resolved = resolve_kernel_config(None)
    activate_kernel_config(resolved)
    assert not indexer_streaming_topk_prefill_enabled()


def test_streaming_topk_prefill_off_with_no_active_config():
    assert not indexer_streaming_topk_prefill_enabled()


def test_slab_rows_absent_is_none():
    assert (
        resolve_kernel_config(None).indexer_prefill_topk_slab_rows is None
    )
    assert (
        resolve_kernel_config(
            {"kernels": []}
        ).indexer_prefill_topk_slab_rows
        is None
    )


def test_slab_rows_round_trips_int():
    resolved = resolve_kernel_config(
        {
            "kernels": [INDEXER_STREAMING_TOPK_PREFILL],
            "indexer_prefill_topk_slab_rows": 8192,
        }
    )
    assert resolved.indexer_prefill_topk_slab_rows == 8192


@pytest.mark.parametrize("bad", [True, False, "16384", 16384.0, 0, -1])
def test_slab_rows_invalid_is_hard_error(bad):
    with pytest.raises(ValueError, match="indexer_prefill_topk_slab_rows"):
        resolve_kernel_config(
            {"kernels": [], "indexer_prefill_topk_slab_rows": bad}
        )


def test_slab_rows_resolves_from_hf_config():
    hf_config = SimpleNamespace(
        vllm={
            "kernels": [INDEXER_STREAMING_TOPK_PREFILL],
            "indexer_prefill_topk_slab_rows": 4096,
        },
    )
    resolved = resolve_kernel_config_from_hf_config(hf_config)
    assert resolved.indexer_prefill_topk_slab_rows == 4096
    assert resolved.roles[ROLE_INDEXER_STREAMING_TOPK_PREFILL] == (
        INDEXER_STREAMING_TOPK_PREFILL
    )


def test_slab_rows_override_reads_active_config():
    assert indexer_prefill_topk_slab_rows_override() is None
    resolved = resolve_kernel_config(
        {"kernels": [], "indexer_prefill_topk_slab_rows": 4096}
    )
    activate_kernel_config(resolved)
    assert indexer_prefill_topk_slab_rows_override() == 4096


# ---------------------------------------------------------------------------
# Indexer gate: the streaming decision reads the config, not a module constant
# ---------------------------------------------------------------------------


def _fake_platform(is_cuda=True, families=(80,)):
    return SimpleNamespace(
        is_cuda=lambda: is_cuda,
        is_device_capability_family=lambda fam: fam in families,
    )


def test_indexer_gate_reads_toggle_from_config(monkeypatch):
    import vllm.model_executor.layers.sparse_attn_indexer as indexer

    monkeypatch.setattr(indexer, "current_platform", _fake_platform())
    assert not indexer.should_use_prefill_streaming_topk(1, False)

    activate_kernel_config(
        resolve_kernel_config(
            {"kernels": [INDEXER_STREAMING_TOPK_PREFILL]}
        )
    )
    assert indexer.should_use_prefill_streaming_topk(1, False)

    activate_kernel_config(resolve_kernel_config({"kernels": []}))
    assert not indexer.should_use_prefill_streaming_topk(1, False)


def test_indexer_gate_guards_unchanged(monkeypatch):
    """The pre-existing platform/DCP/FP4 guards still apply on top of the
    config toggle."""
    import vllm.model_executor.layers.sparse_attn_indexer as indexer

    activate_kernel_config(
        resolve_kernel_config(
            {"kernels": [INDEXER_STREAMING_TOPK_PREFILL]}
        )
    )
    monkeypatch.setattr(indexer, "current_platform", _fake_platform())
    assert indexer.should_use_prefill_streaming_topk(1, False)
    # DCP > 1: off.
    assert not indexer.should_use_prefill_streaming_topk(2, False)
    # FP4 cache: off.
    assert not indexer.should_use_prefill_streaming_topk(1, True)
    # Non-CUDA: off.
    monkeypatch.setattr(
        indexer, "current_platform", _fake_platform(is_cuda=False)
    )
    assert not indexer.should_use_prefill_streaming_topk(1, False)
    # Unsupported capability family: off; family 120 stays supported.
    monkeypatch.setattr(
        indexer, "current_platform", _fake_platform(families=(90,))
    )
    assert not indexer.should_use_prefill_streaming_topk(1, False)
    monkeypatch.setattr(
        indexer, "current_platform", _fake_platform(families=(120,))
    )
    assert indexer.should_use_prefill_streaming_topk(1, False)


def test_indexer_forward_consumes_the_gate():
    import inspect

    import vllm.model_executor.layers.sparse_attn_indexer as indexer

    source = inspect.getsource(indexer.sparse_attn_indexer)
    assert "should_use_prefill_streaming_topk(" in source


def test_indexer_module_constant_retired():
    import vllm.model_executor.layers.sparse_attn_indexer as indexer

    assert not hasattr(indexer, "INDEXER_PREFILL_STREAMING_TOPK")


def test_indexer_slab_rows_resolution():
    import vllm.model_executor.layers.sparse_attn_indexer as indexer

    assert (
        indexer._resolved_prefill_topk_slab_rows()
        == indexer.INDEXER_PREFILL_TOPK_SLAB_ROWS
    )
    activate_kernel_config(
        resolve_kernel_config(
            {"kernels": [], "indexer_prefill_topk_slab_rows": 4096}
        )
    )
    assert indexer._resolved_prefill_topk_slab_rows() == 4096
