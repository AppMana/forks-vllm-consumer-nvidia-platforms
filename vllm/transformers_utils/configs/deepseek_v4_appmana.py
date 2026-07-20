# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified checkpoint-config-driven kernel configuration for AppMana DSV4.

A single ``"appmana"`` block at the top level of the checkpoint's
``config.json`` selects every fork-specific kernel path::

    "appmana": {
      "kernels": ["flash_mla.sparse_mla_decode_fp8",
                  "vllm.models.deepseek_v4.nvidia_sm86.triton_kernels"
                  ".sparse_attention_triton",
                  "vllm._custom_ops.indexer_k_quant_and_cache_int8",
                  "vllm.models.deepseek_v4.common.ops.fused_indexer_q"
                  ".fused_indexer_q_rope_quant_int8"],
      "cache_type": "fp8_ds_mla"
    }

Principles:

1. Every value EXACTLY matches the most salient symbol that gets activated:
   importable FQNs for kernels, the kv-cache dtype string for ``cache_type``.
2. Membership in ``kernels`` activates that path; the role is inferred from
   ``KERNEL_REGISTRY``.
3. Fail closed: an unknown symbol is a hard startup error; two symbols
   claiming the same role is a hard error. A *selector* role (decode fp8,
   decode int8, prefill — roles that always need one kernel) with no symbol
   uses its documented default. When a block with a ``kernels`` list is
   present it is authoritative for the *toggle* roles (indexer cache int8,
   indexer query int8, dense experts int8 activation): unlisted means OFF,
   overriding the legacy checkpoint flag. Without a block, legacy behavior
   applies unchanged.
4. Overriding everything is trivial: vLLM applies dict-valued
   ``--hf-overrides`` entries by *replacing* the whole attribute
   (``ModelConfig._apply_dict_overrides`` does ``setattr`` for plain-dict
   attributes — it does NOT deep-merge), so one
   ``--hf-overrides '{"appmana": {...}}'`` blob replaces the entire block.

Legacy compatibility (deprecated, still honored when no block is present):

* role-keyed HF override strings ``deepseek_v4_sm86_sparse_mla_decode_fp8`` /
  ``..._decode_int8`` / ``..._prefill`` (live LWS manifests still use them);
* the checkpoint flag ``__experimental_enable_imma_from_https://github.com/
  appMana/forks-vllm-ampere`` which enables the dense W4A8 runtime AND rides
  the indexer int8 paths on it.

The new block wins whenever both are present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from vllm.logger import init_logger

logger = init_logger(__name__)

APPMANA_CONFIG_KEY = "appmana"
_ALLOWED_BLOCK_KEYS = frozenset({"kernels", "cache_type", "pp_transport"})
# Sub-keys of the "pp_transport" block. Each toggles a PP intermediate-tensor
# transport optimization. Absent (None) means "not specified": the coordinator
# falls back to the env var, then the built-in default. See parallel_state's
# _pp_metadata_cache_enabled.
_PP_TRANSPORT_CACHE_METADATA_KEY = "cache_metadata"
_ALLOWED_PP_TRANSPORT_KEYS = frozenset({_PP_TRANSPORT_CACHE_METADATA_KEY})

LEGACY_IMMA_CONFIG_KEY = (
    "__experimental_enable_imma_from_https://github.com/appMana/forks-vllm-ampere"
)

# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------

# Selector roles: exactly one kernel is always active; absence = default.
ROLE_SPARSE_MLA_DECODE_FP8 = "sparse_mla_decode_fp8"
ROLE_SPARSE_MLA_DECODE_INT8 = "sparse_mla_decode_int8"
ROLE_SPARSE_MLA_PREFILL = "sparse_mla_prefill"
# Toggle roles: membership turns the path on; absence (in an explicit block)
# turns it off.
ROLE_INDEXER_CACHE_INT8 = "indexer_cache_int8"
ROLE_INDEXER_QUERY_INT8 = "indexer_query_int8"
ROLE_DENSE_EXPERTS_INT8_ACTIVATION = "dense_experts_int8_activation"

# ---------------------------------------------------------------------------
# Symbols (importable FQNs; each is the most salient callable activated)
# ---------------------------------------------------------------------------

SPARSE_MLA_DECODE_FP8_FLASH = "flash_mla.sparse_mla_decode_fp8"
SPARSE_MLA_DECODE_FP8_TRITON = (
    "vllm.models.deepseek_v4.nvidia_sm86.triton_kernels."
    "decode_sparse_attention_triton"
)
SPARSE_MLA_DECODE_INT8_TRITON = "flash_mla.sparse_mla_decode_int8_triton"
SPARSE_MLA_DECODE_INT8_FLASH = "flash_mla.sparse_mla_decode_int8"
SPARSE_MLA_PREFILL_TRITON = (
    "vllm.models.deepseek_v4.nvidia_sm86.triton_kernels.sparse_attention_triton"
)
SPARSE_MLA_PREFILL_FLASH = "flash_mla.sparse_mla_prefill"
INDEXER_CACHE_INT8_WRITER = "vllm._custom_ops.indexer_k_quant_and_cache_int8"
INDEXER_QUERY_INT8_QUANT = (
    "vllm.models.deepseek_v4.common.ops.fused_indexer_q."
    "fused_indexer_q_rope_quant_int8"
)
DENSE_EXPERTS_INT8_ACTIVATION = (
    "vllm.model_executor.layers.quantization.utils.marlin_utils."
    "marlin_act_int8_process_scales"
)

KERNEL_REGISTRY: dict[str, str] = {
    SPARSE_MLA_DECODE_FP8_FLASH: ROLE_SPARSE_MLA_DECODE_FP8,
    SPARSE_MLA_DECODE_FP8_TRITON: ROLE_SPARSE_MLA_DECODE_FP8,
    SPARSE_MLA_DECODE_INT8_TRITON: ROLE_SPARSE_MLA_DECODE_INT8,
    SPARSE_MLA_DECODE_INT8_FLASH: ROLE_SPARSE_MLA_DECODE_INT8,
    SPARSE_MLA_PREFILL_TRITON: ROLE_SPARSE_MLA_PREFILL,
    SPARSE_MLA_PREFILL_FLASH: ROLE_SPARSE_MLA_PREFILL,
    INDEXER_CACHE_INT8_WRITER: ROLE_INDEXER_CACHE_INT8,
    INDEXER_QUERY_INT8_QUANT: ROLE_INDEXER_QUERY_INT8,
    DENSE_EXPERTS_INT8_ACTIVATION: ROLE_DENSE_EXPERTS_INT8_ACTIVATION,
}

SELECTOR_ROLE_DEFAULTS: dict[str, str] = {
    ROLE_SPARSE_MLA_DECODE_FP8: SPARSE_MLA_DECODE_FP8_FLASH,
    ROLE_SPARSE_MLA_DECODE_INT8: SPARSE_MLA_DECODE_INT8_TRITON,
    ROLE_SPARSE_MLA_PREFILL: SPARSE_MLA_PREFILL_TRITON,
}

TOGGLE_ROLES = frozenset(
    {
        ROLE_INDEXER_CACHE_INT8,
        ROLE_INDEXER_QUERY_INT8,
        ROLE_DENSE_EXPERTS_INT8_ACTIVATION,
    }
)

# Deprecated role-keyed HF config attributes -> the role they select.
LEGACY_ALIAS_ROLES: dict[str, str] = {
    "deepseek_v4_sm86_sparse_mla_decode_fp8": ROLE_SPARSE_MLA_DECODE_FP8,
    "deepseek_v4_sm86_sparse_mla_decode_int8": ROLE_SPARSE_MLA_DECODE_INT8,
    "deepseek_v4_sm86_sparse_mla_prefill": ROLE_SPARSE_MLA_PREFILL,
}

_PROOF_ROLE_ORDER = (
    ROLE_SPARSE_MLA_DECODE_FP8,
    ROLE_SPARSE_MLA_DECODE_INT8,
    ROLE_SPARSE_MLA_PREFILL,
    ROLE_INDEXER_CACHE_INT8,
    ROLE_INDEXER_QUERY_INT8,
    ROLE_DENSE_EXPERTS_INT8_ACTIVATION,
)


@dataclass(frozen=True)
class ResolvedAppmanaKernelConfig:
    """Validated role -> symbol assignment for one checkpoint."""

    # True when an "appmana" block with a "kernels" list was present. When
    # False, toggle roles fall back to legacy (URL-flag-derived) behavior at
    # query time via the gates below.
    explicit: bool
    # Active role -> symbol. Selector roles are always present; toggle roles
    # are present iff active (explicit blocks) or resolved lazily (legacy).
    roles: Mapping[str, str] = field(default_factory=dict)
    cache_type: str | None = None
    # Value of the legacy checkpoint flag, used for legacy toggle defaults.
    legacy_dense_flag: bool = False
    # PP intermediate-tensor transport toggles from "appmana.pp_transport".
    # None = "not specified": the coordinator falls back to the env var, then
    # the built-in default (both effectively on today). An explicit bool here
    # rides VllmConfig to every Ray worker, unlike the env var (which is only
    # forwarded to workers via a fixed allowlist).
    pp_cache_metadata: bool | None = None

    def symbol(self, role: str) -> str | None:
        return self.roles.get(role)

    def has_role(self, role: str) -> bool:
        return role in self.roles


def _allowed_cache_types() -> tuple[str, ...]:
    # Lazy import: vllm.config pulls pydantic and more; this module must stay
    # import-light for the kernel/quantization modules that import it.
    from typing import Literal, get_args, get_origin

    from vllm.config.cache import CacheDType

    def _flatten(tp: Any) -> list[str]:
        out: list[str] = []
        for arg in get_args(tp):
            if isinstance(arg, str):
                out.append(arg)
            elif get_origin(arg) is Literal or get_args(arg):
                out.extend(_flatten(arg))
        return out

    return tuple(_flatten(CacheDType))


def resolve_appmana_kernel_config(
    block: Any,
    *,
    legacy_aliases: Mapping[str, str] | None = None,
    legacy_dense_flag: bool = False,
) -> ResolvedAppmanaKernelConfig:
    """Resolve and validate the ``appmana`` block (fail closed).

    ``legacy_aliases`` maps the deprecated role-keyed HF config attribute
    names to their values; they are honored only when no block is present and
    log a deprecation warning when they deviate from the defaults.
    """
    explicit = False
    kernels: list[str] = []
    cache_type: str | None = None
    pp_cache_metadata: bool | None = None

    if block is not None:
        if not isinstance(block, dict):
            raise ValueError(
                f'"{APPMANA_CONFIG_KEY}" config block must be a JSON object, '
                f"got {type(block).__name__}"
            )
        unknown_keys = set(block) - _ALLOWED_BLOCK_KEYS
        if unknown_keys:
            raise ValueError(
                f'Unknown "{APPMANA_CONFIG_KEY}" config key(s) '
                f"{sorted(unknown_keys)}; allowed: {sorted(_ALLOWED_BLOCK_KEYS)}"
            )
        raw_kernels = block.get("kernels")
        if raw_kernels is not None:
            if not isinstance(raw_kernels, (list, tuple)) or not all(
                isinstance(symbol, str) for symbol in raw_kernels
            ):
                raise ValueError(
                    f'"{APPMANA_CONFIG_KEY}.kernels" must be a list of '
                    f"symbol strings, got {raw_kernels!r}"
                )
            explicit = True
            kernels = list(raw_kernels)
        cache_type = block.get("cache_type")
        if cache_type is not None:
            allowed = _allowed_cache_types()
            if cache_type not in allowed:
                raise ValueError(
                    f'Unsupported "{APPMANA_CONFIG_KEY}.cache_type" '
                    f"{cache_type!r}; allowed: {list(allowed)}"
                )
        raw_pp_transport = block.get("pp_transport")
        if raw_pp_transport is not None:
            if not isinstance(raw_pp_transport, dict):
                raise ValueError(
                    f'"{APPMANA_CONFIG_KEY}.pp_transport" must be a JSON '
                    f"object, got {type(raw_pp_transport).__name__}"
                )
            unknown_pp = set(raw_pp_transport) - _ALLOWED_PP_TRANSPORT_KEYS
            if unknown_pp:
                raise ValueError(
                    f'Unknown "{APPMANA_CONFIG_KEY}.pp_transport" key(s) '
                    f"{sorted(unknown_pp)}; allowed: "
                    f"{sorted(_ALLOWED_PP_TRANSPORT_KEYS)}"
                )
            for pp_key in _ALLOWED_PP_TRANSPORT_KEYS:
                if pp_key not in raw_pp_transport:
                    continue
                value = raw_pp_transport[pp_key]
                # Strict bool: JSON true/false only (reject 0/1, matching the
                # fail-closed spirit of the rest of the parser).
                if not isinstance(value, bool):
                    raise ValueError(
                        f'"{APPMANA_CONFIG_KEY}.pp_transport.{pp_key}" must be '
                        f"a boolean, got {value!r}"
                    )
            pp_cache_metadata = raw_pp_transport.get(
                _PP_TRANSPORT_CACHE_METADATA_KEY
            )

    roles: dict[str, str] = {}
    for symbol in kernels:
        role = KERNEL_REGISTRY.get(symbol)
        if role is None:
            raise ValueError(
                f'Unknown kernel symbol {symbol!r} in "{APPMANA_CONFIG_KEY}.'
                f'kernels"; known symbols: {sorted(KERNEL_REGISTRY)}'
            )
        if role in roles:
            raise ValueError(
                f"Kernel symbols {roles[role]!r} and {symbol!r} claim the "
                f"same role {role!r}"
            )
        roles[role] = symbol

    if explicit:
        for alias, value in (legacy_aliases or {}).items():
            role = LEGACY_ALIAS_ROLES.get(alias)
            if role is None or value is None:
                continue
            if value != SELECTOR_ROLE_DEFAULTS[role]:
                logger.warning_once(
                    'Deprecated DSV4 kernel override %s=%r is ignored: the '
                    '"%s" config block wins when both are present.',
                    alias,
                    value,
                    APPMANA_CONFIG_KEY,
                )
    else:
        for alias, value in (legacy_aliases or {}).items():
            role = LEGACY_ALIAS_ROLES.get(alias)
            if role is None or value is None:
                continue
            if KERNEL_REGISTRY.get(value) != role:
                raise ValueError(
                    f"Unsupported value {value!r} for deprecated DSV4 kernel "
                    f"override {alias}; expected one of "
                    f"{sorted(s for s, r in KERNEL_REGISTRY.items() if r == role)}"
                )
            if value != SELECTOR_ROLE_DEFAULTS[role]:
                logger.warning_once(
                    "DSV4 kernel override %s is deprecated; use the "
                    '"%s": {"kernels": [...]} config block (or '
                    "--hf-overrides) instead.",
                    alias,
                    APPMANA_CONFIG_KEY,
                )
            roles[role] = value

    # Selector roles always resolve; absent = documented default.
    for role, default in SELECTOR_ROLE_DEFAULTS.items():
        roles.setdefault(role, default)

    # Cross-role validation: the s8 x s8 IMMA indexer query requires the INT8
    # indexer K cache.
    if ROLE_INDEXER_QUERY_INT8 in roles and ROLE_INDEXER_CACHE_INT8 not in roles:
        raise ValueError(
            f"{roles[ROLE_INDEXER_QUERY_INT8]!r} ({ROLE_INDEXER_QUERY_INT8}) "
            f"requires an {ROLE_INDEXER_CACHE_INT8} symbol "
            f"({INDEXER_CACHE_INT8_WRITER!r}) in the kernels list: the "
            "integer-MMA logits kernel consumes a symmetric INT8 K cache"
        )

    return ResolvedAppmanaKernelConfig(
        explicit=explicit,
        roles=roles,
        cache_type=cache_type,
        legacy_dense_flag=bool(legacy_dense_flag),
        pp_cache_metadata=pp_cache_metadata,
    )


def resolve_appmana_kernel_config_from_hf_config(
    hf_config: Any,
) -> ResolvedAppmanaKernelConfig:
    """Resolve from an HF config object (block, legacy aliases, legacy flag)."""
    legacy_aliases: dict[str, str] = {}
    for alias in LEGACY_ALIAS_ROLES:
        value = getattr(hf_config, alias, None)
        if value is not None:
            legacy_aliases[alias] = value
    legacy_dense_flag = bool(getattr(hf_config, LEGACY_IMMA_CONFIG_KEY, False))
    quantization_config = getattr(hf_config, "quantization_config", None)
    if isinstance(quantization_config, dict):
        legacy_dense_flag = legacy_dense_flag or bool(
            quantization_config.get(LEGACY_IMMA_CONFIG_KEY, False)
        )
    return resolve_appmana_kernel_config(
        getattr(hf_config, APPMANA_CONFIG_KEY, None),
        legacy_aliases=legacy_aliases,
        legacy_dense_flag=legacy_dense_flag,
    )


# ---------------------------------------------------------------------------
# Process-wide activation (set once per process; survives Ray unpickle via
# Dsv4IntConfig.__setstate__ and model build via DeepseekV4Attention.__init__)
# ---------------------------------------------------------------------------

_ACTIVE_CONFIG: ResolvedAppmanaKernelConfig | None = None

# Process-wide PP transport overrides resolved from "appmana.pp_transport".
# None = "not specified" -> the coordinator falls back to the env var, then the
# built-in default. Set at worker init (init_worker_distributed_environment,
# where vllm_config is definitely in hand) AND on every kernel-config
# activation, so the value is stashed before the first PP hop regardless of
# path. The GroupCoordinator reads it via pp_cache_metadata_override().
_PP_CACHE_METADATA_OVERRIDE: bool | None = None


def stash_pp_transport_overrides(pp_cache_metadata: bool | None) -> None:
    """Set the process-wide PP transport override (set-once semantics: only a
    non-None value replaces an already-stashed value, so a later blockless
    activation cannot clobber a value resolved from the checkpoint block)."""
    global _PP_CACHE_METADATA_OVERRIDE
    if pp_cache_metadata is not None:
        _PP_CACHE_METADATA_OVERRIDE = pp_cache_metadata


def pp_cache_metadata_override() -> bool | None:
    return _PP_CACHE_METADATA_OVERRIDE


def activate_appmana_kernel_config(
    resolved: ResolvedAppmanaKernelConfig,
) -> None:
    global _ACTIVE_CONFIG
    if _ACTIVE_CONFIG is not None and _ACTIVE_CONFIG != resolved:
        logger.debug(
            "Replacing active appmana kernel config %s with %s",
            _ACTIVE_CONFIG,
            resolved,
        )
    _ACTIVE_CONFIG = resolved
    # Propagate PP transport overrides so the coordinator sees them even when
    # activation happens via the model build / Ray unpickle path.
    stash_pp_transport_overrides(resolved.pp_cache_metadata)


def active_appmana_kernel_config() -> ResolvedAppmanaKernelConfig | None:
    return _ACTIVE_CONFIG


def _legacy_dense_runtime_active() -> bool:
    try:
        from vllm.model_executor.layers.quantization.dsv4_int import (
            dsv4_int4_experts_int8_dense_active,
        )
    except Exception:
        return False
    return dsv4_int4_experts_int8_dense_active()


def indexer_cache_int8_enabled() -> bool:
    """True when the indexer K cache stores symmetric INT8 instead of FP8.

    Explicit block: on iff ``indexer_k_quant_and_cache_int8`` is listed.
    Legacy: rides the dense W4A8 runtime flag (historical behavior).
    """
    config = _ACTIVE_CONFIG
    if config is not None and config.explicit:
        return config.has_role(ROLE_INDEXER_CACHE_INT8)
    return _legacy_dense_runtime_active()


def indexer_query_int8_enabled() -> bool:
    """True when the indexer query is quantized to symmetric INT8 so the
    logits run as s8 x s8 integer MMA.

    Explicit block: on iff ``fused_indexer_q_rope_quant_int8`` is listed
    (resolution guarantees the INT8 cache is also on). Legacy: rides the
    dense W4A8 runtime flag.
    """
    config = _ACTIVE_CONFIG
    if config is not None and config.explicit:
        return config.has_role(ROLE_INDEXER_QUERY_INT8)
    return _legacy_dense_runtime_active()


def dense_experts_int8_activation_enabled() -> bool:
    """True when the Marlin INT8-activation (W4A8) expert runtime is active."""
    return _legacy_dense_runtime_active()


# ---------------------------------------------------------------------------
# Engine-side application (cache_type default + startup fail-closed check)
# ---------------------------------------------------------------------------

_DENSE_QUANT_METHODS = ("dsv4_int", "dsv4_mxfp4_int8")


def apply_appmana_checkpoint_config(model_config: Any, cache_config: Any) -> None:
    """Validate the checkpoint's ``appmana`` block at engine startup and apply
    ``cache_type`` as the DEFAULT kv-cache dtype.

    An explicit ``--kv-cache-dtype`` always wins: the default is only applied
    when the CLI left ``cache_dtype`` at ``"auto"``.
    """
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None or getattr(hf_config, APPMANA_CONFIG_KEY, None) is None:
        return
    resolved = resolve_appmana_kernel_config_from_hf_config(hf_config)

    if resolved.has_role(ROLE_DENSE_EXPERTS_INT8_ACTIVATION):
        quantization_config = getattr(hf_config, "quantization_config", None)
        quant_method = (
            quantization_config.get("quant_method")
            if isinstance(quantization_config, dict)
            else None
        )
        if quant_method not in _DENSE_QUANT_METHODS:
            raise ValueError(
                f"{DENSE_EXPERTS_INT8_ACTIVATION!r} "
                f"({ROLE_DENSE_EXPERTS_INT8_ACTIVATION}) requires a "
                f"quant_method in {list(_DENSE_QUANT_METHODS)} checkpoint "
                f"(weight-format-implied), got {quant_method!r}"
            )

    if resolved.cache_type is not None and cache_config.cache_dtype == "auto":
        logger.info(
            'Applying checkpoint default kv-cache dtype "%s" from the '
            '"%s.cache_type" config (pass --kv-cache-dtype to override).',
            resolved.cache_type,
            APPMANA_CONFIG_KEY,
        )
        cache_config.cache_dtype = resolved.cache_type


def resolved_proof_line(
    resolved: ResolvedAppmanaKernelConfig, *, kv_cache_dtype: str
) -> str:
    """Single stable-format line stating every active role -> symbol plus the
    resolved kv-cache dtype. This is the benchmark validity check."""
    parts = []
    for role in _PROOF_ROLE_ORDER:
        if role in TOGGLE_ROLES:
            if resolved.explicit:
                active = resolved.has_role(role)
            elif role == ROLE_INDEXER_CACHE_INT8:
                active = indexer_cache_int8_enabled()
            elif role == ROLE_INDEXER_QUERY_INT8:
                active = indexer_query_int8_enabled()
            else:
                active = dense_experts_int8_activation_enabled()
            if active:
                symbol = resolved.symbol(role) or _TOGGLE_ROLE_SYMBOLS[role]
            else:
                symbol = "off"
        else:
            symbol = resolved.roles[role]
        parts.append(f"{role}={symbol}")
    parts.append(f"cache_type={kv_cache_dtype}")
    return "appmana kernels resolved: " + " ".join(parts)


_TOGGLE_ROLE_SYMBOLS: dict[str, str] = {
    ROLE_INDEXER_CACHE_INT8: INDEXER_CACHE_INT8_WRITER,
    ROLE_INDEXER_QUERY_INT8: INDEXER_QUERY_INT8_QUANT,
    ROLE_DENSE_EXPERTS_INT8_ACTIVATION: DENSE_EXPERTS_INT8_ACTIVATION,
}
