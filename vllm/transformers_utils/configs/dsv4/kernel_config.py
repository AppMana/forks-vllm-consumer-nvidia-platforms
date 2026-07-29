# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified checkpoint-config-driven kernel configuration for AppMana DSV4.

A single ``"vllm"`` block at the top level of the checkpoint's
``config.json`` selects every fork-specific kernel path::

    "vllm": {
      "kernels": ["flash_mla.sparse_mla_decode_fp8",
                  "vllm.models.deepseek_v4.nvidia_imma.triton_kernels"
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
   decode int8, prefill: roles that always need one kernel) with no symbol
   uses its documented default. When a block with a ``kernels`` list is
   present it is authoritative for the *toggle* roles (indexer cache int8,
   indexer query int8, dense experts int8 activation, indexer streaming
   top-k prefill): unlisted means OFF. Without a block -- which is what
   official upstream checkpoints ship -- roles are resolved from the
   checkpoint's own dtypes and the device by ``blockless_role_defaults``:
   an int checkpoint gets the integer cache and integer kernels (with its
   indexer/dense integer paths on, because for those weights that is the
   intent rather than an opt-in), and an fp checkpoint gets the fp8 cache
   with sparkinfer on sm_12x or the Triton upcasting kernels wherever the
   device has no accelerated fp8.
4. Overriding everything is trivial: vLLM applies dict-valued
   ``--hf-overrides`` entries by *replacing* the whole attribute
   (``ModelConfig._apply_dict_overrides`` does ``setattr`` for plain-dict
   attributes; it does NOT deep-merge), so one
   ``--hf-overrides '{"vllm": {...}}'`` blob replaces the entire block.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Mapping

from vllm.logger import init_logger

logger = init_logger(__name__)

VLLM_CONFIG_KEY = "vllm"
# Context-slab width (in compressed-token rows) for the streaming prefill
# top-k path. A tuning key: only effective while the
# ``indexer_streaming_topk_prefill`` toggle role is active; absent means the
# indexer module's documented default.
_INDEXER_PREFILL_TOPK_SLAB_ROWS_KEY = "indexer_prefill_topk_slab_rows"
_ALLOWED_BLOCK_KEYS = frozenset(
    {"kernels", "cache_type", _INDEXER_PREFILL_TOPK_SLAB_ROWS_KEY}
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
ROLE_INDEXER_STREAMING_TOPK_PREFILL = "indexer_streaming_topk_prefill"

# ---------------------------------------------------------------------------
# Symbols (importable FQNs; each is the most salient callable activated)
# ---------------------------------------------------------------------------

SPARSE_MLA_DECODE_FP8_FLASH = "flash_mla.sparse_mla_decode_fp8"
SPARSE_MLA_DECODE_FP8_TRITON = (
    "vllm.models.deepseek_v4.nvidia_imma.triton_kernels."
    "decode_sparse_attention_triton"
)
SPARSE_MLA_DECODE_INT8_TRITON = "flash_mla.sparse_mla_decode_int8_triton"
SPARSE_MLA_DECODE_INT8_FLASH = "flash_mla.sparse_mla_decode_int8"
SPARSE_MLA_PREFILL_TRITON = (
    "vllm.models.deepseek_v4.nvidia_imma.triton_kernels.sparse_attention_triton"
)
SPARSE_MLA_PREFILL_FLASH = "flash_mla.sparse_mla_prefill"
# The int8 variant of the same fused native prefill. It is a distinct symbol
# because it is a distinct kernel: it gathers int8 rows and dequantizes
# in-kernel, with none of the fp8 path's whole-cache bf16 dequant pre-pass
# (that pre-pass allocated 2 KiB/slot and OOM'd 24 GB ranks at production pool
# sizes). nvidia_imma/attention.py calls it whenever the cache is int8_ds_mla,
# so a block naming only SPARSE_MLA_PREFILL_FLASH alongside an int8 cache
# describes a kernel that does not run.
SPARSE_MLA_PREFILL_INT8_FLASH = "flash_mla.sparse_mla_prefill_int8"
INDEXER_CACHE_INT8_WRITER = "vllm._custom_ops.indexer_k_quant_and_cache_int8"
INDEXER_QUERY_INT8_QUANT = (
    "vllm.models.deepseek_v4.common.ops.fused_indexer_q."
    "fused_indexer_q_rope_quant_int8"
)
DENSE_EXPERTS_INT8_ACTIVATION = (
    "vllm.model_executor.layers.quantization.utils.marlin_utils."
    "marlin_act_int8_process_scales"
)
INDEXER_STREAMING_TOPK_PREFILL = (
    "vllm.model_executor.layers.sparse_attn_indexer.streaming_prefill_topk"
)
# sm_12x (GB10): sparkinfer's fp8-compute CuTe compressed-MLA kernel, wrapped
# by the nvidia_sm12x launchers. Decode and prefill are distinct symbols
# (scratch modes) of the same underlying op. On sm12x the portable Triton
# symbols above (SPARSE_MLA_DECODE_FP8_TRITON / SPARSE_MLA_PREFILL_TRITON
# fused-decode variant) are the registered fallback — kernel A/B swaps are an
# --hf-overrides edit, no rebuild.
SPARSE_MLA_DECODE_FP8_SPARKINFER = (
    "vllm.models.deepseek_v4.nvidia_sm12x.kernels.sparkinfer_sparse_mla_decode"
)
SPARSE_MLA_PREFILL_SPARKINFER = (
    "vllm.models.deepseek_v4.nvidia_sm12x.kernels.sparkinfer_sparse_mla_extend"
)

KERNEL_REGISTRY: dict[str, str] = {
    SPARSE_MLA_DECODE_FP8_FLASH: ROLE_SPARSE_MLA_DECODE_FP8,
    SPARSE_MLA_DECODE_FP8_TRITON: ROLE_SPARSE_MLA_DECODE_FP8,
    SPARSE_MLA_DECODE_INT8_TRITON: ROLE_SPARSE_MLA_DECODE_INT8,
    SPARSE_MLA_DECODE_INT8_FLASH: ROLE_SPARSE_MLA_DECODE_INT8,
    SPARSE_MLA_PREFILL_TRITON: ROLE_SPARSE_MLA_PREFILL,
    SPARSE_MLA_PREFILL_FLASH: ROLE_SPARSE_MLA_PREFILL,
    SPARSE_MLA_PREFILL_INT8_FLASH: ROLE_SPARSE_MLA_PREFILL,
    SPARSE_MLA_DECODE_FP8_SPARKINFER: ROLE_SPARSE_MLA_DECODE_FP8,
    SPARSE_MLA_PREFILL_SPARKINFER: ROLE_SPARSE_MLA_PREFILL,
    INDEXER_CACHE_INT8_WRITER: ROLE_INDEXER_CACHE_INT8,
    INDEXER_QUERY_INT8_QUANT: ROLE_INDEXER_QUERY_INT8,
    DENSE_EXPERTS_INT8_ACTIVATION: ROLE_DENSE_EXPERTS_INT8_ACTIVATION,
    INDEXER_STREAMING_TOPK_PREFILL: ROLE_INDEXER_STREAMING_TOPK_PREFILL,
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
        ROLE_INDEXER_STREAMING_TOPK_PREFILL,
    }
)

_PROOF_ROLE_ORDER = (
    ROLE_SPARSE_MLA_DECODE_FP8,
    ROLE_SPARSE_MLA_DECODE_INT8,
    ROLE_SPARSE_MLA_PREFILL,
    ROLE_INDEXER_CACHE_INT8,
    ROLE_INDEXER_QUERY_INT8,
    ROLE_DENSE_EXPERTS_INT8_ACTIVATION,
    ROLE_INDEXER_STREAMING_TOPK_PREFILL,
)


@dataclass(frozen=True)
class ResolvedKernelConfig:
    """Validated role -> symbol assignment for one checkpoint."""

    # True when a "vllm" block with a "kernels" list was present. When
    # False (blockless), selector roles carry their documented defaults and
    # every toggle role is off.
    explicit: bool
    # Active role -> symbol. Selector roles are always present; toggle roles
    # are present iff listed in an explicit block.
    roles: Mapping[str, str] = field(default_factory=dict)
    # Roles whose symbol was EXPLICITLY listed in the block's kernels list
    # (as opposed to filled in by SELECTOR_ROLE_DEFAULTS). Lets arch-specific
    # validators distinguish "the user chose this kernel" from "the global
    # default happened to land here" — e.g. sm12x maps unlisted selector
    # roles to sparkinfer while honoring an explicitly listed Triton symbol.
    listed_roles: frozenset[str] = frozenset()
    cache_type: str | None = None
    # Streaming prefill top-k context-slab width in compressed-token rows.
    # None = "not specified": the indexer module's documented default applies.
    # Only consulted while ROLE_INDEXER_STREAMING_TOPK_PREFILL is active.
    indexer_prefill_topk_slab_rows: int | None = None

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


# Module renames the checkpoints cannot follow. A published checkpoint names
# its kernels by FQN and is not editable retroactively, so a rename has to keep
# the old spelling resolving to the same role. nvidia_sm86 -> nvidia_imma: the
# module is named for what it requires (IMMA + cp.async, sm_80 and up) rather
# than for the arch it was written on.
_LEGACY_SYMBOL_PREFIXES = {
    "vllm.models.deepseek_v4.nvidia_sm86.": "vllm.models.deepseek_v4.nvidia_imma.",
}


def _canonical_symbol(symbol: str) -> str:
    """Map a legacy kernel FQN onto its current spelling."""
    for old, new in _LEGACY_SYMBOL_PREFIXES.items():
        if symbol.startswith(old):
            return new + symbol[len(old) :]
    return symbol


def resolve_kernel_config(block: Any) -> ResolvedKernelConfig:
    """Resolve and validate the ``vllm`` block (fail closed)."""
    explicit = False
    kernels: list[str] = []
    cache_type: str | None = None
    indexer_prefill_topk_slab_rows: int | None = None

    if block is not None:
        if not isinstance(block, dict):
            raise ValueError(
                f'"{VLLM_CONFIG_KEY}" config block must be a JSON object, '
                f"got {type(block).__name__}"
            )
        unknown_keys = set(block) - _ALLOWED_BLOCK_KEYS
        if unknown_keys:
            raise ValueError(
                f'Unknown "{VLLM_CONFIG_KEY}" config key(s) '
                f"{sorted(unknown_keys)}; allowed: {sorted(_ALLOWED_BLOCK_KEYS)}"
            )
        raw_kernels = block.get("kernels")
        if raw_kernels is not None:
            if not isinstance(raw_kernels, (list, tuple)) or not all(
                isinstance(symbol, str) for symbol in raw_kernels
            ):
                raise ValueError(
                    f'"{VLLM_CONFIG_KEY}.kernels" must be a list of '
                    f"symbol strings, got {raw_kernels!r}"
                )
            explicit = True
            kernels = list(raw_kernels)
        cache_type = block.get("cache_type")
        if cache_type is not None:
            allowed = _allowed_cache_types()
            if cache_type not in allowed:
                raise ValueError(
                    f'Unsupported "{VLLM_CONFIG_KEY}.cache_type" '
                    f"{cache_type!r}; allowed: {list(allowed)}"
                )
        raw_slab_rows = block.get(_INDEXER_PREFILL_TOPK_SLAB_ROWS_KEY)
        if raw_slab_rows is not None:
            # Strict positive int: reject JSON true/false (bool is an int
            # subclass), floats, strings and non-positive widths.
            if (
                isinstance(raw_slab_rows, bool)
                or not isinstance(raw_slab_rows, int)
                or raw_slab_rows < 1
            ):
                raise ValueError(
                    f'"{VLLM_CONFIG_KEY}.{_INDEXER_PREFILL_TOPK_SLAB_ROWS_KEY}" '
                    f"must be a positive integer, got {raw_slab_rows!r}"
                )
            indexer_prefill_topk_slab_rows = raw_slab_rows

    roles: dict[str, str] = {}
    for symbol in kernels:
        symbol = _canonical_symbol(symbol)
        role = KERNEL_REGISTRY.get(symbol)
        if role is None:
            raise ValueError(
                f'Unknown kernel symbol {symbol!r} in "{VLLM_CONFIG_KEY}.'
                f'kernels"; known symbols: {sorted(KERNEL_REGISTRY)}'
            )
        if role in roles:
            raise ValueError(
                f"Kernel symbols {roles[role]!r} and {symbol!r} claim the "
                f"same role {role!r}"
            )
        roles[role] = symbol

    listed_roles = frozenset(roles)

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

    return ResolvedKernelConfig(
        explicit=explicit,
        roles=roles,
        listed_roles=listed_roles,
        cache_type=cache_type,
        indexer_prefill_topk_slab_rows=indexer_prefill_topk_slab_rows,
    )


def _checkpoint_kernel_family(hf_config: Any) -> str:
    """``"int"`` or ``"fp"`` -- which kernel and cache family a checkpoint needs.

    Read from what the checkpoint declares about its own weights, not from the
    device: an int4/int8 checkpoint needs the integer cache and the IMMA
    kernels on every architecture that has them.
    """
    quant = getattr(hf_config, "quantization_config", None) or {}
    if isinstance(quant, Mapping):
        method = str(quant.get("quant_method", "") or "").lower()
    else:
        method = str(getattr(quant, "quant_method", "") or "").lower()
    expert_dtype = str(getattr(hf_config, "expert_dtype", "") or "").lower()
    if method.startswith("dsv4_int") or expert_dtype.startswith("int"):
        return "int"
    return "fp"


def _flash_mla_has(*symbols: str) -> bool:
    """Whether the installed flash_mla exposes these kernels.

    A probe, not an assumption: the wheel is built per architecture and per
    CUDA version, so presence is a property of the install rather than of the
    source tree.
    """
    try:
        import flash_mla
    except ImportError:
        return False
    return all(hasattr(flash_mla, name) for name in symbols)


def blockless_role_defaults(hf_config: Any) -> tuple[dict[str, str], str]:
    """Kernel roles and cache type for a checkpoint carrying no ``vllm`` block.

    Official DeepSeek checkpoints ship no block, so the defaults cannot be a
    single flat table -- they have to follow the checkpoint's own dtypes and
    the platform:

    * **int checkpoints** take the integer cache and the integer kernels,
      preferring the fused native ones where the arch has IMMA plus
      ``cp.async`` (sm_80 and up, which includes sm_121) and the wheel
      actually carries them, and the portable Triton ones otherwise. Their
      indexer and dense-activation integer paths are on: for these weights
      that is the checkpoint's intent, not an opt-in.
    * **fp checkpoints** take the fp8 cache. Kernel choice is by capability:
      sparkinfer on sm_12x, and the Triton upcasting kernels wherever the
      device has no accelerated fp8 (below sm_89), which is what makes
      sm_80/sm_86 work at all.

    The FlashInfer path is chosen by the attention *class* rather than named
    here (its kernels are not registry symbols), so an mxfp4/mxfp8 checkpoint
    on a FlashInfer-capable device reaches it through ``_select_dsv4_attn_cls``
    with the fp8 cache this returns.
    """
    from vllm.platforms import current_platform

    capability = current_platform.get_device_capability()
    major = capability.major if capability is not None else 0

    if _checkpoint_kernel_family(hf_config) == "int":
        native = major >= 8 and _flash_mla_has(
            "sparse_mla_decode_int8", "sparse_mla_prefill_int8"
        )
        roles = {
            ROLE_SPARSE_MLA_DECODE_INT8: (
                SPARSE_MLA_DECODE_INT8_FLASH if native else SPARSE_MLA_DECODE_INT8_TRITON
            ),
            ROLE_SPARSE_MLA_PREFILL: (
                SPARSE_MLA_PREFILL_INT8_FLASH if native else SPARSE_MLA_PREFILL_TRITON
            ),
            # Inapplicable under an int8 cache; carried so the role is always
            # resolvable, never launched.
            ROLE_SPARSE_MLA_DECODE_FP8: SPARSE_MLA_DECODE_FP8_TRITON,
            ROLE_INDEXER_CACHE_INT8: INDEXER_CACHE_INT8_WRITER,
            ROLE_INDEXER_QUERY_INT8: INDEXER_QUERY_INT8_QUANT,
            ROLE_DENSE_EXPERTS_INT8_ACTIVATION: DENSE_EXPERTS_INT8_ACTIVATION,
        }
        return roles, "int8_ds_mla"

    if major >= 12:
        roles = {
            ROLE_SPARSE_MLA_DECODE_FP8: SPARSE_MLA_DECODE_FP8_SPARKINFER,
            ROLE_SPARSE_MLA_PREFILL: SPARSE_MLA_PREFILL_SPARKINFER,
            ROLE_SPARSE_MLA_DECODE_INT8: SPARSE_MLA_DECODE_INT8_TRITON,
        }
        return roles, "fp8_ds_mla"

    # Native-vs-Triton is a question about the installed wheel, NOT about the
    # device's fp8 support. The fused flash_mla fp8 kernels are what runs on
    # sm_86 -- carrying Ampere is the reason that fork exists -- and they take
    # fp8 cache rows into bf16 tensor-core math rather than needing fp8 dot
    # instructions. cuda_supports_fp8e4nv_in_triton answers a different
    # question (can *Triton* lower tl.dot on fp8), and the Triton kernels
    # consume it internally to pick native fp8 ops or the arithmetic upcast
    # codec. Gating native selection on it would send sm_86 to Triton for a
    # reason that does not apply.
    roles = {
        ROLE_SPARSE_MLA_DECODE_FP8: (
            SPARSE_MLA_DECODE_FP8_FLASH
            if _flash_mla_has("sparse_mla_decode_fp8")
            else SPARSE_MLA_DECODE_FP8_TRITON
        ),
        ROLE_SPARSE_MLA_PREFILL: (
            SPARSE_MLA_PREFILL_FLASH
            if _flash_mla_has("sparse_mla_prefill")
            else SPARSE_MLA_PREFILL_TRITON
        ),
        ROLE_SPARSE_MLA_DECODE_INT8: SPARSE_MLA_DECODE_INT8_TRITON,
    }
    return roles, "fp8_ds_mla"


def resolve_kernel_config_from_hf_config(
    hf_config: Any,
) -> ResolvedKernelConfig:
    """Resolve from an HF config object's ``vllm`` block.

    With no block, the defaults follow the checkpoint's dtypes and the device
    rather than a flat table -- see ``blockless_role_defaults``.
    """
    block = getattr(hf_config, VLLM_CONFIG_KEY, None)
    resolved = resolve_kernel_config(block)
    if resolved.explicit:
        return resolved
    roles, cache_type = blockless_role_defaults(hf_config)
    resolved = dataclasses.replace(
        resolved,
        roles={**resolved.roles, **roles},
        cache_type=resolved.cache_type or cache_type,
    )
    logger.info_once(
        "DeepSeek V4: checkpoint carries no %r block; defaults resolved from "
        "its dtypes and this device -> cache_type=%s",
        VLLM_CONFIG_KEY,
        resolved.cache_type,
    )
    return resolved


# ---------------------------------------------------------------------------
# Process-wide activation (set once per process; survives Ray unpickle via
# Dsv4IntConfig.__setstate__ and model build via DeepseekV4Attention.__init__)
# ---------------------------------------------------------------------------

_ACTIVE_CONFIG: ResolvedKernelConfig | None = None


def activate_kernel_config(
    resolved: ResolvedKernelConfig,
) -> None:
    global _ACTIVE_CONFIG
    if _ACTIVE_CONFIG is not None and _ACTIVE_CONFIG != resolved:
        logger.debug(
            "Replacing active kernel config %s with %s",
            _ACTIVE_CONFIG,
            resolved,
        )
    _ACTIVE_CONFIG = resolved


def active_kernel_config() -> ResolvedKernelConfig | None:
    return _ACTIVE_CONFIG


def indexer_cache_int8_enabled() -> bool:
    """True when the indexer K cache stores symmetric INT8 instead of FP8.

    On iff ``indexer_k_quant_and_cache_int8`` is listed in an explicit
    block; no block (or no active config) means OFF.
    """
    config = _ACTIVE_CONFIG
    if config is not None and config.explicit:
        return config.has_role(ROLE_INDEXER_CACHE_INT8)
    return False


def indexer_query_int8_enabled() -> bool:
    """True when the indexer query is quantized to symmetric INT8 so the
    logits run as s8 x s8 integer MMA.

    On iff ``fused_indexer_q_rope_quant_int8`` is listed in an explicit
    block (resolution guarantees the INT8 cache is also on); no block (or
    no active config) means OFF.
    """
    config = _ACTIVE_CONFIG
    if config is not None and config.explicit:
        return config.has_role(ROLE_INDEXER_QUERY_INT8)
    return False


def dense_experts_int8_activation_enabled() -> bool:
    """True when the Marlin INT8-activation (W4A8) expert runtime is active.

    On iff ``marlin_act_int8_process_scales`` is listed in an explicit
    block; no block (or no active config) means OFF.
    """
    config = _ACTIVE_CONFIG
    if config is not None and config.explicit:
        return config.has_role(ROLE_DENSE_EXPERTS_INT8_ACTIVATION)
    return False


def indexer_streaming_topk_prefill_enabled() -> bool:
    """True when the prefill indexer runs the slab-tiled streaming top-k
    instead of materializing the full [M, N] logits.

    On iff ``streaming_prefill_topk`` is listed in an explicit block; no
    block (or no active config) means OFF -- the one-shot prefill path is
    preserved bit-for-bit.
    """
    config = _ACTIVE_CONFIG
    if config is not None and config.explicit:
        return config.has_role(ROLE_INDEXER_STREAMING_TOPK_PREFILL)
    return False


def indexer_prefill_topk_slab_rows_override() -> int | None:
    """Streaming prefill top-k slab width from ``vllm.
    indexer_prefill_topk_slab_rows``; None = use the indexer module's
    documented default."""
    config = _ACTIVE_CONFIG
    if config is None:
        return None
    return config.indexer_prefill_topk_slab_rows


# ---------------------------------------------------------------------------
# Engine-side application (cache_type default + startup fail-closed check)
# ---------------------------------------------------------------------------

_DENSE_QUANT_METHODS = ("dsv4_int", "dsv4_mxfp4_int8")


def apply_checkpoint_config(model_config: Any, cache_config: Any) -> None:
    """Validate the checkpoint's ``vllm`` block at engine startup and apply
    ``cache_type`` as the DEFAULT kv-cache dtype.

    An explicit ``--kv-cache-dtype`` always wins: the default is only applied
    when the CLI left ``cache_dtype`` at ``"auto"``.
    """
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None or getattr(hf_config, VLLM_CONFIG_KEY, None) is None:
        return
    resolved = resolve_kernel_config_from_hf_config(hf_config)

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
            VLLM_CONFIG_KEY,
        )
        cache_config.cache_dtype = resolved.cache_type


def resolved_proof_line(
    resolved: ResolvedKernelConfig, *, kv_cache_dtype: str
) -> str:
    """Single stable-format line stating every active role -> symbol plus the
    resolved kv-cache dtype. This is the benchmark validity check."""
    parts = []
    for role in _PROOF_ROLE_ORDER:
        if role in TOGGLE_ROLES:
            # Toggle roles carry a symbol iff listed in an explicit block;
            # blockless resolution never activates them.
            symbol = resolved.symbol(role) or "off"
        else:
            symbol = resolved.roles[role]
        parts.append(f"{role}={symbol}")
    parts.append(f"cache_type={kv_cache_dtype}")
    return "vllm kernels resolved: " + " ".join(parts)
