# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The warmup key set must equal the key set the live path launches.

``VllmJitKernel`` subclasses declare, in ``get_warmup_keys()``, the exact
Triton compile keys the serving path will ask for. When the two drift the
failure is silent in every way that matters: the kernel simply JIT-compiles
inside the first real request of that shape, or -- worse -- warmup compiles a
constexpr the live path can never produce (``COMPRESS_RATIO=0``, a divisor of
zero) while never compiling the one it does.

14d3908be2 found three drifts of exactly this kind after the upstream merge.
These tests re-derive the live key set from config the way the serving code
does and compare it to what warmup enumerates, so a fourth cannot land quietly.

Pure Python: dispatch() only computes a frozen dataclass, so no GPU is needed.
"""

from types import SimpleNamespace

import pytest

# Shape of the checkpoint this branch serves. compress_ratios deliberately
# stores 0 for the SWA-only layers, which is what the checkpoint does and what
# made warmup compile COMPRESS_RATIO=0.
INDEX_TOPK = 2048
COMPRESS_RATIOS = (0, 4, 128)
SLIDING_WINDOW = 128
MAX_NUM_BATCHED_TOKENS = 8192


def make_vllm_config(
    *,
    max_model_len: int = 8192,
    index_topk: int = INDEX_TOPK,
    compress_ratios: tuple[int, ...] = COMPRESS_RATIOS,
    sliding_window: int = SLIDING_WINDOW,
    max_num_batched_tokens: int = MAX_NUM_BATCHED_TOKENS,
) -> SimpleNamespace:
    """A VllmConfig stand-in for this checkpoint's shape.

    The warmup paths read config only through ``getattr`` chains, and a real
    VllmConfig would need a model on disk, so a namespace keeps the test
    hermetic without weakening what it checks.
    """
    hf_config = SimpleNamespace(
        index_topk=index_topk,
        compress_ratios=list(compress_ratios),
        sliding_window=sliding_window,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len, hf_config=hf_config),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_num_batched_tokens),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
    )


def live_combine_topk_swa_keys(vllm_config: SimpleNamespace) -> set:
    """The keys ``combine_topk_swa_indices`` actually launches, from config.

    Mirrors ``DeepseekV4FlashMLAImpl._forward_prefill`` (nvidia/flashmla.py):

    * SWA-only layers (ratio <= 1) pass the full-width ``topk_indices_buffer``
      and pin ``TOP_K=0``;
    * C4A layers pass that same buffer and take ``top_k`` from its width;
    * every other ratio passes the C128A prefill buffer, whose width comes
      from ``max_model_len``.

    ``topk_indices_buffer`` is allocated ``[max_num_batched_tokens,
    index_topk]`` in nvidia/model.py, so its width is ``index_topk``.

    Alignment classes: every live caller slices ``seq_lens`` and
    ``gather_lens`` by the same ``chunk_start``, so those two always agree;
    ``query_start_loc`` is offset by ``num_decodes + chunk_start`` and varies
    independently; ``topk_indices`` is a row slice of an int32 buffer whose
    width is a multiple of 4 here, so it is always 16-byte aligned.
    """
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        _COMBINE_TOPK_SWA_INDICES_KERNEL,
    )
    from vllm.models.deepseek_v4.sparse_mla import c128a_prefill_topk_width

    hf_config = vllm_config.model_config.hf_config
    index_topk = hf_config.index_topk
    max_model_len = vllm_config.model_config.max_model_len

    layers: set[tuple[int, int, int]] = set()
    for raw_ratio in hf_config.compress_ratios:
        # models/deepseek_v4/attention.py: max(1, config.compress_ratios[i]).
        ratio = max(1, int(raw_ratio))
        if ratio <= 1:
            layers.add((ratio, 0, index_topk))
        elif ratio == 4:
            layers.add((ratio, index_topk, index_topk))
        else:
            width = c128a_prefill_topk_width(max_model_len, ratio)
            layers.add((ratio, width, width))

    keys = set()
    for ratio, top_k, width in layers:
        for query_start_loc_aligned in (True, False):
            for seq_lens_aligned in (True, False):
                keys.add(
                    _COMBINE_TOPK_SWA_INDICES_KERNEL.dispatch(
                        topk_width=width,
                        topk_indices=True,
                        query_start_loc=query_start_loc_aligned,
                        seq_lens=seq_lens_aligned,
                        gather_lens=seq_lens_aligned,
                        topk=top_k,
                        compress_ratio=ratio,
                        WINDOW_SIZE=hf_config.sliding_window,
                    )
                )
    return keys


@pytest.mark.parametrize("max_model_len", [8192, 163840])
def test_combine_topk_swa_warmup_matches_live(max_model_len: int) -> None:
    """Warmup must enumerate exactly the live key set -- no more, no less.

    Would have caught 14d3908be2: the C128A row was hardcoded
    ``topk=topk_width=8192`` (only correct near a 1M-token context; the live
    width at 163840 is 1280), the C4A rows were hardcoded 512/1024 instead of
    ``hf_config.index_topk``, and the SWA-only row was paired with a width it
    never sees.
    """
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        _COMBINE_TOPK_SWA_INDICES_KERNEL,
    )

    vllm_config = make_vllm_config(max_model_len=max_model_len)
    warmed = set(_COMBINE_TOPK_SWA_INDICES_KERNEL.get_warmup_keys(vllm_config))
    live = live_combine_topk_swa_keys(vllm_config)

    assert warmed == live, (
        "combine_topk_swa warmup/live key drift\n"
        f"  warmed but never launched: {sorted(warmed - live, key=repr)}\n"
        f"  launched but never warmed: {sorted(live - warmed, key=repr)}"
    )


def test_combine_topk_swa_c128a_width_tracks_max_model_len() -> None:
    """The C128A width is a function of max_model_len, not a constant.

    Pins the concrete regression: 8192 was warmed at every context length.
    """
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        _COMBINE_TOPK_SWA_INDICES_KERNEL,
    )
    from vllm.models.deepseek_v4.sparse_mla import c128a_prefill_topk_width
    from vllm.utils.math_utils import next_power_of_2

    assert c128a_prefill_topk_width(163840, 128) == 1280
    assert c128a_prefill_topk_width(8192, 128) == 128

    keys_by_len = {}
    for max_model_len in (8192, 163840):
        keys = _COMBINE_TOPK_SWA_INDICES_KERNEL.get_warmup_keys(
            make_vllm_config(max_model_len=max_model_len)
        )
        keys_by_len[max_model_len] = {
            (key.TOP_K, key.PADDED_TOP_K) for key in keys if key.COMPRESS_RATIO == 128
        }

    # PADDED_TOP_K is next_power_of_2 of the live buffer width. The hardcoded
    # row warmed TOP_K=8192/PADDED_TOP_K=8192 at every context length.
    assert keys_by_len[163840] == {(1280, next_power_of_2(1280))}, keys_by_len
    assert keys_by_len[8192] == {(128, 128)}, keys_by_len


def test_no_warmup_key_has_zero_compress_ratio() -> None:
    """``COMPRESS_RATIO`` is a constexpr divisor; 0 is not a legal value.

    Would have caught 14d3908be2's indexer hunk: a checkpoint storing 0 for
    its SWA-only layers made ``BuildPrefillChunkMetadataKernel`` warm
    ``global_ctx // 0`` and never warm the ``COMPRESS_RATIO=1`` the layer
    (which clamps with ``max(1, ...)``) actually launches.
    """
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        _COMBINE_TOPK_SWA_INDICES_KERNEL,
    )
    from vllm.v1.attention.backends.mla.indexer import (
        _BUILD_PREFILL_CHUNK_METADATA_KERNEL,
    )

    vllm_config = make_vllm_config()
    assert 0 in vllm_config.model_config.hf_config.compress_ratios

    indexer_ratios = {
        key.COMPRESS_RATIO
        for key in _BUILD_PREFILL_CHUNK_METADATA_KERNEL.get_warmup_keys(vllm_config)
    }
    assert 0 not in indexer_ratios, indexer_ratios
    assert 1 in indexer_ratios, (
        f"the clamped SWA-only ratio is never warmed: {sorted(indexer_ratios)}"
    )
    assert indexer_ratios == {1, 4, 128}, sorted(indexer_ratios)

    combine_ratios = {
        key.COMPRESS_RATIO
        for key in _COMBINE_TOPK_SWA_INDICES_KERNEL.get_warmup_keys(vllm_config)
    }
    assert 0 not in combine_ratios, sorted(combine_ratios)


def test_index_topk_is_the_only_width_of_the_shared_topk_buffer() -> None:
    """``topk_tokens`` and the buffer it writes must be the same number.

    ``_fill_short_context_topk_indices`` strides rows by ``TOP_K``
    (``output + row * TOP_K``), and ``TOP_K`` is the indexer's
    ``self.topk_tokens``; the buffer it writes is allocated with
    ``config.index_topk`` columns. If those two ever stop being the same
    config field the short-context path writes each row at the wrong offset,
    which no shape check catches because the buffer is one flat allocation.
    """
    import ast
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]

    attention_src = ast.parse(
        (repo_root / "vllm/models/deepseek_v4/attention.py").read_text(encoding="utf-8")
    )
    topk_tokens_sources = {
        ast.unparse(node.value)
        for node in ast.walk(attention_src)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "topk_tokens"
            for target in node.targets
        )
    }
    assert topk_tokens_sources == {"config.index_topk"}, topk_tokens_sources

    for backend in ("nvidia", "amd", "xpu"):
        model_py = repo_root / f"vllm/models/deepseek_v4/{backend}/model.py"
        if not model_py.exists():
            continue
        model_src = ast.parse(model_py.read_text(encoding="utf-8"))
        allocations = [
            node.value
            for node in ast.walk(model_src)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and target.attr == "topk_indices_buffer"
                for target in node.targets
            )
            and isinstance(node.value, ast.Call)
        ]
        for call in allocations:
            columns = ast.unparse(call.args[-1])
            assert columns == "config.index_topk", (
                f"{backend}/model.py allocates topk_indices_buffer with "
                f"{columns!r}, but the short-context kernel strides rows by "
                f"config.index_topk"
            )
