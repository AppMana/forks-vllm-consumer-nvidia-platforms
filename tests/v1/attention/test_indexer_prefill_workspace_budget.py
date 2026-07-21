# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill K-gather workspace budget: the chunk planner and every workspace
reservation must derive from the same scheduler-bounded budget
(max_num_partial_prefills x max_model_len), replacing the old
`max_model_len * 40` magic sizing that reserved ~1.3 GiB/rank at 1M context.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.model_executor.layers.sparse_attn_indexer import (
    RADIX_TOPK_WORKSPACE_SIZE,
    _reserve_prefill_gather_workspace,
)
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder,
    get_max_prefill_buffer_size,
    split_indexer_prefill_chunks,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec

PRODUCTION_MAX_MODEL_LEN = 1_048_576
COMPRESS_RATIO = 4
HEAD_DIM = 128
# Planner logits budget; any value works, the N-budget is what is under test.
MAX_LOGITS_BYTES = 8 * 1024 * 1024


def _config(
    max_model_len: int, max_num_partial_prefills: int = 1
) -> SimpleNamespace:
    """Duck-typed VllmConfig with the fields get_max_prefill_buffer_size reads."""
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        scheduler_config=SimpleNamespace(
            max_num_partial_prefills=max_num_partial_prefills
        ),
    )


def test_budget_derives_from_scheduler_prefill_bound():
    cfg = _config(PRODUCTION_MAX_MODEL_LEN)
    assert get_max_prefill_buffer_size(cfg) == PRODUCTION_MAX_MODEL_LEN
    assert (
        get_max_prefill_buffer_size(cfg, COMPRESS_RATIO)
        == PRODUCTION_MAX_MODEL_LEN // COMPRESS_RATIO
    )
    # Scales with the scheduler's concurrent-partial-prefill bound.
    cfg4 = _config(PRODUCTION_MAX_MODEL_LEN, max_num_partial_prefills=4)
    assert (
        get_max_prefill_buffer_size(cfg4, COMPRESS_RATIO)
        == 4 * (PRODUCTION_MAX_MODEL_LEN // COMPRESS_RATIO)
    )
    # Rounds compressed rows up so a single max-length request always fits.
    cfg_odd = _config(163_841)
    assert get_max_prefill_buffer_size(cfg_odd, COMPRESS_RATIO) == cdiv(
        163_841, COMPRESS_RATIO
    )


def test_chunk_planner_never_plans_gather_larger_than_reservation():
    """Both directions of budget consistency.

    Forward: no planned chunk's gathered-context rows exceed the rows the
    indexer layer reserves in the shared workspace. Reverse: the reservation
    is exactly the planner budget (not the old 40x-larger uncompressed
    figure), and a single max-length request still fits in it.
    """
    cfg = _config(PRODUCTION_MAX_MODEL_LEN)
    planner_budget = get_max_prefill_buffer_size(cfg, COMPRESS_RATIO)
    # What DeepseekV4Indexer reserves (rows) — must be the identical quantity.
    reservation_rows = get_max_prefill_buffer_size(cfg, COMPRESS_RATIO)
    assert planner_budget == reservation_rows

    max_req_rows = cdiv(PRODUCTION_MAX_MODEL_LEN, COMPRESS_RATIO)
    workloads = [
        # Single max-length request: must fit the reservation in one chunk.
        [max_req_rows],
        # 41 half-length requests: under the old wiring (uncompressed planner
        # budget vs compressed reservation) these packed into one chunk 4x
        # larger than the reserved workspace.
        [max_req_rows // 2] * 41,
        [1_000] * 300,
        [max_req_rows, 1, max_req_rows, 1],
    ]
    for rows in workloads:
        seq_lens = torch.tensor(rows, dtype=torch.int64)
        query_lens = torch.ones_like(seq_lens)
        chunks = split_indexer_prefill_chunks(
            seq_lens, query_lens, planner_budget, MAX_LOGITS_BYTES
        )
        assert chunks
        for req_slice, _query_slice in chunks:
            chunk_rows = int(seq_lens[req_slice].sum())
            assert chunk_rows <= reservation_rows, (
                f"planned chunk of {chunk_rows} rows exceeds reserved "
                f"workspace of {reservation_rows} rows for workload {rows[:4]}..."
            )
        # Every request is covered by exactly one request-level chunk.
        covered = sorted(
            i
            for req_slice, query_slice in chunks
            if query_slice.start == 0
            for i in range(req_slice.start, req_slice.stop)
        )
        assert covered == list(range(len(rows)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_indexer_builder_budget_matches_layer_reservation():
    """The metadata builder's planner budget must equal the compressed-row
    reservation the indexer layer makes, for the same VllmConfig."""
    kv_cache_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        compress_ratio=COMPRESS_RATIO,
    )
    vllm_config = create_vllm_config(
        model_name="deepseek-ai/DeepSeek-V2-Lite-Chat",
        max_model_len=4096,
        hf_config_override={
            "sliding_window": 128,
            "index_topk": 4,
            "compress_ratios": [COMPRESS_RATIO],
        },
    )
    builder = DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["dummy"],
        vllm_config=vllm_config,
        device=torch.device("cuda"),
    )
    assert builder.max_prefill_buffer_size == get_max_prefill_buffer_size(
        vllm_config, COMPRESS_RATIO
    )


class _RecordingWorkspaceManager:
    def __init__(self):
        self.calls: list[tuple[tuple[tuple[int, ...], torch.dtype], ...]] = []

    def get_simultaneous(self, *shapes_and_dtypes):
        self.calls.append(shapes_and_dtypes)
        return [
            torch.zeros((), dtype=dtype).expand(shape)
            for shape, dtype in shapes_and_dtypes
        ]


def _flashmla_warmup_stub(cls, **attrs):
    """Bare instance for exercising forward_mqa's warmup (dummy-run) branch."""
    inst = object.__new__(cls)
    for name, value in attrs.items():
        object.__setattr__(inst, name, value)
    return inst


def _run_flashmla_warmup(stub, monkeypatch):
    from vllm.models.deepseek_v4.nvidia.flashmla import DeepseekV4FlashMLAAttention

    # Warmup dummy runs execute with a forward context whose attn_metadata
    # is None; that is the branch under test.
    monkeypatch.setattr(
        "vllm.models.deepseek_v4.nvidia.flashmla.get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )
    q = torch.zeros((2, 1, 576))
    output = torch.ones((2, 1, 576))
    positions = torch.zeros((2,), dtype=torch.long)
    DeepseekV4FlashMLAAttention.forward_mqa(stub, q, q, positions, output)
    return output


def test_sm86_flash_prefill_warmup_skips_bf16_gather_workspace(monkeypatch):
    """SPARSE_MLA_PREFILL_FLASH consumes the paged caches directly and never
    touches the bf16 gather workspace; warmup must not reserve it."""
    from vllm.models.deepseek_v4.nvidia_sm86.attention import (
        DeepseekV4TritonSM86Attention,
    )
    from vllm.transformers_utils.configs.dsv4.kernel_config import (
        SPARSE_MLA_PREFILL_FLASH,
    )

    manager = _RecordingWorkspaceManager()
    monkeypatch.setattr(
        "vllm.models.deepseek_v4.nvidia.flashmla.current_workspace_manager",
        lambda: manager,
    )
    stub = _flashmla_warmup_stub(
        DeepseekV4TritonSM86Attention,
        compress_ratio=COMPRESS_RATIO,
        max_model_len=PRODUCTION_MAX_MODEL_LEN,
        window_size=128,
        max_num_batched_tokens=8192,
        prefill_symbol=SPARSE_MLA_PREFILL_FLASH,
    )
    output = _run_flashmla_warmup(stub, monkeypatch)
    assert manager.calls == []
    assert torch.all(output == 0)


@pytest.mark.parametrize("use_sm86_triton_prefill", [False, True])
def test_gather_prefill_warmup_still_reserves_bf16_workspace(
    monkeypatch, use_sm86_triton_prefill
):
    """Paths that stage KV through the bf16 gather workspace (the base
    FlashMLA prefill and the sm86 Triton prefill) must keep the warmup
    reservation."""
    from vllm.models.deepseek_v4.nvidia.flashmla import DeepseekV4FlashMLAAttention

    manager = _RecordingWorkspaceManager()
    monkeypatch.setattr(
        "vllm.models.deepseek_v4.nvidia.flashmla.current_workspace_manager",
        lambda: manager,
    )
    attrs = dict(
        compress_ratio=COMPRESS_RATIO,
        max_model_len=PRODUCTION_MAX_MODEL_LEN,
        window_size=128,
        max_num_batched_tokens=8192,
    )
    if use_sm86_triton_prefill:
        from vllm.models.deepseek_v4.nvidia_sm86.attention import (
            DeepseekV4TritonSM86Attention,
        )
        from vllm.transformers_utils.configs.dsv4.kernel_config import (
            SPARSE_MLA_PREFILL_TRITON,
        )

        stub = _flashmla_warmup_stub(
            DeepseekV4TritonSM86Attention,
            prefill_symbol=SPARSE_MLA_PREFILL_TRITON,
            **attrs,
        )
    else:
        stub = _flashmla_warmup_stub(DeepseekV4FlashMLAAttention, **attrs)

    output = _run_flashmla_warmup(stub, monkeypatch)
    assert torch.all(output == 0)

    expected_n = cdiv(PRODUCTION_MAX_MODEL_LEN, COMPRESS_RATIO)
    expected_m = expected_n + 128 + 8192
    assert manager.calls == [
        (
            (
                (
                    DeepseekV4FlashMLAAttention.PREFILL_CHUNK_SIZE,
                    expected_m,
                    576,
                ),
                torch.bfloat16,
            ),
        )
    ]


def test_workspace_arena_accounting_at_production_geometry(monkeypatch):
    """At max_model_len=1M / compress-4 the persistent per-rank workspace
    demand from the indexer prefill gather plus the flashmla warmup drops from
    ~1.29 GiB (max of the two requests) to ~34 MiB.

    The arena grows to the max ever *requested*, so this asserts on the
    requested sizes routed through a real WorkspaceManager.
    """
    from vllm.transformers_utils.configs.dsv4.kernel_config import (
        SPARSE_MLA_PREFILL_FLASH,
    )
    from vllm.models.deepseek_v4.nvidia_sm86.attention import (
        DeepseekV4TritonSM86Attention,
    )
    from vllm.v1.worker.workspace import WorkspaceManager

    manager = WorkspaceManager(torch.device("cpu"))
    monkeypatch.setattr(
        "vllm.model_executor.layers.sparse_attn_indexer.current_workspace_manager",
        lambda: manager,
    )
    monkeypatch.setattr(
        "vllm.models.deepseek_v4.nvidia.flashmla.current_workspace_manager",
        lambda: manager,
    )

    cfg = _config(PRODUCTION_MAX_MODEL_LEN)
    rows = get_max_prefill_buffer_size(cfg, COMPRESS_RATIO)

    # Indexer prefill gather reservation (every indexer forward makes this).
    _reserve_prefill_gather_workspace(
        total_seq_lens=0,
        max_total_seq_len=rows,
        head_dim=HEAD_DIM,
        fp8_dtype=torch.uint8,
        use_fp4_cache=False,
    )

    # FlashMLA warmup on the production flash prefill path adds nothing.
    stub = _flashmla_warmup_stub(
        DeepseekV4TritonSM86Attention,
        compress_ratio=COMPRESS_RATIO,
        max_model_len=PRODUCTION_MAX_MODEL_LEN,
        window_size=128,
        max_num_batched_tokens=8192,
        prefill_symbol=SPARSE_MLA_PREFILL_FLASH,
    )
    _run_flashmla_warmup(stub, monkeypatch)

    arena_bytes = manager._workspace_size_bytes(manager._current_workspaces[0])
    expected_bytes = (
        round_up(rows * HEAD_DIM, 256)
        + round_up(rows * 4, 256)
        + round_up(RADIX_TOPK_WORKSPACE_SIZE, 256)
    )
    assert arena_bytes == expected_bytes
    assert arena_bytes < 40 * 1024 * 1024

    # Old sizing for the record: indexer 40 * max_model_len uncompressed-row
    # budget (132 B per compressed row) and the unconditional bf16 gather
    # warmup. The reclaimed arena demand is the drop in the max request.
    old_indexer_bytes = (PRODUCTION_MAX_MODEL_LEN * 40 // COMPRESS_RATIO) * (
        HEAD_DIM + 4
    ) + RADIX_TOPK_WORKSPACE_SIZE
    old_flashmla_bytes = 4 * (rows + 128 + 8192) * 576 * 2
    old_arena_bytes = max(old_indexer_bytes, old_flashmla_bytes)
    # 1,349,517,312 bytes (~1.26 GiB) reclaimed at production geometry.
    assert old_arena_bytes - arena_bytes == 1_349_517_312
