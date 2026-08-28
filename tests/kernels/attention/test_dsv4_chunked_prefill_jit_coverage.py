# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression coverage for DSV4's direct-warmup JIT boundary.

The reproducer warms a mixed decode + fresh-prefill batch and then advances a
longer prefill through several continuation chunks. These tests deliberately
use the real scheduler-output constructor and the real inference JIT monitor.
A test passes only when the direct warmup covers every Triton specialization
reached by the later continuation chunks.
"""

import os
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace

# Isolate this module from a developer machine's persistent Triton cache.  This
# must be set before importing either kernel module.
_OLD_TRITON_CACHE_DIR = os.environ.get("TRITON_CACHE_DIR")
_TRITON_CACHE_DIR = tempfile.TemporaryDirectory(prefix="triton_dsv4_chunked_")
os.environ["TRITON_CACHE_DIR"] = _TRITON_CACHE_DIR.name

import pytest  # noqa: E402
import torch  # noqa: E402

import vllm.v1.worker.gpu.warmup as gpu_warmup  # noqa: E402
from vllm.model_executor.layers.sparse_attn_indexer import (  # noqa: E402
    warmup_indexer_prefill_logits_kernel,
)
from vllm.models.deepseek_v4.common.ops.cache_utils import (  # noqa: E402
    build_flashinfer_mixed_sparse_indices,
)
from vllm.models.deepseek_v4.nvidia_imma.triton_kernels import (  # noqa: E402
    mqa_logits_workspace_triton,
)
from vllm.triton_utils import triton  # noqa: E402
from vllm.utils import jit_monitor  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 8,
    reason="requires an SM8x CUDA GPU",
)


@dataclass(frozen=True)
class _PrefillFrame:
    num_computed_tokens: int
    num_scheduled_tokens: int

    @property
    def seq_len(self) -> int:
        return self.num_computed_tokens + self.num_scheduled_tokens


_MULTI_CHUNK_PREFILL = (
    _PrefillFrame(0, 1024),
    _PrefillFrame(1024, 1024),
    _PrefillFrame(2048, 1024),
    _PrefillFrame(3072, 928),
)


class _FakeKVConnector:
    def set_disabled(self, disabled: bool) -> None:
        self.disabled = disabled


def _direct_deep_pipeline_warmup_prefill_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> _PrefillFrame:
    """Return the prefill state built by the direct warmup.

    Only the distributed wrapper is replaced: this test models one rank's
    scheduler inputs while retaining a deep pipeline so deferred-state cadence
    is exercised.
    """
    pipeline_parallel_size = 12
    runner = SimpleNamespace(
        is_pooling_model=False,
        max_num_reqs=2,
        parallel_config=SimpleNamespace(pipeline_parallel_size=pipeline_parallel_size),
        kv_connector=_FakeKVConnector(),
        kv_cache_config=SimpleNamespace(
            num_blocks=4096,
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=64))
            ],
        ),
    )
    outputs = []
    monkeypatch.setattr(
        gpu_warmup,
        "run_pp_coupled",
        lambda _runner, _what, body: body(),
    )

    assert gpu_warmup.run_mixed_prefill_decode_warmup(
        runner,
        outputs.append,
        lambda _grammar_output: None,
        1024,
    )

    mixed = next(
        output for output in outputs if output.total_num_scheduled_tokens == 1024
    )
    prefill = next(
        request
        for request in mixed.scheduled_new_reqs
        if request.req_id.endswith("_prefill_")
    )
    frame = _PrefillFrame(
        num_computed_tokens=prefill.num_computed_tokens,
        num_scheduled_tokens=mixed.num_scheduled_tokens[prefill.req_id],
    )
    assert frame == _PrefillFrame(0, 1023)
    # The mixed direct warmup includes one decode token, so the prefill
    # seq_lens pointer is a view beginning one int32 element into the batch.
    assert len(outputs) == 1 + pipeline_parallel_size + 1 + pipeline_parallel_size + 1
    return frame


@pytest.fixture(autouse=True)
def _restore_real_triton_monitor():
    """Keep the process-global Triton hook from leaking between tests."""
    old_hook = triton.knobs.runtime.jit_post_compile_hook
    old_autotuning_print = triton.knobs.autotuning.print
    old_print_env = os.environ.get("TRITON_PRINT_AUTOTUNING")
    old_state = (
        jit_monitor._active,
        jit_monitor._mode,
        jit_monitor._verbose,
        jit_monitor._cutedsl_hook_installed,
        jit_monitor._tilelang_hook_installed,
        jit_monitor._tilelang_jitimpl_compile_depth,
    )
    jit_monitor._active = False
    try:
        yield
    finally:
        triton.knobs.runtime.jit_post_compile_hook = old_hook
        triton.knobs.autotuning.print = old_autotuning_print
        if old_print_env is None:
            os.environ.pop("TRITON_PRINT_AUTOTUNING", None)
        else:
            os.environ["TRITON_PRINT_AUTOTUNING"] = old_print_env
        (
            jit_monitor._active,
            jit_monitor._mode,
            jit_monitor._verbose,
            jit_monitor._cutedsl_hook_installed,
            jit_monitor._tilelang_hook_installed,
            jit_monitor._tilelang_jitimpl_compile_depth,
        ) = old_state


@pytest.fixture(scope="module", autouse=True)
def _restore_triton_cache_directory():
    """Release this module's isolated cache and restore its caller's setting."""
    yield
    if _OLD_TRITON_CACHE_DIR is None:
        os.environ.pop("TRITON_CACHE_DIR", None)
    else:
        os.environ["TRITON_CACHE_DIR"] = _OLD_TRITON_CACHE_DIR
    _TRITON_CACHE_DIR.cleanup()


def _activate_real_triton_monitor(monkeypatch: pytest.MonkeyPatch) -> None:
    # These regressions are specifically Triton cache misses.  Avoid installing
    # process-global wrappers in optional runtimes while retaining activate()'s
    # real Triton post-compile hook and error path.
    monkeypatch.setattr(jit_monitor, "_setup_cutedsl_jit_hook", lambda: None)
    monkeypatch.setattr(jit_monitor, "_setup_tilelang_jit_hook", lambda: None)
    jit_monitor.activate(mode="error", verbose=True)


def _run_sparse_indices_builder(
    frame: _PrefillFrame,
    *,
    compress_ratio: int,
    topk: int,
    mixed_warmup_layout: bool,
) -> None:
    device = torch.device("cuda")
    window_size = 128
    block_size = 64
    max_seq_len = 4000

    decode_swa_indices = torch.empty((0, window_size), dtype=torch.int32, device=device)
    prefill_topk_indices = torch.zeros(
        (frame.num_scheduled_tokens, topk), dtype=torch.int32, device=device
    )
    query_start_loc = torch.tensor(
        [0, frame.num_scheduled_tokens], dtype=torch.int32, device=device
    )
    if mixed_warmup_layout:
        # Mixed batches slice the decode request out of seq_lens. The
        # resulting pointer is offset by four bytes from its allocation.
        seq_lens_storage = torch.tensor(
            [3, frame.seq_len], dtype=torch.int32, device=device
        )
        seq_lens = seq_lens_storage[1:]
        assert seq_lens.data_ptr() % 16 == 4
    else:
        seq_lens = torch.tensor([frame.seq_len], dtype=torch.int32, device=device)
        assert seq_lens.data_ptr() % 16 == 0

    token_to_req_indices = torch.zeros(
        frame.num_scheduled_tokens, dtype=torch.int32, device=device
    )
    swa_block_table = torch.zeros(
        (1, (max_seq_len + block_size - 1) // block_size),
        dtype=torch.int32,
        device=device,
    )
    compressed_block_table = None
    if topk:
        compressed_block_table = torch.zeros_like(swa_block_table)

    build_flashinfer_mixed_sparse_indices(
        decode_swa_indices,
        None,
        None,
        prefill_topk_indices,
        query_start_loc,
        seq_lens,
        token_to_req_indices,
        swa_block_table,
        block_size,
        compressed_block_table,
        block_size,
        window_size,
        compress_ratio,
        topk,
    )
    torch.accelerator.synchronize()


@pytest.mark.parametrize(("compress_ratio", "topk"), [(1, 0), (4, 512)])
def test_direct_warmup_covers_sparse_builder_for_continuations(
    monkeypatch: pytest.MonkeyPatch,
    compress_ratio: int,
    topk: int,
) -> None:
    warmup_frame = _direct_deep_pipeline_warmup_prefill_frame(monkeypatch)
    _run_sparse_indices_builder(
        warmup_frame,
        compress_ratio=compress_ratio,
        topk=topk,
        mixed_warmup_layout=True,
    )

    _activate_real_triton_monitor(monkeypatch)
    for frame in _MULTI_CHUNK_PREFILL:
        _run_sparse_indices_builder(
            frame,
            compress_ratio=compress_ratio,
            topk=topk,
            mixed_warmup_layout=False,
        )


def _run_mqa_workspace(
    frame: _PrefillFrame,
    *,
    seq_len_kv: int,
    unaligned_row_bounds: bool,
) -> None:
    device = torch.device("cuda")
    num_rows = frame.num_scheduled_tokens
    q = torch.randint(-8, 8, (num_rows, 64, 128), dtype=torch.int8, device=device)
    k = torch.randint(-8, 8, (seq_len_kv, 128), dtype=torch.int8, device=device)
    k_scale = torch.rand(seq_len_kv, dtype=torch.float32, device=device)
    weights = torch.rand((num_rows, 64), dtype=torch.float32, device=device)

    if unaligned_row_bounds:
        starts_storage = torch.zeros(num_rows + 1, dtype=torch.int32, device=device)
        ends_storage = torch.full(
            (num_rows + 1,), seq_len_kv, dtype=torch.int32, device=device
        )
        starts = starts_storage[1:]
        ends = ends_storage[1:]
        assert starts.data_ptr() % 16 == 4
        assert ends.data_ptr() % 16 == 4
    else:
        starts = torch.zeros(num_rows, dtype=torch.int32, device=device)
        ends = torch.full((num_rows,), seq_len_kv, dtype=torch.int32, device=device)
        assert starts.data_ptr() % 16 == 0
        assert ends.data_ptr() % 16 == 0

    mqa_logits_workspace_triton(
        q,
        (k, k_scale),
        weights,
        starts,
        ends,
        qk_int8=True,
    )
    torch.accelerator.synchronize()


def test_direct_warmup_covers_mqa_first_scored_continuation_and_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warmup_frame = _direct_deep_pipeline_warmup_prefill_frame(monkeypatch)
    compress_ratio = 4
    topk = 512

    # This is the indexer branch condition. The direct fresh-
    # prefill warmup never invokes MQA because all compressed candidates fit
    # inside top-k, whereas a later continuation is the first state that must
    # score candidates.
    assert warmup_frame.seq_len // compress_ratio <= topk
    first_scored = next(
        frame
        for frame in _MULTI_CHUNK_PREFILL
        if frame.seq_len // compress_ratio > topk
    )
    assert first_scored == _PrefillFrame(2048, 1024)

    # Exercise the same helper used by startup. Calling the kernel directly in
    # this test would let the test pass even if startup forgot to warm it.
    warmup_indexer_prefill_logits_kernel(
        torch.device("cuda"),
        qk_int8=True,
    )

    _activate_real_triton_monitor(monkeypatch)
    for frame in _MULTI_CHUNK_PREFILL[2:]:
        _run_mqa_workspace(
            frame,
            seq_len_kv=frame.seq_len // compress_ratio,
            unaligned_row_bounds=False,
        )


def test_long_prefill_warmup_invokes_indexer_logits_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    runner = SimpleNamespace(
        is_pooling_model=False,
        device=torch.device("cuda"),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2),
        input_batch=None,
    )
    monkeypatch.setattr(
        gpu_warmup,
        "warmup_prefill_chunk_metadata_kernel",
        lambda _device, *, compress_ratio: None,
    )
    monkeypatch.setattr(
        gpu_warmup,
        "warmup_block_table_slot_mapping_kernel",
        lambda _runner, _device: False,
    )
    monkeypatch.setattr(
        gpu_warmup,
        "warmup_indexer_prefill_logits_kernel",
        lambda device: calls.append(device),
    )
    monkeypatch.setattr(
        gpu_warmup,
        "run_mixed_prefill_decode_warmup",
        lambda *_args, **_kwargs: False,
    )

    gpu_warmup.warmup_long_prefill_kernels(
        runner,
        lambda _output: None,
        lambda _grammar: None,
    )

    assert calls == [torch.device("cuda")]
