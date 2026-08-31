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
import statistics
import subprocess
import sys
import tempfile
import textwrap
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
from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers import sparse_attn_indexer  # noqa: E402
from vllm.model_executor.layers.sparse_attn_indexer import (  # noqa: E402
    warmup_indexer_prefill_logits_kernel,
)
from vllm.models.deepseek_v4.common.ops.cache_utils import (  # noqa: E402
    build_flashinfer_mixed_sparse_indices,
)
from vllm.models.deepseek_v4.nvidia_imma import triton_kernels  # noqa: E402
from vllm.models.deepseek_v4.nvidia_imma.triton_kernels import (  # noqa: E402
    mqa_logits_workspace_triton,
)
from vllm.triton_utils import triton  # noqa: E402
from vllm.utils import jit_monitor  # noqa: E402
from vllm.v1.attention.backends.mla.indexer import (  # noqa: E402
    build_prefill_chunk_metadata,
)

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
        max_num_reqs=32,
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
    max_seq_len = max(4000, frame.seq_len)

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


def _run_c128a_metadata_builder(frame: _PrefillFrame, *, active_width: int) -> None:
    from vllm.models.deepseek_v4.sparse_mla import build_c128a_topk_metadata

    device = torch.device("cuda")
    num_tokens = frame.num_scheduled_tokens
    capacity_width = 256
    build_c128a_topk_metadata(
        positions=torch.arange(
            frame.num_computed_tokens,
            frame.seq_len,
            dtype=torch.int64,
            device=device,
        ),
        compress_ratio=128,
        num_decode_tokens=0,
        token_to_req_indices=torch.zeros(num_tokens, dtype=torch.int32, device=device),
        block_table=torch.zeros((1, 1), dtype=torch.int32, device=device),
        block_size=64,
        slot_mapping=torch.zeros(num_tokens, dtype=torch.int64, device=device),
        global_decode_buffer=torch.empty(
            (1, capacity_width), dtype=torch.int32, device=device
        ),
        decode_lens_buffer=torch.empty(1, dtype=torch.int32, device=device),
        prefill_buffer=torch.empty(
            (num_tokens, capacity_width), dtype=torch.int32, device=device
        ),
        max_compressed_tokens=active_width,
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


def test_direct_warmup_covers_c128a_native_prefill_builder_at_8k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.models.deepseek_v4.sparse_mla import (
        c128a_active_topk_width,
        c128a_prefill_topk_width,
    )

    warmup_frame = _direct_deep_pipeline_warmup_prefill_frame(monkeypatch)
    request_frame = _PrefillFrame(0, 8_000)
    capacity_width = c128a_prefill_topk_width(1_000_000, 128)
    warmup_width = c128a_active_topk_width(warmup_frame.seq_len, 128, capacity_width)
    request_width = c128a_active_topk_width(request_frame.seq_len, 128, capacity_width)
    assert warmup_width == request_width == 128

    _run_c128a_metadata_builder(warmup_frame, active_width=warmup_width)
    _run_sparse_indices_builder(
        warmup_frame,
        compress_ratio=128,
        topk=warmup_width,
        mixed_warmup_layout=True,
    )
    _activate_real_triton_monitor(monkeypatch)
    _run_c128a_metadata_builder(request_frame, active_width=request_width)
    _run_sparse_indices_builder(
        request_frame,
        compress_ratio=128,
        topk=request_width,
        mixed_warmup_layout=False,
    )


def _run_mqa_workspace(
    frame: _PrefillFrame,
    *,
    seq_len_kv: int,
) -> None:
    device = torch.device("cuda")
    num_rows = frame.num_scheduled_tokens
    q = torch.randint(-8, 8, (num_rows, 64, 128), dtype=torch.int8, device=device)
    k = torch.randint(-8, 8, (seq_len_kv, 128), dtype=torch.int8, device=device)
    k_scale = torch.rand(seq_len_kv, dtype=torch.float32, device=device)
    weights = torch.rand((num_rows, 64), dtype=torch.float32, device=device)

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


def test_direct_warmup_covers_mqa_first_scored_continuation(
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
        )


def test_real_mixed_continuation_metadata_has_aligned_row_bounds() -> None:
    """The metadata builder allocates row bounds; it never returns offset views."""
    device = torch.device("cuda")
    query_start_loc_cpu = torch.tensor([0, 1, 941], dtype=torch.int32)
    query_start_loc = query_start_loc_cpu.to(device)
    uncompressed_seq_lens = torch.tensor([1, 4000], dtype=torch.int32, device=device)
    compressed_seq_lens_cpu = torch.tensor([1, 1000], dtype=torch.int32)
    compressed_seq_lens = compressed_seq_lens_cpu.to(device)
    block_table = torch.zeros((2, 16), dtype=torch.int32, device=device)

    metadata = build_prefill_chunk_metadata(
        start_idx=1,
        end_idx=2,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        uncompressed_seq_lens=uncompressed_seq_lens,
        compressed_seq_lens=compressed_seq_lens,
        compressed_seq_lens_cpu=compressed_seq_lens_cpu,
        block_table=block_table,
        compress_ratio=4,
        query_slice=slice(5, 929),
    )

    assert metadata is not None
    assert metadata.token_start == 6
    assert metadata.token_end == 930
    for row_bounds in (metadata.cu_seqlen_ks, metadata.cu_seqlen_ke):
        assert row_bounds.shape == (924,)
        assert row_bounds.is_contiguous()
        assert row_bounds.stride() == (1,)
        assert row_bounds.storage_offset() == 0
        assert row_bounds.data_ptr() % 16 == 0


def test_long_prefill_warmup_loads_native_gather_boundary_specializations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup must execute the native templates used at 256 and 512+ rows."""
    calls = []
    synchronizations = []
    real_gather = ops.cp_gather_indexer_k_quant_cache
    real_synchronize = torch.accelerator.synchronize

    def record_real_gather(kv_cache, dst_k, dst_scale, block_table, cu_seq_lens):
        calls.append(
            (
                dst_k.shape,
                dst_k.dtype,
                dst_scale.shape,
                kv_cache.shape,
                block_table.shape,
                tuple(cu_seq_lens.tolist()),
            )
        )
        real_gather(kv_cache, dst_k, dst_scale, block_table, cu_seq_lens)

    def record_real_synchronize():
        real_synchronize()
        synchronizations.append(None)

    monkeypatch.setattr(ops, "cp_gather_indexer_k_quant_cache", record_real_gather)
    monkeypatch.setattr(torch.accelerator, "synchronize", record_real_synchronize)
    monkeypatch.setattr(triton_kernels, "indexer_cache_is_int8", lambda: True)
    monkeypatch.setattr(
        sparse_attn_indexer,
        "_INDEXER_PREFILL_GATHER_KERNEL_WARMUPS",
        set(),
        raising=False,
    )
    monkeypatch.setattr(
        gpu_warmup,
        "warmup_prefill_chunk_metadata_kernel",
        lambda _device, *, compress_ratio: None,
    )
    monkeypatch.setattr(
        gpu_warmup,
        "warmup_indexer_prefill_logits_kernel",
        lambda _device: None,
    )
    monkeypatch.setattr(
        gpu_warmup,
        "warmup_indexer_prefill_topk_kernel",
        lambda _device: None,
    )
    monkeypatch.setattr(
        gpu_warmup,
        "warmup_block_table_slot_mapping_kernel",
        lambda _runner, _device: False,
    )
    runner = SimpleNamespace(
        is_pooling_model=False,
        max_num_reqs=32,
        device=torch.device("cuda"),
        parallel_config=SimpleNamespace(pipeline_parallel_size=12),
        kv_connector=_FakeKVConnector(),
        kv_cache_config=SimpleNamespace(
            num_blocks=4096,
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=64))
            ],
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1024),
        input_batch=None,
    )
    outputs = []
    monkeypatch.setattr(
        gpu_warmup,
        "run_pp_coupled",
        lambda _runner, _what, body: body(),
    )

    gpu_warmup.warmup_long_prefill_kernels(
        runner,
        outputs.append,
        lambda _grammar: None,
    )
    gpu_warmup.warmup_long_prefill_kernels(
        runner,
        lambda _output: None,
        lambda _grammar: None,
    )

    mixed_prefill_rows = [
        output.num_scheduled_tokens[request.req_id] // 4
        for output in outputs
        for request in output.scheduled_new_reqs
        if request.req_id.endswith("_prefill_")
    ]
    assert max(mixed_prefill_rows) == 255

    assert calls == [
        (
            torch.Size((256, 128)),
            torch.int8,
            torch.Size((256, 4)),
            torch.Size((4, 64, 132)),
            torch.Size((1, 4)),
            (0, 256),
        ),
        (
            torch.Size((512, 128)),
            torch.int8,
            torch.Size((512, 4)),
            torch.Size((8, 64, 132)),
            torch.Size((1, 8)),
            (0, 512),
        ),
    ]
    assert synchronizations == [None]


def test_indexer_logits_warmup_covers_production_rows_in_fresh_process() -> None:
    """The startup helper must cover M > 1 without a persistent cache hit."""
    script = textwrap.dedent(
        """
        from unittest.mock import patch

        import torch

        from vllm.model_executor.layers.sparse_attn_indexer import (
            warmup_indexer_prefill_logits_kernel,
        )
        from vllm.models.deepseek_v4.nvidia_imma.triton_kernels import (
            mqa_logits_workspace_triton,
        )
        from vllm.utils import jit_monitor

        warmup_indexer_prefill_logits_kernel(
            torch.device("cuda"),
            qk_int8=True,
        )

        with (
            patch.object(jit_monitor, "_setup_cutedsl_jit_hook", lambda: None),
            patch.object(jit_monitor, "_setup_tilelang_jit_hook", lambda: None),
        ):
            jit_monitor.activate(mode="error", verbose=True)

        for num_rows, seq_len_kv in ((1024, 768), (928, 1000)):
            q = torch.zeros(
                (num_rows, 64, 128), dtype=torch.int8, device="cuda"
            )
            k = torch.zeros((seq_len_kv, 128), dtype=torch.int8, device="cuda")
            k_scale = torch.ones(seq_len_kv, dtype=torch.float32, device="cuda")
            weights = torch.ones(
                (num_rows, 64), dtype=torch.float32, device="cuda"
            )
            row_starts = torch.zeros(num_rows, dtype=torch.int32, device="cuda")
            row_ends = torch.full(
                (num_rows,), seq_len_kv, dtype=torch.int32, device="cuda"
            )
            mqa_logits_workspace_triton(
                q,
                (k, k_scale),
                weights,
                row_starts,
                row_ends,
                qk_int8=True,
            )
            torch.accelerator.synchronize()
        """
    )
    with tempfile.TemporaryDirectory(prefix="triton_dsv4_fresh_process_") as cache:
        env = os.environ.copy()
        env["TRITON_CACHE_DIR"] = cache
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    assert result.returncode == 0, result.stdout + result.stderr


def test_indexer_logits_warmup_covers_unaligned_streaming_scale_slab() -> None:
    """A valid odd slab width must not compile a new scale-pointer variant."""
    script = textwrap.dedent(
        """
        from unittest.mock import patch

        import torch

        from vllm.model_executor.layers.sparse_attn_indexer import (
            warmup_indexer_prefill_logits_kernel,
        )
        from vllm.models.deepseek_v4.nvidia_imma.triton_kernels import (
            mqa_logits_workspace_triton,
        )
        from vllm.utils import jit_monitor

        device = torch.device("cuda")
        warmup_indexer_prefill_logits_kernel(device, qk_int8=True)

        with (
            patch.object(jit_monitor, "_setup_cutedsl_jit_hook", lambda: None),
            patch.object(jit_monitor, "_setup_tilelang_jit_hook", lambda: None),
        ):
            jit_monitor.activate(mode="error", verbose=True)

        # `indexer_prefill_topk_slab_rows=5001` is valid configuration.  Its
        # second slab preserves K's 128-byte row alignment but offsets the
        # float32 scale pointer by four bytes.
        slab_rows = 5001
        context_rows = 513
        k_storage = torch.zeros(
            (slab_rows + context_rows, 128), dtype=torch.int8, device=device
        )
        k_scale_storage = torch.ones(
            slab_rows + context_rows, dtype=torch.float32, device=device
        )
        k = k_storage[slab_rows:]
        k_scale = k_scale_storage[slab_rows:]
        assert k.data_ptr() % 16 == 0
        assert k_scale.data_ptr() % 16 == 4

        q = torch.zeros((1, 64, 128), dtype=torch.int8, device=device)
        weights = torch.ones((1, 64), dtype=torch.float32, device=device)
        row_starts = torch.zeros(1, dtype=torch.int32, device=device)
        row_ends = torch.full(
            (1,), context_rows, dtype=torch.int32, device=device
        )
        mqa_logits_workspace_triton(
            q,
            (k, k_scale),
            weights,
            row_starts,
            row_ends,
            qk_int8=True,
        )
        torch.accelerator.synchronize()
        """
    )
    with tempfile.TemporaryDirectory(prefix="triton_dsv4_unaligned_scale_") as cache:
        env = os.environ.copy()
        env["TRITON_CACHE_DIR"] = cache
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("num_rows", "seq_len_kv"),
    [(1024, 768), (928, 1000)],
)
def test_mqa_scored_prefill_sm86_performance(
    num_rows: int,
    seq_len_kv: int,
) -> None:
    """Keep the real scored 4k-prefill shapes on the vectorized path."""
    if "RTX A5000" not in torch.cuda.get_device_name():
        pytest.skip("performance threshold is calibrated for an RTX A5000")

    device = torch.device("cuda")
    q = torch.randint(
        -8,
        8,
        (num_rows, 64, 128),
        dtype=torch.int8,
        device=device,
    )
    k = torch.randint(
        -8,
        8,
        (seq_len_kv, 128),
        dtype=torch.int8,
        device=device,
    )
    k_scale = torch.rand(seq_len_kv, dtype=torch.float32, device=device)
    weights = torch.rand((num_rows, 64), dtype=torch.float32, device=device)
    starts = torch.zeros(num_rows, dtype=torch.int32, device=device)
    ends = torch.full((num_rows,), seq_len_kv, dtype=torch.int32, device=device)

    # Compile outside the measurement and settle clocks/caches before sampling.
    for _ in range(6):
        mqa_logits_workspace_triton(
            q,
            (k, k_scale),
            weights,
            starts,
            ends,
            qk_int8=True,
        )
    torch.accelerator.synchronize()

    samples_ms = []
    for _ in range(21):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        mqa_logits_workspace_triton(
            q,
            (k, k_scale),
            weights,
            starts,
            ends,
            qk_int8=True,
        )
        end.record()
        end.synchronize()
        samples_ms.append(start.elapsed_time(end))

    median_ms = statistics.median(samples_ms)
    assert median_ms < 0.40, (
        f"scored-prefill MQA regression: M={num_rows}, N={seq_len_kv}, "
        f"median={median_ms:.3f} ms"
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
        "warmup_indexer_prefill_gather_kernel",
        lambda _device: None,
    )
    monkeypatch.setattr(
        gpu_warmup,
        "warmup_indexer_prefill_topk_kernel",
        lambda _device: None,
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


def test_long_prefill_warmup_exercises_native_prefill_topk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    synchronize_calls = 0
    real_topk = ops.top_k_per_row_prefill
    real_synchronize = torch.accelerator.synchronize

    def record_real_topk(*args):
        calls.append(args)
        return real_topk(*args)

    def record_real_synchronize() -> None:
        nonlocal synchronize_calls
        synchronize_calls += 1
        real_synchronize()

    runner = SimpleNamespace(
        is_pooling_model=False,
        max_num_reqs=32,
        device=torch.device("cuda"),
        parallel_config=SimpleNamespace(pipeline_parallel_size=12),
        kv_connector=_FakeKVConnector(),
        kv_cache_config=SimpleNamespace(
            num_blocks=4096,
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=64))
            ],
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1024),
        input_batch=None,
    )
    monkeypatch.setattr(ops, "top_k_per_row_prefill", record_real_topk)
    monkeypatch.setattr(torch.accelerator, "synchronize", record_real_synchronize)
    monkeypatch.setattr(
        gpu_warmup,
        "warmup_prefill_chunk_metadata_kernel",
        lambda _device, *, compress_ratio: None,
    )
    monkeypatch.setattr(
        gpu_warmup,
        "warmup_indexer_prefill_gather_kernel",
        lambda _device: None,
    )
    monkeypatch.setattr(
        gpu_warmup,
        "warmup_indexer_prefill_logits_kernel",
        lambda _device: None,
    )
    monkeypatch.setattr(
        gpu_warmup,
        "warmup_block_table_slot_mapping_kernel",
        lambda _runner, _device: False,
    )
    monkeypatch.setattr(
        sparse_attn_indexer,
        "_INDEXER_PREFILL_TOPK_KERNEL_WARMUPS",
        set(),
        raising=False,
    )
    outputs = []
    monkeypatch.setattr(
        gpu_warmup,
        "run_pp_coupled",
        lambda _runner, _what, body: body(),
    )

    gpu_warmup.warmup_long_prefill_kernels(
        runner,
        outputs.append,
        lambda _grammar: None,
    )
    gpu_warmup.warmup_long_prefill_kernels(
        runner,
        lambda _output: None,
        lambda _grammar: None,
    )

    assert len(calls) == 1
    assert synchronize_calls == 1
    logits, row_starts, row_ends, output, num_rows, _, _, topk = calls[0]
    mixed_prefill_rows = [
        scheduled_output.num_scheduled_tokens[request.req_id] // 4
        for scheduled_output in outputs
        for request in scheduled_output.scheduled_new_reqs
        if request.req_id.endswith("_prefill_")
    ]
    assert max(mixed_prefill_rows) == 255
    assert num_rows == logits.shape[0]
    assert output.shape == (num_rows, topk)
    assert row_starts.min().item() == 0
    assert row_ends.max().item() > topk
    assert logits.shape[1] >= row_ends.max().item()
