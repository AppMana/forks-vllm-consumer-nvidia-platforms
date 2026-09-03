# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.model_executor.warmup.flashinfer_sparse_mla_warmup as sparse_mla_warmup
import vllm.v1.worker.gpu.sample.states as sampling_states
import vllm.v1.worker.gpu.warmup as gpu_warmup


class _FakeKVConnector:
    def set_disabled(self, disabled: bool) -> None:
        self.disabled = disabled


class _FakeBlockTable:
    block_size = 16
    max_num_blocks_per_req = 100
    blocks_per_kv_block = 1


class _FakeMultiGroupBlockTable:
    def __init__(self):
        self.block_tables = [_FakeBlockTable()]
        self.calls = []

    def add_row(self, block_ids, row_idx):
        self.calls.append(("add_row", block_ids, row_idx))

    def commit_block_table(self, num_reqs):
        self.calls.append(("commit_block_table", num_reqs))

    def compute_slot_mapping(self, num_reqs, query_start_loc, positions):
        self.calls.append(
            ("compute_slot_mapping", num_reqs, tuple(query_start_loc.tolist()), len(positions))
        )

    def clear_row(self, row_idx):
        self.calls.append(("clear_row", row_idx))


class _FakeModelRunner:
    is_pooling_model = False
    is_last_pp_rank = False
    is_encoder_decoder = False
    decode_query_len = 1
    device = None
    max_num_reqs = 6
    model_state = SimpleNamespace(max_encoder_len=0)
    kv_connector = _FakeKVConnector()
    kv_cache_config = SimpleNamespace(
        num_blocks=128,
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16)),
        ],
    )
    parallel_config = SimpleNamespace(pipeline_parallel_size=4)
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"])
    )
    scheduler_config = SimpleNamespace(max_num_batched_tokens=16, max_num_seqs=6)


def test_mixed_prefill_decode_warmup_drains_async_pp_slots():
    executed = []

    def execute_model(scheduler_output):
        executed.append(scheduler_output)

    def sample_tokens(_grammar_output):
        return None

    assert gpu_warmup.run_mixed_prefill_decode_warmup(
        _FakeModelRunner(),
        execute_model,
        sample_tokens,
        16,
    )

    scheduled_token_counts = [
        output.total_num_scheduled_tokens for output in executed
    ]
    assert scheduled_token_counts == [
        2,
        0,
        0,
        0,
        0,
        16,
        0,
        0,
        0,
        0,
        0,
    ]
    assert executed[-1].finished_req_ids == {
        "_v2_mixed_warmup_decode_",
        "_v2_mixed_warmup_prefill_",
    }


def test_deepseek_v4_long_prefill_warmup_runs_production_shapes(monkeypatch):
    mixed_sizes = []
    metadata_warmup = []
    runner = SimpleNamespace(
        is_pooling_model=False,
        device=torch.device("cuda", 0),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1024),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
        ),
    )

    monkeypatch.setattr(
        gpu_warmup,
        "warmup_prefill_chunk_metadata_kernel",
        lambda device, *, compress_ratio: metadata_warmup.append(
            (device, compress_ratio)
        ),
    )
    monkeypatch.setattr(
        gpu_warmup,
        "run_mixed_prefill_decode_warmup",
        lambda _runner, _execute, _sample, num_tokens, **_kwargs: (
            mixed_sizes.append(num_tokens) or True
        ),
    )

    gpu_warmup.warmup_long_prefill_kernels(
        runner,
        lambda _scheduler_output: None,
        lambda _grammar_output: None,
    )

    assert metadata_warmup == [(torch.device("cuda", 0), 4)]
    assert mixed_sizes == [16, 1024]


def test_deepseek_v4_long_prefill_warmup_includes_scheduler_cap(monkeypatch):
    from vllm.config import VllmConfig
    from vllm.config.speculative import SpeculativeConfig

    class ParallelDraftSpec:
        method = "dspark"
        num_speculative_tokens = 5
        parallel_drafting = True
        draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(sample_from_anchor=True)
        )

        uses_draft_model = SpeculativeConfig.uses_draft_model
        max_num_new_slots_for_drafting = (
            SpeculativeConfig.max_num_new_slots_for_drafting
        )

    scheduler_config = SimpleNamespace(
        max_num_batched_tokens=1024,
        max_num_scheduled_tokens=None,
        max_num_seqs=6,
    )
    vllm_config = SimpleNamespace(
        speculative_config=ParallelDraftSpec(),
        scheduler_config=scheduler_config,
    )
    VllmConfig._set_max_num_scheduled_tokens(vllm_config)

    mixed_sizes = []
    pure_prefill_sizes = []
    runner = SimpleNamespace(
        is_pooling_model=False,
        device=None,
        scheduler_config=scheduler_config,
        vllm_config=vllm_config,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
        ),
    )
    monkeypatch.setattr(
        gpu_warmup,
        "run_mixed_prefill_decode_warmup",
        lambda _runner, _execute, _sample, num_tokens, **_kwargs: (
            mixed_sizes.append(num_tokens) or True
        ),
    )
    monkeypatch.setattr(
        gpu_warmup,
        "run_pure_prefill_warmup",
        lambda _runner, _execute, _sample, num_tokens, **_kwargs: (
            pure_prefill_sizes.append(num_tokens) or True
        ),
    )

    gpu_warmup.warmup_long_prefill_kernels(
        runner,
        lambda _scheduler_output: None,
        lambda _grammar_output: None,
    )

    assert scheduler_config.max_num_scheduled_tokens == 1024
    assert mixed_sizes == [16, 1024]
    assert pure_prefill_sizes == [1, 28, 60, 1020]


def test_dspark_prepare_input_warmup_covers_every_reachable_block_size(monkeypatch):
    class ParallelDraftSpec:
        method = "dspark"
        num_speculative_tokens = 7
        draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(sample_from_anchor=True)
        )

        max_num_new_slots_for_drafting = 6

    pure_prefill_sizes = []
    runner = SimpleNamespace(
        is_pooling_model=False,
        device=None,
        num_speculative_steps=7,
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=1024,
            max_num_scheduled_tokens=1024,
        ),
        vllm_config=SimpleNamespace(speculative_config=ParallelDraftSpec()),
    )
    monkeypatch.setattr(
        gpu_warmup,
        "run_mixed_prefill_decode_warmup",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        gpu_warmup,
        "run_pure_prefill_warmup",
        lambda _runner, _execute, _sample, num_tokens, **_kwargs: (
            pure_prefill_sizes.append(num_tokens) or True
        ),
    )

    gpu_warmup.warmup_long_prefill_kernels(
        runner,
        lambda _scheduler_output: None,
        lambda _grammar_output: None,
    )

    assert pure_prefill_sizes == [1, 26, 58, 1018]


def test_topk_topp_warmup_covers_each_triton_specialization(monkeypatch):
    calls = []
    runner = SimpleNamespace(
        is_last_pp_rank=True,
        device=torch.device("cpu"),
        decode_query_len=8,
        model_config=SimpleNamespace(get_vocab_size=lambda: 128),
    )

    def apply(logits, top_k, top_p):
        calls.append(
            (
                tuple(logits.shape),
                None if top_k is None else top_k.dtype,
                None if top_p is None else top_p.dtype,
            )
        )
        return logits

    monkeypatch.setattr(
        gpu_warmup, "apply_top_k_top_p_triton", apply, raising=False
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    assert gpu_warmup.warmup_topk_topp_sampler(runner)
    assert calls == [
        ((8, 128), None, torch.float32),
        ((8, 128), torch.int32, None),
        ((8, 128), torch.int32, torch.float32),
    ]


def test_deepseek_v4_long_prefill_warmup_replays_pure_prefill_boundary_shape():
    runner = _FakeModelRunner()
    runner.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=1024,
        max_num_scheduled_tokens=1000,
        max_num_seqs=6,
    )
    runner.vllm_config = SimpleNamespace(speculative_config=None)
    runner.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
    )
    executed = []

    gpu_warmup.warmup_long_prefill_kernels(
        runner,
        executed.append,
        lambda _grammar_output: None,
    )

    boundary_shapes = [
        (
            len(output.num_scheduled_tokens),
            tuple(output.num_scheduled_tokens.values()),
            bool(output.scheduled_spec_decode_tokens),
            all(
                (
                    new_req := next(
                        (
                            req
                            for req in output.scheduled_new_reqs
                            if req.req_id == req_id
                        ),
                        None,
                    )
                )
                is not None
                and new_req.num_computed_tokens + num_scheduled
                < len(new_req.prefill_token_ids)
                for req_id, num_scheduled in output.num_scheduled_tokens.items()
            ),
        )
        for output in executed
        if output.total_num_scheduled_tokens == 1000
    ]
    assert boundary_shapes == [(1, (1000,), False, True)]


def test_deepseek_v4_long_prefill_warmup_directly_warms_slot_mapping(monkeypatch):
    mixed_sizes = []
    metadata_warmup = []
    block_table = _FakeMultiGroupBlockTable()
    runner = SimpleNamespace(
        is_pooling_model=False,
        device=torch.device("cpu"),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
        ),
        input_batch=SimpleNamespace(block_table=block_table),
    )

    monkeypatch.setattr(
        gpu_warmup,
        "warmup_prefill_chunk_metadata_kernel",
        lambda device, *, compress_ratio: metadata_warmup.append(
            (device, compress_ratio)
        ),
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(
        gpu_warmup,
        "run_mixed_prefill_decode_warmup",
        lambda _runner, _execute, _sample, num_tokens, **_kwargs: (
            mixed_sizes.append(num_tokens) or True
        ),
    )

    gpu_warmup.warmup_long_prefill_kernels(
        runner,
        lambda _scheduler_output: None,
        lambda _grammar_output: None,
    )

    assert metadata_warmup == [(torch.device("cpu"), 4)]
    assert mixed_sizes == [16]
    assert block_table.calls == [
        ("add_row", ([1],), 0),
        ("commit_block_table", 1),
        ("compute_slot_mapping", 1, (0, 16), 16),
        ("clear_row", 0),
        ("commit_block_table", 1),
    ]


class _TinyCapacityBlockTable:
    """block_size=16, max_num_blocks_per_req=4 -> a single row can only hold
    4*16=64 tokens. Mirrors a KV-cache manager built against a small
    max_model_len (testbed/shrink configs, unit tests): max_num_batched_tokens
    can still carry its full production-scale default and exceed that."""

    block_size = 16
    max_num_blocks_per_req = 4
    blocks_per_kv_block = 1


def test_block_table_warmup_clamps_to_table_capacity(monkeypatch):
    """Regression test for the fp8-block-checkpoint smoke-test crash this
    session: warmup previously sized its synthetic row from
    max_num_batched_tokens alone, overflowing block_table.np whenever that
    exceeded a single row's real (max_model_len-derived) capacity --
    reproduced independently on the pre-existing MXFP4/Flash path, so this is
    a general block-table-warmup bug, not specific to the fp8-block source."""
    block_table = _FakeMultiGroupBlockTable()
    block_table.block_tables = [_TinyCapacityBlockTable()]
    model_runner = SimpleNamespace(
        input_batch=SimpleNamespace(block_table=block_table),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    result = gpu_warmup.warmup_block_table_slot_mapping_kernel(
        model_runner, torch.device("cpu")
    )

    assert result is True
    # cdiv(8192, 16) = 512 blocks would be requested unclamped; the table can
    # only hold 4, so the clamp must land on exactly 4.
    assert block_table.calls[0] == ("add_row", ([1, 2, 3, 4],), 0)
    # slot mapping must cover only the clamped token count (4*16=64), not
    # the full max_num_batched_tokens=8192 that would have overflowed.
    assert block_table.calls[2] == ("compute_slot_mapping", 1, (0, 64), 64)


def test_long_prefill_warmup_caps_synthetic_requests_at_max_model_len(monkeypatch):
    """max_num_batched_tokens is a batch budget and may exceed max_model_len
    (opt-125m: 8192 vs 2048). The mixed and pure-prefill warmups synthesize
    ONE request from it, and a block-table row only holds
    cdiv(max_model_len, block_size) blocks, so an unclamped size overflows the
    row (staged block-table write overflows row ... len=8191 > 2048)."""
    mixed_sizes = []
    pure_prefill_sizes = []
    runner = SimpleNamespace(
        is_pooling_model=False,
        device=None,
        max_model_len=2048,
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=8192,
            max_num_scheduled_tokens=8000,
        ),
        vllm_config=SimpleNamespace(speculative_config=None),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["OPTForCausalLM"])
        ),
    )
    monkeypatch.setattr(
        gpu_warmup,
        "run_mixed_prefill_decode_warmup",
        lambda _runner, _execute, _sample, num_tokens, **_kwargs: (
            mixed_sizes.append(num_tokens) or True
        ),
    )
    monkeypatch.setattr(
        gpu_warmup,
        "run_pure_prefill_warmup",
        lambda _runner, _execute, _sample, num_tokens, **_kwargs: (
            pure_prefill_sizes.append(num_tokens) or True
        ),
    )

    gpu_warmup.warmup_long_prefill_kernels(
        runner,
        lambda _scheduler_output: None,
        lambda _grammar_output: None,
    )

    # The mixed step's prefill request carries num_tokens - 1 prompt tokens;
    # the pure-prefill request carries num_tokens + 1. Both must fit a row.
    assert mixed_sizes == [16, 2048]
    assert pure_prefill_sizes == [2047]


def test_deepseek_v4_pp_warmup_kernels_run_coupled_production_batch(monkeypatch):
    mixed_sizes = []
    metadata_warmup = []
    sampler_warmup = []
    runner = SimpleNamespace(
        is_pooling_model=False,
        parallel_config=SimpleNamespace(pipeline_parallel_size=5),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
        ),
        device=torch.device("cuda", 0),
        num_speculative_steps=0,
    )

    monkeypatch.setattr(
        gpu_warmup,
        "warmup_prefill_chunk_metadata_kernel",
        lambda device, *, compress_ratio: metadata_warmup.append(
            (device, compress_ratio)
        ),
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(
        gpu_warmup,
        "run_mixed_prefill_decode_warmup",
        lambda _runner, _execute, _sample, num_tokens, **_kwargs: (
            mixed_sizes.append(num_tokens) or True
        ),
    )
    monkeypatch.setattr(
        gpu_warmup,
        "warmup_topk_topp_sampler",
        lambda _runner: sampler_warmup.append(True) or True,
        raising=False,
    )

    gpu_warmup.warmup_kernels(
        runner,
        lambda _scheduler_output: None,
        lambda _grammar_output: None,
    )

    assert mixed_sizes == [16]
    assert metadata_warmup == [(torch.device("cuda", 0), 4)]
    assert sampler_warmup == [True]


def test_non_deepseek_v4_pp_warmup_kernels_keeps_generic_execute_model(monkeypatch):
    executed = []
    sampled = []
    runner = _FakeModelRunner()
    runner.parallel_config = SimpleNamespace(pipeline_parallel_size=5)
    runner.num_speculative_steps = 0

    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    gpu_warmup.warmup_kernels(
        runner,
        lambda scheduler_output: executed.append(scheduler_output),
        lambda grammar_output: sampled.append(grammar_output),
    )

    assert [output.total_num_scheduled_tokens for output in executed] == [
        12,
        6,
        0,
        2,
        0,
        0,
        0,
        0,
        0,
        16,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert len(sampled) == 4


def test_deepseek_v4_spec_warmup_uses_long_prefill_compile_class(monkeypatch):
    executed = []
    runner = _FakeModelRunner()
    runner.parallel_config = SimpleNamespace(pipeline_parallel_size=1)
    runner.num_speculative_steps = 5
    runner.decode_query_len = 6
    runner.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=8192, max_num_seqs=1
    )
    runner.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"]),
        get_vocab_size=lambda: 1024,
    )
    runner.is_last_pp_rank = False

    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(gpu_warmup, "warmup_long_prefill_kernels", lambda *args: None)

    gpu_warmup.warmup_kernels(
        runner,
        lambda scheduler_output: executed.append(scheduler_output),
        lambda _grammar_output: None,
    )

    assert executed[0].total_num_scheduled_tokens == 256


# ---------------------------------------------------------------------------
# Pipeline-parallel collective symmetry of the warmup sequences.
#
# Field evidence (12-rank PP DeepSeek V4 + DSpark, num_speculative_tokens=7):
# the server never became ready, all GPUs idle at 0%, and SIGUSR2 dumps showed
#   rank 0  warmup.py run_spec_verify_warmup -> gpu_worker.py:1116
#           isend_tensor_dict -> send_object -> torch.distributed.send BLOCKED
#   rank 11 warmup.py run_spec_verify_warmup -> gpu_worker.py:1080
#           irecv_tensor_dict -> recv_object -> torch.distributed.recv BLOCKED
#   rank 1  not in the warmup at all, back in worker_busy_loop's dequeue
# i.e. the two end ranks were mid-sequence while a middle rank had already
# left it. The harness below models exactly the transfer rule
# GPUWorker.execute_model implements, and the blocking metadata rendezvous
# both isend_tensor_dict and irecv_tensor_dict perform, so that any
# divergence in per-rank transfer counts shows up as a bounded failure
# instead of a hung test.
# ---------------------------------------------------------------------------

_PP_TIMEOUT = 5.0


class _PPStranded(RuntimeError):
    """A rank blocked in a rendezvous no peer will ever match."""


class _PPLinks:
    """One rendezvous channel per pipeline hop `rank -> rank + 1`.

    Mirrors parallel_state.send_object/recv_object (parallel_state.py:852 and
    :872): the metadata handshake blocks on BOTH sides until it is matched,
    which is why an unmatched send wedges the sender just as an unmatched
    recv wedges the receiver.
    """

    def __init__(self, pp_size: int):
        self.pp_size = pp_size
        self.hops = [queue.Queue() for _ in range(pp_size - 1)]
        self.sends = [0] * pp_size
        self.recvs = [0] * pp_size
        self.stranded: list[tuple[int, str]] = []

    def send(self, rank: int) -> None:
        ack = threading.Event()
        self.sends[rank] += 1
        self.hops[rank].put(ack)
        if not ack.wait(_PP_TIMEOUT):
            self.stranded.append((rank, "send"))
            raise _PPStranded(f"rank {rank} blocked in send")

    def recv(self, rank: int) -> None:
        try:
            ack = self.hops[rank - 1].get(timeout=_PP_TIMEOUT)
        except queue.Empty:
            self.stranded.append((rank, "recv"))
            raise _PPStranded(f"rank {rank} blocked in recv") from None
        self.recvs[rank] += 1
        ack.set()

    @property
    def unmatched(self) -> int:
        return sum(hop.qsize() for hop in self.hops)

    def assert_balanced(self) -> None:
        """Every hop must have as many recvs on its far side as sends on its
        near side, and nothing left in flight."""
        assert self.stranded == [], (
            f"ranks blocked on peers that walked away: {self.stranded}"
        )
        for hop in range(self.pp_size - 1):
            assert self.sends[hop] == self.recvs[hop + 1], (
                f"hop {hop}->{hop + 1}: {self.sends[hop]} sends vs "
                f"{self.recvs[hop + 1]} recvs"
            )
        assert self.sends[self.pp_size - 1] == 0
        assert self.recvs[0] == 0
        assert self.unmatched == 0


class _FakeAllReduce:
    """Barrier-backed MIN all-reduce across the simulated ranks."""

    def __init__(self, pp_size: int):
        self.barrier = threading.Barrier(pp_size, timeout=_PP_TIMEOUT)
        self.slots = [0] * pp_size
        self.rounds = 0

    def __call__(self, tensor, op=None, group=None):
        idx = self.barrier.wait()
        self.slots[idx] = int(tensor.item())
        self.barrier.wait()
        tensor.fill_(min(self.slots))
        if idx == 0:
            self.rounds += 1
        self.barrier.wait()


def _spec_runner(rank: int, pp_size: int, **overrides):
    runner = SimpleNamespace(
        is_pooling_model=False,
        num_speculative_steps=7,
        kv_connector=_FakeKVConnector(),
        kv_cache_config=SimpleNamespace(
            num_blocks=128,
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16)),
            ],
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size),
    )
    for key, value in overrides.items():
        setattr(runner, key, value)
    return runner


def _pp_execute_model(rank: int, pp_size: int, links: _PPLinks):
    """The transfer rule GPUWorker.execute_model implements.

    `forward_pass = scheduler_output.total_num_scheduled_tokens > 0`
    (gpu_worker.py:1036) gates the irecv (gpu_worker.py:1071); a zero-token
    batch also returns from GPUModelRunner.execute_model before producing
    IntermediateTensors (gpu/model_runner.py:1289-1292), so the isend at
    gpu_worker.py:1116 is never reached either.
    """

    def execute_model(scheduler_output):
        if scheduler_output.total_num_scheduled_tokens <= 0:
            return None
        if rank != 0:
            links.recv(rank)
        if rank != pp_size - 1:
            links.send(rank)
        return None

    return execute_model


def _run_ranks(monkeypatch, pp_size, runners, target=gpu_warmup.run_spec_verify_warmup):
    """Drive `target` on every simulated rank concurrently."""
    links = _PPLinks(pp_size)
    all_reduce = _FakeAllReduce(pp_size)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        gpu_warmup, "get_pp_group", lambda: SimpleNamespace(cpu_group=None)
    )

    results: list[object] = [None] * pp_size
    errors: list[BaseException | None] = [None] * pp_size

    def run(rank: int) -> None:
        try:
            results[rank] = target(
                runners[rank],
                _pp_execute_model(rank, pp_size, links),
                lambda _grammar_output: None,
            )
        except BaseException as exc:  # noqa: BLE001 - recorded, asserted on
            errors[rank] = exc

    threads = [
        threading.Thread(target=run, args=(rank,), daemon=True)
        for rank in range(pp_size)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=4 * _PP_TIMEOUT)
    assert not any(thread.is_alive() for thread in threads), (
        "a simulated rank never finished; the warmup can wedge a PP rank"
    )
    return links, all_reduce, results, errors


def test_spec_verify_warmup_transfers_match_on_every_rank(monkeypatch):
    """Healthy 12-rank pipeline: identical transfer counts, warmup preserved."""
    pp_size = 12
    runners = [_spec_runner(rank, pp_size) for rank in range(pp_size)]

    links, _all_reduce, results, errors = _run_ranks(monkeypatch, pp_size, runners)

    assert errors == [None] * pp_size
    links.assert_balanced()
    assert results == [True] * pp_size
    # Three batches carry scheduled tokens (prefill, first verify, steady
    # verify). The pp_size empty drain batches and the cleanup batch must add
    # nothing at all, on any rank.
    assert links.sends == [3] * (pp_size - 1) + [0]
    assert links.recvs == [0] + [3] * (pp_size - 1)
    assert links.unmatched == 0


def test_spec_verify_warmup_uses_scheduler_reachable_atomic_width():
    """Every target verification is one anchor plus its speculative block.

    The scheduler's working PP path never emits a drafts-only target forward:
    its fixed-width transaction is the leading anchor followed by K draft
    tokens.  Warmup must exercise that reachable shape rather than manufacture
    a different protocol.
    """
    num_spec = 7
    runner = _spec_runner(0, 1, num_speculative_steps=num_spec)
    executed = []

    assert gpu_warmup.run_spec_verify_warmup(
        runner,
        executed.append,
        lambda _grammar_output: None,
    )

    verify_frames = [
        output for output in executed if output.scheduled_spec_decode_tokens
    ]
    assert len(verify_frames) == 2
    for frame_idx, output in enumerate(verify_frames):
        req_id, draft_tokens = next(iter(output.scheduled_spec_decode_tokens.items()))
        assert output.num_scheduled_tokens[req_id] == 1 + len(draft_tokens)
        assert output.replayed_pp_anchor_req_ids == ({req_id} if frame_idx else set())


def test_spec_verify_warmup_exercises_runtime_temperature_processing(monkeypatch):
    """Warmup must take the same non-greedy temperature branch as requests."""
    runner = _spec_runner(0, 1, num_speculative_steps=5)
    executed = []

    assert gpu_warmup.run_spec_verify_warmup(
        runner,
        executed.append,
        lambda _grammar_output: None,
    )
    sampling_params = executed[0].scheduled_new_reqs[0].sampling_params
    assert sampling_params is not None

    class _CpuStagedArray:
        def __init__(self, dtype):
            self.np = np.zeros(1, dtype=dtype)
            self.gpu = torch.from_numpy(self.np.copy())

        def copy_to_uva(self):
            self.gpu.copy_(torch.from_numpy(self.np))

    states = object.__new__(sampling_states.SamplingStates)
    states.max_num_reqs = 1
    states.vocab_size = 4
    states.temperature = _CpuStagedArray(np.float32)
    states.top_k = _CpuStagedArray(np.int32)
    states.top_p = _CpuStagedArray(np.float32)
    states.min_p = _CpuStagedArray(np.float32)
    states.seeds = _CpuStagedArray(np.int64)
    states.seeds_set = np.zeros(1, dtype=bool)
    states.num_logprobs = np.full(1, sampling_states.NO_LOGPROBS, dtype=np.int32)
    states.add_request(0, sampling_params)
    states.apply_staged_writes()

    def _emulate_temperature_kernel(logits, idx_mapping, temperatures):
        logits.div_(temperatures[idx_mapping].unsqueeze(1))

    monkeypatch.setattr(
        sampling_states, "apply_temperature", _emulate_temperature_kernel
    )
    logits = torch.tensor([[2.0, 4.0]])
    states.apply_temperature(
        logits,
        torch.tensor([0]),
        np.array([0], dtype=np.int32),
    )

    # A representative non-greedy warmup temperature of 0.5 must run the
    # real SamplingStates branch and transform logits just as a live request
    # does. Greedy temperature=0 bypasses the kernel and leaves these values
    # unchanged, reproducing the request-time first JIT.
    torch.testing.assert_close(logits, torch.tensor([[4.0, 8.0]]))


def test_spec_verify_warmup_respects_per_request_pp_cadence():
    """A request cannot be reused until its PP state has reached every rank."""
    pp_size = 4
    runner = _spec_runner(0, pp_size)
    executed = []

    assert gpu_warmup.run_spec_verify_warmup(
        runner,
        executed.append,
        lambda _grammar_output: None,
    )

    nonzero_indices = [
        idx
        for idx, output in enumerate(executed)
        if output.total_num_scheduled_tokens > 0
    ]
    assert len(nonzero_indices) == 3
    for left, right in zip(nonzero_indices, nonzero_indices[1:]):
        advances = executed[left + 1 : right]
        assert len(advances) == pp_size
        assert all(output.total_num_scheduled_tokens == 0 for output in advances)

    cleanup_idx = next(
        idx for idx, output in enumerate(executed) if output.finished_req_ids
    )
    final_drain = executed[nonzero_indices[-1] + 1 : cleanup_idx]
    assert len(final_drain) == pp_size
    assert all(output.total_num_scheduled_tokens == 0 for output in final_drain)


def test_spec_verify_warmup_is_skipped_on_all_ranks_when_one_rank_declines(
    monkeypatch,
):
    """Regression: the reproduced 2026-08-24 deadlock.

    One middle rank's LOCAL gate fails (its own kv_cache_config cannot
    supply the warmup blocks). Before the collective agreement it returned
    False on its own while every other rank walked into the pipeline, which
    is precisely the captured failure: rank 0 blocked in send_object, rank
    11 blocked in recv_object, the middle rank already back in the worker
    busy loop.
    """
    pp_size = 12
    runners = [_spec_runner(rank, pp_size) for rank in range(pp_size)]
    runners[5].kv_cache_config = SimpleNamespace(
        num_blocks=1,
        kv_cache_groups=runners[5].kv_cache_config.kv_cache_groups,
    )

    links, _all_reduce, results, errors = _run_ranks(monkeypatch, pp_size, runners)

    assert errors == [None] * pp_size
    links.assert_balanced()
    # Uniform decision: all ranks skip, nobody transfers anything.
    assert results == [False] * pp_size
    assert links.sends == [0] * pp_size
    assert links.recvs == [0] * pp_size
    assert links.unmatched == 0


def test_sparse_mla_autotune_rank_local_backend_gate_keeps_pp_sequence_aligned(
    monkeypatch,
    tmp_path,
):
    """A local backend gate cannot skip a collectively coupled PP warmup."""
    pp_size = 12

    class Backend:
        def __init__(self, name):
            self.name = name

        def get_name(self):
            return self.name

    runners = []
    for rank in range(pp_size):
        runner = _spec_runner(
            rank,
            pp_size,
            max_num_reqs=6,
            vllm_config=SimpleNamespace(use_v2_model_runner=True),
        )
        runner.attn_groups = (
            [[SimpleNamespace(backend=Backend("FLASHINFER_MLA_SPARSE_SM120"))]]
            if rank < 4
            else []
        )
        runners.append(runner)

    world = SimpleNamespace(
        rank_in_group=1,
        rank=1,
        broadcast_object=lambda obj, src: None,
        barrier=lambda: None,
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_world_group", lambda: world
    )
    monkeypatch.setattr(sparse_mla_warmup, "has_flashinfer", lambda: True)
    monkeypatch.setattr(
        sparse_mla_warmup,
        "resolve_flashinfer_autotune_file",
        lambda _runner: tmp_path / "autotune-cache",
    )
    monkeypatch.setattr(
        sparse_mla_warmup,
        "flashinfer_autotune",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        sparse_mla_warmup.current_platform,
        "is_device_capability_family",
        lambda _capability: True,
    )
    monkeypatch.setattr(
        gpu_warmup, "run_pp_coupled", lambda _runner, _what, body: body()
    )

    def run_autotune(runner, execute_model, sample_tokens):
        worker = SimpleNamespace(
            model_runner=runner,
            execute_model=execute_model,
            sample_tokens=sample_tokens,
            vllm_config=SimpleNamespace(
                kernel_config=SimpleNamespace(enable_flashinfer_autotune=True)
            ),
        )
        return sparse_mla_warmup._run_flashinfer_sparse_mla_decode_autotune(
            worker,
            16,
            frozenset({"FLASHINFER_MLA_SPARSE_SM120"}),
        )

    links, _all_reduce, results, errors = _run_ranks(
        monkeypatch,
        pp_size,
        runners,
        target=run_autotune,
    )

    links.assert_balanced()
    assert errors == [None] * pp_size
    assert results == [False] * pp_size
    assert links.sends == [0] * pp_size
    assert links.recvs == [0] * pp_size


def test_zero_layer_tail_rank_participates_identically(monkeypatch):
    """The draft/MTP tail rank owns no target layers.

    An attention-free worker gets num_blocks=1 with zero cache groups
    (kv_cache_utils.py:1365-1372), so its block requirement is zero. It must
    reach the same decision as every other rank and post exactly the same
    transfers for its position in the pipeline.
    """
    pp_size = 12
    runners = [_spec_runner(rank, pp_size) for rank in range(pp_size)]
    runners[pp_size - 1].kv_cache_config = SimpleNamespace(
        num_blocks=1, kv_cache_groups=[]
    )

    links, _all_reduce, results, errors = _run_ranks(monkeypatch, pp_size, runners)

    assert errors == [None] * pp_size
    links.assert_balanced()
    assert results == [True] * pp_size
    assert links.recvs[pp_size - 1] == 3
    assert links.sends[pp_size - 1] == 0
    assert links.recvs == [0] + [3] * (pp_size - 1)
    assert links.unmatched == 0


def test_spec_verify_warmup_uniformly_skipped_without_speculation(monkeypatch):
    """num_speculative_steps is config-uniform: every rank returns together,
    and the collective must NOT be entered (a rank that returned before it
    would hang the ranks that reached it)."""
    pp_size = 12
    runners = [
        _spec_runner(rank, pp_size, num_speculative_steps=0) for rank in range(pp_size)
    ]

    links, all_reduce, results, errors = _run_ranks(monkeypatch, pp_size, runners)

    assert errors == [None] * pp_size
    links.assert_balanced()
    assert results == [False] * pp_size
    assert links.sends == [0] * pp_size
    assert links.recvs == [0] * pp_size
    assert all_reduce.rounds == 0


def test_pp_coupled_sequence_fails_fast_rather_than_stranding_peers(monkeypatch):
    """A failure after the first transfer cannot unwind quietly: returning to
    the worker busy loop leaves peers in an unmatched rendezvous forever."""
    exits: list[int] = []
    monkeypatch.setattr(gpu_warmup.os, "_exit", lambda code: exits.append(code))

    runner = _spec_runner(0, 12)

    def _boom():
        raise RuntimeError("kernel blew up mid-sequence")

    gpu_warmup.run_pp_coupled(runner, "test sequence", _boom)

    assert exits == [1]


def test_pp_coupled_sequence_reraises_without_pipeline_parallel(monkeypatch):
    exits: list[int] = []
    monkeypatch.setattr(gpu_warmup.os, "_exit", lambda code: exits.append(code))
    runner = _spec_runner(0, 1)

    def _boom():
        raise RuntimeError("kernel blew up mid-sequence")

    with pytest.raises(RuntimeError):
        gpu_warmup.run_pp_coupled(runner, "test sequence", _boom)
    assert exits == []
