# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import os
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

import pytest
import torch

from tests.utils import multi_gpu_test
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.executor import multiproc_executor as multiproc_executor_module
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc
from vllm.v1.executor.uniproc_executor import (
    ExecutorWithExternalLauncher,
    UniProcExecutor,
)


class Mock: ...


def test_supports_async_scheduling_base_executor():
    assert Executor.supports_async_scheduling() is False


def test_supports_async_scheduling_uniproc_executor():
    assert UniProcExecutor.supports_async_scheduling() is True


def test_supports_async_scheduling_executor_with_external_launcher():
    # ExecutorWithExternalLauncher inherits from UniProcExecutor and does not
    # override supports_async_scheduling, so it should return True.
    assert ExecutorWithExternalLauncher.supports_async_scheduling() is True


def test_supports_async_scheduling_multiproc_executor():
    assert MultiprocExecutor.supports_async_scheduling() is True


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _FakeProcess:
    def __init__(self, clock: _FakeClock, exits_at: float) -> None:
        self.clock = clock
        self.exits_at = exits_at
        self.terminate_called = False

    def is_alive(self) -> bool:
        return self.clock.time() < self.exits_at

    def terminate(self) -> None:
        self.terminate_called = True


@pytest.mark.parametrize(
    ("timeout", "exits_at", "expected_terminate"),
    [
        pytest.param(6, 5, False, id="worker-exits-before-timeout"),
        pytest.param(6, 7, True, id="worker-exceeds-timeout"),
    ],
)
def test_multiproc_executor_worker_termination_timeout(
    monkeypatch, timeout, exits_at, expected_terminate
):
    monkeypatch.setenv("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", str(timeout))
    clock = _FakeClock()
    monkeypatch.setattr(multiproc_executor_module.time, "time", clock.time)
    monkeypatch.setattr(multiproc_executor_module.time, "sleep", clock.sleep)
    executor = MultiprocExecutor.__new__(MultiprocExecutor)
    proc = _FakeProcess(clock, exits_at=exits_at)
    executor._ensure_worker_termination([proc])
    assert proc.terminate_called is expected_terminate


class CustomMultiprocExecutor(MultiprocExecutor):
    def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
        kv_output_aggregator: KVOutputAggregator = None,
    ) -> Any | list[Any] | Future[Any | list[Any]]:
        # Drop marker to show that this was run
        with open(".marker", "w"):
            ...
        return super().collective_rpc(
            method,
            timeout,
            args,
            kwargs,
            non_block,
            unique_reply_rank,
            kv_output_aggregator,
        )


CustomMultiprocExecutorAsync = CustomMultiprocExecutor
MODEL = "Qwen/Qwen3-0.6B"


def test_custom_executor_type_checking():
    with pytest.raises(ValueError):
        engine_args = EngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=Mock,
        )
        LLMEngine.from_engine_args(engine_args)
    with pytest.raises(ValueError):
        engine_args = AsyncEngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=Mock,
        )
        AsyncLLM.from_engine_args(engine_args)


@pytest.mark.parametrize(
    "distributed_executor_backend",
    [
        CustomMultiprocExecutor,
        "tests.v1.executor.test_executor.CustomMultiprocExecutor",
    ],
)
def test_custom_executor(distributed_executor_backend, tmp_path):
    cwd = os.path.abspath(".")
    os.chdir(tmp_path)
    try:
        assert not os.path.exists(".marker")

        engine_args = EngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=distributed_executor_backend,
            enforce_eager=True,  # reduce test time
        )
        engine = LLMEngine.from_engine_args(engine_args)
        sampling_params = SamplingParams(max_tokens=1)

        engine.add_request("0", "foo", sampling_params)
        engine.step()

        assert os.path.exists(".marker")
    finally:
        os.chdir(cwd)


@pytest.mark.parametrize(
    "distributed_executor_backend",
    [
        CustomMultiprocExecutorAsync,
        "tests.v1.executor.test_executor.CustomMultiprocExecutorAsync",
    ],
)
def test_custom_executor_async(distributed_executor_backend, tmp_path):
    cwd = os.path.abspath(".")
    os.chdir(tmp_path)
    try:
        assert not os.path.exists(".marker")

        engine_args = AsyncEngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=distributed_executor_backend,
            enforce_eager=True,  # reduce test time
        )
        engine = AsyncLLM.from_engine_args(engine_args)
        sampling_params = SamplingParams(max_tokens=1)

        async def t():
            stream = engine.generate(
                request_id="0", prompt="foo", sampling_params=sampling_params
            )
            async for x in stream:
                ...

        asyncio.run(t())

        assert os.path.exists(".marker")
    finally:
        os.chdir(cwd)


class _BroadcastMQDrained(Exception):
    """Raised by the fake broadcast MQ when its scripted RPCs run out."""


class _FakeBroadcastMQ:
    def __init__(self, items: list) -> None:
        self._items = list(items)

    def dequeue(self, timeout: float | None = None, indefinite: bool = False):
        if not self._items:
            raise _BroadcastMQDrained
        return self._items.pop(0)


class _FakeResponseMQ:
    def __init__(self) -> None:
        self.enqueued: list = []

    def enqueue(self, item) -> None:
        self.enqueued.append(item)


class _FakeWorker:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def execute_model(self, scheduler_output):
        raise self._error


def _make_worker_proc(rank: int, rpc_items: list, error: Exception) -> WorkerProc:
    proc = WorkerProc.__new__(WorkerProc)
    proc.rank = rank
    proc.worker = _FakeWorker(error)
    proc.rpc_broadcast_mq = _FakeBroadcastMQ(rpc_items)
    proc.worker_response_mq = _FakeResponseMQ()
    proc.use_async_scheduling = False
    return proc


def test_worker_busy_loop_non_output_rank_error_is_fatal(monkeypatch):
    """A kernel error (e.g. CUDA OOM during sparse prefill attention) on a
    worker that is not the reply rank for the RPC (e.g. PP rank 0 of a
    pipeline where only the last stage replies to execute_model) must fail
    the worker instead of being swallowed. A swallowed error desynchronizes
    the pipeline: the failed rank never sends its intermediate tensors, so
    downstream recvs pair with later steps and the engine silently emits
    garbage until a collective desync kills it much later. The worker must
    hard-exit (graceful teardown can hang in NCCL destroy while peers are
    blocked in collectives) so the executor's monitor sees the death."""

    class _HardExit(BaseException):
        pass

    exit_codes: list[int] = []

    def _fake_exit(code: int):
        exit_codes.append(code)
        raise _HardExit

    monkeypatch.setattr(multiproc_executor_module.os, "_exit", _fake_exit)
    oom = torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.23 GiB")
    proc = _make_worker_proc(
        rank=0,
        rpc_items=[("execute_model", (None,), {}, 11)],
        error=oom,
    )
    with pytest.raises(_HardExit):
        proc.worker_busy_loop()
    assert exit_codes == [1]
    # No stray FAILURE may be enqueued: the driver never dequeues this
    # rank's response MQ for a unique-reply RPC, and a stale message would
    # corrupt response ordering for a later collective RPC.
    assert proc.worker_response_mq.enqueued == []


def test_worker_busy_loop_output_rank_error_returns_failure():
    """On the reply rank, an exception is converted to a FAILURE response
    (the driver raises it) and the busy loop keeps serving."""
    oom = torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.23 GiB")
    proc = _make_worker_proc(
        rank=11,
        rpc_items=[("execute_model", (None,), {}, 11)],
        error=oom,
    )
    with pytest.raises(_BroadcastMQDrained):
        proc.worker_busy_loop()
    assert len(proc.worker_response_mq.enqueued) == 1
    status, result = proc.worker_response_mq.enqueued[0]
    assert status == WorkerProc.ResponseStatus.FAILURE
    assert "CUDA out of memory" in result


def test_worker_busy_loop_collective_rpc_error_returns_failure():
    """For collective RPCs (no unique reply rank) every rank replies, so an
    exception is returned as FAILURE and the worker stays alive."""
    err = RuntimeError("boom")
    proc = _make_worker_proc(
        rank=0,
        rpc_items=[("execute_model", (None,), {}, None)],
        error=err,
    )
    with pytest.raises(_BroadcastMQDrained):
        proc.worker_busy_loop()
    assert len(proc.worker_response_mq.enqueued) == 1
    status, result = proc.worker_response_mq.enqueued[0]
    assert status == WorkerProc.ResponseStatus.FAILURE
    assert "boom" in result


def _install_pp_first_rank_oom(worker) -> bool:
    """Fault injector RPC: make the first PP stage's model runner raise CUDA
    OOM on its next forward, mimicking a sparse prefill attention kernel OOM."""
    from vllm.distributed.parallel_state import get_pp_group

    if not get_pp_group().is_first_rank:
        return False

    def _oom(*args, **kwargs):
        raise torch.OutOfMemoryError(
            "CUDA out of memory. Tried to allocate 2.23 GiB (injected)"
        )

    worker.model_runner.execute_model = _oom
    return True


@multi_gpu_test(num_gpus=2)
def test_pp_first_rank_oom_fails_request_fast(monkeypatch):
    """Regression test for the fail-slow bug: with PP, only the last stage
    replies to execute_model, so a CUDA OOM on PP rank 0 was logged in the
    worker and swallowed. The engine limped on desynchronized pipeline
    steps and the client saw nothing. The failure must now propagate to the
    in-flight request promptly."""
    # Needed to send the fault-injector callable over collective_rpc.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    engine_args = AsyncEngineArgs(
        model=MODEL,
        pipeline_parallel_size=2,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.3,
        max_model_len=2048,
        enforce_eager=True,
        disable_log_stats=True,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    try:

        async def t():
            installed = await engine.collective_rpc(_install_pp_first_rank_oom)
            assert installed == [True, False]

            async def consume():
                stream = engine.generate(
                    request_id="oom-req",
                    prompt="foo",
                    sampling_params=SamplingParams(max_tokens=16),
                )
                async for _ in stream:
                    ...

            with pytest.raises((EngineDeadError, EngineGenerateError)):
                # The request must error out promptly (worker death is
                # detected by the executor monitor), not hang or limp.
                await asyncio.wait_for(consume(), timeout=60)

        asyncio.run(t())
    finally:
        engine.shutdown()
