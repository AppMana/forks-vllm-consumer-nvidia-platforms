# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine stall detection.

A dead worker (e.g. Ray worker SIGABRT after a NCCL watchdog timeout) can
leave the engine core blocked forever inside a step while /health keeps
returning 200. The frontend must flip /health to 503 when requests are
outstanding but the engine core has stopped producing step heartbeats.
"""

from unittest.mock import Mock

import pytest

from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.exceptions import EngineStalledError
from vllm.v1.engine.health import EngineStepMonitor


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def advance(self, seconds: float):
        self.now += seconds

    def __call__(self) -> float:
        return self.now


def test_monitor_not_stalled_without_requests():
    clock = FakeClock()
    monitor = EngineStepMonitor(stall_timeout_s=60, clock=clock)
    clock.advance(3600)
    assert not monitor.stalled(has_requests=False)


def test_monitor_stalls_after_timeout_with_requests():
    clock = FakeClock()
    monitor = EngineStepMonitor(stall_timeout_s=60, clock=clock)
    clock.advance(59)
    assert not monitor.stalled(has_requests=True)
    clock.advance(2)
    assert monitor.stalled(has_requests=True)
    assert monitor.seconds_since_activity() == pytest.approx(61)


def test_monitor_activity_resets_stall():
    clock = FakeClock()
    monitor = EngineStepMonitor(stall_timeout_s=60, clock=clock)
    clock.advance(61)
    assert monitor.stalled(has_requests=True)
    monitor.note_activity()
    assert not monitor.stalled(has_requests=True)


def test_monitor_disabled_with_nonpositive_timeout():
    clock = FakeClock()
    monitor = EngineStepMonitor(stall_timeout_s=0, clock=clock)
    clock.advance(3600)
    assert not monitor.stalled(has_requests=True)


def _make_zombie_async_llm(clock: FakeClock) -> AsyncLLM:
    """AsyncLLM whose engine-core step loop has stopped: one request is
    outstanding forever and no step heartbeats arrive."""
    llm = AsyncLLM.__new__(AsyncLLM)
    llm.step_monitor = EngineStepMonitor(stall_timeout_s=60, clock=clock)
    output_processor = Mock()
    output_processor.has_unfinished_requests.return_value = True
    output_processor.get_num_unfinished_requests.return_value = 1
    llm.output_processor = output_processor
    engine_core = Mock()
    engine_core.resources.engine_dead = False
    llm.engine_core = engine_core
    output_handler = Mock()
    output_handler.done.return_value = False
    llm.output_handler = output_handler
    return llm


@pytest.mark.asyncio
async def test_check_health_raises_when_step_loop_stops():
    clock = FakeClock()
    llm = _make_zombie_async_llm(clock)

    # Heartbeats fresh: healthy.
    await llm.check_health()

    # Engine core stops stepping while the request is still outstanding.
    clock.advance(61)
    with pytest.raises(EngineStalledError):
        await llm.check_health()

    # A heartbeat (engine step output) recovers health.
    llm.step_monitor.note_activity()
    await llm.check_health()


@pytest.mark.asyncio
async def test_check_health_ok_when_idle():
    clock = FakeClock()
    llm = _make_zombie_async_llm(clock)
    llm.output_processor.has_unfinished_requests.return_value = False
    clock.advance(3600)
    await llm.check_health()
