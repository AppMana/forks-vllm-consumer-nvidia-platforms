# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frontend-side engine liveness tracking.

The engine core sends the frontend an ``EngineCoreOutputs`` message on every
completed step (stats-only messages included, so chunked prefill steps count
even though they produce no request outputs). Those messages act as step
heartbeats: when requests are outstanding but no heartbeat arrives within the
stall timeout, the engine is considered stalled and health checks must fail.
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class _StartupProgress:
    stage: str = "starting"
    detail: str = ""
    started_at: float = field(default_factory=time.monotonic)


_startup_progress = _StartupProgress()
_startup_progress_lock = threading.Lock()


def reset_startup_progress() -> None:
    """Restart startup progress tracking (also restarts the elapsed clock)."""
    global _startup_progress
    with _startup_progress_lock:
        _startup_progress = _StartupProgress()


def record_startup_stage(stage: str, detail: str = "") -> None:
    """Record the current server startup stage for this process.

    Read by the startup probe HTTP server to answer /health with init
    progress while the engine initializes. Thread-safe; last writer wins.
    """
    with _startup_progress_lock:
        _startup_progress.stage = stage
        _startup_progress.detail = detail


def startup_progress_snapshot() -> dict[str, str | float]:
    with _startup_progress_lock:
        return {
            "stage": _startup_progress.stage,
            "detail": _startup_progress.detail,
            "elapsed_s": round(time.monotonic() - _startup_progress.started_at, 1),
        }


class EngineStepMonitor:
    """Tracks engine-core step heartbeats observed by the frontend.

    ``note_activity`` is called whenever the frontend receives an engine-core
    output message (and when the engine transitions from idle to busy, so an
    idle period never counts against the first step). ``stalled`` reports
    whether the heartbeat is overdue while requests are outstanding.

    A non-positive ``stall_timeout_s`` disables detection.
    """

    def __init__(
        self,
        stall_timeout_s: float,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.stall_timeout_s = stall_timeout_s
        self._clock = clock
        self._last_activity = clock()

    @property
    def enabled(self) -> bool:
        return self.stall_timeout_s > 0

    def note_activity(self) -> None:
        self._last_activity = self._clock()

    def seconds_since_activity(self) -> float:
        return self._clock() - self._last_activity

    def stalled(self, has_requests: bool) -> bool:
        return (
            self.enabled
            and has_requests
            and self.seconds_since_activity() > self.stall_timeout_s
        )
