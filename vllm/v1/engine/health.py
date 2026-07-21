# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frontend-side engine liveness tracking.

The engine core sends the frontend an ``EngineCoreOutputs`` message on every
completed step (stats-only messages included, so chunked prefill steps count
even though they produce no request outputs). Those messages act as step
heartbeats: when requests are outstanding but no heartbeat arrives within the
stall timeout, the engine is considered stalled and health checks must fail.
"""

import time
from collections.abc import Callable


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
