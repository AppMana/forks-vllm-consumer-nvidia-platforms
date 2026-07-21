# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
class EngineGenerateError(Exception):
    """Raised when a AsyncLLM.generate() fails. Recoverable."""

    pass


class EngineStalledError(Exception):
    """Raised by health checks when requests are outstanding but the engine
    core has stopped producing step heartbeats (e.g. a worker died mid-step
    and the engine is blocked forever waiting on it)."""

    def __init__(self, stalled_for_s: float, num_requests: int):
        self.stalled_for_s = stalled_for_s
        self.num_requests = num_requests
        super().__init__(
            f"Engine core has produced no step heartbeat for "
            f"{stalled_for_s:.1f}s with {num_requests} request(s) outstanding."
        )


class EngineDeadError(Exception):
    """Raised when the EngineCore dies. Unrecoverable."""

    def __init__(self, *args, suppress_context: bool = False, **kwargs):
        ENGINE_DEAD_MESSAGE = "EngineCore encountered an issue. See stack trace (above) for the root cause."  # noqa: E501

        super().__init__(ENGINE_DEAD_MESSAGE, *args, **kwargs)
        # Make stack trace clearer when using with LLMEngine by
        # silencing irrelevant ZMQError.
        self.__suppress_context__ = suppress_context
