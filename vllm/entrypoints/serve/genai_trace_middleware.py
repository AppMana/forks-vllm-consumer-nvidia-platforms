# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GenAI inference spans for the OpenAI-compatible server.

An ASGI middleware for ``vllm serve --middleware
vllm.entrypoints.serve.genai_trace_middleware.GenAIContentTraceMiddleware``.
Every ``/v1/chat/completions`` request becomes an OpenTelemetry GenAI
inference span produced by ``opentelemetry-util-genai``: the request's
sampling parameters, the response's finish reasons, ids and token usage
(prompt, completion, prefix-cache hits), and the prompt and completion
themselves, handled exactly as that package is configured to handle them:

* ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` inlines the
  messages into the span (``gen_ai.input.messages`` /
  ``gen_ai.output.messages``); the default keeps them out.
* ``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK=upload`` with
  ``OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH=<fsspec url>`` stores them
  as objects and leaves only ``gen_ai.input.messages_ref`` /
  ``gen_ai.output.messages_ref`` on the span.

The engine's own ``llm_request`` span (``--otlp-traces-endpoint``) joins the
same trace: the middleware injects the span's ``traceparent`` into the
request headers the chat endpoint already reads trace context from. Clients
need no instrumentation and never learn where content is stored. Spans go
to whatever tracer provider is installed; with vLLM tracing enabled that is
the engine's provider and OTLP endpoint.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

TRACED_PATHS = frozenset({"/v1/chat/completions"})


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return "" if content is None else str(content)


class _ChoiceState:
    __slots__ = ("text", "reasoning", "finish_reason")

    def __init__(self) -> None:
        self.text = ""
        self.reasoning = ""
        self.finish_reason: str | None = None


class _ResponseCollector:
    """Folds a chat completion response, streamed or not, into choices."""

    def __init__(self) -> None:
        self.status: int | None = None
        self.content_type = ""
        self.choices: dict[int, _ChoiceState] = {}
        self.usage: dict[str, Any] | None = None
        self.response_id: str | None = None
        self.response_model: str | None = None
        self._buffer = b""
        self._error: str | None = None

    def start(self, message: Message) -> None:
        self.status = int(message.get("status", 0))
        for name, value in message.get("headers", []):
            if name.lower() == b"content-type":
                self.content_type = value.decode("latin-1")

    def feed(self, body: bytes) -> None:
        if not body:
            return
        if self.content_type.startswith("text/event-stream"):
            self._buffer += body
            *lines, self._buffer = self._buffer.split(b"\n")
            for line in lines:
                self._sse_line(line)
        else:
            self._buffer += body

    def _sse_line(self, line: bytes) -> None:
        line = line.strip()
        if not line.startswith(b"data:"):
            return
        data = line[len(b"data:") :].strip()
        if data == b"[DONE]":
            return
        try:
            self._chunk(json.loads(data))
        except ValueError:
            return

    def _chunk(self, chunk: dict[str, Any]) -> None:
        if not isinstance(chunk, dict):
            return
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        self.response_id = chunk.get("id") or self.response_id
        self.response_model = chunk.get("model") or self.response_model
        for choice in chunk.get("choices") or []:
            state = self.choices.setdefault(int(choice.get("index", 0)), _ChoiceState())
            delta = choice.get("delta") or choice.get("message") or {}
            state.text += _text_of(delta.get("content"))
            state.reasoning += _text_of(
                delta.get("reasoning") or delta.get("reasoning_content")
            )
            if choice.get("finish_reason"):
                state.finish_reason = choice["finish_reason"]

    def finish(self) -> None:
        if self.content_type.startswith("text/event-stream"):
            if self._buffer:
                self._sse_line(self._buffer)
            return
        if not self._buffer:
            return
        try:
            body = json.loads(self._buffer)
        except ValueError:
            return
        if self.status is not None and self.status >= 400:
            error = body.get("error") if isinstance(body, dict) else None
            self._error = (
                error.get("message") if isinstance(error, dict) else str(body)
            )
            return
        self._chunk(body)

    @property
    def error(self) -> str | None:
        if self._error:
            return self._error
        if self.status is not None and self.status >= 400:
            return f"HTTP {self.status}"
        return None


class GenAIContentTraceMiddleware:
    """Pure ASGI middleware; see the module docstring."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._handler: Any = None

    def _telemetry_handler(self) -> Any:
        if self._handler is None:
            from opentelemetry.util.genai.completion_hook import load_completion_hook
            from opentelemetry.util.genai.handler import get_telemetry_handler

            self._handler = get_telemetry_handler(completion_hook=load_completion_hook())
        return self._handler

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") not in TRACED_PATHS
        ):
            await self.app(scope, receive, send)
            return

        received: list[Message] = []
        body = b""
        while True:
            message = await receive()
            received.append(message)
            if message["type"] != "http.request":
                break
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break

        async def replay() -> Message:
            if received:
                return received.pop(0)
            return await receive()

        try:
            request = json.loads(body)
        except ValueError:
            request = None
        if not isinstance(request, dict):
            await self.app(scope, replay, send)
            return

        invocation = self._start(request)
        if invocation is not None:
            self._inject_trace_context(scope, invocation)

        collector = _ResponseCollector()

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                collector.start(message)
            elif message["type"] == "http.response.body":
                collector.feed(message.get("body", b""))
            await send(message)

        try:
            await self.app(scope, replay, send_wrapper)
        except BaseException as exc:
            if invocation is not None:
                invocation.fail(exc)
            raise
        if invocation is not None:
            self._finish(invocation, collector)

    def _start(self, request: dict[str, Any]) -> Any:
        try:
            from opentelemetry.util.genai.types import InputMessage, Text

            handler = self._telemetry_handler()
        except Exception:
            logger.warning("GenAI trace middleware disabled", exc_info=True)
            return None
        invocation = handler.start_inference(
            "vllm", request_model=request.get("model"), operation_name="chat"
        )
        input_messages: list[Any] = []
        system_instruction: list[Any] = []
        for message in request.get("messages") or []:
            if not isinstance(message, dict):
                continue
            part = Text(content=_text_of(message.get("content")))
            if message.get("role") == "system":
                system_instruction.append(part)
            else:
                input_messages.append(
                    InputMessage(role=str(message.get("role", "user")), parts=[part])
                )
        invocation.input_messages = input_messages
        invocation.system_instruction = system_instruction
        invocation.temperature = request.get("temperature")
        invocation.top_p = request.get("top_p")
        invocation.frequency_penalty = request.get("frequency_penalty")
        invocation.presence_penalty = request.get("presence_penalty")
        invocation.seed = request.get("seed")
        invocation.max_tokens = request.get("max_completion_tokens") or request.get(
            "max_tokens"
        )
        stop = request.get("stop")
        if isinstance(stop, str):
            stop = [stop]
        invocation.stop_sequences = stop
        return invocation

    @staticmethod
    def _inject_trace_context(scope: Scope, invocation: Any) -> None:
        from opentelemetry.propagate import inject
        from opentelemetry.trace import set_span_in_context

        carrier: dict[str, str] = {}
        inject(carrier, context=set_span_in_context(invocation.span))
        if not carrier:
            return
        injected = {name.lower().encode("latin-1") for name in carrier}
        headers = [
            (name, value)
            for name, value in scope.get("headers", [])
            if name.lower() not in injected
        ]
        headers.extend(
            (name.lower().encode("latin-1"), value.encode("latin-1"))
            for name, value in carrier.items()
        )
        scope["headers"] = headers

    @staticmethod
    def _finish(invocation: Any, collector: _ResponseCollector) -> None:
        from opentelemetry.util.genai.types import (
            Error,
            OutputMessage,
            Reasoning,
            Text,
        )

        collector.finish()
        outputs = []
        for index in sorted(collector.choices):
            state = collector.choices[index]
            parts: list[Any] = []
            if state.reasoning:
                parts.append(Reasoning(content=state.reasoning))
            parts.append(Text(content=state.text))
            outputs.append(
                OutputMessage(
                    role="assistant",
                    parts=parts,
                    finish_reason=state.finish_reason or "",
                )
            )
        invocation.output_messages = outputs
        invocation.response_id = collector.response_id
        invocation.response_model_name = collector.response_model
        usage = collector.usage or {}
        invocation.input_tokens = usage.get("prompt_tokens")
        invocation.output_tokens = usage.get("completion_tokens")
        details = usage.get("prompt_tokens_details") or {}
        if isinstance(details, dict) and details.get("cached_tokens") is not None:
            invocation.cache_read_input_tokens = details["cached_tokens"]
        error = collector.error
        if error:
            invocation.fail(Error(message=error, type=RuntimeError))
        else:
            invocation.stop()
