# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GenAI content trace middleware against a fake chat endpoint.

The fake app streams the user's last message back (or returns it as one
JSON response), reads the ``traceparent`` the middleware injected, and the
test checks the resulting span: request and response attributes, no inlined
content, and ``*_ref`` attributes that resolve to the uploaded objects.
"""

import json
import time
from pathlib import Path

import httpx
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes

pytest.importorskip("opentelemetry.util.genai._upload")

from vllm.entrypoints.serve.genai_trace_middleware import (  # noqa: E402
    GenAIContentTraceMiddleware,
)

_PROVIDER = TracerProvider()
_EXPORTER = InMemorySpanExporter()
_PROVIDER.add_span_processor(SimpleSpanProcessor(_EXPORTER))
trace.set_tracer_provider(_PROVIDER)


async def _read_body(receive):
    body = b""
    while True:
        message = await receive()
        body += message.get("body", b"")
        if not message.get("more_body"):
            return body


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


async def fake_chat_app(scope, receive, send):
    seen_headers = dict(scope["headers"])
    request = json.loads(await _read_body(receive))
    text = request["messages"][-1]["content"]
    words = text.split()
    traceparent = seen_headers.get(b"traceparent", b"").decode()
    if request.get("stream"):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        # Split a chunk across two body messages to exercise line buffering.
        chunks = []
        for i, word in enumerate(words):
            chunks.append(
                _sse(
                    {
                        "id": "chatcmpl-1",
                        "model": "served",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": (" " if i else "") + word},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            )
        chunks.append(
            _sse(
                {
                    "id": "chatcmpl-1",
                    "model": "served",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            )
        )
        chunks.append(
            _sse(
                {
                    "id": "chatcmpl-1",
                    "model": "served",
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": len(words),
                        "prompt_tokens_details": {"cached_tokens": 3},
                    },
                }
            )
        )
        chunks.append(b"data: [DONE]\n\n")
        body = b"".join(chunks)
        cut = len(body) // 2
        await send({"type": "http.response.body", "body": body[:cut], "more_body": True})
        await send({"type": "http.response.body", "body": body[cut:], "more_body": False})
        return

    payload = {
        "id": "chatcmpl-2",
        "model": "served",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text, "reasoning": "hmm"},
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": len(words)},
        "traceparent_seen": traceparent,
    }
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": json.dumps(payload).encode()})


@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK", "upload")
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH", str(tmp_path))
    from opentelemetry.util.genai import handler as genai_handler

    monkeypatch.delattr(
        genai_handler.get_telemetry_handler, "_default_handler", raising=False
    )
    _EXPORTER.clear()
    return tmp_path


def _stored(ref: str) -> Path:
    assert ref.startswith("file:"), ref
    path = Path(ref[len("file:"):])
    # The hook uploads from a worker thread after the span ends.
    deadline = time.monotonic() + 10
    while not (path.is_file() and path.stat().st_size > 0):
        assert time.monotonic() < deadline, f"{ref} was not uploaded"
        time.sleep(0.05)
    return path


def _chat_span():
    spans = [s for s in _EXPORTER.get_finished_spans() if s.name.startswith("chat")]
    assert len(spans) == 1, [s.name for s in _EXPORTER.get_finished_spans()]
    return spans[0]


@pytest.mark.asyncio
async def test_streaming_chat_span_references_uploaded_messages(upload_dir):
    app = GenAIContentTraceMiddleware(fake_chat_app)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "served",
                "messages": [
                    {"role": "system", "content": "be brief"},
                    {"role": "user", "content": "alpha beta gamma delta"},
                ],
                "stream": True,
                "temperature": 0,
                "max_tokens": 32,
            },
        )
    assert response.status_code == 200
    assert response.text.endswith("data: [DONE]\n\n")

    span = _chat_span()
    attrs = dict(span.attributes)
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.request.model"] == "served"
    assert attrs["gen_ai.request.temperature"] == 0
    assert attrs["gen_ai.request.max_tokens"] == 32
    assert attrs["gen_ai.response.id"] == "chatcmpl-1"
    assert attrs["gen_ai.usage.input_tokens"] == 7
    assert attrs["gen_ai.usage.output_tokens"] == 4
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 3
    assert list(attrs["gen_ai.response.finish_reasons"]) == ["stop"]
    assert "gen_ai.input.messages" not in attrs
    assert "gen_ai.output.messages" not in attrs

    inputs = json.load(open(_stored(attrs["gen_ai.input.messages_ref"])))
    assert inputs == [
        {"role": "user", "parts": [{"content": "alpha beta gamma delta", "type": "text"}]}
    ]
    system = json.load(
        open(_stored(attrs[gen_ai_attributes.GEN_AI_SYSTEM_INSTRUCTIONS + "_ref"]))
    )
    assert system == [{"content": "be brief", "type": "text"}]
    outputs = json.load(open(_stored(attrs["gen_ai.output.messages_ref"])))
    assert outputs == [
        {
            "role": "assistant",
            "parts": [{"content": "alpha beta gamma delta", "type": "text"}],
            "finish_reason": "stop",
        }
    ]


@pytest.mark.asyncio
async def test_json_chat_response_joins_trace_and_records_reasoning(upload_dir):
    app = GenAIContentTraceMiddleware(fake_chat_app)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "served", "messages": [{"role": "user", "content": "one two"}]},
        )
    assert response.status_code == 200
    span = _chat_span()
    attrs = dict(span.attributes)

    # The engine-facing request carried this span's context.
    traceparent = response.json()["traceparent_seen"]
    assert format(span.context.trace_id, "032x") in traceparent
    assert format(span.context.span_id, "016x") in traceparent

    assert list(attrs["gen_ai.response.finish_reasons"]) == ["length"]
    outputs = json.load(open(_stored(attrs["gen_ai.output.messages_ref"])))
    assert outputs[0]["parts"] == [
        {"content": "hmm", "type": "reasoning"},
        {"content": "one two", "type": "text"},
    ]


@pytest.mark.asyncio
async def test_other_paths_pass_through_untraced(upload_dir):
    async def health(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    app = GenAIContentTraceMiddleware(health)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")
    assert response.text == "ok"
    assert _EXPORTER.get_finished_spans() == ()
