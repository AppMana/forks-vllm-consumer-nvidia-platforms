# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A real ``vllm serve`` with the GenAI trace middleware and engine tracing.

One streamed chat request must produce, at the OTLP collector, the
middleware's ``chat`` span and the engine's ``llm_request`` span in the same
trace, with the prompt and completion stored through the upload hook and
referenced from the ``chat`` span rather than embedded in it.
"""

import json
import time
from pathlib import Path

import pytest
from opentelemetry.sdk.environment_variables import OTEL_EXPORTER_OTLP_TRACES_INSECURE

from tests.tracing.conftest import (  # noqa: F401
    FAKE_TRACE_SERVER_ADDRESS,
    FakeTraceService,
    trace_service,
)
from tests.utils import RemoteOpenAIServer

pytest.importorskip("opentelemetry.util.genai._upload")

MODEL = "facebook/opt-125m"
MIDDLEWARE = "vllm.entrypoints.serve.genai_trace_middleware.GenAIContentTraceMiddleware"
# opt-125m ships no chat template.
CHAT_TEMPLATE = str(
    Path(__file__).resolve().parents[3] / "examples" / "template_chatml.jinja"
)


def _spans_named(trace_service: FakeTraceService, name: str, deadline_s: float = 30):
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        spans = [s for s in trace_service.get_all_spans() if s["name"] == name]
        if spans:
            return spans
        time.sleep(0.5)
    return []


def _stored(ref: str) -> Path:
    assert ref.startswith("file:"), ref
    path = Path(ref[len("file:"):])
    deadline = time.monotonic() + 10
    while not (path.is_file() and path.stat().st_size > 0):
        assert time.monotonic() < deadline, f"{ref} was not uploaded"
        time.sleep(0.05)
    return path


def test_chat_and_engine_spans_share_a_trace(
    trace_service: FakeTraceService, tmp_path: Path
):
    env = {
        OTEL_EXPORTER_OTLP_TRACES_INSECURE: "true",
        "OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK": "upload",
        "OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH": str(tmp_path),
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }
    args = [
        "--otlp-traces-endpoint",
        FAKE_TRACE_SERVER_ADDRESS,
        "--middleware",
        MIDDLEWARE,
        "--chat-template",
        CHAT_TEMPLATE,
        "--max-model-len",
        "2048",
        "--gpu-memory-utilization",
        "0.3",
        "--enforce-eager",
    ]
    with RemoteOpenAIServer(MODEL, args, env_dict=env) as server:
        client = server.get_client()
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "The capital of France is"}],
            max_tokens=8,
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
        )
        text = "".join(
            chunk.choices[0].delta.content or ""
            for chunk in stream
            if chunk.choices and chunk.choices[0].delta
        )
        assert text

        chat_spans = _spans_named(trace_service, f"chat {MODEL}")
        engine_spans = _spans_named(trace_service, "llm_request")

    assert len(chat_spans) == 1, [s["name"] for s in trace_service.get_all_spans()]
    assert len(engine_spans) == 1
    chat, engine = chat_spans[0], engine_spans[0]
    assert chat["trace_id"] == engine["trace_id"]

    attrs = chat["attributes"]
    assert attrs["gen_ai.usage.output_tokens"] == 8
    assert attrs["gen_ai.usage.input_tokens"] == engine["attributes"][
        "gen_ai.usage.prompt_tokens"
    ]
    assert "gen_ai.input.messages" not in attrs
    assert "gen_ai.output.messages" not in attrs
    inputs = json.load(open(_stored(attrs["gen_ai.input.messages_ref"])))
    assert inputs[0]["parts"][0]["content"] == "The capital of France is"
    outputs = json.load(open(_stored(attrs["gen_ai.output.messages_ref"])))
    assert outputs[0]["parts"][-1]["content"] == text
