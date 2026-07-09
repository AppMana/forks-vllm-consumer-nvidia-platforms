# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio

import pytest

from vllm.benchmarks.lib.endpoint_request_func import (
    RequestFuncInput,
    async_request_openai_completions,
)


class _HangingSSEContent:
    def __init__(self) -> None:
        self._chunks = iter(
            [
                b'data: {"choices":[{"text":"hello"}]}\n\n',
                b'data: {"usage":{"completion_tokens":1,"prompt_tokens":2}}\n\n',
                b"data: [DONE]\n\n",
            ]
        )

    def iter_any(self):
        return self

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            await asyncio.sleep(3600)
            raise StopAsyncIteration


class _FakeResponse:
    status = 200
    reason = "OK"

    def __init__(self) -> None:
        self.content = _HangingSSEContent()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeSession:
    def post(self, **kwargs):
        return _FakeResponse()


@pytest.mark.asyncio
async def test_openai_completions_returns_on_done_for_keepalive_stream():
    output = await asyncio.wait_for(
        async_request_openai_completions(
            RequestFuncInput(
                prompt="hello",
                api_url="http://localhost:8000/v1/completions",
                prompt_len=1,
                output_len=1,
                model="test-model",
            ),
            _FakeSession(),
        ),
        timeout=1.0,
    )

    assert output.success
    assert output.generated_text == "hello"
    assert output.output_tokens == 1
    assert output.prompt_len == 2
