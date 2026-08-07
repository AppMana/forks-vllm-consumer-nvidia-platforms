# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire compatibility for the deprecated ``reasoning_content`` output field.

#33402 removed ``reasoning_content`` from responses in favour of
``reasoning``. Off-the-shelf OpenAI-compatible clients still read the old
name (``@ai-sdk/openai-compatible`` and openwebui both key on
``reasoning_content``), so a reasoning-capable server that emits only
``reasoning`` has its reasoning silently dropped by the client.

These assert the serialized payload, not the model attributes, because the
mismatch is on the wire.
"""

import json

from vllm.entrypoints.openai.chat_completion.protocol import ChatMessage
from vllm.entrypoints.openai.engine.protocol import DeltaMessage

REASONING = "17*20 = 340, 17*3 = 51, sum = 391."
CONTENT = "The product of 17 and 23 is 391."


def test_chat_message_mirrors_reasoning_to_reasoning_content():
    data = json.loads(
        ChatMessage(
            role="assistant", content=CONTENT, reasoning=REASONING
        ).model_dump_json()
    )
    assert data["reasoning"] == REASONING
    assert data["reasoning_content"] == REASONING
    assert data["content"] == CONTENT


def test_delta_message_mirrors_reasoning_to_reasoning_content():
    data = json.loads(DeltaMessage(reasoning=REASONING).model_dump_json())
    assert data["reasoning"] == REASONING
    assert data["reasoning_content"] == REASONING


def test_absent_reasoning_adds_no_compat_key():
    """Non-reasoning responses stay byte-identical to upstream."""
    for msg in (
        ChatMessage(role="assistant", content=CONTENT),
        DeltaMessage(role="assistant", content=CONTENT),
    ):
        data = json.loads(msg.model_dump_json())
        assert "reasoning_content" not in data
        assert data["reasoning"] is None


def test_empty_reasoning_is_mirrored_not_dropped():
    """An empty string is a real value; only None suppresses the key."""
    data = json.loads(DeltaMessage(reasoning="").model_dump_json())
    assert data["reasoning_content"] == ""
