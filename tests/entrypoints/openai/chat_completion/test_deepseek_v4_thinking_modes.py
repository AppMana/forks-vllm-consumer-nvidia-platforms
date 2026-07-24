# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 thinking-mode consistency across the serving path.

The thinking decision is split across three layers, each owning one step:

1. ``ChatCompletionRequest.build_chat_params`` turns a bare
   ``reasoning_effort`` into ``enable_thinking`` (unless the user set it
   explicitly in ``chat_template_kwargs``).
2. ``DeepseekV4Tokenizer.apply_chat_template`` renders the prompt in
   ``thinking`` or ``chat`` mode from those kwargs.
3. ``DeepSeekV4Parser`` picks its initial state (REASONING vs CONTENT)
   from the same kwargs.

These tests pin the invariant that all three layers agree for every way a
request can express a thinking mode. A disagreement is silent at runtime:
the prompt renders ``<think>`` but the parser emits the reasoning as
content, or vice versa.
"""

from typing import Any

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.parser.deepseek_v4 import (
    DSML_THINK_END,
    DSML_THINK_START,
    DSML_TOOL_END,
    DSML_TOOL_START,
    DeepSeekV4Parser,
)
from vllm.parser.engine.parser_engine_config import ParserState
from vllm.tokenizers.deepseek_v4 import get_deepseek_v4_tokenizer

MESSAGES = [{"role": "user", "content": "hi"}]

_PARSER_VOCAB = {
    DSML_THINK_START: 1,
    DSML_THINK_END: 2,
    DSML_TOOL_START: 3,
    DSML_TOOL_END: 4,
}


class _FakeHfTokenizer:
    vocab_size = 100

    def get_added_vocab(self) -> dict[str, int]:
        return {DSML_THINK_END: 100}

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs):
        return [len(text)]


def _chat_template_kwargs(**request_fields: Any) -> dict[str, Any]:
    request = ChatCompletionRequest(
        model="m", messages=MESSAGES, **request_fields
    )
    return request.build_chat_params(None, "auto").chat_template_kwargs


def _rendered_prompt(chat_template_kwargs: dict[str, Any]) -> str:
    tokenizer = get_deepseek_v4_tokenizer(_FakeHfTokenizer())
    prompt = tokenizer.apply_chat_template(
        MESSAGES, tokenize=False, **chat_template_kwargs
    )
    assert isinstance(prompt, str)
    return prompt


def _parser_initial_state(chat_template_kwargs: dict[str, Any]) -> ParserState:
    parser = DeepSeekV4Parser(
        make_mock_tokenizer(_PARSER_VOCAB),
        chat_template_kwargs=chat_template_kwargs,
    )
    return parser.parser_engine_config.initial_state


@pytest.mark.parametrize(
    ("request_fields", "expect_thinking"),
    [
        # reasoning_effort alone must enable thinking: it is the only
        # knob the OpenAI API defines, and the protocol layer maps it to
        # enable_thinking for templates that need the explicit opt-in.
        ({"reasoning_effort": "minimal"}, True),
        ({"reasoning_effort": "low"}, True),
        ({"reasoning_effort": "medium"}, True),
        ({"reasoning_effort": "high"}, True),
        ({"reasoning_effort": "max"}, True),
        ({"reasoning_effort": "xhigh"}, True),
        # "none" is the explicit opt-out.
        ({"reasoning_effort": "none"}, False),
        # No signal at all defaults to chat mode.
        ({}, False),
        # The template-level flags still work on their own.
        ({"chat_template_kwargs": {"thinking": True}}, True),
        ({"chat_template_kwargs": {"enable_thinking": True}}, True),
        # "none" wins over a template-level thinking flag.
        (
            {
                "chat_template_kwargs": {"thinking": True},
                "reasoning_effort": "none",
            },
            False,
        ),
        # An explicit user enable_thinking=False is respected even when
        # an effort level is supplied: the protocol layer only injects
        # enable_thinking when the user did not set it.
        (
            {
                "chat_template_kwargs": {"enable_thinking": False},
                "reasoning_effort": "high",
            },
            False,
        ),
    ],
    ids=lambda v: repr(v),
)
def test_thinking_mode_consistent_across_layers(
    request_fields: dict[str, Any], expect_thinking: bool
):
    kwargs = _chat_template_kwargs(**request_fields)

    prompt = _rendered_prompt(kwargs)
    renders_thinking = prompt.endswith(DSML_THINK_START)
    if not renders_thinking:
        # Chat mode renders a closed think block for this template.
        assert prompt.endswith(DSML_THINK_END)

    parser_state = _parser_initial_state(kwargs)

    assert renders_thinking == expect_thinking
    assert (parser_state == ParserState.REASONING) == expect_thinking


def test_max_effort_changes_the_rendered_prompt():
    """max/xhigh select the max-effort system preamble; every other level
    maps to the plain thinking prompt (the template defines only high and
    max)."""
    high = _rendered_prompt(_chat_template_kwargs(reasoning_effort="high"))
    medium = _rendered_prompt(_chat_template_kwargs(reasoning_effort="medium"))
    max_ = _rendered_prompt(_chat_template_kwargs(reasoning_effort="max"))
    xhigh = _rendered_prompt(_chat_template_kwargs(reasoning_effort="xhigh"))
    bare = _rendered_prompt(
        _chat_template_kwargs(chat_template_kwargs={"thinking": True})
    )

    assert medium == high == bare
    assert xhigh == max_
    assert max_ != high


def test_none_effort_matches_default_chat_prompt():
    assert _rendered_prompt(
        _chat_template_kwargs(reasoning_effort="none")
    ) == _rendered_prompt(_chat_template_kwargs())
