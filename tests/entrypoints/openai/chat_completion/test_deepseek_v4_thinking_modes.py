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
from vllm.tokenizers.deepseek_v4_encoding import (
    DEFAULT_REASONING_EFFORT,
    REASONING_EFFORT_PROMPTS,
    bos_token,
)

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


def test_none_effort_matches_default_chat_prompt():
    assert _rendered_prompt(
        _chat_template_kwargs(reasoning_effort="none")
    ) == _rendered_prompt(_chat_template_kwargs())


# ---------------------------------------------------------------------------
# Reasoning effort (DeepSeek-V4-Flash-0731 three-level contract)
# ---------------------------------------------------------------------------
#
# 0731 replaced a single "max"-only preamble with three levels. The mechanism
# is *entirely* a text prefix at position 0 of the prompt, so every failure
# mode here is silent: a wrong mapping serves a different model persona with no
# error, no warning, and a perfectly well-formed response.
#
# Two properties need pinning, and they fail independently:
#   1. the prefix TEXT for each DeepSeek level (drift = wrong instructions)
#   2. the OpenAI-level -> DeepSeek-level MAPPING (drift = right text, wrong
#      request)
# The old contract made these indistinguishable, because "high" mapped to a
# level that emitted nothing.


def _prefix_of(prompt: str, baseline: str) -> str:
    """The reasoning-effort preamble a prompt carries, relative to baseline.

    The preamble is prepended by ``render_message`` at message index 0, which
    lands it *after* the conversation's BOS token rather than at the very front
    of the string. Everything from there on must be byte-identical to the
    baseline: a level that changed anything else has leaked out of its preamble
    and into the body of the encoding.
    """
    assert baseline.startswith(bos_token)
    assert prompt.startswith(bos_token)
    body = baseline[len(bos_token):]
    prompt = prompt[len(bos_token):]
    assert prompt.endswith(body), (
        "a reasoning-effort prompt must be the baseline thinking prompt with a "
        "preamble inserted after BOS; anything else means the level leaked "
        "into the body of the encoding"
    )
    return prompt[: len(prompt) - len(body)]


def _thinking_baseline() -> str:
    """The thinking-mode prompt with no effort preamble at all."""
    return _rendered_prompt(_chat_template_kwargs(chat_template_kwargs={"thinking": True}))


def test_reasoning_effort_prompts_match_the_released_encoding():
    """The three preambles are verbatim from 0731's encoding_dsv4.py.

    Pinned as literals rather than compared to the constant they come from:
    a test that reads REASONING_EFFORT_PROMPTS would pass no matter how the
    text was reworded. The prefix IS the feature.
    """
    assert REASONING_EFFORT_PROMPTS["low"] == ""
    assert REASONING_EFFORT_PROMPTS["high"] == (
        "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
        "You MUST be very thorough in your thinking and comprehensively "
        "decompose the problem to resolve the root cause, rigorously "
        "stress-testing your logic against all potential paths, edge cases, "
        "and adversarial scenarios.\n"
        "Explicitly write out your entire deliberation process, documenting "
        "every intermediate step, considered alternative, and rejected "
        "hypothesis to ensure absolutely no assumption is left unchecked.\n\n"
    )
    assert REASONING_EFFORT_PROMPTS["max"] == (
        "Reasoning Effort: Beyond maximum — exhaustive, relentless, and "
        "uncompromising.\n"
        "You MUST reason with the utmost depth and rigor, leaving absolutely "
        "nothing to chance: exhaustively decompose the problem into its most "
        "fundamental components, trace every causal chain to its root, and "
        "resolve the underlying cause rather than any surface symptom.\n"
        "Do not stop reasoning until you have independently verified the "
        "solution from multiple angles and are certain that no assumption "
        "remains unchecked and no error remains undiscovered.\n\n"
    )
    # The three levels must be mutually distinguishable, or the mapping tests
    # below would pass vacuously.
    assert len(set(REASONING_EFFORT_PROMPTS.values())) == 3


@pytest.mark.parametrize(
    ("openai_effort", "dsv4_level"),
    [
        # The OpenAI API's seven levels collapse onto DeepSeek's three in
        # order. This is the whole mapping; it is not derived from anything,
        # so it is enumerated exhaustively.
        ("minimal", "low"),
        ("low", "low"),
        ("medium", "high"),
        ("high", "high"),
        ("xhigh", "max"),
        ("max", "max"),
    ],
)
def test_openai_effort_maps_to_the_right_dsv4_preamble(
    openai_effort: str, dsv4_level: str
):
    baseline = _thinking_baseline()
    prompt = _rendered_prompt(_chat_template_kwargs(reasoning_effort=openai_effort))

    assert _prefix_of(prompt, baseline) == REASONING_EFFORT_PROMPTS[dsv4_level]


def test_effort_levels_are_ordered_and_distinct():
    """low < high < max must stay three observably different prompts.

    Before 0731 "high" emitted nothing, so low and high were identical and a
    mapping bug between them was undetectable. Guard the separation itself.
    """
    baseline = _thinking_baseline()
    rendered = {
        level: _rendered_prompt(_chat_template_kwargs(reasoning_effort=effort))
        for effort, level in (("low", "low"), ("high", "high"), ("max", "max"))
    }

    # "low" is the default: it adds nothing on top of a bare thinking prompt.
    assert rendered["low"] == baseline
    assert rendered["high"] != rendered["low"]
    assert rendered["max"] != rendered["high"]
    assert len(set(rendered.values())) == 3
    # Each heavier level is strictly longer -- the preamble only ever grows.
    assert len(rendered["low"]) < len(rendered["high"]) < len(rendered["max"])


def test_effort_has_no_effect_in_chat_mode():
    """0731's encoding README: reasoning_effort is inert outside thinking mode.

    A leak here would inject "Reasoning Effort: ..." into a non-thinking
    request, which the model has no reasoning block to satisfy.
    """
    chat_baseline = _rendered_prompt(_chat_template_kwargs())

    for effort in ("minimal", "low", "medium", "high", "xhigh", "max"):
        prompt = _rendered_prompt(
            _chat_template_kwargs(
                reasoning_effort=effort,
                chat_template_kwargs={"enable_thinking": False},
            )
        )
        assert prompt == chat_baseline, f"{effort} leaked into chat mode"
        assert "Reasoning Effort" not in prompt


def test_effort_preamble_appears_once_at_the_front_of_a_conversation():
    """The preamble is a conversation-level prefix, not a per-message one.

    ``render_message`` gates it on ``index == 0``; a regression that dropped
    the gate would repeat the preamble before every turn, which still produces
    a valid-looking prompt.
    """
    conversation = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]
    request = ChatCompletionRequest(
        model="m", messages=conversation, reasoning_effort="max"
    )
    kwargs = request.build_chat_params(None, "auto").chat_template_kwargs
    tokenizer = get_deepseek_v4_tokenizer(_FakeHfTokenizer())
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False, **kwargs)

    assert isinstance(prompt, str)
    assert prompt.count("Reasoning Effort:") == 1
    assert prompt.startswith(bos_token + REASONING_EFFORT_PROMPTS["max"])


def test_unknown_effort_falls_back_to_the_default_level():
    """An unrecognized level must degrade to "low", not raise or escalate.

    ``chat_template_kwargs`` is free-form, so a string the OpenAI protocol
    Literal never validates can reach the tokenizer. The encoding asserts on
    unknown levels, so the mapping layer has to absorb them.
    """
    baseline = _thinking_baseline()
    prompt = _rendered_prompt(
        {"thinking": True, "reasoning_effort": "ludicrous"}
    )

    assert prompt == baseline
    assert DEFAULT_REASONING_EFFORT == "low"
