import asyncio
import types

from tools.ampere.dsv4_needle_bench import build_prompt, run_request


class _WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return text.split()

    def decode(self, token_ids):
        return " ".join(token_ids)


def test_run_seed_makes_repeated_benchmark_prompts_unique():
    tokenizer = _WhitespaceTokenizer()

    prompt_a, needle_a = build_prompt(tokenizer, 0, 128, 16, 1, run_seed=1)
    prompt_b, needle_b = build_prompt(tokenizer, 0, 128, 16, 1, run_seed=2)

    assert prompt_a != prompt_b
    assert needle_a != needle_b
    assert prompt_a.startswith("run 00000001, document 000 of 001.")
    assert prompt_b.startswith("run 00000002, document 000 of 001.")


def _chunk(content=None, reasoning=None, finish=None, usage=None):
    delta = types.SimpleNamespace(content=content, reasoning=reasoning)
    choice = types.SimpleNamespace(delta=delta, finish_reason=finish)
    return types.SimpleNamespace(choices=[choice], usage=usage)


class _StubStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for chunk in self._chunks:
                yield chunk

        return gen()


class _StubClient:
    def __init__(self, chunks):
        async def create(**kwargs):
            del kwargs
            return _StubStream(chunks)

        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create)
        )


def _usage(prompt_tokens, completion_tokens):
    return types.SimpleNamespace(
        model_dump=lambda: {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
    )


def test_reasoning_only_response_is_not_reported_as_a_recall_miss():
    """A response whose tokens all went to reasoning is a distinct outcome.

    The DeepSeek V4 needle rows give the model needle_tokens+128 to reproduce
    the needle. A thinking block that opens and never closes inside that
    budget consumes it entirely and the response carries no content at all.
    Counting that as a verbatim miss makes it indistinguishable from the model
    reproducing the wrong text, and leaves ttft/prefill unmeasured so the row
    reports a nonsensical 0 tok/s prefill rate.
    """
    chunks = [
        _chunk(reasoning="Let me look for the needle"),
        _chunk(reasoning=" in this document."),
        _chunk(finish="length", usage=_usage(8000, 1128)),
    ]
    result = asyncio.run(
        run_request(_StubClient(chunks), "m", "prompt", "the needle", 1128)
    )

    assert result["text"] == ""
    assert result["reasoning_only"] is True
    assert result["needle_verbatim"] is False
    # The request is not evidence about recall either way, so it must not be
    # scored as a 0.0 match against the needle.
    assert result["match_ratio"] is None
    # Prefill is still measurable: the first reasoning chunk is a real
    # first token, so the row must not report a 0 tok/s prefill rate.
    assert result["ttft_s"] is None
    assert result["ttft_any_s"] is not None
    assert result["prefill_tok_s"] is not None


def test_content_response_still_scores_recall_normally():
    chunks = [
        _chunk(reasoning="thinking"),
        _chunk(content="the needle"),
        _chunk(finish="stop", usage=_usage(8000, 3)),
    ]
    result = asyncio.run(
        run_request(_StubClient(chunks), "m", "prompt", "the needle", 128)
    )

    assert result["reasoning_only"] is False
    assert result["needle_verbatim"] is True
    assert result["match_ratio"] == 1.0
    assert result["ttft_s"] is not None
