# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from typing import Any

from transformers import TokenizersBackend

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam

from .deepseek_v4_encoding import DEFAULT_REASONING_EFFORT, encode_messages
from .hf import HfTokenizer, get_cached_tokenizer
from .protocol import TokenizerLike

# The OpenAI API exposes seven reasoning-effort levels; DeepSeek V4's encoding
# has three ("low"/"high"/"max", where "low" emits no prefix at all). This is
# the order-preserving collapse of one onto the other.
#
# Getting this wrong is silent: before DeepSeek-V4-Flash-0731 the encoding only
# emitted a prefix for "max", so mapping everything non-max onto "high" was a
# harmless no-op. From 0731 on, "high" emits the text that "max" used to, so
# that same mapping would escalate every minimal/low/medium request to a heavy
# "Absolute maximum" reasoning prefix without any error surfacing.
#
# "none" is handled separately by the caller: it selects chat mode outright
# rather than a thinking-mode effort level.
_OPENAI_TO_DSV4_REASONING_EFFORT = {
    "minimal": "low",
    "low": "low",
    "medium": "high",
    "high": "high",
    "xhigh": "max",
    "max": "max",
}


def get_deepseek_v4_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer:
    """
    Wraps a tokenizer to use the custom DeepSeek V4 chat template encoding.
    """
    dsv4_tokenizer = copy.copy(tokenizer)

    added_vocab = tokenizer.get_added_vocab()
    added_vocab_size = len(added_vocab)
    tokenizer_vocab_size = tokenizer.vocab_size

    class _DeepseekV4Tokenizer(tokenizer.__class__):  # type: ignore
        def apply_chat_template(
            self,
            messages: list["ChatCompletionMessageParam"],
            tools: list[dict[str, Any]] | None = None,
            **kwargs,
        ) -> str | list[int]:
            thinking = kwargs.get("thinking", False)
            enable_thinking = kwargs.get("enable_thinking", False)
            thinking = thinking or enable_thinking
            thinking_mode = "thinking" if thinking else "chat"

            conversation = kwargs.get("conversation", messages)
            messages = conversation.copy()
            if tools is not None and len(tools) > 0:
                if messages and messages[0].get("role") == "system":
                    messages[0] = dict(messages[0])
                else:
                    messages.insert(0, {"role": "system"})
                messages[0]["tools"] = tools  # type: ignore[typeddict-unknown-key]

            reasoning_effort = kwargs.get("reasoning_effort")
            if not isinstance(reasoning_effort, str):
                reasoning_effort = None
            elif reasoning_effort == "none":
                thinking_mode = "chat"
                reasoning_effort = None
            else:
                reasoning_effort = _OPENAI_TO_DSV4_REASONING_EFFORT.get(
                    reasoning_effort, DEFAULT_REASONING_EFFORT
                )

            encode_config = dict(
                thinking_mode=thinking_mode,
                drop_thinking=kwargs.get("drop_thinking", True),
                reasoning_effort=reasoning_effort,
            )

            prompt_str = encode_messages(messages, **encode_config)  # type: ignore

            if kwargs.get("tokenize", True):
                tokenizer_kwargs = {
                    k: kwargs[k] for k in ("truncation", "max_length") if k in kwargs
                }
                return self.encode(
                    prompt_str,
                    add_special_tokens=False,
                    **tokenizer_kwargs,
                )

            return prompt_str

        def num_special_tokens_to_add(self) -> int:
            return len(self.encode(""))

        def __len__(self) -> int:
            return tokenizer_vocab_size + added_vocab_size

        def get_added_vocab(self) -> dict[str, int]:
            return added_vocab.copy()

        def __reduce__(self):
            return get_deepseek_v4_tokenizer, (tokenizer,)

    _DeepseekV4Tokenizer.__name__ = f"DSV4{tokenizer.__class__.__name__}"

    dsv4_tokenizer.__class__ = _DeepseekV4Tokenizer
    return dsv4_tokenizer


class DeepseekV4Tokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> HfTokenizer:
        tokenizer = TokenizersBackend.from_pretrained(*args, **kwargs)
        return get_cached_tokenizer(get_deepseek_v4_tokenizer(tokenizer))
