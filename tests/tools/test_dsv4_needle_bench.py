from tools.ampere.dsv4_needle_bench import build_prompt


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
