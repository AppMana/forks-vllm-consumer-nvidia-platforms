# SPDX-License-Identifier: Apache-2.0
"""Concurrent needle-in-a-haystack benchmark for a live DSV4 server.

Each request carries a unique ~needle-tokens passage embedded at a
request-specific depth inside a unique haystack, sized with the real model
tokenizer so prompt lengths hit the requested input-token target. The model
is asked to reproduce the passage verbatim, which makes the output length
equal to the needle length and lets correctness be checked by exact
containment.

Every haystack begins with a request-unique salt so prefix caching cannot
serve any request's prefill from another's.

Example:

    python tools/ampere/dsv4_needle_bench.py \
        --base-url http://10.2.0.56:8080 \
        --model appmana/deepseek-v4-int4-int8 \
        --input-tokens 8000 --needle-tokens 1000 --concurrency 64 \
        --output-json results.json
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import random
import statistics
import sys
import time

import aiohttp
from transformers import AutoTokenizer

WORDS = (
    "quartz lattice ember violet copper meadow harbor drift anchor tide "
    "signal ledger canyon summit prairie tundra glacier basalt marble "
    "cobalt saffron indigo crimson umber juniper cedar willow aspen "
    "falcon heron plover osprey kestrel gannet petrel skua tern brant "
    "keel hull mast spar boom tiller rudder winch cleat davit gudgeon"
).split()

INSTRUCTION = (
    "The document above contains exactly one passage delimited by the "
    "markers BEGIN_NEEDLE and END_NEEDLE. Reproduce that passage verbatim, "
    "without the markers and without any commentary before or after it."
)


def _words_text(rng: random.Random, n_words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def _text_of_tokens(tokenizer, rng: random.Random, n_tokens: int) -> str:
    """Deterministic filler text measuring exactly ``n_tokens`` tokens."""
    text = _words_text(rng, n_tokens)  # ~1.2 tokens/word, always enough
    ids = tokenizer.encode(text, add_special_tokens=False)
    while len(ids) < n_tokens:
        text += " " + _words_text(rng, n_tokens - len(ids) + 8)
        ids = tokenizer.encode(text, add_special_tokens=False)
    text = tokenizer.decode(ids[:n_tokens])
    # decode/encode must round-trip for the verbatim check to be meaningful
    assert (
        len(tokenizer.encode(text, add_special_tokens=False)) == n_tokens
    ), "tokenizer round-trip changed the token count"
    return text


def build_prompt(
    tokenizer, index: int, input_tokens: int, needle_tokens: int, total: int
) -> tuple[str, str]:
    """Return (user_message, needle_text) for request ``index``."""
    rng = random.Random(0xD5F4 + index)
    # No label prefix: the seeded word sequence is already unique per
    # request, and a label gets interpreted as metadata and dropped from
    # the "verbatim" reproduction.
    needle = _text_of_tokens(tokenizer, rng, needle_tokens)
    framed = f"\nBEGIN_NEEDLE\n{needle}\nEND_NEEDLE\n"

    overhead = len(
        tokenizer.encode(framed + "\n\n" + INSTRUCTION, add_special_tokens=False)
    )
    salt = f"document {index:03d} of {total:03d}. "
    filler_budget = input_tokens - overhead - len(
        tokenizer.encode(salt, add_special_tokens=False)
    )
    filler = _text_of_tokens(tokenizer, rng, filler_budget)

    depth = index / max(total - 1, 1)
    cut = int(len(filler) * depth)
    cut = filler.rfind(" ", 0, cut) + 1 if cut else 0
    document = salt + filler[:cut] + framed + filler[cut:]
    return document + "\n\n" + INSTRUCTION, needle


async def run_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    needle: str,
    max_tokens: int,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = None
    text = ""
    usage = {}
    finish = None
    async with session.post(
        f"{base_url}/v1/chat/completions", json=payload
    ) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {}).get("content")
                if delta:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text += delta
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    elapsed = time.perf_counter() - t0
    return {
        "ttft_s": ttft,
        "elapsed_s": elapsed,
        "finish_reason": finish,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "needle_verbatim": needle in text,
        # autojunk=False: with a small word vocabulary every word is
        # "popular" and the default heuristic degenerates ratio() to 0.
        "match_ratio": difflib.SequenceMatcher(
            None, needle.split(), text.split(), autojunk=False
        ).ratio(),
        "text": text,
    }


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None, help="defaults to --model")
    ap.add_argument("--input-tokens", type=int, default=8000)
    ap.add_argument("--needle-tokens", type=int, default=1000)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="defaults to needle-tokens + 128")
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    max_tokens = args.max_tokens or args.needle_tokens + 128
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)

    prompts = [
        build_prompt(
            tokenizer, i, args.input_tokens, args.needle_tokens,
            args.concurrency,
        )
        for i in range(args.concurrency)
    ]

    timeout = aiohttp.ClientTimeout(total=3600)
    bench_t0 = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(
            *(
                run_request(
                    session, args.base_url, args.model, prompt, needle,
                    max_tokens,
                )
                for prompt, needle in prompts
            )
        )
    wall_s = time.perf_counter() - bench_t0

    passed = sum(r["needle_verbatim"] for r in results)
    ratios = [r["match_ratio"] for r in results]
    ttfts = [r["ttft_s"] for r in results if r["ttft_s"] is not None]
    out_tokens = sum(r["completion_tokens"] or 0 for r in results)
    in_tokens = sum(r["prompt_tokens"] or 0 for r in results)
    decode_tps = [
        (r["completion_tokens"] - 1) / (r["elapsed_s"] - r["ttft_s"])
        for r in results
        if r["ttft_s"] and r["completion_tokens"] and r["completion_tokens"] > 1
    ]

    summary = {
        "concurrency": args.concurrency,
        "target_input_tokens": args.input_tokens,
        "target_needle_tokens": args.needle_tokens,
        "wall_s": round(wall_s, 2),
        "needle_verbatim_passed": passed,
        "match_ratio": {
            "mean": round(statistics.mean(ratios), 4),
            "p50": round(statistics.median(ratios), 4),
            "min": round(min(ratios), 4),
        },
        "prompt_tokens_total": in_tokens,
        "completion_tokens_total": out_tokens,
        "aggregate_output_tok_s": round(out_tokens / wall_s, 2),
        "aggregate_prefill_tok_s_upper_bound": round(
            in_tokens / max(ttfts) if ttfts else 0.0, 2
        ),
        "ttft_s": {
            "mean": round(statistics.mean(ttfts), 2) if ttfts else None,
            "p50": round(statistics.median(ttfts), 2) if ttfts else None,
            "max": round(max(ttfts), 2) if ttfts else None,
        },
        "per_request_decode_tok_s": {
            "mean": round(statistics.mean(decode_tps), 2) if decode_tps else None,
            "p50": round(statistics.median(decode_tps), 2) if decode_tps else None,
            "min": round(min(decode_tps), 2) if decode_tps else None,
        },
        "finish_reasons": {
            fr: sum(1 for r in results if r["finish_reason"] == fr)
            for fr in {r["finish_reason"] for r in results}
        },
    }
    print(json.dumps(summary, indent=2))

    if args.output_json:
        with open(args.output_json, "w") as fh:
            json.dump({"summary": summary, "results": results}, fh, indent=2)

    return 0 if passed == args.concurrency else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
