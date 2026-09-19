# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Summarise needle-bench JSONs in the style of `vllm bench serve`.

usage: needle_matrix_report.py [--markdown] <needle-bench dir> [more dirs...]

--markdown prints the model-card table instead: per stream is the median
request of a batch (TTFT, prompt tokens over TTFT, completion tokens over
the decode time), aggregate is the whole batch (prompt tokens over the
slowest TTFT, completion tokens over the wall), verbatim is passes over C.

Per row (one JSON = one (input_len, concurrency) cell, all requests sent at
once): request throughput, output/total token throughput over the wall, and
mean/median/p99 of TTFT, TPOT, ITL and E2EL in milliseconds. TPOT per
request is (e2el - ttft)/(completion_tokens - 1); ITL is the streamed chunk
inter-arrival series recorded by the tool.
"""

import glob
import json
import os
import statistics
import sys


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def row_stats(path):
    with open(path) as handle:
        j = json.load(handle)
    s, rs = j["summary"], j["results"]
    ok = [r for r in rs if r.get("ttft_s") and r.get("completion_tokens")]
    wall = s["wall_s"]
    ttft = [r["ttft_s"] * 1000 for r in ok]
    e2el = [r["elapsed_s"] * 1000 for r in ok]
    tpot = [
        (r["elapsed_s"] - r["ttft_s"]) / (r["completion_tokens"] - 1) * 1000
        for r in ok
        if r["completion_tokens"] > 1
    ]
    itl = [x * 1000 for r in ok for x in r.get("itl_s", [])]
    out_tok = sum(r["completion_tokens"] for r in ok)
    in_tok = sum(r["prompt_tokens"] or 0 for r in ok)
    return {
        "input": s["target_input_tokens"],
        "C": s["concurrency"],
        "completed": len(ok),
        "failed": len(rs) - len(ok),
        "verbatim": s["needle_verbatim_passed"],
        "wall_s": wall,
        "req_per_s": len(ok) / wall,
        "out_tok_s": out_tok / wall,
        "total_tok_s": (in_tok + out_tok) / wall,
        "prefill_tok_s_p50": statistics.median(
            [r["prefill_tok_s"] for r in ok if r.get("prefill_tok_s")] or [float("nan")]
        ),
        "prompt_tokens": in_tok,
        "ttft": ttft,
        "tpot": tpot,
        "itl": itl,
        "e2el": e2el,
    }


def fmt(x, w=9, d=1):
    return f"{x:{w}.{d}f}" if x == x else " " * (w - 3) + "nan"


MARKDOWN_HEADER = (
    "| Input tokens | C | TTFT s | Prefill tok/s per stream | Prefill tok/s aggregate "
    "| Output tok/s per stream | Output tok/s aggregate | Verbatim |\n"
    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
)


def markdown_row(r):
    ttft_s = statistics.median(r["ttft"]) / 1000 if r["ttft"] else float("nan")
    decode = [
        1000.0 / t for t in r["tpot"]
    ]  # completion tokens per second of decode, per request
    per_stream_out = statistics.median(decode) if decode else float("nan")
    prefill_aggregate = (
        r["prompt_tokens"] / (max(r["ttft"]) / 1000) if r["ttft"] else float("nan")
    )
    return (
        f"| {r['input']:,} | {r['C']} | {ttft_s:.1f} | {r['prefill_tok_s_p50']:,.0f} "
        f"| {prefill_aggregate:,.0f} | {per_stream_out:.1f} | {r['out_tok_s']:.1f} "
        f"| {r['verbatim']}/{r['C']} |"
    )


def markdown_table(rows):
    return "\n".join([MARKDOWN_HEADER] + [markdown_row(r) for r in rows])


def main():
    args = sys.argv[1:]
    markdown = "--markdown" in args
    rows = []
    for d in [a for a in args if a != "--markdown"]:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                rows.append(row_stats(f))
            except Exception as e:  # noqa: BLE001
                print(f"skip {f}: {e}", file=sys.stderr)
    rows.sort(key=lambda r: (r["input"], r["C"]))
    if markdown:
        print(markdown_table(rows))
        return
    hdr = (
        f"{'input':>7} {'C':>3} {'ok':>3} {'verb':>4} {'wall s':>8} {'req/s':>7} "
        f"{'out tok/s':>9} {'tot tok/s':>10} {'prefill/req':>11} | "
        f"{'TTFT mean':>9} {'med':>9} {'p99':>9} | "
        f"{'TPOT mean':>9} {'med':>9} {'p99':>9} | "
        f"{'ITL mean':>9} {'med':>9} {'p99':>9} | "
        f"{'E2EL mean':>9} {'med':>9} {'p99':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        m = lambda xs: statistics.fmean(xs) if xs else float("nan")  # noqa: E731
        md = lambda xs: statistics.median(xs) if xs else float("nan")  # noqa: E731
        print(
            f"{r['input']:>7} {r['C']:>3} {r['completed']:>3} {r['verbatim']:>4} "
            f"{r['wall_s']:>8.1f} {r['req_per_s']:>7.3f} {r['out_tok_s']:>9.1f} "
            f"{r['total_tok_s']:>10.1f} {fmt(r['prefill_tok_s_p50'], 11, 0)} | "
            f"{fmt(m(r['ttft']))} {fmt(md(r['ttft']))} {fmt(pct(r['ttft'], 99))} | "
            f"{fmt(m(r['tpot']), 9, 2)} {fmt(md(r['tpot']), 9, 2)} "
            f"{fmt(pct(r['tpot'], 99), 9, 2)} | "
            f"{fmt(m(r['itl']), 9, 2)} {fmt(md(r['itl']), 9, 2)} "
            f"{fmt(pct(r['itl'], 99), 9, 2)} | "
            f"{fmt(m(r['e2el']))} {fmt(md(r['e2el']))} {fmt(pct(r['e2el'], 99))}"
        )
    print(
        "\nlatencies in ms; TPOT/ITL exclude the first token; "
        "prefill/req = median prompt_tokens/TTFT of one request"
    )


if __name__ == "__main__":
    main()
