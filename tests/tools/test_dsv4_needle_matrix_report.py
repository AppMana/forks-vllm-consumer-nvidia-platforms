# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The markdown mode emits the model-card table from needle-bench JSONs."""

import json

from tools.ampere.dsv4_needle_matrix_report import markdown_table, row_stats


def _write_cell(tmp_path, name, input_tokens, concurrency, verbatim, wall_s, results):
    payload = {
        "summary": {
            "target_input_tokens": input_tokens,
            "concurrency": concurrency,
            "needle_verbatim_passed": verbatim,
            "wall_s": wall_s,
        },
        "results": results,
    }
    (tmp_path / name).write_text(json.dumps(payload))


def test_markdown_table_reports_per_stream_and_aggregate_rates(tmp_path):
    # Two requests of 8,000 prompt tokens each, 1,000 output tokens each.
    # The slower one decides the aggregate prefill rate; the wall clock
    # decides the aggregate output rate.
    _write_cell(
        tmp_path,
        "c2.json",
        8000,
        2,
        2,
        30.0,
        [
            {
                "ttft_s": 2.0,
                "elapsed_s": 27.0,
                "completion_tokens": 1000,
                "prompt_tokens": 8000,
                "prefill_tok_s": 4000.0,
            },
            {
                "ttft_s": 4.0,
                "elapsed_s": 29.0,
                "completion_tokens": 1000,
                "prompt_tokens": 8000,
                "prefill_tok_s": 2000.0,
            },
        ],
    )
    _write_cell(
        tmp_path,
        "c1.json",
        8000,
        1,
        0,
        25.0,
        [
            {
                "ttft_s": 2.5,
                "elapsed_s": 22.5,
                "completion_tokens": 1000,
                "prompt_tokens": 8000,
                "prefill_tok_s": 3200.0,
            }
        ],
    )
    rows = sorted(
        (row_stats(p) for p in tmp_path.glob("*.json")),
        key=lambda r: (r["input"], r["C"]),
    )
    lines = markdown_table(rows).splitlines()
    assert lines[0] == (
        "| Input tokens | C | TTFT s | Prefill tok/s per stream "
        "| Prefill tok/s aggregate | Output tok/s per stream "
        "| Output tok/s aggregate | Verbatim |"
    )
    assert lines[1] == "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    # C=1: median TTFT 2.5 s; 8,000/2.5 = 3,200 per stream and aggregate; 1,000
    # output tokens over 20 s of decode = 50.0 per stream, over 25 s of wall = 40.0.
    assert lines[2] == "| 8,000 | 1 | 2.5 | 3,200 | 3,200 | 50.0 | 40.0 | 0/1 |"
    # C=2: median TTFT 3.0 s; per-stream prefill is the median 3,000; aggregate is
    # 16,000 tokens over the 4.0 s until the last prefill finished = 4,000; per-stream
    # decode is the median of 1,000/25 and 1,000/25 = 40.0; aggregate 2,000/30 = 66.7.
    assert lines[3] == "| 8,000 | 2 | 3.0 | 3,000 | 4,000 | 40.0 | 66.7 | 2/2 |"
