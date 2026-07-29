# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static name-resolution gates for the hand-reconciled DSV4 trees.

Two NameError-shaped faults survived the 316-commit upstream merge into this
branch, both the same mistake: the reconciliation kept a call site and dropped
the definition it needs (``_COMBINE_TOPK_SWA_NUM_WORKERS`` and the four warmup
tables beside it, recovered in 771f44a69b). Neither is reachable until KV cache
allocation, so every kernel unit test passed while the first real request died.

An attribute read of a *deleted upstream field* has the same shape and the same
blast radius -- ``scheduler_config.max_num_partial_prefills`` (7e4c1e1559) took
every worker down at init -- but no undefined-name checker sees it, so it gets
its own audit here.

Both checks are pure-Python and need no GPU.
"""

import ast
import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

# The trees the merge reconciled by hand. Scoped rather than repo-wide so a
# finding here is always actionable by whoever owns this branch.
MERGED_TREES = (
    "vllm/models/deepseek_v4",
    "vllm/v1/attention/backends/mla",
    "vllm/model_executor/layers/fused_moe/router",
    "vllm/model_executor/warmup",
    "vllm/model_executor/kernels/mhc",
)


def _python_files() -> list[Path]:
    files: list[Path] = []
    for tree in MERGED_TREES:
        root = REPO_ROOT / tree
        assert root.is_dir(), f"{tree} is gone; update MERGED_TREES"
        files.extend(sorted(p for p in root.rglob("*.py")))
    assert files
    return files


def _undefined_names_pyflakes(files: list[Path]) -> list[str]:
    from pyflakes import messages as pyflakes_messages
    from pyflakes.checker import Checker

    # Undefined *names* only. Unused imports and shadowing are style findings
    # that ruff already owns; these three are the ones that raise at runtime.
    fatal = (
        pyflakes_messages.UndefinedName,
        pyflakes_messages.UndefinedLocal,
        pyflakes_messages.UndefinedExport,
    )

    findings: list[str] = []
    for path in files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for message in Checker(tree, filename=str(path)).messages:
            if isinstance(message, fatal):
                rel = path.relative_to(REPO_ROOT)
                text = message.message % message.message_args
                findings.append(f"{rel}:{message.lineno}: {text}")
    return findings


def _undefined_names_ruff(files: list[Path]) -> list[str]:
    ruff = shutil.which("ruff")
    assert ruff is not None
    proc = subprocess.run(
        [
            ruff,
            "check",
            "--no-cache",
            "--select",
            "F821,F811,F822",
            "--output-format",
            "json",
        ]
        + [str(p) for p in files],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"ruff failed: {proc.stderr}")
    return [
        f"{Path(item['filename']).relative_to(REPO_ROOT)}:"
        f"{(item.get('location') or {}).get('row')}: {item['code']} {item['message']}"
        for item in json.loads(proc.stdout or "[]")
    ]


def test_no_undefined_names_in_merged_trees() -> None:
    """No call site may outlive the definition it reads.

    Would have caught 771f44a69b: ``_COMBINE_TOPK_SWA_NUM_WORKERS`` and the
    warmup input tables were called and never defined.
    """
    files = _python_files()
    try:
        import pyflakes  # noqa: F401
    except ImportError:
        if shutil.which("ruff") is None:
            pytest.skip(
                "neither pyflakes nor ruff is installed; "
                "add pyflakes to requirements/test/*.in to make this a hard gate"
            )
        findings = _undefined_names_ruff(files)
    else:
        findings = _undefined_names_pyflakes(files)

    assert not findings, "undefined names in the merged DSV4 trees:\n" + "\n".join(
        findings
    )


# ---------------------------------------------------------------------------
# scheduler_config attribute audit
# ---------------------------------------------------------------------------


def _scheduler_config_attribute_reads(files: list[Path]) -> set[tuple[Path, int, str]]:
    """Every ``...scheduler_config.<name>`` attribute access in ``files``.

    Matches ``scheduler_config.x``, ``self.scheduler_config.x`` and
    ``vllm_config.scheduler_config.x`` -- the three spellings the DSV4 tree
    uses -- by looking only at the immediately preceding attribute/name.
    """
    reads: set[tuple[Path, int, str]] = set()
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            base = node.value
            base_name = (
                base.id
                if isinstance(base, ast.Name)
                else base.attr
                if isinstance(base, ast.Attribute)
                else None
            )
            if base_name == "scheduler_config":
                reads.add((path, node.lineno, node.attr))
    return reads


def test_every_scheduler_config_attribute_still_exists() -> None:
    """A field upstream deleted must not survive as a live read.

    Would have caught 7e4c1e1559: ``get_max_prefill_buffer_size`` read
    ``scheduler_config.max_num_partial_prefills`` after V1 unified chunked
    prefill and removed it, so the indexer raised AttributeError at worker
    init -- before the sparse indexer ever ran on this branch.
    """
    from vllm.config.scheduler import SchedulerConfig

    known = set(dir(SchedulerConfig))
    fields = getattr(SchedulerConfig, "__dataclass_fields__", None)
    if fields:
        known |= set(fields)
    known |= set(getattr(SchedulerConfig, "__annotations__", {}))

    missing = sorted(
        f"{path.relative_to(REPO_ROOT)}:{lineno}: SchedulerConfig has no {attr!r}"
        for path, lineno, attr in _scheduler_config_attribute_reads(_python_files())
        if attr not in known and not attr.startswith("__")
    )
    assert not missing, (
        "scheduler_config reads that SchedulerConfig no longer defines:\n"
        + "\n".join(missing)
    )
