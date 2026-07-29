# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build metadata that no runtime test can reach.

``docker/versions.json`` is generated from the Dockerfile's ARG defaults and
fed back to ``docker buildx bake`` as the build's variable set. When it goes
stale the bake build silently produces a *different* image from the plain
``docker build``: a248f91ec5 found it still carrying the pre-fork
``TORCH_CUDA_ARCH_LIST`` with no ``12.1a``, i.e. an image with no GB10 SASS
at all.

The sparkinfer install has the same shape of failure: its setup.py builds the
AOT CUDA extensions only if it can import ``torch.utils.cpp_extension`` and
falls back to a pure-Python install otherwise, silently, so building under
isolation ships zero ``.so`` and the first collective runs nvcc inside a live
request (419d9d92b6).

Both are text checks and cost nothing.
"""

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"
VERSIONS_JSON = REPO_ROOT / "docker" / "versions.json"

# The architectures this branch exists to serve: sm_86 (Ampere consumer) and
# sm_121a (GB10 / DGX Spark).
REQUIRED_ARCHES = ("8.6", "12.1a")


def dockerfile_arg_defaults(name: str) -> list[str]:
    """Every ``ARG <name>=<default>`` default in the Dockerfile, unquoted."""
    pattern = re.compile(
        rf"^\s*ARG\s+{re.escape(name)}=(.*?)\s*$", re.IGNORECASE | re.MULTILINE
    )
    values = []
    for match in pattern.finditer(DOCKERFILE.read_text(encoding="utf-8")):
        value = match.group(1).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values.append(value)
    return values


def bake_variable_default(name: str) -> str:
    versions = json.loads(VERSIONS_JSON.read_text(encoding="utf-8"))
    return versions["variable"][name]["default"]


def dockerfile_run_blocks() -> list[str]:
    """The Dockerfile's RUN instructions with line continuations joined."""
    joined = re.sub(r"\\\n", " ", DOCKERFILE.read_text(encoding="utf-8"))
    return [
        line for line in joined.splitlines() if line.lstrip().upper().startswith("RUN ")
    ]


def test_torch_cuda_arch_list_arg_is_consistent() -> None:
    """Every stage that sets the arch list must set the same one."""
    defaults = dockerfile_arg_defaults("torch_cuda_arch_list")
    assert defaults, "no ARG torch_cuda_arch_list in docker/Dockerfile"
    assert len(set(defaults)) == 1, (
        f"docker/Dockerfile stages disagree on torch_cuda_arch_list: {defaults}"
    )


@pytest.mark.parametrize("arch", REQUIRED_ARCHES)
def test_torch_cuda_arch_list_covers_both_platforms(arch: str) -> None:
    """One image serves sm_86 and sm_121; a missing arch means no SASS."""
    arch_list = dockerfile_arg_defaults("torch_cuda_arch_list")[0]
    assert arch in arch_list.split(), (
        f"TORCH_CUDA_ARCH_LIST={arch_list!r} does not build for {arch}"
    )


def test_versions_json_matches_the_dockerfile() -> None:
    """``docker buildx bake`` must not build something else.

    Would have caught a248f91ec5: versions.json still carried the pre-fork
    arch list, so a bake build produced an image with no GB10 SASS.
    """
    for arg_name, bake_name in (
        ("torch_cuda_arch_list", "TORCH_CUDA_ARCH_LIST"),
        ("max_jobs", "MAX_JOBS"),
        ("nvcc_threads", "NVCC_THREADS"),
    ):
        defaults = dockerfile_arg_defaults(arg_name)
        if not defaults:
            continue
        assert bake_variable_default(bake_name) == defaults[0], (
            f"docker/versions.json {bake_name}={bake_variable_default(bake_name)!r} "
            f"but docker/Dockerfile ARG {arg_name}={defaults[0]!r}; "
            f"regenerate with tools/generate_versions_json.py"
        )


def test_sparkinfer_is_installed_without_build_isolation() -> None:
    """Under isolation sparkinfer's setup.py silently ships no extensions.

    Would have caught 419d9d92b6.
    """
    install_lines = [
        line
        for line in dockerfile_run_blocks()
        if "SPARKINFER_REPO" in line and "pip install" in line
    ]
    assert install_lines, "no sparkinfer install found in docker/Dockerfile"
    for line in install_lines:
        assert "--no-build-isolation" in line, (
            "sparkinfer is installed with build isolation, so its setup.py "
            "falls back to a pure-Python install and the AOT CUDA extensions "
            f"are never built: {line.strip()}"
        )


def test_sparkinfer_extension_artifacts_are_asserted() -> None:
    """The pure-Python fallback is silent, so the image must check for the
    artifacts rather than trust the install succeeded."""
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert "sparkinfer_pcie_" in dockerfile, (
        "docker/Dockerfile no longer asserts the sparkinfer AOT extensions "
        "exist; the pure-Python fallback would regress unnoticed"
    )
