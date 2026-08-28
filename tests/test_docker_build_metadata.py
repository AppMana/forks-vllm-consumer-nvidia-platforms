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
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"
VERSIONS_JSON = REPO_ROOT / "docker" / "versions.json"
BAKE_HCL = REPO_ROOT / "docker" / "docker-bake.hcl"
CMAKE_LISTS = REPO_ROOT / "CMakeLists.txt"
CMAKE_UTILS = REPO_ROOT / "cmake" / "utils.cmake"
PYTHON_OVERLAY_DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile.python-overlay"

# The architectures this branch exists to serve: sm_86 (Ampere consumer) and
# sm_121a (GB10 / DGX Spark).
REQUIRED_ARCHES = ("8.6", "12.1a")
REQUIRED_SPARKINFER_REF = "78cc92eaad3bf0378d199c44621bbaee75d0cb47"
RUNTIME_OVERLAY_FILES = (
    "vllm/model_executor/model_loader/default_loader.py",
    "vllm/models/deepseek_v4/nvidia/dspark.py",
    "vllm/v1/executor/ray_executor.py",
    "vllm/v1/executor/ray_executor_v2.py",
    "vllm/v1/executor/ray_utils.py",
    "vllm/v1/worker/gpu/warmup.py",
    "vllm/v1/worker/gpu/model_runner.py",
    "vllm/v1/worker/gpu/pp_utils.py",
    "vllm/v1/worker/gpu_worker.py",
)


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


def bake_hcl_variable_default(name: str) -> str:
    """The ``default = "..."`` of a top-level ``variable`` block in the bake file."""
    pattern = re.compile(
        rf'variable\s+"{re.escape(name)}"\s*\{{[^}}]*?default\s*=\s*"([^"]*)"',
        re.DOTALL,
    )
    match = pattern.search(BAKE_HCL.read_text(encoding="utf-8"))
    assert match, f'no variable "{name}" with a string default in {BAKE_HCL.name}'
    return match.group(1)


def dockerfile_run_blocks() -> list[str]:
    """The Dockerfile's RUN instructions with line continuations joined."""
    joined = re.sub(r"\\\n", " ", DOCKERFILE.read_text(encoding="utf-8"))
    return [
        line for line in joined.splitlines() if line.lstrip().upper().startswith("RUN ")
    ]


def test_python_overlay_is_generic_and_installs_runtime_fixes() -> None:
    """The fast validation image must be explicit and base-image agnostic."""
    dockerfile = PYTHON_OVERLAY_DOCKERFILE.read_text(encoding="utf-8")
    assert re.search(r"^ARG BASE_IMAGE\s*$", dockerfile, re.MULTILINE)
    assert "ARG BASE_IMAGE=" not in dockerfile
    assert "FROM ${BASE_IMAGE}" in dockerfile
    for relative in RUNTIME_OVERLAY_FILES:
        assert f"COPY {relative} " in dockerfile
        assert f'"{relative.removeprefix("vllm/")}"' in dockerfile


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
        ("APPMANA_NCCL_GIT_REF", "APPMANA_NCCL_GIT_REF"),
        ("USB4_RDMA_PROVIDER_VERSION", "USB4_RDMA_PROVIDER_VERSION"),
        ("APPMANA_THUNDERBOLT_RELEASE_TAG", "APPMANA_THUNDERBOLT_RELEASE_TAG"),
    ):
        defaults = dockerfile_arg_defaults(arg_name)
        if not defaults:
            continue
        assert bake_variable_default(bake_name) == defaults[0], (
            f"docker/versions.json {bake_name}={bake_variable_default(bake_name)!r} "
            f"but docker/Dockerfile ARG {arg_name}={defaults[0]!r}; "
            f"regenerate with tools/generate_versions_json.py"
        )


def test_bake_hcl_arch_list_matches_the_dockerfile() -> None:
    """``docker buildx bake`` reads the .hcl even when versions.json is absent.

    ``docker/docker-bake.hcl`` declares its own ``TORCH_CUDA_ARCH_LIST``
    default and ``_common`` passes it into every target as
    ``torch_cuda_arch_list``, so it *overrides* the Dockerfile ARG. The
    README-documented ``cd docker && docker buildx bake`` passes no
    ``-f docker/versions.json``, so versions.json cannot rescue a stale value
    here. This is the a248f91ec5 failure mode at a third location.
    """
    dockerfile_default = dockerfile_arg_defaults("torch_cuda_arch_list")[0]
    assert bake_hcl_variable_default("TORCH_CUDA_ARCH_LIST") == dockerfile_default, (
        f"docker/docker-bake.hcl TORCH_CUDA_ARCH_LIST="
        f"{bake_hcl_variable_default('TORCH_CUDA_ARCH_LIST')!r} but "
        f"docker/Dockerfile ARG torch_cuda_arch_list={dockerfile_default!r}; "
        f"a bake build would ship different SASS from a plain docker build"
    )


def test_bake_openai_target_uses_requested_image_tag() -> None:
    """The immutable release tag must name the image, not only label it."""
    openai_target = re.search(
        r'target\s+"openai"\s*\{(?P<body>.*?)^\}',
        BAKE_HCL.read_text(encoding="utf-8"),
        re.DOTALL | re.MULTILINE,
    )
    assert openai_target, 'no target "openai" in docker/docker-bake.hcl'
    assert re.search(
        r"^\s*tags\s*=\s*\[\s*VLLM_IMAGE_TAG\s*\]\s*$",
        openai_target.group("body"),
        re.MULTILINE,
    ), "the openai bake target ignores VLLM_IMAGE_TAG and emits a mutable local tag"


def test_cuda13_supported_arch_filter_preserves_sm121a(tmp_path: Path) -> None:
    """The global CUDA-13 allow-list must not collapse 12.1a to 12.0.

    A native GB10 build can contain apparently relevant SM12x code while the
    AllSpark translation units are actually only sm_120a.  On SM121 that
    launched but returned corrupt values, so checking the Docker ARG alone is
    insufficient.
    """
    cuda13_branch = re.search(
        r"CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13\.0\).*?"
        r'set\(CUDA_SUPPORTED_ARCHS "([^"]+)"\)',
        CMAKE_LISTS.read_text(encoding="utf-8"),
        re.DOTALL,
    )
    assert cuda13_branch, "could not find the CUDA >= 13 supported-arch list"
    supported_archs = cuda13_branch.group(1)

    script = tmp_path / "check-sm121-filter.cmake"
    script.write_text(
        "cmake_minimum_required(VERSION 3.26)\n"
        f'include("{CMAKE_UTILS.as_posix()}")\n'
        "cuda_archs_loose_intersection("
        f'FILTERED "{supported_archs}" "8.6;12.1a")\n'
        "list(LENGTH FILTERED FILTERED_COUNT)\n"
        'if(NOT "8.6" IN_LIST FILTERED OR '
        'NOT "12.1a" IN_LIST FILTERED OR '
        "NOT FILTERED_COUNT EQUAL 2)\n"
        '  message(FATAL_ERROR "SM121 collapsed to: ${FILTERED}")\n'
        "endif()\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["cmake", "-P", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_gdrcopy_os_version_tracks_ubuntu_version() -> None:
    """The gdrcopy .deb is per-distro-release and the URL is built from this.

    ``tools/install_gdrcopy.sh`` interpolates ``GDRCOPY_OS_VERSION`` into both
    the download path and the package filename. Upstream defaults it to
    ``Ubuntu22_04`` and corrects it only inside the ``-ubuntu2404`` bake
    targets; this fork's base image is 24.04, so the Dockerfile default has to
    carry the correction or every plain ``openai``/``test`` build installs the
    22.04 package into a 24.04 image.
    """
    ubuntu_version = dockerfile_arg_defaults("UBUNTU_VERSION")[0]
    expected = f"Ubuntu{ubuntu_version.replace('.', '_')}"
    actual = dockerfile_arg_defaults("GDRCOPY_OS_VERSION")[0]
    assert actual == expected, (
        f"docker/Dockerfile ARG UBUNTU_VERSION={ubuntu_version!r} implies "
        f"GDRCOPY_OS_VERSION={expected!r}, found {actual!r}"
    )
    assert bake_variable_default("GDRCOPY_OS_VERSION") == expected, (
        f"docker/versions.json GDRCOPY_OS_VERSION="
        f"{bake_variable_default('GDRCOPY_OS_VERSION')!r} but the Dockerfile "
        f"implies {expected!r}; regenerate with tools/generate_versions_json.py"
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


def test_sparkinfer_ref_includes_native_int8_indexer_kernels() -> None:
    """The shared image needs both paged decode and contiguous prefill."""
    defaults = dockerfile_arg_defaults("SPARKINFER_REF")
    assert defaults == [REQUIRED_SPARKINFER_REF]
    assert bake_variable_default("SPARKINFER_REF") == REQUIRED_SPARKINFER_REF


def test_sparkinfer_extension_artifacts_are_asserted() -> None:
    """The pure-Python fallback is silent, so the image must check for the
    artifacts rather than trust the install succeeded."""
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert "sparkinfer_pcie_" in dockerfile, (
        "docker/Dockerfile no longer asserts the sparkinfer AOT extensions "
        "exist; the pure-Python fallback would regress unnoticed"
    )
