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
AMPERE_BUILD_HELPER = REPO_ROOT / "tools" / "ampere" / "build_vllm_ampere_image.sh"

# The architectures this branch exists to serve: sm_86 (Ampere consumer) and
# sm_121a (GB10 / DGX Spark).
REQUIRED_ARCHES = ("8.6", "12.1a")
REQUIRED_SPARKINFER_REF = "d91c0a600c0a18d008c649d2aa30e89514ffba16"
REQUIRED_SPARKINFER_SPEC = "sparkinfer==1.0.2.dev1"
RUNTIME_OVERLAY_FILES = (
    "vllm/model_executor/layers/quantization/dsv4_int.py",
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
    assert '"scatter_draft_tokens"' in dockerfile
    assert '"steady_verify.total_num_scheduled_tokens = 1 + num_spec"' in dockerfile


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
        ("BUILD_DEEPEP", "BUILD_DEEPEP"),
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


def test_ampere_build_includes_flash_attention_for_model_inspection() -> None:
    """The serving image must import model metadata before backend selection."""
    assert dockerfile_arg_defaults("VLLM_SKIP_FLASH_ATTN_BUILD") == ["0"]
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    base, csrc = dockerfile.split("FROM base AS csrc-build", 1)
    csrc, build = csrc.split("FROM base AS build", 1)
    build = build.split("FROM ${BUILD_BASE_IMAGE} AS appmana-nccl-build", 1)[0]
    assignment = "ENV VLLM_SKIP_FLASH_ATTN_BUILD=${VLLM_SKIP_FLASH_ATTN_BUILD}"
    assert assignment not in base
    assert assignment in csrc
    assert assignment in build

    helper = AMPERE_BUILD_HELPER.read_text(encoding="utf-8")
    assert 'skip_flash_attn_build="${VLLM_SKIP_FLASH_ATTN_BUILD:-0}"' in helper
    assert '--build-arg "VLLM_SKIP_FLASH_ATTN_BUILD=${skip_flash_attn_build}"' in helper

    setup_source = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")
    guard = 'os.environ.get("VLLM_SKIP_FLASH_ATTN_BUILD") != "1"'
    assert guard in setup_source
    guarded_body = setup_source.split(guard, 1)[1].split("# FA4 CuteDSL", 1)[0]
    assert "_vllm_fa2_C" in guarded_body
    assert "_vllm_fa3_C" in guarded_body


def test_sccache_has_a_persistent_buildkit_local_cache() -> None:
    """A source-layer rebuild must not require a remote round trip per object."""
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert re.search(
        r"RUN\s+--mount=type=cache,target=/workspace/tmp/sccache,sharing=shared"
        r".*?python3 setup\.py bdist_wheel",
        dockerfile,
        re.DOTALL,
    )


def test_rust_sccache_uses_a_prepared_persistent_temp_directory() -> None:
    """Rust caching must not rely on an image's transient /tmp directory."""
    rust_builds = [
        block for block in dockerfile_run_blocks() if "build_rust.sh" in block
    ]
    assert any(
        "--mount=type=cache,target=/workspace/tmp/sccache,sharing=shared" in block
        and "mkdir -p /workspace/tmp /workspace/tmp/sccache" in block
        and "export TMPDIR=/workspace/tmp" in block
        and "export SCCACHE_DIR=/workspace/tmp/sccache" in block
        and "export SCCACHE_SERVER_UDS=/workspace/tmp/sccache/sccache.sock" in block
        for block in rust_builds
    )


def test_cmake_dependencies_have_a_persistent_buildkit_cache() -> None:
    """A source rebuild must not clone every pinned CMake dependency again."""
    csrc_builds = [
        block
        for block in dockerfile_run_blocks()
        if "python3 setup.py bdist_wheel" in block
    ]
    assert any(
        "--mount=type=cache,target=/workspace/tmp/sccache,sharing=shared" in block
        and "--mount=type=cache,target=/workspace/.deps,sharing=shared" in block
        for block in csrc_builds
    )


def test_sm86_skips_high_end_cuda_external_projects() -> None:
    """Ampere-only builds must not fetch or package newer-architecture code."""
    cmake = CMAKE_LISTS.read_text(encoding="utf-8")
    match = re.search(
        r"cuda_archs_loose_intersection\(VLLM_HIGH_END_EXTERNAL_ARCHS"
        r'\s+"(?P<supported>[^"]+)"\s+"\$\{CUDA_ARCHS\}"\)'
        r"(?P<body>.*?)# vllm-flash-attn should be last",
        cmake,
        re.DOTALL,
    )
    assert match, "high-end external projects are not architecture-gated"
    assert "8.6" not in match.group("supported").split(";")
    body = match.group("body")
    assert "if(VLLM_HIGH_END_EXTERNAL_ARCHS)" in body
    for project in ("deepgemm", "fmha_sm100", "flashmla", "qutlass", "tml_fa4"):
        assert f"include(cmake/external_projects/{project}.cmake)" in body
    for target in (
        "_deep_gemm_C",
        "fmha_sm100",
        "_flashmla_C",
        "_flashmla_extension_C",
        "_qutlass_C",
        "tml_fa4",
    ):
        assert f"add_custom_target({target})" in body


def test_ampere_build_skips_unsupported_deepep_extensions() -> None:
    """SM86 builds must not compile DeepEP's SM90/SM100-only wheel."""
    assert dockerfile_arg_defaults("BUILD_DEEPEP") == ["1"]

    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert re.search(
        r'if \[ "\$\{BUILD_DEEPEP\}" = "1" \]; then.*?'
        r"/tmp/install_python_libraries\.sh",
        dockerfile,
        re.DOTALL,
    )
    assert re.search(
        r'if \[ "\$\{BUILD_DEEPEP\}" = "1" \]; then.*?'
        r"uv pip install --system ep_kernels/dist/\*\.whl",
        dockerfile,
        re.DOTALL,
    )

    helper = AMPERE_BUILD_HELPER.read_text(encoding="utf-8")
    assert 'build_deepep="${BUILD_DEEPEP:-0}"' in helper
    assert '--build-arg "BUILD_DEEPEP=${build_deepep}"' in helper


def test_deepep_disabled_marker_reaches_the_final_image() -> None:
    """DeepEP-off builds must checksum the marker instead of requiring a wheel."""
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert "touch /tmp/ep_kernels_workspace/dist/deepep-disabled" in dockerfile
    assert "sha256sum /tmp/ep_kernels_workspace/dist/*" in dockerfile
    assert "sha256sum /tmp/ep_kernels_workspace/dist/*.whl" not in dockerfile


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


def test_sparkinfer_is_installed_from_the_published_wheel() -> None:
    """Image builds must not recompile Sparkinfer from a mutable source tree."""
    install_lines = [
        line
        for line in dockerfile_run_blocks()
        if "SPARKINFER_SPEC" in line and "pip install" in line
    ]
    assert install_lines, "no sparkinfer install found in docker/Dockerfile"
    for line in install_lines:
        assert "--extra-index-url" in line
        assert "git+" not in line
        assert "--no-build-isolation" not in line

    assert dockerfile_arg_defaults("SPARKINFER_SPEC") == [REQUIRED_SPARKINFER_SPEC]


def test_flashmla_release_check_uses_distribution_metadata() -> None:
    """The package's kernel ABI version is distinct from its wheel release."""
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert "import importlib.metadata as importlib_metadata" in dockerfile
    assert "importlib_metadata.version('flash-mla') == '2.0.1.dev1'" in dockerfile
    assert "flash_mla.__version__ == '2.0.1.dev1'" not in dockerfile


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


def test_final_image_validates_the_nccl_elf_mapped_by_pytorch() -> None:
    """Validate the ELF selected by the runtime loader after all installs."""
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    stage_match = re.search(
        r"^FROM\s+\S+\s+AS\s+vllm-openai-base\s*$\n"
        r"(?P<body>.*?)(?=^FROM\s|\Z)",
        dockerfile,
        re.DOTALL | re.MULTILINE,
    )
    assert stage_match, "no vllm-openai-base stage in docker/Dockerfile"
    stage = stage_match.group("body")

    python_run_blocks = list(
        re.finditer(
            r"^RUN\s+[^\n]*python3\s+-\s+<<'PY'\n(?P<body>.*?)^PY\s*$",
            stage,
            re.DOTALL | re.MULTILINE,
        )
    )
    validators = [
        match for match in python_run_blocks if "libnccl.so" in match.group("body")
    ]
    assert validators, "no final-image NCCL ELF validation block"
    validator = validators[-1]

    dependency_installs = list(
        re.finditer(r"\b(?:uv\s+pip|python3\s+-m\s+pip|apt-get)\s+install\b", stage)
    )
    assert dependency_installs, "no dependency installs found in vllm-openai-base"
    assert validator.start() > max(match.end() for match in dependency_installs), (
        "NCCL validation must run after every dependency install"
    )

    final_stage_match = re.search(
        r"^FROM\s+vllm-openai-base\s+AS\s+vllm-openai\s*$\n"
        r"(?P<body>.*?)(?=^FROM\s|\Z)",
        dockerfile,
        re.DOTALL | re.MULTILINE,
    )
    assert final_stage_match, "no vllm-openai final stage in docker/Dockerfile"
    final_stage = final_stage_match.group("body")
    assert not re.search(
        r"\b(?:uv\s+pip|python3\s+-m\s+pip|apt-get)\s+install\b", final_stage
    ), "the final image installs dependencies after NCCL validation"

    validation = validator.group("body")
    assert re.search(r"^import\s+torch\b", validation, re.MULTILINE), (
        "NCCL validation must load PyTorch"
    )
    assert re.search(r"^import\s+ctypes\b", validation, re.MULTILINE), (
        "NCCL validation must use the runtime ELF loader"
    )
    assert "libtorch_cuda.so" in validation, (
        "NCCL validation must load PyTorch's CUDA ELF before inspecting mappings"
    )
    assert "/proc/self/maps" in validation, (
        "NCCL validation must inspect the NCCL ELF actually mapped at runtime"
    )
    assert "torch.cuda.nccl.version" not in validation, (
        "the compiled NCCL version does not identify the ELF mapped at runtime"
    )
