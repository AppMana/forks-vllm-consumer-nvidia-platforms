# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.v1.executor.ray_executor_v2 import RayExecutorV2


def _assert_bounded_nsight_capture(runtime_env: dict) -> None:
    nsight = runtime_env["nsight"]
    trace_kinds = set(nsight["t"].split(","))

    assert {"cuda", "nvtx", "osrt"} <= trace_kinds
    assert nsight["capture-range"] == "cudaProfilerApi"
    assert nsight["capture-range-end"] == "repeat"
    assert nsight["cuda-graph-trace"] == "node"


def test_ray_executor_v2_uses_bounded_nsight_capture() -> None:
    executor = object.__new__(RayExecutorV2)
    executor.parallel_config = SimpleNamespace(
        ray_runtime_env=None,
        ray_workers_use_nsight=True,
    )

    runtime_env = executor._build_runtime_env()

    _assert_bounded_nsight_capture(runtime_env)
