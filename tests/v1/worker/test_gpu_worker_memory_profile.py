# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.v1.worker.gpu_worker import Worker


class _FakeModelRunner:
    def __init__(self) -> None:
        self.profile_runs = 0

    def profile_run(self) -> None:
        self.profile_runs += 1


def _make_worker(*, model_type: str, pipeline_parallel_size: int) -> Worker:
    worker = Worker.__new__(Worker)
    worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=1234)
    worker.init_snapshot = SimpleNamespace(free_memory=8 * 1024**3)
    worker.model_runner = _FakeModelRunner()
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type=model_type,
                architectures=["DeepseekV4ForCausalLM"] if model_type == "deepseek_v4" else [],
            ),
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=pipeline_parallel_size),
    )
    return worker


def test_deepseek_v4_pp_explicit_kv_cache_skips_profile_run() -> None:
    worker = _make_worker(model_type="deepseek_v4", pipeline_parallel_size=5)

    assert worker.determine_available_memory() == 1234
    assert worker.model_runner.profile_runs == 0


def test_non_deepseek_v4_pp_explicit_kv_cache_keeps_profile_run() -> None:
    worker = _make_worker(model_type="llama", pipeline_parallel_size=5)

    assert worker.determine_available_memory() == 1234
    assert worker.model_runner.profile_runs == 1
