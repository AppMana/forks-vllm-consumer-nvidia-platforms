# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Live chunked-prefill observability in the stat loggers.

During a long chunked prefill no output batches reach the frontend, so
iteration-stats-based counters freeze. The loggers must consume the
per-step ``SchedulerStats.num_scheduled_prefill_tokens`` so both
Prometheus and the periodic log line show prefill progress live.
"""

import pytest
from prometheus_client import REGISTRY

from vllm.config import ModelConfig, VllmConfig
from vllm.v1.metrics.loggers import LoggingStatLogger, PrometheusStatLogger
from vllm.v1.metrics.stats import SchedulerStats

MODEL_NAME = "facebook/opt-125m"


@pytest.fixture(scope="module")
def vllm_config() -> VllmConfig:
    return VllmConfig(
        model_config=ModelConfig(model=MODEL_NAME, dtype="float16", seed=42)
    )


def _sample(name: str) -> float | None:
    return REGISTRY.get_sample_value(
        name, {"model_name": MODEL_NAME, "engine": "0"}
    )


def test_prometheus_scheduled_prefill_tokens_counter(vllm_config):
    """The counter must advance on every stats-only record (no outputs)."""
    logger = PrometheusStatLogger(vllm_config, engine_indexes=[0])

    # Mid-prefill records carry scheduler stats but no iteration stats.
    logger.record(
        scheduler_stats=SchedulerStats(num_scheduled_prefill_tokens=128),
        iteration_stats=None,
        engine_idx=0,
    )
    logger.record(
        scheduler_stats=SchedulerStats(num_scheduled_prefill_tokens=64),
        iteration_stats=None,
        engine_idx=0,
    )

    assert _sample("vllm:scheduled_prefill_tokens_total") == 192


def test_logging_stat_logger_live_prompt_throughput(vllm_config):
    """Prompt throughput must be nonzero mid-prefill, before any output."""
    logger = LoggingStatLogger(vllm_config, engine_index=0)

    logger.record(
        scheduler_stats=SchedulerStats(
            num_running_reqs=1, num_scheduled_prefill_tokens=4096
        ),
        iteration_stats=None,
    )
    logger.log()

    assert logger.last_prompt_throughput > 0
    assert not logger.engine_is_idle


def test_logging_stat_logger_not_idle_with_running_requests(vllm_config):
    """A stalled engine (running request, zero throughput) must keep the
    periodic log line at INFO level rather than going silent."""
    logger = LoggingStatLogger(vllm_config, engine_index=0)

    logger.record(
        scheduler_stats=SchedulerStats(num_running_reqs=1),
        iteration_stats=None,
    )
    logger.log()

    assert logger.last_prompt_throughput == 0
    assert not logger.engine_is_idle


def test_logging_stat_logger_idle_when_no_work(vllm_config):
    logger = LoggingStatLogger(vllm_config, engine_index=0)

    logger.record(scheduler_stats=SchedulerStats(), iteration_stats=None)
    logger.log()

    assert logger.engine_is_idle
