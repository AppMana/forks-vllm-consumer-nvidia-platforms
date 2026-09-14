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
    return REGISTRY.get_sample_value(name, {"model_name": MODEL_NAME, "engine": "0"})


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


def test_completed_sequence_throughput_is_mean_of_rates(vllm_config):
    """Unequal durations must not turn arithmetic means into weighted rates."""
    from vllm.v1.engine import FinishReason
    from vllm.v1.metrics.stats import FinishedRequestStats, IterationStats

    logger = PrometheusStatLogger(vllm_config, engine_indexes=[0])
    stats = IterationStats()
    stats.finished_requests = [
        FinishedRequestStats(
            FinishReason.STOP,
            num_prompt_tokens=1100,
            num_cached_tokens=1000,
            prefill_time=1,
            num_generation_tokens=11,
            decode_time=1,
        ),
        FinishedRequestStats(
            FinishReason.STOP,
            num_prompt_tokens=300,
            prefill_time=3,
            num_generation_tokens=61,
            decode_time=3,
        ),
    ]
    logger.record(scheduler_stats=None, iteration_stats=stats, engine_idx=0)
    assert _sample("vllm:request_prefill_tokens_per_second_sum") == 200
    assert _sample("vllm:request_prefill_tokens_per_second_count") == 2
    assert _sample("vllm:request_decode_tokens_per_second_sum") == 30
    assert _sample("vllm:request_decode_tokens_per_second_count") == 2
    # Decode mean is15, not70/4=17.5; prompt excludes1000cachehits.


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_sequence_throughput_omits_undefined_intervals(vllm_config, duration):
    from vllm.v1.engine import FinishReason
    from vllm.v1.metrics.stats import FinishedRequestStats, IterationStats

    logger = PrometheusStatLogger(vllm_config, engine_indexes=[0])
    stats = IterationStats()
    stats.finished_requests = [
        FinishedRequestStats(
            FinishReason.STOP,
            num_prompt_tokens=100,
            prefill_time=duration,
            num_generation_tokens=1,
            decode_time=duration,
        )
    ]
    logger.record(scheduler_stats=None, iteration_stats=stats, engine_idx=0)
    assert _sample("vllm:request_prefill_tokens_per_second_count") == 0
    assert _sample("vllm:request_decode_tokens_per_second_count") == 0
