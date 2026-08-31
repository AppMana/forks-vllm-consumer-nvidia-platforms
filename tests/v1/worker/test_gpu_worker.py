# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.sequence import IntermediateTensors
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker import startup_plan
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.startup_plan import (
    maybe_apply_startup_plan,
    maybe_save_startup_plan,
)

# Startup-plan persistence (vllm/v1/worker/startup_plan.py), applied and
# saved by Worker.determine_available_memory / compile_or_warm_up_model.


def test_pp_orders_previous_send_before_entering_next_receive(monkeypatch):
    """Schedule the send dependency before synchronous receive metadata."""
    events = []

    class PreviousSend:
        completed = False

        def is_completed(self):
            return self.completed

        def wait(self):
            assert not self.completed
            events.append("enqueue_previous_send_dependency")

    class PPGroup:
        is_first_rank = False
        is_last_rank = False

        def irecv_tensor_dict(self, **_kwargs):
            events.append("enter_receive_metadata")
            assert events == [
                "enqueue_previous_send_dependency",
                "enter_receive_metadata",
            ]
            return {"hidden_states": torch.zeros(1)}, [], []

        def isend_tensor_dict(self, _tensor_dict, **_kwargs):
            events.append("post_next_send")
            return []

    class ModelRunner:
        intermediate_tensors = None
        is_pooling_model = False

        def execute_model(self, _scheduler_output, intermediate_tensors):
            assert intermediate_tensors is not None
            events.append("execute_forward")
            return IntermediateTensors({"hidden_states": torch.zeros(1)})

    pp_group = PPGroup()
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_worker.get_pp_group", lambda: pp_group
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_worker.get_tp_group", lambda: SimpleNamespace()
    )

    worker = SimpleNamespace(
        _pp_send_work=[PreviousSend()],
        use_v2_model_runner=True,
        model_runner=ModelRunner(),
        annotate_profile=lambda _scheduler_output: nullcontext(),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                pass_config=SimpleNamespace(enable_sp=False)
            ),
            parallel_config=SimpleNamespace(
                pipeline_parallel_size=3,
                distributed_executor_backend="mp",
            ),
        ),
    )
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.total_num_scheduled_tokens = 1

    execute_model = Worker.execute_model.__wrapped__
    execute_model(worker, scheduler_output)

    assert events == [
        "enqueue_previous_send_dependency",
        "enter_receive_metadata",
        "execute_forward",
        "post_next_send",
    ]


def test_pp_receive_reuses_storage_after_send_dependency_is_enqueued(
    monkeypatch,
):
    """Stream ordering permits persistent receive storage without a host wait."""
    events = []
    persistent_tensors = {"hidden_states": torch.zeros(1)}

    class PreviousSend:
        completed = False

        def is_completed(self):
            return self.completed

        def wait(self):
            assert not self.completed
            events.append("enqueue_previous_send_dependency")

    class PPGroup:
        is_first_rank = False
        is_last_rank = False

        def irecv_tensor_dict(self, *, recv_tensor_dict, **_kwargs):
            storage = "fresh" if recv_tensor_dict is None else "persistent"
            events.append(f"enter_receive_metadata:{storage}")
            return {"hidden_states": torch.zeros(1)}, [], []

        def isend_tensor_dict(self, _tensor_dict, **_kwargs):
            events.append("post_next_send")
            return []

    class ModelRunner:
        intermediate_tensors = IntermediateTensors(persistent_tensors)
        is_pooling_model = False

        def execute_model(self, _scheduler_output, intermediate_tensors):
            assert intermediate_tensors is not None
            events.append("execute_forward")
            return IntermediateTensors({"hidden_states": torch.zeros(1)})

    pp_group = PPGroup()
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_worker.get_pp_group", lambda: pp_group
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_worker.get_tp_group", lambda: SimpleNamespace()
    )

    worker = SimpleNamespace(
        _pp_send_work=[PreviousSend()],
        use_v2_model_runner=True,
        model_runner=ModelRunner(),
        annotate_profile=lambda _scheduler_output: nullcontext(),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                pass_config=SimpleNamespace(enable_sp=False)
            ),
            parallel_config=SimpleNamespace(
                pipeline_parallel_size=3,
                distributed_executor_backend="mp",
            ),
        ),
    )
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.total_num_scheduled_tokens = 1

    execute_model = Worker.execute_model.__wrapped__
    execute_model(worker, scheduler_output)

    assert events == [
        "enqueue_previous_send_dependency",
        "enter_receive_metadata:persistent",
        "execute_forward",
        "post_next_send",
    ]


def test_pp_empty_control_turn_preserves_pending_activation_send(monkeypatch):
    """Empty PP turns advance control state without retiring activation sends."""
    events = []

    class PreviousSend:
        def wait(self):
            events.append("wait_previous_send")

    class PPGroup:
        is_first_rank = False
        is_last_rank = False

    class ModelRunner:
        is_pooling_model = False

        def execute_model(self, scheduler_output, intermediate_tensors):
            assert scheduler_output.total_num_scheduled_tokens == 0
            assert intermediate_tensors is None
            events.append("advance_control_state")
            return None

    previous_send = PreviousSend()
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_worker.get_pp_group", lambda: PPGroup()
    )

    worker = SimpleNamespace(
        _pp_send_work=[previous_send],
        use_v2_model_runner=True,
        model_runner=ModelRunner(),
        annotate_profile=lambda _scheduler_output: nullcontext(),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                pass_config=SimpleNamespace(enable_sp=False)
            ),
            parallel_config=SimpleNamespace(
                pipeline_parallel_size=3,
                distributed_executor_backend="mp",
            ),
        ),
    )

    execute_model = Worker.execute_model.__wrapped__
    execute_model(worker, SchedulerOutput.make_empty())

    assert events == ["advance_control_state"]
    assert worker._pp_send_work == [previous_send]


def _plan_worker(config_hash="abc123", free_memory=78 * GiB_bytes, kv_bytes=None):
    """The minimal Worker surface the startup-plan entry points touch."""
    return SimpleNamespace(
        vllm_config=SimpleNamespace(compute_hash=lambda: config_hash),
        rank=0,
        parallel_config=SimpleNamespace(world_size=1),
        init_snapshot=SimpleNamespace(free_memory=free_memory),
        cache_config=SimpleNamespace(kv_cache_memory_bytes=kv_bytes),
    )


def _plan_platform(name="NVIDIA H100 PCIe"):
    return SimpleNamespace(
        get_device_name=lambda device_id=0: name,
        get_device_total_memory=lambda device_id=0: 80 * GiB_bytes,
        get_device_capability=lambda device_id=0: (9, 0),
    )


@pytest.fixture
def plan_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Enable the startup plan, isolated under a tmp cache root."""
    monkeypatch.setenv("VLLM_ENABLE_STARTUP_PLAN", "1")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    with patch.object(startup_plan, "current_platform", _plan_platform()):
        yield


def test_startup_plan_fingerprint_sensitivity(plan_env):
    """The fingerprint is the OOM-safety key: stable for identical inputs,
    different for anything the profiled value depends on."""
    fp = startup_plan.compute_plan_fingerprint
    base = fp(_plan_worker().vllm_config, 0, 1)
    assert base == fp(_plan_worker().vllm_config, 0, 1)
    assert base != fp(_plan_worker("other").vllm_config, 0, 1)
    assert base != fp(_plan_worker().vllm_config, 1, 2)
    with patch.object(startup_plan, "current_platform", _plan_platform("NVIDIA A100")):
        assert base != fp(_plan_worker().vllm_config, 0, 1)
    with patch("vllm.__version__", "0.0.0+plan-test"):
        assert base != fp(_plan_worker().vllm_config, 0, 1)


def test_startup_plan_apply_gate(plan_env):
    """Only a fingerprint-matching, memory-safe plan is ever applied."""
    maybe_save_startup_plan(_plan_worker(), 50 * GiB_bytes)

    applied = _plan_worker()
    maybe_apply_startup_plan(applied)
    assert applied.cache_config.kv_cache_memory_bytes == 50 * GiB_bytes

    less_memory = _plan_worker(free_memory=60 * GiB_bytes)
    other_config = _plan_worker(config_hash="zzz999")
    for refused in (less_memory, other_config):
        maybe_apply_startup_plan(refused)
        assert refused.cache_config.kv_cache_memory_bytes is None

    # An explicit --kv-cache-memory is never overridden.
    explicit = _plan_worker(kv_bytes=7 * GiB_bytes)
    maybe_apply_startup_plan(explicit)
    assert explicit.cache_config.kv_cache_memory_bytes == 7 * GiB_bytes
