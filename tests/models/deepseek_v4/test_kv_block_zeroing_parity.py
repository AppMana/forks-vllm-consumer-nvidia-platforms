# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The two GPU model runners must agree on KV-block zeroing.

The scheduler emits ``new_block_ids_to_zero`` only when
``kv_cache_config.needs_kv_cache_zeroing`` is set, and DSV4 sets it: the fp32
compressor state shares the cache with the fp8_ds_mla KV, so a freshly
allocated block that is not zeroed decodes stale bytes to NaN.

Two runners took two different positions on a zeroer that was never built. The
v2 runner asserted and died on the first request; the v1 runner guards with
``hasattr`` and skips the zeroing silently, which is the worse of the two --
those blocks are exactly the ones that produce NaN. 3ff0b7bc2a fixed the v2
runner to build it on demand; the v1 runner still skips.

Neither test needs a GPU: the v1 side calls the real ``_zero_block_ids`` on a
stub receiver, and the v2 side reads the source of the branch, which is inlined
in ``update_requests`` and cannot be called without a SchedulerOutput.
"""

import ast
import inspect
import textwrap


class _ZeroerSpy:
    def __init__(self):
        self.zeroed: list[list[int]] = []

    def zero_block_ids(self, block_ids):
        self.zeroed.append(list(block_ids))


class _V1Receiver:
    """Stand-in for a v1 GPUModelRunner that the worker never eagerly inited."""

    def __init__(self, *, zeroer=None):
        if zeroer is not None:
            self._kv_block_zeroer = zeroer

    def _init_kv_zero_meta(self):
        self._kv_block_zeroer = _ZeroerSpy()


def _v1_zero_block_ids(receiver, block_ids) -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    GPUModelRunner._zero_block_ids(receiver, block_ids)


def test_v1_runner_zeroes_when_the_zeroer_exists() -> None:
    """Baseline, so a red result below is about the missing-zeroer case."""
    spy = _ZeroerSpy()
    _v1_zero_block_ids(_V1Receiver(zeroer=spy), [3, 4, 5])
    assert spy.zeroed == [[3, 4, 5]]


def test_v1_runner_never_silently_skips_zeroing() -> None:
    """A runner handed block ids must zero them or fail loudly.

    Same invariant 3ff0b7bc2a established for the v2 runner. The v1 runner's
    ``hasattr`` guard returns quietly instead, so this is currently RED and
    reports a real remaining bug rather than a test defect.
    """
    receiver = _V1Receiver(zeroer=None)
    _v1_zero_block_ids(receiver, [7])

    zeroer = getattr(receiver, "_kv_block_zeroer", None)
    assert zeroer is not None, (
        "v1 GPUModelRunner._zero_block_ids was asked to zero blocks with no "
        "zeroer built and did nothing. Those blocks decode stale bytes to NaN "
        "under DSV4's mixed-precision cache; the v2 runner builds the zeroer "
        "on first use (3ff0b7bc2a) and v1 must do the same."
    )
    assert zeroer.zeroed == [[7]]


def _v2_new_block_ids_branch() -> ast.If:
    """The ``if scheduler_output.new_block_ids_to_zero:`` body in update_requests."""
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    tree = ast.parse(textwrap.dedent(inspect.getsource(GPUModelRunner.update_requests)))
    branches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "new_block_ids_to_zero" in ast.unparse(node.test)
    ]
    assert len(branches) == 1, (
        f"expected one new_block_ids_to_zero branch, found {len(branches)}"
    )
    return branches[0]


def test_v2_runner_builds_the_zeroer_on_demand() -> None:
    """The v2 branch must recover, not assert and not skip.

    Would have caught 3ff0b7bc2a: the branch opened with
    ``assert self.kv_block_zeroer is not None`` and died on the first request
    of every DSV4 serve.
    """
    body = ast.unparse(_v2_new_block_ids_branch())

    assert "_init_kv_zero_meta" in body, (
        "the v2 runner does not build the KV block zeroer when it is missing"
    )
    assert "zero_block_ids" in body
    assert "assert self.kv_block_zeroer is not None" not in body, (
        "the v2 runner asserts on the zeroer instead of building it; that "
        "kills the worker on the first request"
    )


def test_worker_eager_init_covers_both_runners() -> None:
    """``gpu_worker`` builds the zeroer only if the runner exposes the hook,
    so both runners must expose it under the same name."""
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as V2GPUModelRunner
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as V1GPUModelRunner

    for runner_cls in (V1GPUModelRunner, V2GPUModelRunner):
        assert hasattr(runner_cls, "_init_kv_zero_meta"), runner_cls.__name__
