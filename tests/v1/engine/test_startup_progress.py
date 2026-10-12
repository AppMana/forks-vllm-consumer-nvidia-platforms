# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-core init progress streaming over the startup handshake.

The engine core sends PROGRESS messages on the handshake socket while it
initializes (executor init, KV cache profiling, graph capture). The
front-end's wait loop records them so the startup probe can serve
per-stage /health during engine init.
"""

import threading

import msgspec
import zmq

from vllm.config import CacheConfig, ParallelConfig
from vllm.v1.engine.health import (
    reset_startup_progress,
    startup_progress_snapshot,
)
from vllm.v1.engine.utils import (
    CoreEngine,
    CoreEngineLaunch,
    EngineZmqAddresses,
    wait_for_engine_startup,
)


def test_wait_for_engine_startup_records_progress_messages():
    reset_startup_progress()

    ctx = zmq.Context()
    addr = "inproc://startup-progress-test"
    router = ctx.socket(zmq.ROUTER)
    router.bind(addr)

    def engine():
        dealer = ctx.socket(zmq.DEALER)
        dealer.setsockopt(zmq.IDENTITY, (0).to_bytes(2, "little"))
        dealer.connect(addr)
        dealer.setsockopt(zmq.RCVTIMEO, 5000)
        common = {"local": True, "headless": False}
        dealer.send(msgspec.msgpack.encode({"status": "HELLO", **common}))
        try:
            dealer.recv()  # EngineHandshakeMetadata
        except zmq.Again:
            dealer.close(linger=0)
            return
        dealer.send(
            msgspec.msgpack.encode(
                {
                    "status": "PROGRESS",
                    "stage": "executor_init",
                    "detail": "engine 0",
                    **common,
                }
            )
        )
        dealer.send(msgspec.msgpack.encode({"status": "READY", **common}))
        dealer.close()

    thread = threading.Thread(target=engine)
    thread.start()
    try:
        wait_for_engine_startup(
            handshake_socket=router,
            core_engines=[CoreEngine(index=0, local=True)],
            parallel_config=ParallelConfig(),
            coordinated_dp=False,
            cache_config=CacheConfig(),
            launch=CoreEngineLaunch(
                engine_manager=None,
                coordinator=None,
                tensor_queue=None,
                addresses=EngineZmqAddresses(
                    inputs=["inproc://in"], outputs=["inproc://out"]
                ),
            ),
        )
    finally:
        thread.join(timeout=10)
        router.close(linger=0)
        ctx.term()

    snap = startup_progress_snapshot()
    # READY does not clobber the last reported stage; the caller flips to
    # the post-engine stages once engine startup returns.
    assert snap["stage"] == "executor_init"
    assert snap["detail"] == "engine 0"
