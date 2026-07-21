# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Early startup probe server.

During engine initialization (NCCL init, weight load, CUDA graph capture)
the real API server is not up yet, historically leaving /health and /metrics
connection-refused for many minutes. The startup probe binds the server
socket immediately and serves 503 with init-stage detail until the real
server takes over.
"""

import socket

import pytest
import requests

from vllm.entrypoints.serve.instrumentator.startup import StartupProbeServer
from vllm.v1.engine.health import (
    record_startup_stage,
    reset_startup_progress,
    startup_progress_snapshot,
)


@pytest.fixture
def server_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    yield sock
    sock.close()


def _get(port: int, path: str) -> requests.Response:
    return requests.get(f"http://127.0.0.1:{port}{path}", timeout=5)


def test_startup_progress_snapshot():
    reset_startup_progress()
    snap = startup_progress_snapshot()
    assert snap["stage"] == "starting"
    assert snap["elapsed_s"] >= 0

    record_startup_stage("weight_load", "shard 3/12")
    snap = startup_progress_snapshot()
    assert snap["stage"] == "weight_load"
    assert snap["detail"] == "shard 3/12"


def test_probe_serves_health_and_metrics_during_init(server_socket):
    reset_startup_progress()
    record_startup_stage("engine_core_launch")
    port = server_socket.getsockname()[1]

    probe = StartupProbeServer(server_socket)
    probe.start()
    try:
        response = _get(port, "/health")
        assert response.status_code == 503
        body = response.json()
        assert body["stage"] == "engine_core_launch"
        assert body["elapsed_s"] >= 0

        # Stage updates are visible immediately.
        record_startup_stage("graph_capture", "capturing CUDA graphs")
        body = _get(port, "/health").json()
        assert body["stage"] == "graph_capture"
        assert body["detail"] == "capturing CUDA graphs"

        # /metrics serves the prometheus registry even before engine init.
        metrics = _get(port, "/metrics")
        assert metrics.status_code == 200
        assert "text/plain" in metrics.headers["content-type"]

        # Every other path also reports initializing rather than 404.
        other = _get(port, "/v1/models")
        assert other.status_code == 503
        assert other.json()["stage"] == "graph_capture"
    finally:
        probe.stop()


def test_probe_releases_socket_for_real_server(server_socket):
    """After stop() the underlying socket must remain usable, so the real
    uvicorn server can take over the same address (no connection refused)."""
    reset_startup_progress()
    port = server_socket.getsockname()[1]

    probe = StartupProbeServer(server_socket)
    probe.start()
    assert _get(port, "/health").status_code == 503
    probe.stop()

    # Simulate the real server taking over the same socket.
    takeover = StartupProbeServer(server_socket)
    takeover.start()
    try:
        assert _get(port, "/health").status_code == 503
    finally:
        takeover.stop()
