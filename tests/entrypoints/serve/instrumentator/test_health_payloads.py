# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""/health payloads for stalled and dead engines (no GPU required)."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.instrumentator.health import router
from vllm.v1.engine.exceptions import EngineDeadError, EngineStalledError


class _FakeEngineClient:
    def __init__(self, exc: Exception | None = None):
        self._exc = exc

    async def check_health(self) -> None:
        if self._exc is not None:
            raise self._exc


def _client(engine_client) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.engine_client = engine_client
    return TestClient(app)


def test_health_ok():
    client = _client(_FakeEngineClient())
    response = client.get("/health")
    assert response.status_code == 200


def test_health_stalled_returns_503_with_stage():
    client = _client(
        _FakeEngineClient(EngineStalledError(stalled_for_s=61.2, num_requests=1))
    )
    response = client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["stage"] == "stalled"
    assert body["elapsed_s"] == 61.2
    assert "1 request" in body["detail"]


def test_health_dead_returns_503_with_stage():
    client = _client(_FakeEngineClient(EngineDeadError()))
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["stage"] == "dead"
