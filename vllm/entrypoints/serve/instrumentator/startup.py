# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Early startup probe HTTP server.

Engine initialization (NCCL communicator init, weight loading, CUDA graph
capture) can take many minutes, during which the real API server is not
serving yet and /health and /metrics would connection-refuse. The startup
probe serves both from the already-bound server socket immediately:

* ``/health`` (and any other path) returns 503 with a JSON body
  ``{"stage": ..., "detail": ..., "elapsed_s": ...}`` describing init
  progress (see ``vllm.v1.engine.health.record_startup_stage``).
* ``/metrics`` returns the prometheus registry (process metrics during
  init; engine metrics appear once registered).

The probe runs uvicorn in a daemon thread on a dup of the server socket so
that stopping it never closes the underlying socket: the real server then
takes over the same listening socket without a connection-refused window
(connections arriving in between queue in the accept backlog).
"""

import contextlib
import json
import socket
import threading
import time
from argparse import Namespace

import prometheus_client
import uvicorn

from vllm.logger import init_logger
from vllm.v1.engine.health import startup_progress_snapshot
from vllm.v1.metrics.prometheus import get_prometheus_registry

logger = init_logger(__name__)


async def _probe_app(scope, receive, send) -> None:
    """Minimal ASGI app: /metrics from the prometheus registry, everything
    else 503 with the current startup stage."""
    if scope["type"] != "http":
        raise RuntimeError(f"Unsupported ASGI scope type: {scope['type']}")

    if scope["path"].rstrip("/") == "/metrics":
        registry = get_prometheus_registry()
        body = prometheus_client.generate_latest(registry)
        status = 200
        content_type = prometheus_client.CONTENT_TYPE_LATEST.encode()
    else:
        body = json.dumps(startup_progress_snapshot()).encode()
        status = 503
        content_type = b"application/json"

    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", content_type),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class StartupProbeServer:
    """Serves the startup probe app on a dup of an already-bound socket."""

    def __init__(self, sock: socket.socket, args: Namespace | None = None):
        self._sock = sock
        self._args = args
        self._dup: socket.socket | None = None
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        ssl_kwargs = {}
        if self._args is not None and getattr(self._args, "ssl_keyfile", None):
            ssl_kwargs = dict(
                ssl_keyfile=self._args.ssl_keyfile,
                ssl_certfile=self._args.ssl_certfile,
                ssl_ca_certs=self._args.ssl_ca_certs,
                ssl_cert_reqs=self._args.ssl_cert_reqs,
                ssl_ciphers=self._args.ssl_ciphers,
            )
        config = uvicorn.Config(
            _probe_app,
            log_config=None,
            access_log=False,
            lifespan="off",
            **ssl_kwargs,
        )
        self._server = uvicorn.Server(config)
        self._dup = self._sock.dup()
        self._thread = threading.Thread(
            target=self._server.run,
            kwargs={"sockets": [self._dup]},
            daemon=True,
            name="StartupProbeServer",
        )
        self._thread.start()
        while not self._server.started and self._thread.is_alive():
            time.sleep(0.01)
        if not self._thread.is_alive():
            logger.warning("Startup probe server failed to start.")
        else:
            logger.info(
                "Startup probe serving /health and /metrics during engine "
                "initialization."
            )

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                logger.warning("Startup probe server did not stop in time.")
        if self._dup is not None:
            with contextlib.suppress(OSError):
                self._dup.close()
        self._server = None
        self._thread = None
        self._dup = None
