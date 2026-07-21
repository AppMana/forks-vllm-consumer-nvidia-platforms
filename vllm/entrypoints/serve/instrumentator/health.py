# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError, EngineStalledError

logger = init_logger(__name__)


router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.get("/health", response_class=Response)
async def health(raw_request: Request) -> Response:
    """Health check."""
    client = engine_client(raw_request)
    if client is None:
        # Render-only servers have no engine; they are always healthy.
        return Response(status_code=200)
    try:
        await client.check_health()
        return Response(status_code=200)
    except EngineStalledError as e:
        return JSONResponse(
            {
                "stage": "stalled",
                "detail": str(e),
                "elapsed_s": round(e.stalled_for_s, 1),
            },
            status_code=503,
        )
    except EngineDeadError as e:
        return JSONResponse({"stage": "dead", "detail": str(e)}, status_code=503)
