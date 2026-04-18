"""
Health and readiness endpoints.

Two separate probes following Kubernetes conventions:
    GET /health  — liveness:  is the process alive?
    GET /ready   — readiness: is the process ready to serve traffic?

These endpoints must never depend on external services or the ML model
to answer the liveness probe, as a failing model should not cause
Kubernetes to restart the pod unnecessarily.
"""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Monitoring"])


@router.get("/health", summary="Liveness probe")
async def health() -> dict:
    """
    Liveness probe — confirms the process is running.

    Returns HTTP 200 as long as the server process is alive.
    Kubernetes uses this to decide whether to restart the container.

    Returns:
        dict: {"status": "ok", "version": <app_version>}
    """
    settings = get_settings()
    return {"status": "ok", "version": settings.APP_VERSION}


@router.get("/ready", summary="Readiness probe")
async def ready(request: Request) -> JSONResponse:
    """
    Readiness probe — confirms ML artifacts are loaded and ready to serve.

    Returns HTTP 200 when the PredictorService is fully initialised.
    Returns HTTP 503 during startup before artifacts finish loading.

    Kubernetes uses this to gate traffic: no requests are routed to the
    container until this probe returns 200.

    Args:
        request: FastAPI request (used to access app.state.predictor).

    Returns:
        JSONResponse: 200 if ready, 503 otherwise.
    """
    predictor = getattr(request.app.state, "predictor", None)
    model_loaded = predictor is not None and predictor.is_loaded

    if model_loaded:
        return JSONResponse(
            status_code=200,
            content={"status": "ready", "model_loaded": True},
        )
    return JSONResponse(
        status_code=503,
        content={"status": "starting", "model_loaded": False},
    )
