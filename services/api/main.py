"""
FastAPI application factory.

This module creates and configures the FastAPI application instance.
It uses the lifespan context manager to load ML artifacts once at startup
and release them cleanly at shutdown, avoiding per-request loading costs.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .monitoring.health import router as health_router
from .routes.predict import router as predict_router
from .routes.train import router as train_router
from .services.predictor import ArtifactLoadError, PredictorService
from .services.stub_predictor import StubPredictorService
from .services.stub_training import StubTrainingService
from .services.training import TrainingService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan handler — manages startup and shutdown lifecycle.

    On startup:
        - Instantiates PredictorService and loads all ML artifacts from disk.
        - Stores the ready service in app.state.predictor for dependency injection.

    On shutdown:
        - Removes the predictor from app state to release memory.

    Args:
        app: The FastAPI application instance.

    Yields:
        None: Control is yielded to FastAPI to serve requests.

    Raises:
        ArtifactLoadError: Propagated if any ML artifact cannot be loaded,
            which causes the server to exit immediately with a non-zero code.
            This is intentional: a server that cannot load its model must not
            start serving traffic silently.
    """
    settings = get_settings()
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)

    if settings.STUB_MODE:
        logger.warning(
            "STUB_MODE is enabled — returning random predictions. "
            "Do NOT use in production."
        )
        predictor: PredictorService | StubPredictorService = StubPredictorService()
        predictor.load_artifacts()
    else:
        predictor = PredictorService(settings)
        try:
            predictor.load_artifacts()
        except ArtifactLoadError as exc:
            logger.critical("Cannot start: %s", exc)
            raise

    app.state.predictor = predictor
    logger.info("Application ready.")

    # Training service — same STUB_MODE flag governs real vs stub behaviour.
    if settings.STUB_MODE:
        trainer: TrainingService | StubTrainingService = StubTrainingService()
    else:
        trainer = TrainingService(settings)
    app.state.trainer = trainer
    logger.info("Training service ready (stub=%s).", settings.STUB_MODE)

    yield

    logger.info("Shutting down. Releasing resources.")

    # Cancel any active training job and wait briefly for clean termination.
    active = trainer._active_job()
    if active is not None and active._task is not None:
        active._task.cancel()
        try:
            import asyncio as _asyncio
            await _asyncio.wait_for(active._task, timeout=5.0)
        except (Exception,):
            pass

    del app.state.predictor
    del app.state.trainer


def create_app() -> FastAPI:
    """
    Application factory — create and configure the FastAPI instance.

    Using a factory function (rather than a module-level app variable) makes
    the app easier to test: each test can get a fresh instance with clean state.

    Returns:
        FastAPI: Fully configured application ready to be served by Uvicorn.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=settings.APP_DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Allow all origins in development. Restrict in production via environment
    # variable or a dedicated CORS_ORIGINS setting.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(predict_router)
    app.include_router(train_router)

    return app


# Module-level app instance used by Uvicorn:
#   uvicorn main:app --host 0.0.0.0 --port 8000
app = create_app()
