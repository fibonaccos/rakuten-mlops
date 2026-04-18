"""
Prediction router.

Exposes two endpoints:
    POST /predict        — single product classification
    POST /predict/batch  — batch product classification

Both endpoints delegate to PredictorService which is injected via
FastAPI Depends(), keeping the routing layer thin and independently testable.
"""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ..config import get_settings
from ..schemas.inference import (
    BatchPredictionInput,
    BatchPredictionOutput,
    PredictionOutputMetadata,
    SinglePredictionInput,
    SinglePredictionOutput,
)
from ..services.predictor import PredictorService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Prediction"])


def get_predictor(request: Request) -> PredictorService:
    """
    FastAPI dependency that retrieves the PredictorService from app state.

    The service is loaded once at startup via the lifespan handler and stored
    in app.state.predictor. Overriding this dependency in tests allows the
    router to be tested without any real ML artifacts.

    Args:
        request: Incoming HTTP request (FastAPI injects this automatically).

    Returns:
        PredictorService: The shared inference service.
    """
    return request.app.state.predictor


@router.post(
    "",
    response_model=SinglePredictionOutput,
    summary="Classify a single product",
    description=(
        "Returns the predicted Rakuten product category (`prdtypecode`) for a "
        "single product described by its designation and optional description."
    ),
)
async def predict_single(
    body: SinglePredictionInput,
    predictor: PredictorService = Depends(get_predictor),
) -> SinglePredictionOutput:
    """
    Classify a single Rakuten product.

    Args:
        body: Input payload with product text and optional response options.
        predictor: Injected PredictorService instance.

    Returns:
        SinglePredictionOutput: Predicted label with metadata.

    Raises:
        HTTPException 500: If the predictor raises an unexpected error.
    """
    settings = get_settings()

    start = time.perf_counter()
    try:
        result = predictor.predict(body.inputs, body.options)
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. Check server logs for details.",
        ) from exc
    elapsed_ms = (time.perf_counter() - start) * 1000

    return SinglePredictionOutput(
        results=result,
        metadata=PredictionOutputMetadata(
            model_name=predictor.model_name,
            model_version=settings.APP_VERSION,
            inference_time_ms=elapsed_ms,
        ),
    )


@router.post(
    "/batch",
    response_model=BatchPredictionOutput,
    summary="Classify a batch of products",
    description=(
        "Returns predicted Rakuten product categories for a list of products. "
        "Batch inference is more efficient than sequential single predictions "
        "as the model processes the full matrix in a single pass."
    ),
)
async def predict_batch(
    body: BatchPredictionInput,
    predictor: PredictorService = Depends(get_predictor),
) -> BatchPredictionOutput:
    """
    Classify a batch of Rakuten products.

    Args:
        body: Input payload with a list of products and optional response options.
        predictor: Injected PredictorService instance.

    Returns:
        BatchPredictionOutput: One result per input, with shared metadata.

    Raises:
        HTTPException 422: If the inputs list is empty.
        HTTPException 500: If the predictor raises an unexpected error.
    """
    if not body.inputs:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="inputs must contain at least one item.",
        )

    settings = get_settings()

    start = time.perf_counter()
    try:
        results = predictor.predict_batch(body.inputs, body.options)
    except Exception as exc:
        logger.exception("Batch prediction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch prediction failed. Check server logs for details.",
        ) from exc
    elapsed_ms = (time.perf_counter() - start) * 1000

    return BatchPredictionOutput(
        results=results,
        metadata=PredictionOutputMetadata(
            model_name=predictor.model_name,
            model_version=settings.APP_VERSION,
            inference_time_ms=elapsed_ms,
        ),
    )
