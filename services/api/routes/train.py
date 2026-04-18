"""
Training job router.

Exposes four endpoints:
    POST   /train                    — submit a new training job (202 Accepted)
    GET    /train/jobs               — list recent jobs, newest first
    GET    /train/{job_id}           — get status of a specific job
    DELETE /train/{job_id}/cancel    — request cancellation of an active job

Design notes
------------
All endpoints are thin: validation + service call + response mapping.
Business logic (job state machine, subprocess management) lives entirely in
BaseTrainingService and its subclasses.

The ``get_trainer`` dependency function follows the same pattern as
``get_predictor`` in ``routes/predict.py``: it reads ``app.state.trainer``
injected at startup. Overriding it in tests via ``app.dependency_overrides``
lets routes be tested without any real training infrastructure.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from ..schemas.training import (
    TrainJobListResponse,
    TrainJobRequest,
    TrainJobStatus,
)
from ..services.training import BaseTrainingService, TrainingConflictError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/train", tags=["Training"])


def get_trainer(request: Request) -> BaseTrainingService:
    """
    FastAPI dependency that retrieves the training service from app state.

    The service is created once at startup (lifespan) and stored in
    ``app.state.trainer``. Override this in tests via
    ``app.dependency_overrides[get_trainer]``.

    Args:
        request: Incoming HTTP request.

    Returns:
        BaseTrainingService: The shared training service instance.
    """
    return request.app.state.trainer


@router.post(
    "",
    response_model=TrainJobStatus,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a training job",
    description=(
        "Enqueues a new model training run. Returns immediately with HTTP 202 and "
        "a job ID that can be polled via GET /train/{job_id}. "
        "Only one job may be active at a time — a second submission while a job is "
        "QUEUED or RUNNING returns HTTP 409 Conflict."
    ),
)
async def submit_training(
    body: TrainJobRequest,
    trainer: BaseTrainingService = Depends(get_trainer),
) -> TrainJobStatus:
    """
    Submit a new training job.

    Args:
        body: Optional run name.
        trainer: Injected training service.

    Returns:
        TrainJobStatus: The new job in QUEUED state.

    Raises:
        HTTPException 409: If a job is already active.
    """
    try:
        job = await trainer.submit(body)
    except TrainingConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    return job.to_status()


@router.get(
    "/jobs",
    response_model=TrainJobListResponse,
    summary="List training jobs",
    description="Returns recent training jobs ordered from newest to oldest.",
)
async def list_jobs(
    limit: int = Query(default=20, ge=1, le=100, description="Maximum number of jobs to return."),
    trainer: BaseTrainingService = Depends(get_trainer),
) -> TrainJobListResponse:
    """
    List recent training jobs.

    Args:
        limit: Maximum number of jobs to return (1–100).
        trainer: Injected training service.

    Returns:
        TrainJobListResponse: Ordered list of job statuses.
    """
    jobs = trainer.list_jobs(limit=limit)
    return TrainJobListResponse(
        jobs=[j.to_status() for j in jobs],
        total=len(jobs),
    )


@router.get(
    "/{job_id}",
    response_model=TrainJobStatus,
    summary="Get training job status",
    description="Returns the current status of a specific training job.",
)
async def get_job(
    job_id: str,
    trainer: BaseTrainingService = Depends(get_trainer),
) -> TrainJobStatus:
    """
    Get the status of a training job by ID.

    Args:
        job_id: UUID of the job.
        trainer: Injected training service.

    Returns:
        TrainJobStatus: Current state of the job.

    Raises:
        HTTPException 404: If no job with that ID exists.
    """
    job = trainer.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No training job found with id '{job_id}'.",
        )
    return job.to_status()


@router.delete(
    "/{job_id}/cancel",
    response_model=TrainJobStatus,
    summary="Cancel a training job",
    description=(
        "Requests cancellation of a QUEUED or RUNNING job. "
        "The job transitions to CANCELLED asynchronously — poll GET /train/{job_id} "
        "to confirm. Returns 422 if the job is already in a terminal state."
    ),
)
async def cancel_job(
    job_id: str,
    trainer: BaseTrainingService = Depends(get_trainer),
) -> TrainJobStatus:
    """
    Cancel an active training job.

    Args:
        job_id: UUID of the job to cancel.
        trainer: Injected training service.

    Returns:
        TrainJobStatus: Job state after the cancellation signal was sent.

    Raises:
        HTTPException 404: If the job does not exist.
        HTTPException 422: If the job is already in a terminal state.
    """
    job = trainer.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No training job found with id '{job_id}'.",
        )

    cancelled = trainer.cancel_job(job_id)
    if not cancelled:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Job {job_id} is in state '{job.status.value}' and cannot be cancelled. "
                "Only QUEUED or RUNNING jobs can be cancelled."
            ),
        )
    return job.to_status()
