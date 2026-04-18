"""
Pydantic schemas for the training job API.

These schemas define the HTTP contract between the routing layer and the
training service. They are intentionally decoupled from the core training
implementation so the API can evolve independently.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Lifecycle states of a training job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainJobRequest(BaseModel):
    """Input payload for POST /train."""

    run_name: str | None = Field(
        default=None,
        description="Optional human-readable label for this training run.",
        max_length=128,
    )


class TrainJobStatus(BaseModel):
    """Full status snapshot of a training job, returned by every /train endpoint."""

    job_id: str = Field(description="Unique identifier (UUID4) of the job.")
    status: JobStatus = Field(description="Current lifecycle state.")
    run_name: str | None = Field(default=None, description="Label given at submission time.")
    created_at: datetime = Field(description="ISO 8601 timestamp of job submission.")
    started_at: datetime | None = Field(default=None, description="When execution began.")
    completed_at: datetime | None = Field(
        default=None, description="When execution finished (success or failure)."
    )
    duration_s: float | None = Field(
        default=None, description="Wall-clock training duration in seconds."
    )
    metrics: dict[str, Any] | None = Field(
        default=None,
        description="Evaluation metrics produced by the run. Populated on COMPLETED.",
    )
    error: str | None = Field(
        default=None, description="Error message. Populated on FAILED."
    )


class TrainJobListResponse(BaseModel):
    """Response schema for GET /train/jobs."""

    jobs: list[TrainJobStatus]
    total: int = Field(description="Total number of jobs returned.")
