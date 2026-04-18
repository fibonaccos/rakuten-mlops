"""
Training job management service.

Manages the full lifecycle of asynchronous training jobs: submission, status
tracking, subprocess execution, and graceful cancellation.

Architecture — Template Method pattern
---------------------------------------
``BaseTrainingService`` contains all job-management logic (submit, get, list,
cancel, execute) and delegates only the actual "run the training" step to a
single abstract method ``_run_training(job)``.

Two concrete implementations share the same interface:
    - ``TrainingService``     — spawns a real subprocess via asyncio.
    - ``StubTrainingService`` — simulates a run with asyncio.sleep (see stub module).

Design decisions
----------------
In-memory job store
    Jobs are kept in a plain dict. This is intentional for a single-instance
    development API. A Redis or DB-backed store would replace it for multi-pod
    deployments without changing the public interface.

Subprocess isolation
    The training pipeline lives in ``core/`` and depends on keras/tensorflow,
    which are NOT installed in the ``api`` dependency group. Running training
    as a subprocess keeps the API process lean and avoids import-time side
    effects (GPU initialisation, TF logging spam, etc.).

One-at-a-time policy
    Only one job may be QUEUED or RUNNING simultaneously. This mirrors the
    reality of GPU-constrained training: launching two training runs in
    parallel would thrash memory. The 409 Conflict response is intentional.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import Settings
from ..schemas.training import JobStatus, TrainJobRequest, TrainJobStatus

logger = logging.getLogger(__name__)


class TrainingConflictError(Exception):
    """Raised when a job is submitted while another is already active."""


class TrainingJob:
    """
    Mutable state container for a single training run.

    All mutation happens inside ``BaseTrainingService._execute()``, which runs
    in an asyncio Task. Reads from the HTTP layer are safe because Python's GIL
    ensures that individual attribute reads/writes are atomic.

    Attributes:
        job_id:       UUID4 string identifying this run.
        status:       Current lifecycle state (see JobStatus).
        run_name:     Optional human-readable label supplied at submission.
        created_at:   UTC timestamp of job creation.
        started_at:   UTC timestamp when _execute() began (None until RUNNING).
        completed_at: UTC timestamp when _execute() finished (None until terminal).
        metrics:      Evaluation metrics dict. Populated on COMPLETED.
        error:        Error description string. Populated on FAILED.
        _task:        asyncio.Task handle — used internally for cancellation.
    """

    def __init__(self, job_id: str, run_name: str | None) -> None:
        self.job_id: str = job_id
        self.run_name: str | None = run_name
        self.status: JobStatus = JobStatus.QUEUED
        self.created_at: datetime = datetime.now(timezone.utc)
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None
        self.metrics: dict[str, Any] | None = None
        self.error: str | None = None
        self._task: asyncio.Task | None = None

    @property
    def duration_s(self) -> float | None:
        """Wall-clock duration in seconds, available once the job is terminal."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_status(self) -> TrainJobStatus:
        """Serialise this job into the API response schema."""
        return TrainJobStatus(
            job_id=self.job_id,
            status=self.status,
            run_name=self.run_name,
            created_at=self.created_at,
            started_at=self.started_at,
            completed_at=self.completed_at,
            duration_s=self.duration_s,
            metrics=self.metrics,
            error=self.error,
        )


class BaseTrainingService(ABC):
    """
    Abstract base class for training services.

    Provides all job-management operations. Subclasses only need to implement
    ``_run_training(job)``, which does the actual work and returns a metrics dict.

    This is the Template Method design pattern: the algorithm skeleton (submit →
    mark running → run → mark done) is fixed here; the variable step is deferred
    to subclasses.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, TrainingJob] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    async def submit(self, request: TrainJobRequest) -> TrainingJob:
        """
        Submit a new training job.

        Raises:
            TrainingConflictError: If a job is already QUEUED or RUNNING.

        Returns:
            TrainingJob: The newly created job (status QUEUED).
        """
        active = self._active_job()
        if active is not None:
            raise TrainingConflictError(
                f"Job {active.job_id} is already {active.status.value}. "
                "Wait for it to finish or cancel it before submitting a new one."
            )

        job = TrainingJob(job_id=str(uuid.uuid4()), run_name=request.run_name)
        self._jobs[job.job_id] = job
        job._task = asyncio.create_task(self._execute(job))
        logger.info("Training job %s submitted (run_name=%r)", job.job_id, job.run_name)
        return job

    def get_job(self, job_id: str) -> TrainingJob | None:
        """Return the job with the given ID, or None if not found."""
        return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 20) -> list[TrainingJob]:
        """Return the most recent jobs, newest first."""
        jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def cancel_job(self, job_id: str) -> bool:
        """
        Request cancellation of a QUEUED or RUNNING job.

        Returns:
            True if the cancellation signal was sent, False otherwise (job not
            found, already terminal, or has no associated task).
        """
        job = self._jobs.get(job_id)
        if job is None or job._task is None:
            return False
        if job.status not in (JobStatus.QUEUED, JobStatus.RUNNING):
            return False
        job._task.cancel()
        logger.info("Cancellation requested for job %s", job_id)
        return True

    # ── Internal ───────────────────────────────────────────────────────────────

    def _active_job(self) -> TrainingJob | None:
        """Return the job currently QUEUED or RUNNING, if any."""
        for job in self._jobs.values():
            if job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                return job
        return None

    async def _execute(self, job: TrainingJob) -> None:
        """
        Orchestrate job execution: lifecycle transitions + error handling.

        This is the Template Method algorithm skeleton. The actual training
        is delegated to the abstract ``_run_training`` hook.

        All state mutations are wrapped in try/except/finally so that even a
        pre-start cancellation (where Python 3.12 injects CancelledError before
        any coroutine code runs) is caught and reflected in ``job.status``.
        """
        try:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(timezone.utc)
            logger.info("Training job %s started", job.job_id)
            job.metrics = await self._run_training(job)
            job.status = JobStatus.COMPLETED
            logger.info("Training job %s completed in %.1fs", job.job_id, job.duration_s)
        except asyncio.CancelledError:
            job.status = JobStatus.CANCELLED
            logger.warning("Training job %s was cancelled", job.job_id)
            raise
        except Exception as exc:
            job.error = str(exc)
            job.status = JobStatus.FAILED
            logger.error("Training job %s failed: %s", job.job_id, exc)
        finally:
            job.completed_at = datetime.now(timezone.utc)

    @abstractmethod
    async def _run_training(self, job: TrainingJob) -> dict[str, Any]:
        """
        Execute the training pipeline and return the resulting metrics.

        This is the variable step that subclasses must implement.

        Args:
            job: The job being executed (for logging / context).

        Returns:
            dict: Evaluation metrics to store on the job.

        Raises:
            Any exception will be caught by _execute and transition the job to FAILED.
        """


class TrainingService(BaseTrainingService):
    """
    Production training service.

    Runs the training pipeline as an external subprocess so that heavy
    dependencies (keras, tensorflow, pandas) are never imported into the
    API process. The command is fully configurable via ``settings.TRAIN_COMMAND``.

    After the subprocess exits successfully, the metrics file produced by the
    training script is read and returned.
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings

    async def _run_training(self, job: TrainingJob) -> dict[str, Any]:
        logger.info(
            "Job %s — launching subprocess: %s", job.job_id, self._settings.TRAIN_COMMAND
        )
        proc = await asyncio.create_subprocess_shell(
            self._settings.TRAIN_COMMAND,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(
                f"Training subprocess exited with code {proc.returncode}. "
                f"stderr: {stderr.decode(errors='replace')[-2000:]}"
            )

        metrics_path = Path(self._settings.METRICS_PATH)
        if metrics_path.exists():
            with open(metrics_path) as f:
                return json.load(f)

        logger.warning(
            "Job %s — metrics file not found at %s", job.job_id, metrics_path
        )
        return {}
