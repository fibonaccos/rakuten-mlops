"""
Tests for the /train HTTP endpoints.

These tests validate HTTP-level behaviour: status codes, response schemas,
and error handling. The BaseTrainingService is replaced by an AsyncMock so
tests run without any real training infrastructure or asyncio task overhead.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from services.api.main import app
from services.api.routes.train import get_trainer
from services.api.schemas.training import JobStatus, TrainJobStatus
from services.api.services.training import TrainingConflictError, TrainingJob


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_job(
    job_id: str = "test-job-id",
    status: JobStatus = JobStatus.QUEUED,
    run_name: str | None = None,
    metrics: dict | None = None,
    error: str | None = None,
) -> TrainingJob:
    """Build a TrainingJob with predetermined values for assertions."""
    job = TrainingJob(job_id=job_id, run_name=run_name)
    job.status = status
    job.metrics = metrics
    job.error = error
    return job


def _make_trainer_mock(job: TrainingJob | None = None) -> MagicMock:
    """Build a MagicMock trainer that returns the given job from all reads."""
    if job is None:
        job = _make_job()
    mock = MagicMock()
    mock.submit = AsyncMock(return_value=job)
    mock.get_job.return_value = job
    mock.list_jobs.return_value = [job]
    mock.cancel_job.return_value = True
    return mock


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def override_trainer():
    """
    Replace get_trainer with a mock for every test in this module.

    autouse=True means every test gets a clean override automatically.
    The fixture tears down the override after each test.
    """
    mock = _make_trainer_mock()
    app.dependency_overrides[get_trainer] = lambda: mock
    yield mock
    app.dependency_overrides.pop(get_trainer, None)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


# ── POST /train ────────────────────────────────────────────────────────────────


def test_submit_returns_202(client):
    """POST /train must return 202 Accepted."""
    resp = client.post("/train", json={})
    assert resp.status_code == 202


def test_submit_response_has_job_id(client):
    """Response body must contain a job_id field."""
    resp = client.post("/train", json={})
    assert "job_id" in resp.json()


def test_submit_with_run_name(client, override_trainer):
    """run_name in the request body must be forwarded to the service."""
    job = _make_job(run_name="ci-run")
    override_trainer.submit = AsyncMock(return_value=job)
    resp = client.post("/train", json={"run_name": "ci-run"})
    assert resp.status_code == 202
    assert resp.json()["run_name"] == "ci-run"


def test_submit_conflict_returns_409(client, override_trainer):
    """When the service raises TrainingConflictError, return 409 Conflict."""
    override_trainer.submit = AsyncMock(
        side_effect=TrainingConflictError("already running")
    )
    resp = client.post("/train", json={})
    assert resp.status_code == 409


def test_submit_status_is_queued(client):
    """The status field of a fresh submission must be 'queued'."""
    resp = client.post("/train", json={})
    assert resp.json()["status"] == "queued"


# ── GET /train/jobs ────────────────────────────────────────────────────────────


def test_list_jobs_returns_200(client):
    """GET /train/jobs must return 200."""
    resp = client.get("/train/jobs")
    assert resp.status_code == 200


def test_list_jobs_response_shape(client):
    """Response must have 'jobs' list and 'total' integer."""
    resp = client.get("/train/jobs")
    body = resp.json()
    assert "jobs" in body
    assert "total" in body
    assert isinstance(body["jobs"], list)


def test_list_jobs_limit_param(client, override_trainer):
    """The limit query param must be forwarded to the service."""
    client.get("/train/jobs?limit=5")
    override_trainer.list_jobs.assert_called_once_with(limit=5)


# ── GET /train/{job_id} ────────────────────────────────────────────────────────


def test_get_job_returns_200(client):
    """GET /train/{job_id} for a known job must return 200."""
    resp = client.get("/train/test-job-id")
    assert resp.status_code == 200


def test_get_job_response_has_correct_id(client):
    """Response job_id must match the requested ID."""
    resp = client.get("/train/test-job-id")
    assert resp.json()["job_id"] == "test-job-id"


def test_get_job_unknown_returns_404(client, override_trainer):
    """GET /train/{unknown_id} must return 404 when service returns None."""
    override_trainer.get_job.return_value = None
    resp = client.get("/train/does-not-exist")
    assert resp.status_code == 404


# ── DELETE /train/{job_id}/cancel ──────────────────────────────────────────────


def test_cancel_job_returns_200(client):
    """DELETE /train/{job_id}/cancel for a cancellable job must return 200."""
    resp = client.delete("/train/test-job-id/cancel")
    assert resp.status_code == 200


def test_cancel_unknown_job_returns_404(client, override_trainer):
    """Cancelling an unknown job must return 404."""
    override_trainer.get_job.return_value = None
    resp = client.delete("/train/ghost/cancel")
    assert resp.status_code == 404


def test_cancel_terminal_job_returns_422(client, override_trainer):
    """Cancelling a COMPLETED job must return 422 (cannot cancel terminal jobs)."""
    completed_job = _make_job(status=JobStatus.COMPLETED)
    override_trainer.get_job.return_value = completed_job
    override_trainer.cancel_job.return_value = False
    resp = client.delete("/train/test-job-id/cancel")
    assert resp.status_code == 422


# ── Completed job response ─────────────────────────────────────────────────────


def test_completed_job_includes_metrics(client, override_trainer):
    """A COMPLETED job's response must include the metrics dict."""
    completed = _make_job(
        status=JobStatus.COMPLETED,
        metrics={"global": {"accuracy": 0.74}},
    )
    override_trainer.get_job.return_value = completed
    resp = client.get("/train/test-job-id")
    assert resp.json()["metrics"] == {"global": {"accuracy": 0.74}}


def test_failed_job_includes_error(client, override_trainer):
    """A FAILED job's response must include the error string."""
    failed = _make_job(status=JobStatus.FAILED, error="OOM on epoch 3")
    override_trainer.get_job.return_value = failed
    resp = client.get("/train/test-job-id")
    assert resp.json()["error"] == "OOM on epoch 3"
