"""
Unit tests for the training service layer.

These tests exercise BaseTrainingService behaviour through StubTrainingService
(the concrete implementation available without ML artefacts). A fast stub
with ``simulated_duration_s=0.0`` is used so tests complete instantly.

Coverage:
    - Job submission lifecycle transitions
    - Conflict detection (one-at-a-time policy)
    - Job retrieval and listing
    - Cancellation
    - Full async execution with StubTrainingService
"""

import asyncio

import pytest

from services.api.schemas.training import JobStatus, TrainJobRequest
from services.api.services.stub_training import StubTrainingService
from services.api.services.training import TrainingConflictError


@pytest.fixture
def stub() -> StubTrainingService:
    """Fast stub with zero simulated duration so tests complete immediately."""
    return StubTrainingService(simulated_duration_s=0.0)


@pytest.fixture
def request_no_name() -> TrainJobRequest:
    return TrainJobRequest()


@pytest.fixture
def request_with_name() -> TrainJobRequest:
    return TrainJobRequest(run_name="my-run")


# ── Submission ─────────────────────────────────────────────────────────────────


async def test_submit_returns_queued_job(stub, request_no_name):
    """A freshly submitted job must be in QUEUED state."""
    job = await stub.submit(request_no_name)
    assert job.status == JobStatus.QUEUED


async def test_submit_assigns_uuid(stub, request_no_name):
    """Each submission must produce a unique job ID."""
    job1 = await stub.submit(request_no_name)
    # Wait for completion so the second submit is not blocked
    await job1._task
    job2 = await stub.submit(request_no_name)
    assert job1.job_id != job2.job_id


async def test_submit_stores_run_name(stub, request_with_name):
    """run_name supplied at submission should be preserved on the job."""
    job = await stub.submit(request_with_name)
    assert job.run_name == "my-run"


async def test_submit_conflict_while_running(stub, request_no_name):
    """Submitting a second job while one is active must raise TrainingConflictError."""
    await stub.submit(request_no_name)
    # Do not await the first task — it is still running
    with pytest.raises(TrainingConflictError):
        await stub.submit(request_no_name)


# ── Retrieval & listing ────────────────────────────────────────────────────────


async def test_get_job_returns_known_job(stub, request_no_name):
    """get_job must return the job for a known ID."""
    job = await stub.submit(request_no_name)
    found = stub.get_job(job.job_id)
    assert found is job


async def test_get_job_unknown_returns_none(stub):
    """get_job must return None for an unknown ID."""
    assert stub.get_job("does-not-exist") is None


async def test_list_jobs_empty_initially(stub):
    """list_jobs must return an empty list when no jobs have been submitted."""
    assert stub.list_jobs() == []


async def test_list_jobs_after_submit(stub, request_no_name):
    """list_jobs must include the submitted job."""
    job = await stub.submit(request_no_name)
    jobs = stub.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].job_id == job.job_id


# ── Cancellation ───────────────────────────────────────────────────────────────


async def test_cancel_unknown_job_returns_false(stub):
    """Cancelling an unknown job ID must return False without raising."""
    assert stub.cancel_job("ghost-id") is False


async def test_cancel_active_job_returns_true():
    """cancel_job transitions a RUNNING job to CANCELLED.

    Uses an asyncio.Event to guarantee the task is actually RUNNING before
    we request cancellation — avoiding the race where cancel() fires before
    _execute has run any code (Python 3.12 pre-start cancellation behaviour).
    """
    running_event = asyncio.Event()

    class _SynchronizedStub(StubTrainingService):
        async def _run_training(self, job):
            running_event.set()        # signal: we are now inside _run_training
            await asyncio.sleep(60.0)  # park here until cancelled
            return {}

    service = _SynchronizedStub()
    job = await service.submit(TrainJobRequest())
    # Wait until _run_training has definitely started (job is RUNNING).
    await asyncio.wait_for(running_event.wait(), timeout=1.0)
    assert job.status == JobStatus.RUNNING

    assert service.cancel_job(job.job_id) is True
    done, _ = await asyncio.wait({job._task}, timeout=1.0)
    assert job._task in done
    assert job.status == JobStatus.CANCELLED


# ── Full lifecycle (stub) ──────────────────────────────────────────────────────


async def test_stub_completes_with_metrics(stub, request_no_name):
    """After the async task finishes, status must be COMPLETED with metrics."""
    job = await stub.submit(request_no_name)
    await job._task  # wait for simulated run (0 s)
    assert job.status == JobStatus.COMPLETED
    assert job.metrics is not None
    assert "global" in job.metrics


async def test_stub_duration_populated_on_completion(stub, request_no_name):
    """duration_s must be a non-negative float after completion."""
    job = await stub.submit(request_no_name)
    await job._task
    assert job.duration_s is not None
    assert job.duration_s >= 0.0


async def test_to_status_reflects_job_state(stub, request_no_name):
    """to_status() must produce a TrainJobStatus matching all job fields."""
    job = await stub.submit(request_no_name)
    await job._task
    ts = job.to_status()
    assert ts.job_id == job.job_id
    assert ts.status == job.status
    assert ts.metrics == job.metrics
