"""
Stub training service for development and CI.

``StubTrainingService`` implements the exact same interface as
``TrainingService`` but never spawns a subprocess or touches the file system.
Instead it sleeps for a configurable duration and returns a pre-built metrics
dict that mirrors the shape of a real evaluation report.

Usage
-----
Selected automatically when ``STUB_MODE=true`` in ``.env`` (set by default in
the development environment). The duration can be shortened in tests by passing
``simulated_duration_s=0.0`` to the constructor.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .training import BaseTrainingService, TrainingJob

_FAKE_METRICS: dict[str, Any] = {
    "10": {"accuracy": 0.97, "precision": 0.69, "recall": 0.26, "f1-score": 0.38, "auc": 0.96, "support": 623},
    "40": {"accuracy": 0.98, "precision": 0.74, "recall": 0.39, "f1-score": 0.51, "auc": 0.97, "support": 502},
    "50": {"accuracy": 0.99, "precision": 0.76, "recall": 0.60, "f1-score": 0.67, "auc": 0.98, "support": 336},
    "global": {
        "accuracy": 0.74,
        "precision_macro": 0.73,
        "precision_weighted": 0.74,
        "recall_macro": 0.70,
        "recall_weighted": 0.74,
        "f1_macro": 0.71,
        "f1_weighted": 0.74,
        "auc_macro": 0.977,
        "auc_weighted": 0.978,
        "support": 16984,
    },
}


class StubTrainingService(BaseTrainingService):
    """
    Drop-in development stub for ``TrainingService``.

    Simulates the full async lifecycle (QUEUED → RUNNING → COMPLETED) without
    any real computation. Useful for:
      - Local development without ML artefacts or a GPU.
      - CI pipelines that must test the HTTP layer only.
      - Demonstrating the API without triggering an expensive training run.

    The ``simulated_duration_s`` parameter makes it possible to keep tests fast
    while still exercising the async task machinery.
    """

    def __init__(self, simulated_duration_s: float = 3.0) -> None:
        super().__init__()
        self._simulated_duration_s = simulated_duration_s

    async def _run_training(self, job: TrainingJob) -> dict[str, Any]:
        """Simulate a training run by sleeping, then return fake metrics."""
        await asyncio.sleep(self._simulated_duration_s)
        return _FAKE_METRICS
