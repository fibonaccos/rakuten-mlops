"""
Client for the Airflow REST API (stable v1).

Airflow is protected by basic auth in the local stack, so credentials come
from the front-end settings. Every method degrades gracefully: the MLOps page
must stay readable even when the scheduler is not running.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests

JsonDict = dict[str, Any]


@dataclass
class DagSummary:
    """A DAG as listed by ``GET /api/v1/dags``."""

    dag_id: str
    description: str | None
    is_paused: bool
    schedule: str | None
    tags: list[str]


@dataclass
class DagRunSummary:
    """A DAG run as listed by ``GET /api/v1/dags/{dag_id}/dagRuns``."""

    dag_id: str
    run_id: str
    state: str
    run_type: str
    start_date: datetime | None
    end_date: datetime | None

    @property
    def duration_s(self) -> float | None:
        """Wall-clock duration of the run, when it is finished."""
        if self.start_date is None or self.end_date is None:
            return None
        return (self.end_date - self.start_date).total_seconds()


def _parse_date(value: Any) -> datetime | None:
    """Parse an ISO 8601 timestamp returned by Airflow."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class AirflowClient:
    """Read-and-trigger client over the Airflow stable REST API."""

    def __init__(
        self,
        base_url: str,
        username: str = "admin",
        password: str = "admin",
        timeout: float = 10.0,
    ) -> None:
        """
        Args:
            base_url: Root URL of the Airflow webserver.
            username: Basic auth user.
            password: Basic auth password.
            timeout: Per-request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.auth = (username, password)
        self.timeout = timeout

    def _get(self, path: str, params: JsonDict | None = None) -> JsonDict:
        """Send a GET to the Airflow API and return the decoded body."""
        response = requests.get(
            f"{self.base_url}{path}",
            params=params,
            auth=self.auth,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload: Any = response.json()
        return payload if isinstance(payload, dict) else {}

    def health(self) -> JsonDict:
        """
        Return the ``GET /health`` payload of the webserver.

        Returns:
            JsonDict: Component statuses, or an empty dict when unreachable.
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            response.raise_for_status()
            payload: Any = response.json()
        except (requests.RequestException, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def is_up(self) -> bool:
        """Return True when the metadata database and scheduler are healthy."""
        health = self.health()
        if not health:
            return False
        return health.get("metadatabase", {}).get("status") == "healthy"

    def list_dags(self) -> list[DagSummary]:
        """
        Return the DAGs known to the scheduler.

        Returns:
            list[DagSummary]: Parsed DAGs, empty when Airflow is unreachable.
        """
        try:
            payload = self._get("/api/v1/dags", {"limit": 100})
        except (requests.RequestException, ValueError):
            return []

        dags: list[DagSummary] = []
        for entry in payload.get("dags", []):
            schedule = entry.get("schedule_interval")
            if isinstance(schedule, dict):
                schedule = schedule.get("value")
            dags.append(
                DagSummary(
                    dag_id=entry.get("dag_id", ""),
                    description=entry.get("description"),
                    is_paused=bool(entry.get("is_paused", True)),
                    schedule=schedule,
                    tags=[tag.get("name", "") for tag in entry.get("tags", [])],
                )
            )
        return dags

    def list_runs(self, dag_id: str, limit: int = 10) -> list[DagRunSummary]:
        """
        Return the most recent runs of a DAG.

        Args:
            dag_id: Identifier of the DAG.
            limit: Maximum number of runs to return.

        Returns:
            list[DagRunSummary]: Runs ordered from newest to oldest.
        """
        try:
            payload = self._get(
                f"/api/v1/dags/{dag_id}/dagRuns",
                {"limit": limit, "order_by": "-start_date"},
            )
        except (requests.RequestException, ValueError):
            return []

        return [
            DagRunSummary(
                dag_id=entry.get("dag_id", dag_id),
                run_id=entry.get("dag_run_id", ""),
                state=entry.get("state", "unknown"),
                run_type=entry.get("run_type", ""),
                start_date=_parse_date(entry.get("start_date")),
                end_date=_parse_date(entry.get("end_date")),
            )
            for entry in payload.get("dag_runs", [])
        ]

    def trigger_dag(self, dag_id: str, conf: JsonDict | None = None) -> JsonDict:
        """
        Trigger a manual run of a DAG.

        Args:
            dag_id: Identifier of the DAG to trigger.
            conf: Optional run configuration passed to the DAG.

        Returns:
            JsonDict: The created DAG run.

        Raises:
            requests.RequestException: If Airflow refuses the trigger.
        """
        response = requests.post(
            f"{self.base_url}/api/v1/dags/{dag_id}/dagRuns",
            json={"conf": conf or {}},
            auth=self.auth,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload: Any = response.json()
        return payload if isinstance(payload, dict) else {}
