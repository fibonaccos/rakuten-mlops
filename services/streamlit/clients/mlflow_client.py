"""
Client for the MLflow tracking server, with a local file-store fallback.

Two situations happen in practice on this project:

* the stack runs with ``docker compose`` and the tracking server answers on
  port 5000 — the REST API is then the source of truth;
* someone ran ``core/src/models/train.py`` directly on their machine, where
  MLflow logs to the ``tracking/mlflow/mlruns`` directory.

The client covers both so the MLOps page always has something to show.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import yaml

JsonDict = dict[str, Any]


@dataclass
class MlflowRun:
    """A training run, normalised across the REST API and the file store."""

    run_id: str
    run_name: str
    experiment_id: str
    status: str
    start_time: datetime | None
    end_time: datetime | None
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def duration_s(self) -> float | None:
        """Wall-clock duration of the run, when both timestamps are known."""
        if self.start_time is None or self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()

    @property
    def git_commit(self) -> str | None:
        """Return the Git commit MLflow attached to the run, if any."""
        return self.tags.get("mlflow.source.git.commit")


def _to_datetime(value: Any) -> datetime | None:
    """Convert an MLflow epoch-milliseconds timestamp to a datetime."""
    try:
        millis = int(value)
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(millis / 1000) if millis > 0 else None


class MlflowClient:
    """Read-only client over the MLflow tracking REST API."""

    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        """
        Args:
            base_url: Root URL of the tracking server.
            timeout: Per-request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_up(self) -> bool:
        """Return True when the tracking server answers its health endpoint."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        except requests.RequestException:
            return False
        return response.ok

    def _post(self, path: str, body: JsonDict) -> JsonDict:
        """Send a POST to the MLflow REST API and return the decoded body."""
        response = requests.post(f"{self.base_url}{path}", json=body, timeout=self.timeout)
        response.raise_for_status()
        payload: Any = response.json()
        return payload if isinstance(payload, dict) else {}

    def list_experiments(self) -> list[JsonDict]:
        """
        Return the experiments registered on the server.

        Returns:
            list[JsonDict]: Raw experiment entries, or an empty list on failure.
        """
        try:
            payload = self._post("/api/2.0/mlflow/experiments/search", {"max_results": 100})
        except (requests.RequestException, ValueError):
            return []
        experiments = payload.get("experiments", [])
        return experiments if isinstance(experiments, list) else []

    def search_runs(self, experiment_ids: list[str], limit: int = 25) -> list[MlflowRun]:
        """
        Return the most recent runs of the given experiments.

        Args:
            experiment_ids: MLflow experiment identifiers.
            limit: Maximum number of runs to return.

        Returns:
            list[MlflowRun]: Runs ordered from newest to oldest.
        """
        if not experiment_ids:
            return []
        body = {
            "experiment_ids": experiment_ids,
            "max_results": limit,
            "order_by": ["attributes.start_time DESC"],
        }
        try:
            payload = self._post("/api/2.0/mlflow/runs/search", body)
        except (requests.RequestException, ValueError):
            return []

        runs: list[MlflowRun] = []
        for entry in payload.get("runs", []):
            info = entry.get("info", {})
            data = entry.get("data", {})
            tags = {t["key"]: t["value"] for t in data.get("tags", []) if "key" in t}
            runs.append(
                MlflowRun(
                    run_id=info.get("run_id", ""),
                    run_name=info.get("run_name") or tags.get("mlflow.runName", ""),
                    experiment_id=info.get("experiment_id", ""),
                    status=info.get("status", "UNKNOWN"),
                    start_time=_to_datetime(info.get("start_time")),
                    end_time=_to_datetime(info.get("end_time")),
                    metrics={
                        m["key"]: float(m["value"]) for m in data.get("metrics", []) if "key" in m
                    },
                    params={p["key"]: str(p["value"]) for p in data.get("params", []) if "key" in p},
                    tags=tags,
                )
            )
        return runs


def read_local_runs(mlruns_dir: Path, limit: int = 25) -> list[MlflowRun]:
    """
    Read runs directly from an MLflow file store.

    Used as a fallback when the tracking server is down: the directory layout
    is ``mlruns/<experiment_id>/<run_id>/{meta.yaml,metrics/,params/,tags/}``.

    Args:
        mlruns_dir: Path to the ``mlruns`` directory.
        limit: Maximum number of runs to return.

    Returns:
        list[MlflowRun]: Runs ordered from newest to oldest, empty if the
        directory does not exist or contains no readable run.
    """
    if not mlruns_dir.is_dir():
        return []

    runs: list[MlflowRun] = []
    for meta_file in mlruns_dir.glob("*/*/meta.yaml"):
        run_dir = meta_file.parent
        try:
            meta = yaml.safe_load(meta_file.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(meta, dict) or "run_id" not in meta:
            continue

        runs.append(
            MlflowRun(
                run_id=str(meta.get("run_id", run_dir.name)),
                run_name=str(meta.get("run_name") or run_dir.name),
                experiment_id=str(meta.get("experiment_id", run_dir.parent.name)),
                status=str(meta.get("status", "UNKNOWN")),
                start_time=_to_datetime(meta.get("start_time")),
                end_time=_to_datetime(meta.get("end_time")),
                metrics=_read_metric_dir(run_dir / "metrics"),
                params=_read_key_value_dir(run_dir / "params"),
                tags=_read_key_value_dir(run_dir / "tags"),
            )
        )

    runs.sort(key=lambda run: run.start_time or datetime.min, reverse=True)
    return runs[:limit]


def _read_key_value_dir(directory: Path) -> dict[str, str]:
    """Read a file-store directory where each file name is a key."""
    if not directory.is_dir():
        return {}
    values: dict[str, str] = {}
    for item in directory.iterdir():
        if item.is_file():
            try:
                values[item.name] = item.read_text(encoding="utf-8").strip()
            except OSError:
                continue
    return values


def _read_metric_dir(directory: Path) -> dict[str, float]:
    """
    Read the last recorded value of each metric in a file store.

    Each metric file holds one ``<timestamp> <value> <step>`` line per record,
    so the latest value is the last line of the file.
    """
    metrics: dict[str, float] = {}
    for name, content in _read_key_value_dir(directory).items():
        lines = [line for line in content.splitlines() if line.strip()]
        if not lines:
            continue
        parts = lines[-1].split()
        if len(parts) < 2:
            continue
        try:
            metrics[name] = float(parts[1])
        except ValueError:
            continue
    return metrics
