"""
Client for the Rakuten prediction API.

The front-end never imports the FastAPI application: it only speaks HTTP, the
same way any other consumer would. That keeps the demo honest — what the jury
sees on screen is exactly what the API returns over the network.

Every call is timed and pushed to an optional recorder, which feeds the
"Santé & monitoring" page with real latency measurements.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests

JsonDict = dict[str, Any]


class ApiError(RuntimeError):
    """Raised when the API is unreachable or answers with an error status."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        """
        Args:
            message: Human-readable explanation, displayed as-is in the UI.
            status_code: HTTP status returned by the API, when there is one.
        """
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class CallRecord:
    """A single HTTP call, kept in session state to build latency charts."""

    method: str
    path: str
    status_code: int | None
    elapsed_ms: float
    at: datetime
    ok: bool


Recorder = Callable[[CallRecord], None]


class ApiClient:
    """
    Thin synchronous wrapper around the API endpoints used by the front-end.

    The client is intentionally stateless apart from the bearer token: a new
    instance is built on every Streamlit rerun, which keeps the UI in sync with
    the sidebar settings without any cache invalidation logic.
    """

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        timeout: float = 30.0,
        recorder: Recorder | None = None,
    ) -> None:
        """
        Args:
            base_url: Root URL of the API, e.g. ``http://localhost:8000``.
            token: JWT access token obtained from ``POST /auth/login``.
            timeout: Per-request timeout in seconds.
            recorder: Optional callback invoked once per call with a CallRecord.
        """
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._recorder = recorder

    # ── Plumbing ──────────────────────────────────────────────────────────────

    @property
    def is_authenticated(self) -> bool:
        """Return True when a bearer token is available."""
        return bool(self.token)

    def _headers(self, extra: JsonDict | None = None) -> dict[str, str]:
        """Build request headers, adding the bearer token when present."""
        headers: dict[str, str] = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if extra:
            headers.update({k: str(v) for k, v in extra.items()})
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        data: JsonDict | None = None,
        params: JsonDict | None = None,
        expected: Sequence[int] = (200,),
    ) -> JsonDict:
        """
        Send a request, record its latency and return the decoded JSON body.

        Args:
            method: HTTP verb.
            path: Path starting with a slash, e.g. ``/predict``.
            json_body: Payload serialised as JSON.
            data: Payload serialised as a form (used by the login endpoint).
            params: Query string parameters.
            expected: Status codes considered successful.

        Returns:
            JsonDict: Decoded response body.

        Raises:
            ApiError: On network failure or unexpected status code.
        """
        url = f"{self.base_url}{path}"
        started = time.perf_counter()
        status: int | None = None
        try:
            response = requests.request(
                method,
                url,
                json=json_body,
                data=data,
                params=params,
                headers=self._headers(),
                timeout=self.timeout,
            )
            status = response.status_code
        except requests.RequestException as exc:
            self._record(method, path, None, started, ok=False)
            raise ApiError(f"API injoignable sur {url} ({exc.__class__.__name__}).") from exc

        ok = status in expected
        self._record(method, path, status, started, ok=ok)

        if not ok:
            raise ApiError(self._error_message(response), status_code=status)

        if not response.content:
            return {}
        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise ApiError(f"Réponse non JSON depuis {url}.", status_code=status) from exc
        return payload if isinstance(payload, dict) else {"data": payload}

    @staticmethod
    def _error_message(response: requests.Response) -> str:
        """Extract the most useful error message from an API response."""
        try:
            body = response.json()
        except ValueError:
            return f"HTTP {response.status_code} — {response.text[:200]}"
        detail = body.get("detail") if isinstance(body, dict) else None
        if isinstance(detail, list) and detail:
            detail = detail[0].get("msg", detail[0]) if isinstance(detail[0], dict) else detail[0]
        return f"HTTP {response.status_code} — {detail or response.text[:200]}"

    def _record(
        self,
        method: str,
        path: str,
        status: int | None,
        started: float,
        *,
        ok: bool,
    ) -> None:
        """Push a CallRecord to the recorder, if one was provided."""
        if self._recorder is None:
            return
        self._recorder(
            CallRecord(
                method=method,
                path=path,
                status_code=status,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                at=datetime.now(),
                ok=ok,
            )
        )

    # ── Monitoring ────────────────────────────────────────────────────────────

    def health(self) -> JsonDict:
        """Return the liveness payload of ``GET /health``."""
        return self._request("GET", "/health")

    def ready(self) -> tuple[bool, JsonDict]:
        """
        Probe ``GET /ready`` without raising when the API is still starting.

        Returns:
            tuple[bool, JsonDict]: Readiness flag and the raw payload.
        """
        try:
            payload = self._request("GET", "/ready", expected=(200, 503))
        except ApiError:
            return False, {}
        return bool(payload.get("model_loaded")), payload

    def openapi(self) -> JsonDict:
        """Return the OpenAPI schema, used to list the exposed routes."""
        return self._request("GET", "/openapi.json")

    # ── Authentication ────────────────────────────────────────────────────────

    def login(self, username: str, password: str) -> str:
        """
        Exchange credentials for a JWT access token.

        Args:
            username: Account name.
            password: Plain-text password, sent as form data (OAuth2 password flow).

        Returns:
            str: The bearer token.

        Raises:
            ApiError: If the credentials are rejected.
        """
        payload = self._request(
            "POST",
            "/auth/login",
            data={"username": username, "password": password},
        )
        token = payload.get("access_token")
        if not isinstance(token, str):
            raise ApiError("Réponse de login inattendue : aucun access_token.")
        return token

    def me(self) -> JsonDict:
        """Return the profile of the authenticated user (``GET /auth/me``)."""
        return self._request("GET", "/auth/me")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        designation: str,
        description: str | None = None,
        *,
        with_confidence: bool = True,
        with_distribution: bool = True,
    ) -> JsonDict:
        """
        Classify a single product.

        Args:
            designation: Product title.
            description: Optional longer description.
            with_confidence: Ask the API for the probability of the winning class.
            with_distribution: Ask the API for the full probability distribution.

        Returns:
            JsonDict: Body of ``POST /predict``.
        """
        body = {
            "inputs": {"designation": designation, "description": description or None},
            "options": {
                "return_confidence": with_confidence,
                "return_distribution": with_distribution,
            },
        }
        return self._request("POST", "/predict", json_body=body)

    def predict_batch(
        self,
        items: Sequence[JsonDict],
        *,
        with_confidence: bool = True,
        with_distribution: bool = False,
    ) -> JsonDict:
        """
        Classify a list of products in a single call.

        Args:
            items: Dicts with a ``designation`` key and an optional ``description``.
            with_confidence: Include the winning probability in each result.
            with_distribution: Include the full distribution in each result.

        Returns:
            JsonDict: Body of ``POST /predict/batch``.
        """
        body = {
            "inputs": [
                {
                    "designation": str(item.get("designation", "")),
                    "description": item.get("description") or None,
                }
                for item in items
            ],
            "options": {
                "return_confidence": with_confidence,
                "return_distribution": with_distribution,
            },
        }
        return self._request("POST", "/predict/batch", json_body=body)

    # ── Training jobs ─────────────────────────────────────────────────────────

    def submit_training(self, run_name: str | None = None) -> JsonDict:
        """Submit a training job (``POST /train``) and return its initial status."""
        return self._request(
            "POST",
            "/train",
            json_body={"run_name": run_name} if run_name else {},
            expected=(202,),
        )

    def list_jobs(self, limit: int = 20) -> list[JsonDict]:
        """Return recent training jobs, newest first."""
        payload = self._request("GET", "/train/jobs", params={"limit": limit})
        jobs = payload.get("jobs", [])
        return jobs if isinstance(jobs, list) else []

    def get_job(self, job_id: str) -> JsonDict:
        """Return the current status of one training job."""
        return self._request("GET", f"/train/{job_id}")

    def cancel_job(self, job_id: str) -> JsonDict:
        """Request cancellation of an active training job."""
        return self._request("DELETE", f"/train/{job_id}/cancel")
