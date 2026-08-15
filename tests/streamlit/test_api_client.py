"""Tests for the HTTP client used by the front-end."""

from typing import Any

import pytest
import requests

from services.streamlit.clients import api_client as module
from services.streamlit.clients.api_client import ApiClient, ApiError, CallRecord


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, status_code: int, payload: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.content = b"x" if payload is not None or text else b""

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


@pytest.fixture
def calls() -> list[CallRecord]:
    """Collect the call records emitted by the client."""
    return []


def _client(monkeypatch: pytest.MonkeyPatch, response: Any, calls: list[CallRecord]) -> ApiClient:
    """Build a client whose transport always returns *response*."""

    def fake_request(method: str, url: str, **kwargs: Any) -> Any:
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(module.requests, "request", fake_request)
    return ApiClient("http://api:8000", token="jwt", recorder=calls.append)


def test_predict_builds_the_expected_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """The single prediction payload matches the API schema."""
    captured: dict[str, Any] = {}

    def fake_request(method: str, url: str, **kwargs: Any) -> FakeResponse:
        captured.update({"method": method, "url": url, **kwargs})
        return FakeResponse(200, {"results": {"label": "1560"}, "metadata": {}})

    monkeypatch.setattr(module.requests, "request", fake_request)

    client = ApiClient("http://api:8000/", token="jwt")
    client.predict("Table basse", "Chêne clair")

    assert captured["url"] == "http://api:8000/predict"
    assert captured["json"]["inputs"] == {
        "designation": "Table basse",
        "description": "Chêne clair",
    }
    assert captured["json"]["options"]["return_distribution"] is True
    assert captured["headers"]["Authorization"] == "Bearer jwt"


def test_empty_description_is_sent_as_null(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty description becomes ``None``, which the API schema allows."""
    captured: dict[str, Any] = {}

    def fake_request(method: str, url: str, **kwargs: Any) -> FakeResponse:
        captured.update(kwargs)
        return FakeResponse(200, {"results": {}, "metadata": {}})

    monkeypatch.setattr(module.requests, "request", fake_request)
    ApiClient("http://api:8000").predict("Figurine", "")

    assert captured["json"]["inputs"]["description"] is None


def test_login_returns_the_token(monkeypatch: pytest.MonkeyPatch, calls: list[CallRecord]) -> None:
    """A successful login returns the bearer token."""
    client = _client(monkeypatch, FakeResponse(200, {"access_token": "abc"}), calls)
    assert client.login("admin", "changeme") == "abc"


def test_error_status_raises_with_the_api_detail(
    monkeypatch: pytest.MonkeyPatch, calls: list[CallRecord]
) -> None:
    """The message shown in the UI carries the detail returned by the API."""
    client = _client(monkeypatch, FakeResponse(401, {"detail": "Not authenticated"}), calls)

    with pytest.raises(ApiError) as excinfo:
        client.me()

    assert excinfo.value.status_code == 401
    assert "Not authenticated" in str(excinfo.value)


def test_network_failure_is_wrapped(
    monkeypatch: pytest.MonkeyPatch, calls: list[CallRecord]
) -> None:
    """A connection error becomes an ApiError instead of leaking requests' own."""
    client = _client(monkeypatch, requests.ConnectionError("refused"), calls)

    with pytest.raises(ApiError):
        client.health()

    assert calls[-1].ok is False
    assert calls[-1].status_code is None


def test_every_call_is_recorded(monkeypatch: pytest.MonkeyPatch, calls: list[CallRecord]) -> None:
    """Latency measurements feed the monitoring page."""
    client = _client(monkeypatch, FakeResponse(200, {"status": "ok"}), calls)
    client.health()

    assert len(calls) == 1
    assert calls[0].path == "/health"
    assert calls[0].method == "GET"
    assert calls[0].ok is True
    assert calls[0].elapsed_ms >= 0


def test_ready_does_not_raise_while_starting(
    monkeypatch: pytest.MonkeyPatch, calls: list[CallRecord]
) -> None:
    """A 503 on /ready means "not ready yet", not an error to display."""
    client = _client(monkeypatch, FakeResponse(503, {"model_loaded": False}), calls)
    ready, payload = client.ready()

    assert ready is False
    assert payload == {"model_loaded": False}


def test_submit_training_accepts_202(
    monkeypatch: pytest.MonkeyPatch, calls: list[CallRecord]
) -> None:
    """POST /train answers 202 Accepted, which is a success for the client."""
    client = _client(monkeypatch, FakeResponse(202, {"job_id": "abc", "status": "queued"}), calls)
    assert client.submit_training("demo")["job_id"] == "abc"
