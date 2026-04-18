"""
Tests for the /predict and /predict/batch HTTP endpoints.

These tests validate HTTP-level behaviour: status codes, response schemas,
query validation, and error handling. The PredictorService is mocked via
FastAPI dependency override so tests run without any ML artifacts.
"""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from services.api.main import app
from services.api.routes.predict import get_predictor
from services.api.schemas.inference import (
    BasePredictionOutput,
    PredictionInputOptions,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_predictor_mock(
    label: str = "40",
    confidence: float | None = None,
    distribution: dict[str, float] | None = None,
) -> MagicMock:
    """Build a PredictorService mock with configurable return values."""
    mock = MagicMock()
    mock.is_loaded = True
    mock.model_name = "model.keras"
    mock.predict.return_value = BasePredictionOutput(
        label=label,
        confidence=confidence,
        distribution=distribution,
    )
    mock.predict_batch.return_value = [
        BasePredictionOutput(label=label, confidence=confidence, distribution=distribution)
    ]
    return mock


@pytest.fixture(autouse=True)
def override_predictor():
    """
    Override the get_predictor dependency for every test in this module.

    autouse=True means this fixture is applied automatically to all tests
    without needing to declare it explicitly.
    """
    mock = _make_predictor_mock()
    app.dependency_overrides[get_predictor] = lambda: mock
    yield mock
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def valid_single_payload() -> dict:
    return {
        "inputs": {
            "designation": "Livre de cuisine francaise",
            "description": "Recettes traditionnelles",
        }
    }


@pytest.fixture
def valid_batch_payload() -> dict:
    return {
        "inputs": [
            {"designation": "Livre de cuisine", "description": "Recettes"},
            {"designation": "Smartphone Samsung", "description": None},
        ]
    }


# ── POST /predict ─────────────────────────────────────────────────────────────


class TestPredictSingle:
    def test_returns_200(self, client: TestClient, valid_single_payload: dict) -> None:
        response = client.post("/predict", json=valid_single_payload)
        assert response.status_code == 200

    def test_response_contains_label(
        self, client: TestClient, valid_single_payload: dict
    ) -> None:
        response = client.post("/predict", json=valid_single_payload)
        body = response.json()
        assert "results" in body
        assert "label" in body["results"]

    def test_response_contains_metadata(
        self, client: TestClient, valid_single_payload: dict
    ) -> None:
        response = client.post("/predict", json=valid_single_payload)
        body = response.json()
        assert "metadata" in body
        assert "model_name" in body["metadata"]
        assert "inference_time_ms" in body["metadata"]
        assert "timestamp" in body["metadata"]

    def test_confidence_absent_by_default(
        self, client: TestClient, valid_single_payload: dict
    ) -> None:
        response = client.post("/predict", json=valid_single_payload)
        assert response.json()["results"]["confidence"] is None

    def test_confidence_present_when_requested(
        self, client: TestClient, override_predictor: MagicMock
    ) -> None:
        override_predictor.predict.return_value = BasePredictionOutput(
            label="40", confidence=0.87
        )
        payload = {
            "inputs": {"designation": "Test product"},
            "options": {"return_confidence": True},
        }
        response = client.post("/predict", json=payload)
        assert response.json()["results"]["confidence"] == pytest.approx(0.87)

    def test_distribution_present_when_requested(
        self, client: TestClient, override_predictor: MagicMock
    ) -> None:
        override_predictor.predict.return_value = BasePredictionOutput(
            label="40",
            distribution={"10": 0.2, "40": 0.8},
        )
        payload = {
            "inputs": {"designation": "Test product"},
            "options": {"return_distribution": True},
        }
        response = client.post("/predict", json=payload)
        dist = response.json()["results"]["distribution"]
        assert dist is not None
        assert "40" in dist

    def test_missing_designation_returns_422(self, client: TestClient) -> None:
        response = client.post("/predict", json={"inputs": {"description": "only desc"}})
        assert response.status_code == 422

    def test_empty_body_returns_422(self, client: TestClient) -> None:
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_description_is_optional(self, client: TestClient) -> None:
        payload = {"inputs": {"designation": "Only designation"}}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_service_error_returns_500(
        self, client: TestClient, override_predictor: MagicMock, valid_single_payload: dict
    ) -> None:
        override_predictor.predict.side_effect = RuntimeError("Model failure")
        response = client.post("/predict", json=valid_single_payload)
        assert response.status_code == 500


# ── POST /predict/batch ───────────────────────────────────────────────────────


class TestPredictBatch:
    def test_returns_200(self, client: TestClient, valid_batch_payload: dict) -> None:
        response = client.post("/predict/batch", json=valid_batch_payload)
        assert response.status_code == 200

    def test_response_is_list(
        self, client: TestClient, override_predictor: MagicMock, valid_batch_payload: dict
    ) -> None:
        override_predictor.predict_batch.return_value = [
            BasePredictionOutput(label="40"),
            BasePredictionOutput(label="10"),
        ]
        response = client.post("/predict/batch", json=valid_batch_payload)
        body = response.json()
        assert "results" in body
        assert isinstance(body["results"], list)
        assert len(body["results"]) == 2

    def test_empty_inputs_returns_422(self, client: TestClient) -> None:
        response = client.post("/predict/batch", json={"inputs": []})
        assert response.status_code == 422

    def test_metadata_present(
        self, client: TestClient, valid_batch_payload: dict
    ) -> None:
        response = client.post("/predict/batch", json=valid_batch_payload)
        assert "metadata" in response.json()
