"""
Tests for PredictorService.

These tests cover the feature engineering pipeline and prediction logic
in isolation, without loading real ML artifacts. Heavy dependencies
(Keras, SentenceTransformer, joblib) are mocked so the suite runs fast
in any environment, including CI without GPU or large model downloads.
"""

import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from services.api.config import Settings
from services.api.schemas.inference import BasePredictionInput, PredictionInputOptions
from services.api.services.predictor import ArtifactLoadError, PredictorService


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_keras_mock(proba_output: np.ndarray) -> MagicMock:
    """Return a fake keras module that behaves like the real one for our use-case."""
    mock_keras = MagicMock()
    mock_model = MagicMock()
    mock_model.predict.return_value = proba_output
    mock_keras.models.load_model.return_value = mock_model
    return mock_keras, mock_model


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def settings() -> Settings:
    """Minimal Settings with fake artifact paths."""
    return Settings(
        MODEL_PATH="fake/model.keras",
        SCALER_PATH="fake/scaler.joblib",
        PCA_PATH="fake/pca.joblib",
        LABELS_MAP_PATH="fake/labels_map.json",
        EMBEDDER_MODEL_NAME="paraphrase-multilingual-MiniLM-L12-v2",
        EMBED_CHUNK_SIZE=150,
        EMBED_OVERLAP=30,
    )


@pytest.fixture
def loaded_predictor(settings: Settings) -> PredictorService:
    """
    PredictorService with all heavy dependencies mocked.

    Keras is injected into sys.modules so 'import keras as ks' inside
    load_artifacts() resolves to our mock without needing keras installed.
    SentenceTransformer and joblib are patched at the module level.
    """
    proba = np.array([[0.2, 0.8]])
    mock_keras, mock_model = _make_keras_mock(proba)

    mock_scaler = MagicMock()
    mock_scaler.transform.side_effect = lambda x: x  # identity

    mock_pca = MagicMock()
    mock_pca.transform.side_effect = lambda x: x  # identity

    mock_embedder = MagicMock()
    # encode() must return a 2-D array: (n_chunks, embedding_dim=384)
    mock_embedder.encode.return_value = np.zeros((1, 384))

    predictor = PredictorService(settings)

    with (
        patch.dict(sys.modules, {"keras": mock_keras, "keras.models": mock_keras.models}),
        patch("services.api.services.predictor.joblib.load", side_effect=[mock_scaler, mock_pca]),
        patch("services.api.services.predictor.SentenceTransformer", return_value=mock_embedder),
        patch("builtins.open", MagicMock()),
        patch("json.load", return_value={"10": 0, "40": 1}),
    ):
        predictor.load_artifacts()

    # Re-inject mocks directly after load so tests can configure return values.
    predictor._model = mock_model
    predictor._scaler = mock_scaler
    predictor._pca = mock_pca
    predictor._labels_map = {0: "10", 1: "40"}
    predictor._embedder = mock_embedder

    return predictor


@pytest.fixture
def single_input() -> BasePredictionInput:
    return BasePredictionInput(
        designation="Livre de cuisine francaise",
        description="Recettes traditionnelles et modernes",
    )


@pytest.fixture
def minimal_input() -> BasePredictionInput:
    """Input with no description (optional field)."""
    return BasePredictionInput(designation="Smartphone Samsung Galaxy")


# ── _compute_stats ────────────────────────────────────────────────────────────


class TestComputeStats:
    def test_returns_12_features(self, loaded_predictor: PredictorService) -> None:
        stats = loaded_predictor._compute_stats("hello world", "desc text")
        assert stats.shape == (12,)

    def test_character_count(self, loaded_predictor: PredictorService) -> None:
        stats = loaded_predictor._compute_stats("hello", "")
        assert stats[0] == 5.0  # len("hello")

    def test_word_count(self, loaded_predictor: PredictorService) -> None:
        stats = loaded_predictor._compute_stats("one two three", "")
        assert stats[1] == 3.0

    def test_empty_description_yields_zeros(
        self, loaded_predictor: PredictorService
    ) -> None:
        stats = loaded_predictor._compute_stats("hello world", "")
        assert stats[6] == 0.0  # length of ""
        assert stats[7] == 0.0  # word count of ""

    def test_numeric_count(self, loaded_predictor: PredictorService) -> None:
        stats = loaded_predictor._compute_stats("abc123", "")
        assert stats[4] == 3.0  # 3 digits

    def test_punctuation_count(self, loaded_predictor: PredictorService) -> None:
        stats = loaded_predictor._compute_stats("hello, world!", "")
        assert stats[5] == 2.0  # comma + exclamation


# ── _chunk_text ───────────────────────────────────────────────────────────────


class TestChunkText:
    def test_short_text_not_chunked(self, loaded_predictor: PredictorService) -> None:
        assert loaded_predictor._chunk_text("short text") == ["short text"]

    def test_empty_text_returns_placeholder(
        self, loaded_predictor: PredictorService
    ) -> None:
        assert loaded_predictor._chunk_text("") == [""]

    def test_long_text_is_chunked(self, loaded_predictor: PredictorService) -> None:
        long_text = " ".join(["word"] * 200)
        assert len(loaded_predictor._chunk_text(long_text)) > 1

    def test_chunk_size_respected(self, loaded_predictor: PredictorService) -> None:
        long_text = " ".join([f"w{i}" for i in range(300)])
        for chunk in loaded_predictor._chunk_text(long_text):
            assert len(chunk.split()) <= 150


# ── _build_feature_vector ─────────────────────────────────────────────────────


class TestBuildFeatureVector:
    def test_output_shape(
        self,
        loaded_predictor: PredictorService,
        single_input: BasePredictionInput,
    ) -> None:
        # 12 stat features + 384 embedding dims = 396, shaped (1, 396)
        vec = loaded_predictor._build_feature_vector(single_input)
        assert vec.shape == (1, 396)

    def test_missing_description_handled(
        self,
        loaded_predictor: PredictorService,
        minimal_input: BasePredictionInput,
    ) -> None:
        vec = loaded_predictor._build_feature_vector(minimal_input)
        assert vec.shape == (1, 396)


# ── predict ───────────────────────────────────────────────────────────────────


class TestPredict:
    def test_returns_label(
        self,
        loaded_predictor: PredictorService,
        single_input: BasePredictionInput,
    ) -> None:
        # mock returns [0.2, 0.8] -> argmax=1 -> label "40"
        loaded_predictor._model.predict.return_value = np.array([[0.2, 0.8]])
        result = loaded_predictor.predict(single_input)
        assert result.label == "40"

    def test_confidence_none_by_default(
        self,
        loaded_predictor: PredictorService,
        single_input: BasePredictionInput,
    ) -> None:
        loaded_predictor._model.predict.return_value = np.array([[0.2, 0.8]])
        assert loaded_predictor.predict(single_input).confidence is None

    def test_confidence_returned_when_requested(
        self,
        loaded_predictor: PredictorService,
        single_input: BasePredictionInput,
    ) -> None:
        loaded_predictor._model.predict.return_value = np.array([[0.2, 0.8]])
        options = PredictionInputOptions(return_confidence=True)
        result = loaded_predictor.predict(single_input, options)
        assert result.confidence == pytest.approx(0.8)

    def test_distribution_returned_when_requested(
        self,
        loaded_predictor: PredictorService,
        single_input: BasePredictionInput,
    ) -> None:
        loaded_predictor._model.predict.return_value = np.array([[0.2, 0.8]])
        options = PredictionInputOptions(return_distribution=True)
        result = loaded_predictor.predict(single_input, options)
        assert result.distribution is not None
        assert set(result.distribution.keys()) == {"10", "40"}
        assert sum(result.distribution.values()) == pytest.approx(1.0)

    def test_raises_if_not_loaded(self, settings: Settings) -> None:
        predictor = PredictorService(settings)
        with pytest.raises(RuntimeError, match="Artifacts not loaded"):
            predictor.predict(BasePredictionInput(designation="test"))


# ── predict_batch ─────────────────────────────────────────────────────────────


class TestPredictBatch:
    def test_returns_one_result_per_input(
        self, loaded_predictor: PredictorService
    ) -> None:
        loaded_predictor._model.predict.return_value = np.array(
            [[0.2, 0.8], [0.9, 0.1]]
        )
        inputs = [
            BasePredictionInput(designation="Product A"),
            BasePredictionInput(designation="Product B"),
        ]
        assert len(loaded_predictor.predict_batch(inputs)) == 2

    def test_batch_labels_match_argmax(
        self, loaded_predictor: PredictorService
    ) -> None:
        loaded_predictor._model.predict.return_value = np.array(
            [[0.2, 0.8], [0.9, 0.1]]
        )
        inputs = [
            BasePredictionInput(designation="Product A"),
            BasePredictionInput(designation="Product B"),
        ]
        results = loaded_predictor.predict_batch(inputs)
        assert results[0].label == "40"  # argmax=1
        assert results[1].label == "10"  # argmax=0


# ── ArtifactLoadError ─────────────────────────────────────────────────────────


class TestArtifactLoadError:
    def test_raised_on_missing_model_file(self, settings: Settings) -> None:
        predictor = PredictorService(settings)
        mock_keras = MagicMock()
        mock_keras.models.load_model.side_effect = OSError("file not found")
        with patch.dict(sys.modules, {"keras": mock_keras, "keras.models": mock_keras.models}):
            with pytest.raises(ArtifactLoadError, match="Failed to load"):
                predictor.load_artifacts()
