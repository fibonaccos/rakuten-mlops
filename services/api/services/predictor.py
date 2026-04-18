"""
Core ML inference service.

This module implements the full feature engineering pipeline required for inference,
replicating exactly the training pipeline from core/src/data/build_features.py.

Warning:
    Any divergence between this pipeline and the training pipeline introduces
    training-serving skew, causing silent degradation of prediction quality.
    Always keep both in sync.
"""

import json
import logging
import unicodedata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import Settings
from ..schemas.inference import (
    BasePredictionInput,
    BasePredictionOutput,
    PredictionInputOptions,
)

logger = logging.getLogger(__name__)


class ArtifactLoadError(Exception):
    """Raised when one or more ML artifacts cannot be loaded."""


class PredictorService:
    """
    Stateful service that encapsulates the full inference pipeline.

    The service must be initialised once via load_artifacts() before any
    call to predict(). This is enforced by the FastAPI lifespan handler.

    Pipeline (must match training exactly):
        1. Combine designation + description into a single text.
        2. Embed with SentenceTransformer (chunk_size=150, overlap=30).
        3. Compute 12 statistical features (6 stats x 2 text columns).
        4. Concatenate: [12 stats | 384 embeddings] = 396 features.
        5. StandardScaler.transform().
        6. PCA.transform() -> 120 components.
        7. Keras model.predict() -> softmax over 27 classes.
        8. Map argmax index to label via labels_map.json.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any = None
        self._scaler: Any = None
        self._pca: Any = None
        self._labels_map: dict[int, str] = {}
        self._embedder: SentenceTransformer | None = None
        self._loaded: bool = False

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        """Return True once all artifacts are loaded and ready."""
        return self._loaded

    @property
    def model_name(self) -> str:
        """Return the Keras model file name (used in response metadata)."""
        return Path(self._settings.MODEL_PATH).name

    def load_artifacts(self) -> None:
        """
        Load all ML artifacts from disk into memory.

        Must be called once before serving predictions. Designed to be invoked
        from the FastAPI lifespan handler so the cost is paid at startup, not
        per-request.

        Raises:
            ArtifactLoadError: If any artifact file is missing or corrupted.
        """
        logger.info("Loading ML artifacts...")
        try:
            import keras as ks

            self._model = ks.models.load_model(self._settings.MODEL_PATH)
            logger.info("Keras model loaded from %s", self._settings.MODEL_PATH)

            self._scaler = joblib.load(self._settings.SCALER_PATH)
            logger.info("Scaler loaded from %s", self._settings.SCALER_PATH)

            self._pca = joblib.load(self._settings.PCA_PATH)
            logger.info("PCA loaded from %s", self._settings.PCA_PATH)

            with open(self._settings.LABELS_MAP_PATH, "r", encoding="utf-8") as f:
                raw: dict[str, str] = json.load(f)
            # labels_map.json stores {prdtypecode_str: index_int}; we invert it.
            self._labels_map = {int(v): k for k, v in raw.items()}
            logger.info(
                "Labels map loaded: %d classes", len(self._labels_map)
            )

            self._embedder = SentenceTransformer(
                self._settings.EMBEDDER_MODEL_NAME
            )
            logger.info(
                "SentenceTransformer loaded: %s",
                self._settings.EMBEDDER_MODEL_NAME,
            )

            self._loaded = True
            logger.info("All artifacts ready.")
        except (OSError, ValueError, KeyError) as exc:
            raise ArtifactLoadError(
                f"Failed to load ML artifacts: {exc}"
            ) from exc

    def predict(
        self,
        inputs: BasePredictionInput,
        options: PredictionInputOptions | None = None,
    ) -> BasePredictionOutput:
        """
        Run the full inference pipeline on a single product input.

        Args:
            inputs: Product designation and optional description.
            options: Optional flags to include confidence or distribution.

        Returns:
            BasePredictionOutput: Predicted label with optional confidence/distribution.

        Raises:
            RuntimeError: If artifacts have not been loaded yet.
        """
        if not self._loaded:
            raise RuntimeError("Artifacts not loaded. Call load_artifacts() first.")

        feature_vector = self._build_feature_vector(inputs)
        scaled = self._scaler.transform(feature_vector)
        reduced = self._pca.transform(scaled)

        raw_probs: np.ndarray = self._model.predict(reduced, verbose=0)[0]

        predicted_index = int(np.argmax(raw_probs))
        label = self._labels_map[predicted_index]

        confidence: float | None = None
        distribution: dict[str, float] | None = None

        if options is not None:
            if options.return_confidence:
                confidence = float(raw_probs[predicted_index])
            if options.return_distribution:
                distribution = {
                    self._labels_map[i]: float(p)
                    for i, p in enumerate(raw_probs)
                }

        return BasePredictionOutput(
            label=label,
            confidence=confidence,
            distribution=distribution,
        )

    def predict_batch(
        self,
        inputs: list[BasePredictionInput],
        options: PredictionInputOptions | None = None,
    ) -> list[BasePredictionOutput]:
        """
        Run inference on a list of product inputs in a single model pass.

        Batching is more efficient than calling predict() in a loop because the
        Keras model processes the entire matrix in one GPU/CPU operation.

        Args:
            inputs: List of product inputs.
            options: Optional flags to include confidence or distribution.

        Returns:
            list[BasePredictionOutput]: One result per input, in the same order.

        Raises:
            RuntimeError: If artifacts have not been loaded yet.
        """
        if not self._loaded:
            raise RuntimeError("Artifacts not loaded. Call load_artifacts() first.")

        feature_matrix = np.vstack(
            [self._build_feature_vector(inp) for inp in inputs]
        )
        scaled = self._scaler.transform(feature_matrix)
        reduced = self._pca.transform(scaled)

        all_probs: np.ndarray = self._model.predict(reduced, verbose=0)

        results: list[BasePredictionOutput] = []
        for raw_probs in all_probs:
            predicted_index = int(np.argmax(raw_probs))
            label = self._labels_map[predicted_index]

            confidence = None
            distribution = None
            if options is not None:
                if options.return_confidence:
                    confidence = float(raw_probs[predicted_index])
                if options.return_distribution:
                    distribution = {
                        self._labels_map[i]: float(p)
                        for i, p in enumerate(raw_probs)
                    }

            results.append(
                BasePredictionOutput(
                    label=label,
                    confidence=confidence,
                    distribution=distribution,
                )
            )
        return results

    # ── Private helpers — feature engineering ────────────────────────────────

    def _build_feature_vector(self, inputs: BasePredictionInput) -> np.ndarray:
        """
        Transform a single product input into the 396-dimensional feature vector.

        Feature layout (must match training column order):
            [0:12]   12 statistical features (length, num_words, mean_word_len,
                     max_word_len, num_digits, num_punctuation) x 2 columns
            [12:396] 384-dimensional SentenceTransformer embedding

        Args:
            inputs: Product designation and optional description.

        Returns:
            np.ndarray: Shape (1, 396) — ready for scaler.transform().
        """
        designation = inputs.designation
        description = inputs.description or ""

        stat_features = self._compute_stats(designation, description)
        embedding = self._embed(designation + " " + description)

        feature_vector = np.concatenate([stat_features, embedding]).reshape(1, -1)
        return feature_vector

    def _compute_stats(self, designation: str, description: str) -> np.ndarray:
        """
        Compute 12 statistical features over two text columns.

        Features per column (6 total, applied to designation then description):
            - Character count
            - Word count
            - Mean word length
            - Maximum word length
            - Number of numeric characters (Unicode category N*)
            - Number of punctuation characters (Unicode category P*)

        Args:
            designation: Main product text.
            description: Optional product description (empty string if absent).

        Returns:
            np.ndarray: Shape (12,) with float values.
        """
        features: list[float] = []
        for text in (designation, description):
            words = text.split() if text else []
            word_lengths = [len(w) for w in words]
            features.extend([
                float(len(text)),
                float(len(words)),
                float(np.mean(word_lengths)) if word_lengths else 0.0,
                float(max(word_lengths)) if word_lengths else 0.0,
                float(sum(unicodedata.category(c).startswith("N") for c in text)),
                float(sum(unicodedata.category(c).startswith("P") for c in text)),
            ])
        return np.array(features, dtype=np.float64)

    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks to fit the transformer token limit.

        Args:
            text: Input text string.

        Returns:
            list[str]: Word-level chunks with the configured overlap.
        """
        chunk_size = self._settings.EMBED_CHUNK_SIZE
        overlap = self._settings.EMBED_OVERLAP

        if not isinstance(text, str) or not text.strip():
            return [""]
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
        chunks: list[str] = []
        for i in range(0, len(words), chunk_size - overlap):
            chunks.append(" ".join(words[i : i + chunk_size]))
        return chunks

    def _embed(self, text: str) -> np.ndarray:
        """
        Embed a text string into a 384-dimensional vector.

        Long texts are chunked and the final embedding is the mean of all chunks,
        matching the behaviour used during training.

        Args:
            text: Text to embed (designation + description concatenated).

        Returns:
            np.ndarray: Shape (384,).
        """
        assert self._embedder is not None
        chunks = self._chunk_text(text)
        chunk_embeddings: np.ndarray = self._embedder.encode(chunks)
        return np.mean(chunk_embeddings, axis=0)
