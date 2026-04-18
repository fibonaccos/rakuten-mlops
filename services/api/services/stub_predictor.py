"""
Stub predictor — development replacement for PredictorService.

Used when STUB_MODE=true in .env (i.e. no ML artifacts available yet).
Satisfies exactly the same interface as PredictorService so the router,
lifespan handler, and health checks require zero changes.

The stub draws random labels from the known Rakuten prdtypecode set and
generates plausible-looking confidence scores. It is explicit about being
a stub through its model_name property so responses are never confused
with real predictions.

Warning:
    Never enable STUB_MODE in production. It returns meaningless predictions.
"""

import random
import time
from typing import Final

from ..schemas.inference import (
    BasePredictionInput,
    BasePredictionOutput,
    PredictionInputOptions,
)

# 27 Rakuten product category codes — matches labels_map.json
_KNOWN_LABELS: Final[list[str]] = [
    "10", "40", "50", "60", "1140", "1160", "1180", "1280", "1281",
    "1300", "1301", "1302", "1320", "1560", "1920", "1940", "2060",
    "2220", "2280", "2403", "2462", "2522", "2582", "2583", "2585",
    "2705", "2905",
]


class StubPredictorService:
    """
    Drop-in replacement for PredictorService that requires no ML artifacts.

    Designed for local development and CI environments where the real model,
    scaler and PCA are not yet available. Activate via STUB_MODE=true.
    """

    def __init__(self) -> None:
        self._loaded: bool = False

    # ── Same public interface as PredictorService ─────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_name(self) -> str:
        return "stub-model"

    def load_artifacts(self) -> None:
        """
        No-op: the stub has nothing to load.

        Called by the lifespan handler exactly like the real service.
        """
        self._loaded = True

    def predict(
        self,
        inputs: BasePredictionInput,
        options: PredictionInputOptions | None = None,
    ) -> BasePredictionOutput:
        """
        Return a random prediction drawn from the known label set.

        Args:
            inputs: Product input (ignored — stub only uses designation length
                as a deterministic seed so the same input always returns the
                same label within a single process run).
            options: Optional flags for confidence / distribution.

        Returns:
            BasePredictionOutput: A plausible-looking but random prediction.
        """
        # Seed from designation so identical inputs → identical stub label.
        rng = random.Random(hash(inputs.designation))

        label = rng.choice(_KNOWN_LABELS)
        raw_scores = _softmax_noise(rng, len(_KNOWN_LABELS))
        predicted_idx = _KNOWN_LABELS.index(label)

        confidence = None
        distribution = None

        if options is not None:
            if options.return_confidence:
                confidence = round(raw_scores[predicted_idx], 4)
            if options.return_distribution:
                distribution = {
                    lbl: round(score, 4)
                    for lbl, score in zip(_KNOWN_LABELS, raw_scores)
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
        Return one random prediction per input.

        Args:
            inputs: List of product inputs.
            options: Optional flags for confidence / distribution.

        Returns:
            list[BasePredictionOutput]: One stub result per input.
        """
        return [self.predict(inp, options) for inp in inputs]


# ── Internal helpers ──────────────────────────────────────────────────────────


def _softmax_noise(rng: random.Random, n: int) -> list[float]:
    """
    Generate a normalised probability distribution of length *n* using
    random logits — mimics the shape of a real softmax output.

    Args:
        rng: Seeded random generator for reproducibility.
        n: Number of classes.

    Returns:
        list[float]: Probabilities that sum to 1.0.
    """
    logits = [rng.gauss(0, 1) for _ in range(n)]
    exp_logits = [2.718 ** l for l in logits]
    total = sum(exp_logits)
    return [e / total for e in exp_logits]
