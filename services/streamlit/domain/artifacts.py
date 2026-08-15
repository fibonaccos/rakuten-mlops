"""
Read access to the artifacts produced by the training pipeline.

The front-end reads the same files the API and MLflow rely on
(``core/artifacts/``): metrics, learning curves, class weights, label map and
the confusion matrix. Nothing is recomputed here — the page shows what the
pipeline actually wrote, and stays silent when a file is missing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from services.streamlit.domain.labels import category_family, category_name

# Share of the dataset held out for testing, from core/params.yaml (train_size: 0.8).
TEST_SPLIT_RATIO = 0.2


def _read_json(path: Path) -> dict[str, Any]:
    """Return the decoded JSON file, or an empty dict when unreadable."""
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return content if isinstance(content, dict) else {}


@dataclass
class MetricsReport:
    """Evaluation report built from ``core/artifacts/metrics.json``."""

    overall: dict[str, float] = field(default_factory=dict)
    per_class: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def available(self) -> bool:
        """Return True when the report holds usable numbers."""
        return bool(self.overall) and not self.per_class.empty

    @property
    def test_size(self) -> int:
        """Number of products in the test set."""
        return int(self.overall.get("support", 0))

    @property
    def dataset_size(self) -> int:
        """Estimated size of the full dataset, derived from the test split."""
        return int(round(self.test_size / TEST_SPLIT_RATIO)) if self.test_size else 0

    @property
    def weakest_classes(self) -> pd.DataFrame:
        """The five categories with the lowest F1-score."""
        if self.per_class.empty:
            return self.per_class
        return self.per_class.nsmallest(5, "f1")

    @property
    def strongest_classes(self) -> pd.DataFrame:
        """The five categories with the highest F1-score."""
        if self.per_class.empty:
            return self.per_class
        return self.per_class.nlargest(5, "f1")


def load_metrics(artifacts_dir: Path) -> MetricsReport:
    """
    Load the evaluation report written by ``core/src/models/evaluate.py``.

    The file stores one entry per ``prdtypecode`` plus a ``global`` entry.
    Per-class rows are turned into a DataFrame enriched with readable category
    names so every chart and table in the app shares the same vocabulary.

    Args:
        artifacts_dir: Directory holding ``metrics.json``.

    Returns:
        MetricsReport: Parsed report; empty when the file is missing.
    """
    raw = _read_json(artifacts_dir / "metrics.json")
    if not raw:
        return MetricsReport()

    overall = {k: float(v) for k, v in raw.get("global", {}).items()}

    rows: list[dict[str, Any]] = []
    for code, values in raw.items():
        if code == "global" or not isinstance(values, dict):
            continue
        rows.append(
            {
                "code": code,
                "categorie": category_name(code),
                "famille": category_family(code),
                "accuracy": float(values.get("accuracy", 0.0)),
                "precision": float(values.get("precision", 0.0)),
                "recall": float(values.get("recall", 0.0)),
                "f1": float(values.get("f1-score", 0.0)),
                "auc": float(values.get("auc", 0.0)),
                "support": int(values.get("support", 0)),
            }
        )

    per_class = pd.DataFrame(rows)
    if not per_class.empty:
        per_class = per_class.sort_values("support", ascending=False).reset_index(drop=True)
    return MetricsReport(overall=overall, per_class=per_class)


def load_history(artifacts_dir: Path) -> pd.DataFrame:
    """
    Load the Keras training history as a tidy DataFrame.

    Args:
        artifacts_dir: Directory holding ``history.json``.

    Returns:
        pd.DataFrame: One row per epoch, one column per tracked metric,
        with a 1-based ``epoch`` column. Empty when the file is missing.
    """
    raw = _read_json(artifacts_dir / "history.json")
    if not raw:
        return pd.DataFrame()

    history = pd.DataFrame({key: values for key, values in raw.items() if isinstance(values, list)})
    if history.empty:
        return history
    history.insert(0, "epoch", range(1, len(history) + 1))
    return history


def load_class_weights(artifacts_dir: Path) -> pd.DataFrame:
    """
    Load the class weights applied during training.

    ``class_weights.json`` is keyed by ``prdtypecode``, the same vocabulary as
    the rest of the app — no remapping through ``labels_map.json`` is needed.

    Args:
        artifacts_dir: Directory holding ``class_weights.json``.

    Returns:
        pd.DataFrame: Columns ``code``, ``categorie`` and ``poids``.
    """
    weights = _read_json(artifacts_dir / "class_weights.json")
    if not weights:
        return pd.DataFrame()

    rows = [
        {
            "code": str(code),
            "categorie": category_name(code),
            "poids": float(weight),
        }
        for code, weight in weights.items()
    ]
    return pd.DataFrame(rows).sort_values("poids", ascending=False).reset_index(drop=True)


def load_labels_map(artifacts_dir: Path) -> dict[str, int]:
    """
    Load the ``prdtypecode`` to model-output-index mapping.

    Args:
        artifacts_dir: Directory holding ``labels_map.json``.

    Returns:
        dict[str, int]: Mapping of product type codes to output indices.
    """
    raw = _read_json(artifacts_dir / "labels_map.json")
    return {str(code): int(index) for code, index in raw.items()}


def confusion_matrix_file(artifacts_dir: Path) -> Path | None:
    """
    Return the path of the confusion matrix image, when it exists.

    Args:
        artifacts_dir: Directory holding ``confusion_matrix.png``.

    Returns:
        Path | None: Path to the image, or None when it has not been generated.
    """
    path = artifacts_dir / "confusion_matrix.png"
    return path if path.is_file() else None


def load_pca_variance(features_metadata_file: Path) -> float | None:
    """
    Return the share of variance the PCA keeps, as recorded by the pipeline.

    ``core/data/features/metadata.json`` is written by ``build_features.py`` and
    is not versioned, so this returns None on a fresh clone.

    Args:
        features_metadata_file: Path to the metadata file.

    Returns:
        float | None: Explained variance ratio, or None when unavailable.
    """
    metadata = _read_json(features_metadata_file)
    variance = metadata.get("pca", {}).get("params", {}).get("explained_variance")
    return float(variance) if isinstance(variance, (int, float)) else None


def load_params(params_file: Path) -> dict[str, Any]:
    """
    Load ``core/params.yaml``, the single source of truth of the pipeline.

    Args:
        params_file: Path to the YAML file.

    Returns:
        dict[str, Any]: Parsed parameters, empty when unreadable.
    """
    try:
        content = yaml.safe_load(params_file.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return content if isinstance(content, dict) else {}
