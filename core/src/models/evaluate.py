import json
import keras as ks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from box import Box
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from typing import Any

from ..utils.loaders import load_params, load_model


def make_report(
        y_test: np.ndarray,
        y_pred: np.ndarray,
        labels_map: dict,
        filepath: str
    ) -> dict[str, Any]:
    """
    Build a report from the model evaluation on the test dataset.

    Args:
        y_test (np.ndarray): The model predictions on test dataset.
        y_pred (np.ndarray): The real classes of the test dataset.
        labels_map (dict): The mapping between labels and integers.
        filepath (str): The path to save the report.

    Returns:
        dict[str, Any]: The built report.
    """

    report = {}
    labels_map_r: dict = {v: k for k, v in labels_map.items()}

    y_test_oh = np.array(ks.utils.to_categorical(y_test))

    y_pred_int = np.argmax(y_pred, axis=1)

    report = {}
    for i in range(y_pred.shape[1]):
        y_true_class = y_test_oh[:, i]
        y_pred_class = y_pred[:, i]
        y_pred_bin = (y_pred_class >= 0.5).astype(int)
        report[labels_map_r[i]] = {
            "accuracy": accuracy_score(y_true_class, y_pred_bin),
            "precision": precision_score(y_true_class, y_pred_bin, zero_division=0),
            "recall": recall_score(y_true_class, y_pred_bin, zero_division=0),
            "f1-score": f1_score(y_true_class, y_pred_bin, zero_division=0),
            "auc": roc_auc_score(y_true_class, y_pred_class),
            "support": int(y_true_class.sum())
        }
    report["global"] = {
        "accuracy": accuracy_score(y_test, y_pred_int),
        "precision_macro": precision_score(y_test, y_pred_int, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_test, y_pred_int, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred_int, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred_int, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred_int, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred_int, average="weighted", zero_division=0),
        "auc_macro": roc_auc_score(y_test_oh, y_pred, average="macro"),
        "auc_weighted": roc_auc_score(y_test_oh, y_pred, average="weighted"),
        "support": int(len(y_test))
    }

    with open(filepath, "w") as f:
        json.dump(report, f, indent=2)
    return report


def make_cm(
        y_test: np.ndarray,
        y_pred: np.ndarray,
        labels_map: dict,
        filepath: str
    ) -> None:
    """
    Build a confusion matrix from predictions on the test dataset.

    Args:
        y_test (np.ndarray): The model predictions on test dataset.
        y_pred (np.ndarray): The real classes of the test dataset.
        labels_map (dict): The mapping between labels and integers.
        filepath (str): The path to save the report.

    Returns:
        None:
    """

    labels_map_r: dict = {v: k for k, v in labels_map.items()}

    y_pred_int = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_test, y_pred_int)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(15, 12))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='cividis',
                xticklabels=[labels_map_r[i] for i in range(len(labels_map_r))],
                yticklabels=[labels_map_r[i] for i in range(len(labels_map_r))])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filepath)
    return None


def evaluate_model() -> None:
    """
    Build metrics from a trained model computed on a test dataset. It uses the
    parameters loaded from the `params.yaml` file at the root of the `core` directory.

    Metrics and artefacts are saved in a dedicated folder.

    The fields used for evaluating the model can be found under the `evaluate` key of
    the `params.yaml` file.

    Returns:
        None:
    """

    CORE_DIR: Path = Path(__file__).parent.parent.parent
    params: Box = Box(load_params(str(CORE_DIR / "params.yaml"))).evaluate

    X_test = pd.read_parquet(CORE_DIR / params.input.x_test)
    y_test = pd.read_parquet(CORE_DIR / params.input.y_test)[params.target]\
        .to_numpy()

    X_test = X_test.drop(columns=["productid"]).to_numpy()

    with open(CORE_DIR / params.input.labels_map, "r") as f:
        labels_map: dict = json.load(f)

    y_test = np.array([labels_map[str(code)] for code in y_test])

    model = load_model(CORE_DIR / params.model)
    y_pred = model.predict(X_test)

    report = make_report(y_test, y_pred, labels_map, str(CORE_DIR / params.output.metrics))
    make_cm(y_test, y_pred, labels_map, str(CORE_DIR / params.output.confusion_matrix))
    return None


if __name__ == "__main__":
    evaluate_model()
