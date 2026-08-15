"""Tests for artifact loading and category naming."""

import json
from pathlib import Path

from services.streamlit.domain.artifacts import (
    confusion_matrix_file,
    load_class_weights,
    load_history,
    load_labels_map,
    load_metrics,
    load_params,
    load_pca_variance,
)
from services.streamlit.domain.catalog import load_catalogue, load_demo_products
from services.streamlit.domain.labels import (
    CATEGORY_FAMILIES,
    CATEGORY_NAMES,
    category_label,
    category_name,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT_ROOT / "core" / "artifacts"
ASSETS = PROJECT_ROOT / "services" / "streamlit" / "assets"


def test_every_trained_class_has_a_readable_name() -> None:
    """The label map and the naming table describe the same 27 categories."""
    codes = set(load_labels_map(ARTIFACTS))
    assert codes == set(CATEGORY_NAMES)
    assert codes == set(CATEGORY_FAMILIES)


def test_catalogue_matches_the_challenge_dataset() -> None:
    """The shipped evidence describes the full labelled catalogue."""
    catalogue = load_catalogue(ASSETS)

    assert catalogue.available
    assert catalogue.total_products == 84_916
    assert set(catalogue.categories) == set(CATEGORY_NAMES)
    assert sum(item.count for item in catalogue.categories.values()) == catalogue.total_products
    assert 0 < catalogue.description_gap < 1


def test_catalogue_frame_is_ordered_and_labelled() -> None:
    """The frame feeding the charts is sorted by size and carries readable labels."""
    frame = load_catalogue(ASSETS).frame()

    assert len(frame) == 27
    assert frame["produits"].is_monotonic_decreasing
    assert frame.iloc[0]["code"] == "2583"
    assert frame["libelle"].str.contains(" · ").all()


def test_every_category_carries_its_evidence() -> None:
    """Each category ships characteristic terms and real product titles."""
    for evidence in load_catalogue(ASSETS).categories.values():
        assert evidence.terms, evidence.code
        assert evidence.examples, evidence.code
        assert 0 <= evidence.description_filled <= 1


def test_demo_products_are_real_and_held_out() -> None:
    """The demo covers every category with products carrying a ground truth."""
    products = load_demo_products(ASSETS)
    codes = set(load_labels_map(ARTIFACTS))

    assert len(products) == 27
    assert {product.true_code for product in products} == codes
    assert all(product.designation.strip() for product in products)
    assert all(product.productid > 0 for product in products)
    assert all(" · " in product.true_label for product in products)


def test_missing_assets_degrade_quietly(tmp_path: Path) -> None:
    """A missing asset directory yields empty structures, never an exception."""
    assert not load_catalogue(tmp_path).available
    assert load_catalogue(tmp_path).frame().empty
    assert load_demo_products(tmp_path) == ()


def test_metrics_report_is_complete() -> None:
    """The evaluation report exposes global metrics and one row per class."""
    report = load_metrics(ARTIFACTS)

    assert report.available
    assert len(report.per_class) == 27
    assert 0 < report.overall["accuracy"] <= 1
    assert report.test_size > 0
    assert report.dataset_size > report.test_size
    assert set(report.per_class.columns) >= {"code", "categorie", "f1", "support"}


def test_metrics_report_ranks_classes() -> None:
    """Best and worst classes are ordered by F1-score."""
    report = load_metrics(ARTIFACTS)

    assert len(report.weakest_classes) == 5
    assert report.weakest_classes["f1"].max() <= report.strongest_classes["f1"].min()


def test_history_is_indexed_by_epoch() -> None:
    """Learning curves carry a 1-based epoch column."""
    history = load_history(ARTIFACTS)

    assert not history.empty
    assert history["epoch"].iloc[0] == 1
    assert {"accuracy", "val_accuracy", "loss", "val_loss"} <= set(history.columns)


def test_class_weights_are_mapped_back_to_product_codes() -> None:
    """Weights are indexed by model output, and displayed by product code."""
    weights = load_class_weights(ARTIFACTS)

    assert len(weights) == 27
    assert set(weights["code"]) == set(CATEGORY_NAMES)
    assert weights["poids"].is_monotonic_decreasing


def test_confusion_matrix_is_found() -> None:
    """The evaluation image is picked up from the artifacts directory."""
    assert confusion_matrix_file(ARTIFACTS) is not None


def test_params_expose_the_pipeline_configuration() -> None:
    """The front-end reads the same params.yaml as the pipeline."""
    params = load_params(PROJECT_ROOT / "core" / "params.yaml")

    assert params["features"]["pca"]["n_components"] == 120
    assert params["train"]["n_classes"] == 27


def test_pca_variance_is_read_from_the_pipeline_metadata(tmp_path: Path) -> None:
    """The explained variance comes from the file build_features.py writes."""
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps({"pca": {"params": {"n_components": 120, "explained_variance": 0.9028}}}),
        encoding="utf-8",
    )

    assert load_pca_variance(metadata) == 0.9028
    assert load_pca_variance(tmp_path / "absent.json") is None


def test_missing_artifacts_degrade_quietly(tmp_path: Path) -> None:
    """An empty artifacts directory yields empty structures, never an exception."""
    assert not load_metrics(tmp_path).available
    assert load_history(tmp_path).empty
    assert load_class_weights(tmp_path).empty
    assert load_labels_map(tmp_path) == {}
    assert confusion_matrix_file(tmp_path) is None
    assert load_params(tmp_path / "absent.yaml") == {}


def test_corrupted_metrics_do_not_raise(tmp_path: Path) -> None:
    """A truncated JSON file is treated as a missing report."""
    (tmp_path / "metrics.json").write_text("{ broken", encoding="utf-8")
    assert not load_metrics(tmp_path).available


def test_category_naming() -> None:
    """Known codes get their name, unknown codes get a readable fallback."""
    assert category_name("1560") == "Mobilier d'intérieur"
    assert category_label("1560").startswith("1560 · ")
    assert category_name("9999") == "Catégorie 9999"


def test_category_names_are_backed_by_the_catalogue_evidence() -> None:
    """
    Spot-check the naming decisions the raw data settles.

    These three codes are the ones a reader is most likely to guess wrong, so
    they are pinned to the terms actually observed in the catalogue.
    """
    categories = load_catalogue(ASSETS).categories

    assert "pokemon" in categories["1160"].terms  # cards, not books
    assert "telechargement" in categories["2905"].terms  # downloads, not construction toys
    assert "piscine" in categories["2583"].terms


def test_labels_map_file_is_inverted_the_same_way_as_the_api(tmp_path: Path) -> None:
    """The map is read as ``{code: index}``, matching the API convention."""
    (tmp_path / "labels_map.json").write_text(json.dumps({"10": 0, "40": 1}), encoding="utf-8")
    assert load_labels_map(tmp_path) == {"10": 0, "40": 1}
