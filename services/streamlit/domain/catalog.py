"""
Reference data extracted from the raw Rakuten catalogue.

Two JSON assets are shipped with the front-end, both generated from the
challenge files (``X_train_update.csv`` + ``Y_train_CVw08PX.csv``), which are
far too large to version:

``assets/categories.json``
    Per ``prdtypecode``: the real number of products, the share of the corpus,
    how often a description is actually filled, the terms that characterise the
    category, and three real product titles.

``assets/demo_products.json``
    One real product per category, taken from the **model's own test split**
    (``core/data/features/x_test.parquet``) and therefore never seen during
    training. Each one carries its true ``prdtypecode``, so a demonstration can
    be checked live instead of being taken on trust.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from services.streamlit.domain.labels import category_family, category_name


@dataclass(frozen=True)
class CategoryEvidence:
    """What the raw catalogue says about one ``prdtypecode``."""

    code: str
    count: int
    share: float
    description_filled: float
    terms: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        """Readable name of the category."""
        return category_name(self.code)

    @property
    def family(self) -> str:
        """Coarse family the category belongs to."""
        return category_family(self.code)

    @property
    def label(self) -> str:
        """Display label combining code and name."""
        return f"{self.code} · {self.name}"


@dataclass(frozen=True)
class DemoProduct:
    """A real catalogue entry the model never trained on."""

    designation: str
    description: str
    true_code: str
    productid: int

    @property
    def true_label(self) -> str:
        """Display label of the ground-truth category."""
        return f"{self.true_code} · {category_name(self.true_code)}"


@dataclass(frozen=True)
class Catalogue:
    """Aggregate view of the raw dataset."""

    source: str
    total_products: int
    without_description: int
    categories: dict[str, CategoryEvidence] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        """Return True when the asset was found and parsed."""
        return bool(self.categories)

    @property
    def description_gap(self) -> float:
        """Share of products shipped without any description."""
        if not self.total_products:
            return 0.0
        return self.without_description / self.total_products

    def frame(self) -> pd.DataFrame:
        """
        Return the categories as a DataFrame ordered from largest to smallest.

        Returns:
            pd.DataFrame: Columns ``code``, ``categorie``, ``famille``,
            ``libelle``, ``produits``, ``part``, ``description_remplie``,
            ``termes``.
        """
        rows = [
            {
                "code": evidence.code,
                "categorie": evidence.name,
                "famille": evidence.family,
                "libelle": evidence.label,
                "produits": evidence.count,
                "part": evidence.share,
                "description_remplie": evidence.description_filled,
                "termes": ", ".join(evidence.terms[:6]),
            }
            for evidence in self.categories.values()
        ]
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        return frame.sort_values("produits", ascending=False).reset_index(drop=True)


def _read_json(path: Path) -> dict[str, Any]:
    """Return the decoded JSON file, or an empty dict when unreadable."""
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return content if isinstance(content, dict) else {}


@lru_cache
def load_catalogue(assets_dir: Path) -> Catalogue:
    """
    Load the per-category evidence extracted from the raw catalogue.

    Args:
        assets_dir: Directory holding ``categories.json``.

    Returns:
        Catalogue: Parsed evidence; empty when the asset is missing.
    """
    raw = _read_json(assets_dir / "categories.json")
    if not raw:
        return Catalogue(source="", total_products=0, without_description=0)

    categories = {
        code: CategoryEvidence(
            code=code,
            count=int(entry.get("count", 0)),
            share=float(entry.get("share", 0.0)),
            description_filled=float(entry.get("description_filled", 0.0)),
            terms=list(entry.get("terms", [])),
            examples=list(entry.get("examples", [])),
        )
        for code, entry in raw.get("categories", {}).items()
    }
    return Catalogue(
        source=str(raw.get("source", "")),
        total_products=int(raw.get("total_products", 0)),
        without_description=int(raw.get("without_description", 0)),
        categories=categories,
    )


@lru_cache
def load_demo_products(assets_dir: Path) -> tuple[DemoProduct, ...]:
    """
    Load the held-out products used for the live demonstration.

    Args:
        assets_dir: Directory holding ``demo_products.json``.

    Returns:
        tuple[DemoProduct, ...]: One product per category, empty when missing.
    """
    raw = _read_json(assets_dir / "demo_products.json")
    return tuple(
        DemoProduct(
            designation=str(item.get("designation", "")),
            description=str(item.get("description", "")),
            true_code=str(item.get("true_code", "")),
            productid=int(item.get("productid", 0)),
        )
        for item in raw.get("products", [])
        if item.get("designation")
    )
