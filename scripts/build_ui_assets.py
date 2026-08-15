"""
Regenerate the reference assets shipped with the Streamlit front-end.

The raw challenge files are far too large to version (60 MB of CSV, 2.4 GB of
images), so the front-end ships two small JSON extracts instead:

``services/streamlit/assets/categories.json``
    Per ``prdtypecode``: the real number of products, its share of the corpus,
    how often a description is actually filled, the terms that characterise the
    category, and three real product titles. This is the evidence behind the
    category names displayed everywhere in the app — Rakuten never published
    what its codes mean.

``services/streamlit/assets/demo_products.json``
    One real product per category, restricted to the model's own test split so
    none of them was seen during training. Each carries its true
    ``prdtypecode``, which lets the demo verify a prediction live.

Usage::

    uv run python scripts/build_ui_assets.py --raw-dir <dossier des CSV Rakuten>

Expected in ``--raw-dir``: ``X_train_update.csv`` and ``Y_train_CVw08PX.csv``.
The held-out split is read from ``core/data/features/x_test.parquet``.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from html import unescape
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "services" / "streamlit" / "assets"
HELD_OUT_FILE = PROJECT_ROOT / "core" / "data" / "features" / "x_test.parquet"

# Function words and units, in the three languages present in the catalogue.
STOP_WORDS = set(
    """le la les un une des du de d l et ou a au aux en pour par avec sans sur sous dans
    ce cet cette ces son sa ses leur leurs il elle est sont plus tres tout tous toute
    the and for with of to in on you your is are it this that der die das und mit fur
    ein eine zu von im den dem ist sie auch als auf nicht se ne pas qui que quoi dont
    x cm mm m kg g ml cl l pcs pc lot new neuf""".split()
)

# Minimum number of documents a term must appear in to be considered typical.
MIN_TERM_DOCS = 15
MIN_TERM_SHARE = 0.02

TERMS_PER_CATEGORY = 8
EXAMPLES_PER_CATEGORY = 3
SEED = 3


def tokenize(text: str) -> list[str]:
    """
    Split a designation into comparable tokens.

    Args:
        text: Raw product designation.

    Returns:
        list[str]: Lowercase, accent-free alphabetic tokens of length >= 3,
        stripped of function words.
    """
    normalised = unicodedata.normalize("NFKD", str(text).lower())
    normalised = "".join(char for char in normalised if not unicodedata.combining(char))
    return [
        token
        for token in re.findall(r"[a-z]+", normalised)
        if len(token) >= 3 and token not in STOP_WORDS
    ]


def clean_text(text: object) -> str:
    """
    Strip the markup the raw catalogue carries, as the pipeline does.

    Roughly a third of the descriptions are raw HTML fragments
    (``<p>…</p><br/>``) and entities are left encoded (``&#34;``). The training
    pipeline removes them in ``core/src/data/clean_data.py`` via
    ``lxml.html.text_content()``; this mirrors that step without pulling lxml
    into the script, so the demo sends the model the kind of text it was
    trained on rather than a soup of tags.

    Args:
        text: Raw value read from the CSV.

    Returns:
        str: Display-ready text, empty for missing values.
    """
    if not isinstance(text, str):
        return ""
    value = re.sub(r"<[^>]+>", " ", text)
    value = unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def build_categories(frame: pd.DataFrame) -> dict[str, dict]:
    """
    Compute the per-category evidence.

    A term is kept when it is far more frequent inside the category than in the
    corpus as a whole — a plain document-frequency ratio, which is enough to
    surface what a category is about without pulling in a vectoriser.

    Args:
        frame: Catalogue with ``designation``, ``description`` and ``prdtypecode``.

    Returns:
        dict[str, dict]: One entry per product type code.
    """
    frame = frame.assign(tokens=frame["designation"].map(tokenize))

    corpus_counter: Counter[str] = Counter()
    for tokens in frame["tokens"]:
        corpus_counter.update(set(tokens))
    corpus_size = len(frame)

    categories: dict[str, dict] = {}
    for code, group in frame.groupby("prdtypecode"):
        counter: Counter[str] = Counter()
        for tokens in group["tokens"]:
            counter.update(set(tokens))
        size = len(group)
        threshold = max(MIN_TERM_DOCS, size * MIN_TERM_SHARE)

        scored = sorted(
            (
                (counter[term] / size) / (corpus_counter[term] / corpus_size),
                term,
            )
            for term in counter
            if counter[term] >= threshold
        )

        examples = (
            group["designation"]
            .loc[lambda series: series.str.len().between(30, 85)]
            .drop_duplicates()
            .head(600)
            .sample(EXAMPLES_PER_CATEGORY, random_state=SEED)
            .tolist()
        )

        categories[str(code)] = {
            "count": int(size),
            "share": round(size / corpus_size, 4),
            "description_filled": round(float(1 - group["description"].isna().mean()), 3),
            "terms": [term for _, term in reversed(scored[-TERMS_PER_CATEGORY:])],
            "examples": examples,
        }

    return dict(sorted(categories.items(), key=lambda item: -item[1]["count"]))


def build_demo_products(frame: pd.DataFrame, categories: dict[str, dict]) -> list[dict]:
    """
    Pick one held-out product per category for the live demonstration.

    Args:
        frame: Catalogue with designation, description, productid and code.
        categories: Evidence dict, used to order the demo by category size.

    Returns:
        list[dict]: Products with their ground-truth product type code.

    Raises:
        FileNotFoundError: If the model's test split cannot be read.
    """
    if not HELD_OUT_FILE.is_file():
        raise FileNotFoundError(
            f"{HELD_OUT_FILE} is missing — run the feature pipeline first so the "
            "demo can be restricted to products the model never trained on."
        )

    held_out = set(pd.read_parquet(HELD_OUT_FILE)["productid"].astype("int64"))
    pool = frame[frame["productid"].astype("int64").isin(held_out)]
    pool = pool[pool["designation"].str.len().between(35, 100)]

    products: list[dict] = []
    for code in categories:
        subset = pool[pool["prdtypecode"] == code]
        if subset.empty:
            continue
        row = subset.sample(1, random_state=17).iloc[0]
        products.append(
            {
                "designation": row["designation"],
                "description": clean_text(row["description"]),
                "true_code": code,
                "productid": int(row["productid"]),
            }
        )
    return products


def main() -> None:
    """Read the raw catalogue and write both JSON assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory holding X_train_update.csv and Y_train_CVw08PX.csv.",
    )
    args = parser.parse_args()

    designations = pd.read_csv(args.raw_dir / "X_train_update.csv", index_col=0)
    labels = pd.read_csv(args.raw_dir / "Y_train_CVw08PX.csv", index_col=0)

    frame = designations.join(labels)
    frame["prdtypecode"] = frame["prdtypecode"].astype(str)
    frame["designation"] = frame["designation"].map(clean_text)

    categories = build_categories(frame)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    (ASSETS_DIR / "categories.json").write_text(
        json.dumps(
            {
                "source": (
                    "X_train_update.csv + Y_train_CVw08PX.csv "
                    "(challenge Rakuten France Multimodal)"
                ),
                "total_products": int(len(frame)),
                "without_description": int(frame["description"].isna().sum()),
                "categories": categories,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    products = build_demo_products(frame, categories)
    (ASSETS_DIR / "demo_products.json").write_text(
        json.dumps(
            {
                "source": (
                    "produits du jeu de test du modèle (core/data/features/x_test.parquet)"
                ),
                "products": products,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"categories.json    {len(frame):,} produits, {len(categories)} catégories")
    print(f"demo_products.json {len(products)} produits jamais vus à l'entraînement")


if __name__ == "__main__":
    main()
