"""
Data page: what the catalogue actually contains, and how a product becomes
120 numbers.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
import yaml

from services.streamlit.domain.artifacts import (
    load_class_weights,
    load_metrics,
    load_params,
    load_pca_variance,
)
from services.streamlit.domain.catalog import load_catalogue
from services.streamlit.ui import charts, layout, state, theme

PIPELINE_STEPS = [
    (
        "1 · Nettoyage",
        "`clean_data.py` retire le HTML, normalise les espaces et les accents, "
        "et conserve `designation`, `description`, `prdtypecode`.",
    ),
    (
        "2 · Embeddings",
        "`paraphrase-multilingual-MiniLM-L12-v2` encode le texte en 384 dimensions. "
        "Les textes longs sont découpés (150 mots, 30 de recouvrement) puis moyennés.",
    ),
    (
        "3 · Statistiques",
        "6 mesures par colonne de texte (longueur, nombre de mots, longueur moyenne "
        "et maximale, chiffres, ponctuation), soit 12 features.",
    ),
    (
        "4 · Mise à l'échelle",
        "`StandardScaler` centre et réduit les 396 colonnes. L'objet ajusté est "
        "sérialisé pour être réutilisé à l'identique en inférence.",
    ),
    (
        "5 · Réduction",
        "`PCA` ramène 396 dimensions à 120. Moins de bruit, un modèle plus léger, "
        "et une inférence plus rapide.",
    ),
    (
        "6 · Modèle",
        "Réseau dense Keras, softmax sur 27 classes. Le code retenu est l'`argmax`, "
        "la confiance est la probabilité associée.",
    ),
]


def _tab_distribution(frame: pd.DataFrame) -> None:
    """Render the real class distribution of the catalogue."""
    st.markdown(
        "Chaque barre est le nombre réel de produits d'une catégorie dans le jeu "
        "d'entraînement complet. C'est cet écart entre la tête et la queue de "
        "distribution qui explique une bonne partie des écarts de performance."
    )

    families = ["Toutes"] + sorted(frame["famille"].unique())
    selected = st.selectbox("Famille de produits", families, index=0)
    filtered = frame if selected == "Toutes" else frame[frame["famille"] == selected]

    st.plotly_chart(
        charts.horizontal_bar(
            filtered,
            label_column="libelle",
            value_column="produits",
            value_format=",.0f",
            height=max(260, 24 * len(filtered) + 80),
        ),
        width="stretch",
    )

    largest, smallest = frame.iloc[0], frame.iloc[-1]
    layout.note(
        f"La plus grosse catégorie, **{largest['libelle']}**, pèse "
        f"{largest['part']:.1%} du catalogue à elle seule, soit "
        f"{largest['produits'] / smallest['produits']:.0f} fois la plus petite "
        f"(**{smallest['libelle']}**, {smallest['produits']} produits). "
        "Un modèle entraîné sans correction apprendrait surtout à répondre "
        f"« {largest['code']} »."
    )


def _tab_categories(frame: pd.DataFrame) -> None:
    """Render the evidence behind each category name."""
    st.markdown(
        "Rakuten ne documente pas ses codes : le jeu de données ne contient que des "
        "nombres. Les libellés utilisés dans cette application ont donc été établis "
        "en lisant le catalogue — pour chaque code, les termes nettement "
        "sur-représentés par rapport au reste du corpus, et de vrais titres de fiches."
    )

    table = frame[["code", "categorie", "famille", "produits", "part", "termes"]].copy()
    table["part"] = table["part"].map(lambda value: f"{value:.2%}")
    st.dataframe(
        table.rename(
            columns={
                "categorie": "libellé retenu",
                "part": "part du corpus",
                "termes": "termes caractéristiques",
            }
        ),
        hide_index=True,
        width="stretch",
        height=420,
    )

    catalogue = load_catalogue(state.settings().assets_path)
    choice = st.selectbox(
        "Voir de vrais produits d'une catégorie",
        frame["libelle"].tolist(),
        index=0,
    )
    code = choice.split(" · ")[0]
    evidence = catalogue.categories.get(code)
    if evidence is None:
        return

    left, right = st.columns([2, 3], gap="large")
    with left:
        st.markdown("**Termes caractéristiques**")
        st.markdown("\n".join(f"- `{term}`" for term in evidence.terms))
        st.caption(
            f"Description remplie pour {evidence.description_filled:.0%} des fiches "
            "de cette catégorie."
        )
    with right:
        st.markdown("**Titres réels tirés du catalogue**")
        for example in evidence.examples:
            st.markdown(f'<div class="step">{example}</div>', unsafe_allow_html=True)
            st.write("")


def _tab_pipeline() -> None:
    """Render the feature engineering chain."""
    settings = state.settings()

    st.markdown(
        "Le point critique est que **l'API rejoue exactement cette chaîne**. "
        "Un écart d'une seule étape entre entraînement et service — un scaler "
        "réajusté, un ordre de colonnes inversé — dégrade les prédictions sans "
        "qu'aucune erreur ne soit levée."
    )
    st.write("")
    layout.steps(PIPELINE_STEPS[:3])
    st.write("")
    layout.steps(PIPELINE_STEPS[3:])

    st.write("")
    st.markdown("**Dimensions le long de la chaîne**")
    flow = pd.DataFrame(
        {
            "etape": ["Statistiques", "Embeddings", "Concaténation", "Après PCA", "Sortie modèle"],
            "dimensions": [12, 384, 396, 120, 27],
        }
    )
    st.plotly_chart(
        charts.vertical_bar(flow, label_column="etape", value_column="dimensions", height=280),
        width="stretch",
    )
    st.caption(
        "L'ordre des colonnes compte : les 12 statistiques d'abord, puis les 384 embeddings."
    )

    variance = load_pca_variance(settings.features_metadata_path)
    if variance is not None:
        layout.note(
            f"La PCA ajustée sur ce jeu conserve **{variance:.1%}** de la variance avec "
            "120 composantes — chiffre relevé dans `core/data/features/metadata.json`, "
            "écrit par `build_features.py`. Passer de 396 à 120 dimensions coûte donc "
            "moins de 10 % d'information."
        )


def _tab_weights() -> None:
    """Render the class weights used during training."""
    weights = load_class_weights(state.settings().artifacts_path)
    if weights.empty:
        st.info("`class_weights.json` n'est pas disponible.")
        return

    st.markdown(
        "Pour compenser le déséquilibre, l'entraînement pondère chaque classe à "
        "l'inverse de sa fréquence : une erreur sur une catégorie rare coûte plus "
        "cher qu'une erreur sur une catégorie très représentée."
    )
    weights["libelle"] = weights["code"] + " · " + weights["categorie"]
    st.plotly_chart(
        charts.horizontal_bar(
            weights,
            label_column="libelle",
            value_column="poids",
            value_format=".2f",
            color=theme.SERIES_ORANGE,
            height=max(260, 24 * len(weights) + 80),
        ),
        width="stretch",
    )


def render() -> None:
    """Render the data and feature engineering page."""
    layout.page_header(
        "🗂️",
        "Les données et la fabrique de features",
        "Le catalogue Rakuten est du texte libre, multilingue, souvent incomplet et "
        "très déséquilibré. Cette page montre ce qu'il contient réellement et par "
        "quelles transformations il passe avant d'atteindre le modèle.",
    )

    settings = state.settings()
    catalogue = load_catalogue(settings.assets_path)
    report = load_metrics(settings.artifacts_path)

    if not catalogue.available:
        st.warning(
            "Les données de référence du catalogue (`services/streamlit/assets/"
            "categories.json`) sont introuvables."
        )
        return

    frame = catalogue.frame()
    largest = frame.iloc[0]

    layout.tiles(
        [
            (
                "Produits",
                f"{catalogue.total_products:,}".replace(",", " "),
                "jeu d'entraînement du challenge",
            ),
            (
                "Sans description",
                f"{catalogue.description_gap:.0%}",
                "le titre est parfois la seule information",
            ),
            (
                "Classe majoritaire",
                f"{largest['part']:.1%}",
                f"{largest['code']} · {largest['categorie']}",
            ),
            (
                "Jeu de test",
                f"{report.test_size:,}".replace(",", " ") if report.available else "—",
                "produits mis de côté pour l'évaluation",
            ),
        ]
    )

    st.write("")
    tab_distribution, tab_categories, tab_pipeline, tab_weights, tab_params = st.tabs(
        [
            "Répartition",
            "Les 27 catégories",
            "Chaîne de features",
            "Poids de classes",
            "params.yaml",
        ]
    )

    with tab_distribution:
        _tab_distribution(frame)
    with tab_categories:
        _tab_categories(frame)
    with tab_pipeline:
        _tab_pipeline()
    with tab_weights:
        _tab_weights()
    with tab_params:
        st.markdown(
            "`core/params.yaml` est la source de vérité du pipeline : chemins, taille des "
            "chunks, nombre de composantes PCA, graine aléatoire. L'API relit les mêmes "
            "valeurs, ce qui évite les divergences silencieuses."
        )
        params = load_params(settings.params_path)
        if params:
            st.code(yaml.safe_dump(params, sort_keys=False, allow_unicode=True), language="yaml")
        else:
            st.info("`core/params.yaml` est introuvable depuis cette machine.")

    st.caption(f"Source des chiffres de cette page : {catalogue.source}.")
