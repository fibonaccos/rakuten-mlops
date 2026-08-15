"""
Data page: what the dataset looks like and how a product becomes 120 numbers.
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
]


def _class_distribution(per_class: pd.DataFrame) -> pd.DataFrame:
    """Return the per-class support, sorted from the largest class to the smallest."""
    frame = per_class[["code", "categorie", "famille", "support"]].copy()
    frame["libelle"] = frame["code"] + " · " + frame["categorie"]
    return frame.sort_values("support", ascending=False).reset_index(drop=True)


def render() -> None:
    """Render the data and feature engineering page."""
    layout.page_header(
        "🗂️",
        "Les données et la fabrique de features",
        "Le jeu Rakuten est du texte libre, multilingue et fortement déséquilibré. "
        "Cette page montre à quoi ressemble la matière première et par quelles "
        "transformations elle passe avant d'atteindre le modèle.",
    )

    settings = state.settings()
    report = load_metrics(settings.artifacts_path)

    if not report.available:
        st.warning(
            "Les artefacts d'évaluation (`core/artifacts/metrics.json`) sont introuvables. "
            "Lancez `python -m core.src.models.evaluate` pour les régénérer."
        )
        return

    distribution = _class_distribution(report.per_class)
    largest = distribution.iloc[0]
    smallest = distribution.iloc[-1]

    layout.tiles(
        [
            ("Produits", f"{report.dataset_size:,}".replace(",", " "), "estimation sur split 80/20"),
            ("Jeu de test", f"{report.test_size:,}".replace(",", " "), "produits évalués"),
            (
                "Classe majoritaire",
                f"{largest['support']:,}".replace(",", " "),
                f"{largest['code']} · {largest['categorie']}",
            ),
            (
                "Déséquilibre",
                f"×{largest['support'] / max(smallest['support'], 1):.0f}",
                "entre la plus grande et la plus petite classe",
            ),
        ]
    )

    st.write("")
    tab_distribution, tab_pipeline, tab_weights, tab_params = st.tabs(
        ["Répartition des catégories", "Chaîne de features", "Poids de classes", "params.yaml"]
    )

    with tab_distribution:
        st.markdown(
            "Chaque barre est le nombre de produits **du jeu de test** dans une catégorie. "
            "L'écart entre la tête et la queue de distribution explique une bonne partie "
            "des écarts de performance visibles page suivante."
        )
        families = ["Toutes"] + sorted(distribution["famille"].unique())
        selected = st.selectbox("Famille de produits", families, index=0)
        filtered = (
            distribution if selected == "Toutes" else distribution[distribution["famille"] == selected]
        )

        st.plotly_chart(
            charts.horizontal_bar(
                filtered,
                label_column="libelle",
                value_column="support",
                value_format=",.0f",
                height=max(260, 24 * len(filtered) + 80),
            ),
            width="stretch",
        )
        st.caption(
            "Les libellés de catégories sont une lecture métier faite par l'équipe : "
            "Rakuten ne publie que les codes numériques."
        )

    with tab_pipeline:
        st.markdown(
            "Le point critique est que **l'API rejoue exactement cette chaîne**. "
            "Un écart d'une seule étape entre entraînement et service — un scaler "
            "réajusté, un ordre de colonnes inversé — dégrade les prédictions sans "
            "qu'aucune erreur ne soit levée."
        )
        st.write("")
        layout.steps(PIPELINE_STEPS[:3])
        st.write("")
        layout.steps(PIPELINE_STEPS[3:] + [("6 · Modèle", "Réseau dense Keras, softmax sur 27 classes.")])

        st.write("")
        st.markdown("**Dimensions le long de la chaîne**")
        flow = pd.DataFrame(
            {
                "etape": [
                    "Statistiques",
                    "Embeddings",
                    "Concaténation",
                    "Après PCA",
                    "Sortie modèle",
                ],
                "dimensions": [12, 384, 396, 120, 27],
            }
        )
        st.plotly_chart(
            charts.vertical_bar(
                flow,
                label_column="etape",
                value_column="dimensions",
                height=280,
            ),
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

    with tab_weights:
        weights = load_class_weights(settings.artifacts_path)
        if weights.empty:
            st.info("`class_weights.json` n'est pas disponible.")
        else:
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
