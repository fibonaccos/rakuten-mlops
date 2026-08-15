"""
Performance page: what the model is worth, globally and category by category.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from services.streamlit.domain.artifacts import (
    confusion_matrix_file,
    load_history,
    load_metrics,
)
from services.streamlit.ui import charts, layout, state, theme

CURVES = [
    ("Exactitude", "accuracy", "val_accuracy"),
    ("AUC", "auc", "val_auc"),
    ("Perte", "loss", "val_loss"),
]


def _tab_per_class(per_class: pd.DataFrame, f1_macro: float) -> None:
    """Render the per-category breakdown."""
    st.markdown(
        "Le F1-score équilibre précision et rappel. La ligne pointillée est la moyenne "
        "macro : chaque catégorie compte autant, quelle que soit sa taille."
    )

    left, right = st.columns([1, 1])
    with left:
        sort_key = st.selectbox(
            "Trier par", ["F1-score", "Support", "Précision", "Rappel", "AUC"], index=0
        )
    with right:
        families = ["Toutes"] + sorted(per_class["famille"].unique())
        family = st.selectbox("Famille", families, index=0)

    column = {
        "F1-score": "f1",
        "Support": "support",
        "Précision": "precision",
        "Rappel": "recall",
        "AUC": "auc",
    }[sort_key]

    frame = per_class if family == "Toutes" else per_class[per_class["famille"] == family]
    frame = frame.sort_values(column, ascending=False).copy()
    frame["libelle"] = frame["code"] + " · " + frame["categorie"]

    st.plotly_chart(
        charts.horizontal_bar(
            frame,
            label_column="libelle",
            value_column="f1",
            value_format=".2f",
            height=max(260, 24 * len(frame) + 90),
            reference=f1_macro,
            reference_label="F1 macro",
        ),
        width="stretch",
    )

    st.markdown("**Détail des métriques**")
    st.dataframe(
        frame[["code", "categorie", "famille", "precision", "recall", "f1", "auc", "support"]]
        .rename(
            columns={
                "categorie": "catégorie",
                "precision": "précision",
                "recall": "rappel",
                "f1": "F1",
                "auc": "AUC",
            }
        )
        .style.format(
            {"précision": "{:.3f}", "rappel": "{:.3f}", "F1": "{:.3f}", "AUC": "{:.3f}"}
        ),
        hide_index=True,
        width="stretch",
        height=420,
    )


def _tab_learning(history: pd.DataFrame) -> None:
    """Render the learning curves."""
    if history.empty:
        st.info("`core/artifacts/history.json` est introuvable.")
        return

    st.markdown(
        f"L'entraînement a duré **{len(history)} époques**. Les trois courbes sont "
        "séparées volontairement : superposer une perte et une exactitude sur deux axes "
        "rend le graphique illisible et trompeur."
    )

    for title, train_column, val_column in CURVES:
        st.markdown(f"**{title}**")
        st.plotly_chart(
            charts.line_chart(
                history,
                x_column="epoch",
                series=[(train_column, "entraînement"), (val_column, "validation")],
                x_title="époque",
                y_title=title.lower(),
                height=260,
            ),
            width="stretch",
        )

    final = history.iloc[-1]
    gap = float(final["accuracy"]) - float(final["val_accuracy"])
    st.write("")
    layout.note(
        f"À la dernière époque, l'exactitude d'entraînement atteint "
        f"{final['accuracy']:.3f} contre {final['val_accuracy']:.3f} en validation, "
        f"soit un écart de {gap:.3f}. "
        + (
            "L'écart reste contenu : le modèle généralise correctement."
            if gap < 0.08
            else "L'écart se creuse : c'est le signe d'un surapprentissage à surveiller."
        )
    )


def _tab_confusion() -> None:
    """Render the confusion matrix image."""
    path = confusion_matrix_file(state.settings().artifacts_path)
    if path is None:
        st.info("`core/artifacts/confusion_matrix.png` est introuvable.")
        return
    st.markdown(
        "Chaque ligne est la vraie catégorie, chaque colonne la catégorie prédite. "
        "La diagonale porte les bonnes réponses ; les taches hors diagonale sont les "
        "confusions récurrentes — le plus souvent entre catégories voisines : les quatre "
        "familles de livres (10, 2280, 2403, 2705), le mobilier d'intérieur et de jardin "
        "(1560, 2582), ou les jeux vidéo neufs et d'occasion (40, 2462)."
    )
    st.image(str(path), width="stretch")
    st.caption("Image produite par `core/src/models/evaluate.py`.")


def _tab_reading(report_frame: pd.DataFrame, overall: dict[str, float]) -> None:
    """Render the narrative analysis of the results."""
    strongest = report_frame.nlargest(5, "f1")
    weakest = report_frame.nsmallest(5, "f1")

    st.markdown(
        f"L'exactitude globale est de **{overall.get('accuracy', 0):.1%}** et l'AUC macro "
        f"de **{overall.get('auc_macro', 0):.3f}**. L'écart entre les deux dit quelque "
        "chose d'important : le modèle ordonne très bien les classes (AUC élevée), mais "
        "la décision finale par `argmax` reste fragile sur les catégories rares."
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Les cinq catégories les mieux traitées**")
        st.dataframe(
            strongest[["code", "categorie", "f1", "support"]].rename(
                columns={"categorie": "catégorie", "f1": "F1"}
            ),
            hide_index=True,
            width="stretch",
        )
    with right:
        st.markdown("**Les cinq catégories les plus difficiles**")
        st.dataframe(
            weakest[["code", "categorie", "f1", "support"]].rename(
                columns={"categorie": "catégorie", "f1": "F1"}
            ),
            hide_index=True,
            width="stretch",
        )

    st.write("")
    correlation = report_frame["support"].corr(report_frame["f1"])
    st.markdown(
        f"La corrélation entre la taille d'une classe et son F1 vaut **{correlation:.2f}** : "
        "le déséquilibre du jeu de données explique une part réelle des écarts, mais pas "
        "tout — certaines petites catégories très typées s'en sortent bien, tandis que "
        "des catégories volumineuses au vocabulaire générique restent confuses."
    )

    st.markdown(
        "**Pistes d'amélioration identifiées**\n\n"
        "- fusionner ou hiérarchiser les catégories que le modèle confond systématiquement ;\n"
        "- exploiter les images produit, aujourd'hui non utilisées ;\n"
        "- calibrer un seuil de confiance par catégorie pour router les cas douteux "
        "vers une revue manuelle plutôt que de forcer une décision."
    )


def render() -> None:
    """Render the performance page."""
    layout.page_header(
        "📊",
        "Performance du modèle",
        "Les chiffres viennent directement de `core/artifacts/metrics.json`, écrit par "
        "l'étape d'évaluation du pipeline. Rien n'est recalculé ici.",
    )

    settings = state.settings()
    report = load_metrics(settings.artifacts_path)

    if not report.available:
        st.warning(
            "Aucun rapport d'évaluation trouvé. Lancez `python -m core.src.models.evaluate` "
            "pour générer `metrics.json`."
        )
        return

    overall = report.overall
    layout.tiles(
        [
            ("Exactitude", f"{overall.get('accuracy', 0):.1%}", "sur le jeu de test"),
            ("F1 macro", f"{overall.get('f1_macro', 0):.3f}", "toutes classes à poids égal"),
            (
                "F1 pondéré",
                f"{overall.get('f1_weighted', 0):.3f}",
                "pondéré par la taille des classes",
            ),
            ("AUC macro", f"{overall.get('auc_macro', 0):.3f}", "capacité de discrimination"),
        ]
    )

    st.write("")
    tab_class, tab_learning, tab_confusion, tab_reading = st.tabs(
        [
            "Par catégorie",
            "Courbes d'apprentissage",
            "Matrice de confusion",
            "Lecture des résultats",
        ]
    )

    with tab_class:
        _tab_per_class(report.per_class, float(overall.get("f1_macro", 0)))
    with tab_learning:
        _tab_learning(load_history(settings.artifacts_path))
    with tab_confusion:
        _tab_confusion()
    with tab_reading:
        _tab_reading(report.per_class, overall)

    st.markdown(
        f'<div style="height:4px;background:{theme.SERIES_BLUE};border-radius:2px;'
        'margin-top:1.5rem;opacity:0.12"></div>',
        unsafe_allow_html=True,
    )
