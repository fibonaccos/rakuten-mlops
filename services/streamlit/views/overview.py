"""
Landing page: what the project does, how the stack is wired, and what state
it is in right now.
"""

from __future__ import annotations

import streamlit as st

from services.streamlit.domain.artifacts import load_metrics
from services.streamlit.domain.catalog import load_catalogue
from services.streamlit.ui import layout, state, theme

ARCHITECTURE_DOT = """
digraph stack {
    rankdir=LR;
    bgcolor="transparent";
    node [shape=box style="rounded,filled" fontname="Segoe UI" fontsize=11
          color="#e1e0d9" penwidth=1.2 margin="0.18,0.12"];
    edge [color="#898781" fontname="Segoe UI" fontsize=9 arrowsize=0.7];

    raw      [label="data.csv\\ncatalogue brut" fillcolor="#f9f9f7" fontcolor="#52514e"];
    clean    [label="clean_data.py" fillcolor="#f9f9f7" fontcolor="#52514e"];
    features [label="build_features.py\\n396 → 120 dims" fillcolor="#f9f9f7" fontcolor="#52514e"];
    train    [label="train.py\\nKeras" fillcolor="#f9f9f7" fontcolor="#52514e"];

    airflow  [label="Airflow\\norchestration" fillcolor="#eb6834" fontcolor="white"];
    mlflow   [label="MLflow\\ntracking" fillcolor="#1baf7a" fontcolor="white"];
    api      [label="FastAPI\\n/predict · /train" fillcolor="#2a78d6" fontcolor="white"];
    ui       [label="Streamlit\\ncette interface" fillcolor="#bf0000" fontcolor="white"];

    artifacts [label="artifacts/\\nmodel.keras · scaler · pca"
               shape=note fillcolor="#f9f9f7" fontcolor="#52514e"];

    raw -> clean -> features -> train -> artifacts;
    artifacts -> api [label="chargés au démarrage"];
    api -> ui [label="HTTP + JWT"];
    airflow -> clean [style=dashed label="déclenche"];
    train -> mlflow [style=dashed label="logue runs"];
    mlflow -> ui [style=dashed];
    airflow -> ui [style=dashed];
}
"""

SERVICES = [
    (
        "`api`",
        "FastAPI",
        "8000",
        "Authentification JWT, inférence unitaire et par lot, jobs d'entraînement",
    ),
    (
        "`mlflow`",
        "MLflow",
        "5000",
        "Suivi des runs : paramètres, métriques, artefacts, commit Git",
    ),
    (
        "`airflow-webserver`",
        "Airflow",
        "8080",
        "Orchestration du pipeline de données et de réentraînement",
    ),
    (
        "`streamlit`",
        "Streamlit",
        "8501",
        "Interface de démonstration et de pilotage (cette application)",
    ),
]

WALKTHROUGH = [
    ("1 · Vue d'ensemble", "Le problème métier, l'architecture, l'état du stack."),
    ("2 · Données & features", "Le jeu de données, son déséquilibre, la chaîne de features."),
    ("3 · Prédiction", "La démonstration : un produit, un lot, le contrat d'API."),
    ("4 · Performance", "Ce que vaut le modèle, catégorie par catégorie."),
    ("5 · MLOps", "Airflow, MLflow, réentraînement piloté, CI/CD."),
    ("6 · Monitoring", "Disponibilité, temps de réponse, suite du chantier supervision."),
]


def render() -> None:
    """Render the overview page."""
    layout.page_header(
        "🏠",
        "Classification automatique du catalogue Rakuten",
        "Un produit arrive avec un titre et une description libres. Le modèle lui attribue "
        "l'une des 27 catégories du catalogue, et toute la chaîne — données, entraînement, "
        "service, supervision — est automatisée et reproductible.",
    )

    settings = state.settings()
    report = load_metrics(settings.artifacts_path)
    catalogue = load_catalogue(settings.assets_path)

    layout.tiles(
        [
            ("Catégories", "27", "codes `prdtypecode`"),
            (
                "Produits",
                f"{catalogue.total_products:,}".replace(",", " ") if catalogue.available else "—",
                "fiches du catalogue étiquetées",
            ),
            ("Features", "396 → 120", "statistiques + embeddings, réduits par PCA"),
            (
                "F1 macro",
                f"{report.overall.get('f1_macro', 0):.3f}" if report.available else "—",
                "sur le jeu de test",
            ),
        ]
    )

    st.write("")
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Le problème")
        st.markdown(
            "Un catalogue e-commerce reçoit des milliers de fiches produit par jour, "
            "rédigées par des vendeurs différents, dans plusieurs langues, avec des "
            "descriptions parfois vides. Les ranger à la main ne passe pas à l'échelle, "
            "et une fiche mal classée est une fiche que personne ne trouve.\n\n"
            "Le modèle transforme le texte libre en un vecteur de 120 dimensions, puis "
            "un réseau dense produit une distribution de probabilités sur les 27 "
            "catégories. La catégorie retenue est l'`argmax`, et l'API renvoie aussi la "
            "confiance associée — c'est elle qui permet de router les cas douteux vers "
            "une validation humaine."
        )

    with right:
        st.subheader("État du stack")
        for status in layout.current_statuses():
            st.markdown(
                layout.pill_html(status.name, status.up, status.detail),
                unsafe_allow_html=True,
            )
        st.caption(
            "Sondes `GET /health`, `/ready` sur l'API, `/health` sur MLflow et Airflow. "
            "Un service éteint n'empêche jamais les autres pages de fonctionner."
        )

    st.divider()

    st.subheader("Architecture")
    st.graphviz_chart(ARCHITECTURE_DOT, width="stretch")
    st.caption(
        "Trait plein : le chemin de la donnée. Trait pointillé : le pilotage et "
        "l'observation. Les artefacts ne sont pas empaquetés dans les images Docker, "
        "ils sont montés au démarrage — mettre à jour le modèle ne demande pas de rebuild."
    )

    st.write("")
    st.markdown("**Les services du `docker-compose`**")
    st.dataframe(
        {
            "Service": [row[0] for row in SERVICES],
            "Techno": [row[1] for row in SERVICES],
            "Port": [row[2] for row in SERVICES],
            "Rôle": [row[3] for row in SERVICES],
        },
        hide_index=True,
        width="stretch",
    )

    st.divider()

    st.subheader("Déroulé de la démonstration")
    columns = st.columns(3)
    for index, (title, description) in enumerate(WALKTHROUGH):
        with columns[index % 3]:
            layout.step_card(title, description)
        if index % 3 == 2:
            st.write("")

    st.write("")
    layout.note(
        "Cette interface ne contient aucune logique de machine learning : elle appelle "
        "l'API en HTTP, exactement comme le ferait n'importe quel client. Ce qui est "
        "affiché ici est ce que le service renvoie réellement."
    )

    st.markdown(
        f'<div style="height:4px;background:{theme.BRAND};border-radius:2px;'
        'margin-top:1.5rem;opacity:0.15"></div>',
        unsafe_allow_html=True,
    )
