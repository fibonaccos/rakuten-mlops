"""
Entry point of the Streamlit front-end.

Run it from the repository root:

    streamlit run services/streamlit/app.py

Streamlit puts the script's own directory at the head of ``sys.path``, not the
project root, so the repository root is prepended here before importing the
application packages.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from services.streamlit.ui import layout, state  # noqa: E402
from services.streamlit.views import (  # noqa: E402
    data,
    mlops,
    monitoring,
    overview,
    performance,
    predict,
)

PAGES = [
    (overview.render, "Vue d'ensemble", "🏠", "vue-ensemble"),
    (data.render, "Données & features", "🗂️", "donnees"),
    (predict.render, "Prédiction", "🎯", "prediction"),
    (performance.render, "Performance", "📊", "performance"),
    (mlops.render, "MLOps", "⚙️", "mlops"),
    (monitoring.render, "Santé", "🩺", "sante"),
]


def main() -> None:
    """Configure the app, build the navigation and run the selected page."""
    st.set_page_config(
        page_title="Rakuten MLOps",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    state.init_state()
    layout.inject_css()

    navigation = st.navigation(
        [
            st.Page(render, title=title, icon=icon, url_path=url_path, default=index == 0)
            for index, (render, title, icon, url_path) in enumerate(PAGES)
        ]
    )

    layout.sidebar()
    navigation.run()


main()
