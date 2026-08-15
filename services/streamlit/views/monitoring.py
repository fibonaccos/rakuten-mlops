"""
Monitoring page: availability of the stack, response times measured from the
client side, and the state of the observability roadmap.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from services.streamlit.clients.api_client import ApiError
from services.streamlit.domain.catalog import load_demo_products
from services.streamlit.ui import charts, layout, state, theme

BENCHMARK_CALLS = 10

ROADMAP = [
    (
        "Fait — sondes applicatives",
        "`GET /health` (le process répond) et `GET /ready` (les artefacts sont chargés) "
        "sont branchées sur les healthchecks Docker et sur cette interface.",
    ),
    (
        "En cours — export Prometheus",
        "Exposer `/metrics` sur l'API (latence par route, volume, taux d'erreur, "
        "distribution des classes prédites). Le dossier `infra/prometheus/` attend "
        "la configuration de scrape.",
    ),
    (
        "À venir — tableaux Grafana",
        "Deux tableaux de bord : santé applicative (temps de réponse, CPU, mémoire) "
        "et dérive (distribution des prédictions comparée au jeu d'entraînement, "
        "confiance moyenne, part des cas sous seuil). Dossier `infra/grafana/`.",
    ),
]


def _summary_tiles(frame: pd.DataFrame) -> None:
    """Render the latency summary computed from the session call log."""
    latencies = frame["latence_ms"]
    error_rate = 1 - frame["ok"].mean()
    layout.tiles(
        [
            ("Appels mesurés", str(len(frame)), "depuis l'ouverture de la session"),
            ("Latence médiane", f"{latencies.median():.0f} ms", "aller-retour complet"),
            ("95e centile", f"{latencies.quantile(0.95):.0f} ms", "les 5 % les plus lents"),
            (
                "Taux d'erreur",
                f"{error_rate:.0%}",
                "réponses hors 2xx ou API injoignable",
            ),
        ]
    )


def _benchmark() -> None:
    """Run a small burst of calls to populate the latency chart."""
    client = state.api_client()
    products = load_demo_products(state.settings().assets_path)
    progress = st.progress(0.0, text="Mesure en cours…")
    failures = 0

    for index in range(BENCHMARK_CALLS):
        try:
            if state.is_authenticated() and products and index % 2 == 1:
                product = products[index % len(products)]
                client.predict(
                    product.designation,
                    product.description,
                    with_distribution=False,
                )
            else:
                client.health()
        except ApiError:
            failures += 1
        progress.progress(
            (index + 1) / BENCHMARK_CALLS, text=f"Appel {index + 1}/{BENCHMARK_CALLS}"
        )

    progress.empty()
    if failures:
        st.warning(f"{failures} appel(s) en échec pendant la mesure.")
    else:
        st.success(f"{BENCHMARK_CALLS} appels effectués.")


def render() -> None:
    """Render the monitoring page."""
    layout.page_header(
        "🩺",
        "Monitoring de l'application",
        "Disponibilité des services et temps de réponse réels, mesurés depuis ce client. "
        "Les mesures portent sur la session en cours : c'est une observation de bout en "
        "bout, pas un remplacement de la supervision serveur.",
    )

    statuses = layout.current_statuses()
    columns = st.columns(len(statuses))
    for column, status in zip(columns, statuses):
        with column:
            st.markdown(
                f'<div class="tile"><div class="label">{status.name}</div>'
                f'<div class="value" style="font-size:1.15rem;color:'
                f'{theme.STATUS["good"] if status.up else theme.STATUS["critical"]}">'
                f"{'disponible' if status.up else 'indisponible'}</div>"
                f'<div class="hint">{status.detail}</div></div>',
                unsafe_allow_html=True,
            )

    st.divider()

    left, right = st.columns([3, 1], gap="large")
    with left:
        st.subheader("Temps de réponse")
    with right:
        st.write("")
        if st.button("Lancer une mesure", type="primary", width="stretch"):
            _benchmark()

    frame = state.call_log_frame()
    if frame.empty:
        st.info(
            "Aucun appel enregistré pour l'instant. Lancez une mesure, ou utilisez la page "
            "de prédiction : chaque requête HTTP émise par l'interface est chronométrée ici."
        )
    else:
        _summary_tiles(frame)
        st.write("")
        st.plotly_chart(charts.latency_chart(frame), width="stretch")

        by_route = (
            frame.groupby("route")
            .agg(
                appels=("latence_ms", "size"),
                mediane_ms=("latence_ms", "median"),
                p95_ms=("latence_ms", lambda values: values.quantile(0.95)),
                max_ms=("latence_ms", "max"),
                erreurs=("ok", lambda values: int((~values).sum())),
            )
            .reset_index()
            .sort_values("appels", ascending=False)
        )
        st.markdown("**Par route**")
        st.dataframe(
            by_route.rename(
                columns={
                    "route": "route",
                    "mediane_ms": "médiane (ms)",
                    "p95_ms": "p95 (ms)",
                    "max_ms": "max (ms)",
                }
            ).style.format({"médiane (ms)": "{:.0f}", "p95 (ms)": "{:.0f}", "max (ms)": "{:.0f}"}),
            hide_index=True,
            width="stretch",
        )

        layout.note(
            "Le premier appel à `/predict` est toujours plus lent : le modèle de plongement "
            "et le réseau Keras sont chargés au démarrage, mais les premières inférences "
            "paient encore l'initialisation des graphes de calcul."
        )

        with st.expander("Journal détaillé des appels"):
            st.dataframe(frame.iloc[::-1], hide_index=True, width="stretch", height=320)

    st.divider()
    st.subheader("Supervision : où nous en sommes")
    for title, description in ROADMAP:
        st.markdown(f'<div class="step"><b>{title}</b>{description}</div>', unsafe_allow_html=True)
        st.write("")

    layout.note(
        "Suivre la dérive suppose de comparer la distribution des prédictions en "
        "production à celle du jeu d'entraînement, visible page « Données ». C'est cette "
        "comparaison qui déclenchera le réentraînement automatique côté Airflow."
    )
