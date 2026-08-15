"""
MLOps page: orchestration, experiment tracking, retraining and continuous
integration — the parts of the project that live around the model.
"""

from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

from services.streamlit.clients.api_client import ApiError
from services.streamlit.clients.mlflow_client import MlflowRun, read_local_runs
from services.streamlit.settings import PROJECT_ROOT
from services.streamlit.ui import charts, layout, state, theme

CI_JOBS = [
    (
        "lint",
        "`ruff` (format + règles + tri des imports) et `mypy`, via les hooks pre-commit "
        "joués sur tout le dépôt. Le même hook tourne en local avant chaque commit.",
    ),
    (
        "test-api",
        "Installation des groupes `api` et `dev` avec `uv`, puis `pytest tests/api`. "
        "Le mode stub permet de tester les routes sans embarquer le modèle.",
    ),
    (
        "docker-build",
        "Construction des images `api`, `mlflow` et `airflow` en parallèle, avec cache "
        "GitHub Actions. Ce job ne démarre que si lint et tests passent.",
    ),
]

RUN_STATE_LABELS = {
    "success": "succès",
    "failed": "échec",
    "running": "en cours",
    "queued": "en attente",
}


def _runs_frame(runs: list[MlflowRun]) -> pd.DataFrame:
    """Turn MLflow runs into a display-ready frame."""
    return pd.DataFrame(
        [
            {
                "run": run.run_name or run.run_id[:8],
                "statut": run.status,
                "début": run.start_time,
                "durée (s)": round(run.duration_s, 1) if run.duration_s else None,
                "accuracy": run.metrics.get("accuracy"),
                "val_accuracy": run.metrics.get("val_accuracy"),
                "val_auc": run.metrics.get("val_auc"),
                "commit git": (run.git_commit or "")[:8],
            }
            for run in runs
        ]
    )


def _tab_airflow() -> None:
    """Render the Airflow orchestration tab."""
    st.markdown(
        "Airflow porte le pipeline de bout en bout : récupération des nouvelles données, "
        "nettoyage, construction des features, réentraînement, puis publication des "
        "artefacts que l'API recharge. Les DAG vivent dans `services/airflow/dags/`."
    )

    client = state.airflow_client()
    health = client.health()
    if not health:
        st.warning(
            f"Airflow ne répond pas sur `{st.session_state['airflow_url']}`. "
            "Démarrez le stack avec `docker compose up -d airflow-webserver airflow-scheduler`."
        )
        return

    components = [
        ("Base de métadonnées", health.get("metadatabase", {}).get("status")),
        ("Scheduler", health.get("scheduler", {}).get("status")),
        ("Triggerer", health.get("triggerer", {}).get("status")),
    ]
    for name, status in components:
        st.markdown(
            layout.pill_html(name, status == "healthy", status or "inconnu"),
            unsafe_allow_html=True,
        )

    dags = client.list_dags()
    if not dags:
        st.info(
            "Aucun DAG n'est chargé pour l'instant. Le dossier `services/airflow/dags/` "
            "est monté dans le conteneur : tout fichier déposé y est détecté automatiquement."
        )
        return

    st.write("")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "DAG": dag.dag_id,
                    "actif": not dag.is_paused,
                    "planification": dag.schedule or "—",
                    "description": dag.description or "",
                    "tags": ", ".join(dag.tags),
                }
                for dag in dags
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    selected = st.selectbox("DAG à inspecter", [dag.dag_id for dag in dags])
    runs = client.list_runs(selected)
    if runs:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "exécution": run.run_id,
                        "état": RUN_STATE_LABELS.get(run.state, run.state),
                        "type": run.run_type,
                        "début": run.start_date,
                        "durée (s)": round(run.duration_s, 1) if run.duration_s else None,
                    }
                    for run in runs
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    else:
        st.caption("Ce DAG n'a pas encore été exécuté.")

    confirm = st.checkbox(f"Confirmer le déclenchement manuel de `{selected}`")
    if st.button("Déclencher une exécution", disabled=not confirm):
        try:
            run = client.trigger_dag(selected)
        except requests.RequestException as exc:
            st.error(f"Déclenchement refusé par Airflow : {exc}")
        else:
            st.success(f"Exécution `{run.get('dag_run_id', '?')}` créée.")
            st.rerun()


def _tab_mlflow() -> None:
    """Render the MLflow tracking tab."""
    st.markdown(
        "Chaque entraînement écrit dans MLflow ses paramètres, ses métriques, ses "
        "artefacts et **le hash du commit Git** qui a produit le code. C'est ce qui "
        "permet de remonter d'un modèle en production jusqu'à la ligne de code exacte."
    )

    client = state.mlflow_client()
    runs: list[MlflowRun] = []
    source = ""

    if client.is_up():
        experiments = client.list_experiments()
        if experiments:
            names = {exp.get("name", exp.get("experiment_id", "")): exp for exp in experiments}
            chosen = st.selectbox("Expérience", list(names))
            runs = client.search_runs([names[chosen].get("experiment_id", "")])
            source = f"serveur de tracking `{st.session_state['mlflow_url']}`"
        else:
            st.info("Le serveur MLflow répond mais ne contient aucune expérience.")
    else:
        runs = read_local_runs(state.settings().mlruns_path)
        if runs:
            source = f"dossier local `{state.settings().mlruns_path}`"
            st.info(
                "Le serveur MLflow ne répond pas : les runs affichés sont lus directement "
                "dans le magasin de fichiers local produit par les entraînements lancés "
                "hors Docker.",
                icon="ℹ️",
            )
        else:
            st.warning(
                f"MLflow est injoignable sur `{st.session_state['mlflow_url']}` et aucun run "
                "local n'a été trouvé. Démarrez `docker compose up -d mlflow`, ou lancez un "
                "entraînement pour peupler `tracking/mlflow/mlruns`."
            )
            return

    if not runs:
        return

    frame = _runs_frame(runs)
    st.caption(f"Source : {source} · {len(runs)} run(s)")
    st.dataframe(frame, hide_index=True, width="stretch")

    tracked = frame.dropna(subset=["val_accuracy"])
    if len(tracked) > 1:
        st.markdown("**Exactitude de validation par run**")
        st.plotly_chart(
            charts.horizontal_bar(
                tracked.sort_values("val_accuracy", ascending=False),
                label_column="run",
                value_column="val_accuracy",
                value_format=".3f",
                color=theme.SERIES_AQUA,
                height=max(220, 38 * len(tracked) + 60),
            ),
            width="stretch",
        )

    latest = runs[0]
    if latest.params:
        with st.expander(f"Paramètres du run `{latest.run_name or latest.run_id[:8]}`"):
            st.json(latest.params)


def _tab_training() -> None:
    """Render the training-job tab."""
    st.markdown(
        "L'API expose l'entraînement comme une ressource : `POST /train` répond "
        "immédiatement en `202 Accepted` avec un identifiant de job, et le client "
        "interroge `GET /train/{id}` pour suivre l'avancement. Un seul job actif à la "
        "fois — une seconde soumission renvoie `409 Conflict`."
    )

    if not state.is_authenticated():
        st.warning("Connectez-vous depuis la barre latérale pour piloter l'entraînement.")
        return

    client = state.api_client()

    left, right = st.columns([2, 1], gap="large")
    with left:
        run_name = st.text_input("Nom du run", placeholder="reentrainement-demo")
    with right:
        st.write("")
        if st.button("Lancer un entraînement", type="primary"):
            try:
                job = client.submit_training(run_name or None)
            except ApiError as exc:
                st.error(str(exc))
            else:
                st.session_state["training_job_id"] = job.get("job_id")
                st.success(f"Job `{job.get('job_id')}` soumis.")

    try:
        jobs = client.list_jobs(limit=10)
    except ApiError as exc:
        st.error(str(exc))
        return

    if not jobs:
        st.caption("Aucun entraînement n'a encore été soumis dans cette instance de l'API.")
        return

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "job": job.get("job_id", "")[:8],
                    "état": job.get("status"),
                    "nom": job.get("run_name") or "—",
                    "soumis": job.get("created_at"),
                    "durée (s)": job.get("duration_s"),
                    "erreur": job.get("error") or "",
                }
                for job in jobs
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    active = next((job for job in jobs if job.get("status") in {"queued", "running"}), None)
    if active is None:
        st.caption("Aucun job actif.")
        return

    st.info(f"Job `{active.get('job_id')}` en cours ({active.get('status')}).")
    columns = st.columns(2)
    with columns[0]:
        if st.button("Rafraîchir l'état"):
            st.rerun()
    with columns[1]:
        if st.button("Annuler le job"):
            try:
                client.cancel_job(str(active.get("job_id")))
            except ApiError as exc:
                st.error(str(exc))
            else:
                st.rerun()


def _tab_ci() -> None:
    """Render the continuous integration tab."""
    st.markdown(
        "Chaque `push` sur `master` et chaque pull request déclenchent le pipeline "
        "GitHub Actions. Les exécutions périmées d'une même branche sont annulées "
        "automatiquement pour ne pas gaspiller de minutes de CI."
    )
    st.write("")
    layout.steps([(name, description) for name, description in CI_JOBS])

    st.write("")
    layout.note(
        "Le job `docker-build` dépend de `lint` et `test-api` : une image n'est jamais "
        "construite à partir d'un code qui ne passe ni le lint ni les tests."
    )

    workflow = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
    if workflow.is_file():
        with st.expander("Voir le workflow"):
            st.code(workflow.read_text(encoding="utf-8"), language="yaml")


def render() -> None:
    """Render the MLOps page."""
    layout.page_header(
        "⚙️",
        "La chaîne MLOps",
        "Orchestration, traçabilité des expériences, réentraînement à la demande et "
        "intégration continue : ce qui transforme un notebook en service qu'on peut "
        "exploiter et faire évoluer.",
    )

    tab_airflow, tab_mlflow, tab_training, tab_ci = st.tabs(
        ["Airflow", "MLflow", "Réentraînement", "CI/CD"]
    )

    with tab_airflow:
        _tab_airflow()
    with tab_mlflow:
        _tab_mlflow()
    with tab_training:
        _tab_training()
    with tab_ci:
        _tab_ci()
