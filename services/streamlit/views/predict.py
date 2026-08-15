"""
Prediction page: the live demonstration.

Three tabs, in the order they are shown to a jury: one product, then a batch,
then the raw HTTP contract behind both.
"""

from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd
import streamlit as st

from services.streamlit.clients.api_client import ApiError
from services.streamlit.domain.labels import category_label, category_name
from services.streamlit.domain.samples import BATCH_SAMPLE_CSV, SAMPLES
from services.streamlit.ui import charts, layout, state, theme

TOP_K = 5
MAX_BATCH_ROWS = 200


def _require_auth() -> bool:
    """Show a hint and return False when the user is not authenticated."""
    if state.is_authenticated():
        return True
    st.warning(
        "Les routes d'inférence sont protégées. Connectez-vous depuis la barre latérale "
        "pour lancer une prédiction."
    )
    return False


def _distribution_frame(distribution: dict[str, float]) -> pd.DataFrame:
    """Turn the API probability distribution into a sorted, labelled frame."""
    frame = pd.DataFrame(
        [
            {"code": code, "libelle": category_label(code), "probabilite": float(probability)}
            for code, probability in distribution.items()
        ]
    )
    return frame.sort_values("probabilite", ascending=False).reset_index(drop=True)


def _render_single_result(payload: dict[str, Any], expected: str | None) -> None:
    """Render the outcome of a single prediction."""
    results = payload.get("results", {})
    metadata = payload.get("metadata", {})
    label = str(results.get("label", "?"))
    confidence = results.get("confidence")
    distribution = results.get("distribution") or {}

    match_hint = ""
    if expected:
        match_hint = "conforme à l'attendu" if expected == label else f"attendu : {expected}"

    layout.tiles(
        [
            ("Catégorie prédite", label, category_name(label)),
            (
                "Confiance",
                f"{confidence:.1%}" if isinstance(confidence, (int, float)) else "—",
                "probabilité de la classe retenue",
            ),
            (
                "Temps d'inférence",
                f"{metadata.get('inference_time_ms', 0):.0f} ms",
                "mesuré côté serveur",
            ),
            ("Modèle", str(metadata.get("model_name", "—")), match_hint),
        ]
    )

    if str(metadata.get("model_name", "")).startswith("stub"):
        st.info(
            "L'API tourne en **mode stub** (`API_STUB_MODE=true`) : les prédictions sont "
            "tirées au hasard. Le contrat HTTP est réel, pas le résultat.",
            icon="ℹ️",
        )

    if not distribution:
        return

    st.write("")
    frame = _distribution_frame(distribution)
    top = frame.head(TOP_K)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown(f"**Les {TOP_K} catégories les plus probables**")
        st.plotly_chart(
            charts.horizontal_bar(
                top,
                label_column="libelle",
                value_column="probabilite",
                value_format=".1%",
                height=60 * len(top) + 60,
            ),
            width="stretch",
        )
    with right:
        st.markdown("**Lecture**")
        first, second = top.iloc[0], top.iloc[1]
        gap = first["probabilite"] - second["probabilite"]
        st.markdown(
            f"Le modèle place `{first['code']}` en tête avec "
            f"**{first['probabilite']:.1%}**, devant `{second['code']}` "
            f"({second['probabilite']:.1%}) — un écart de {gap:.1%}."
        )
        if gap < 0.15:
            st.markdown(
                "Écart faible : c'est typiquement le cas qu'on enverrait en "
                "validation humaine plutôt que de le classer automatiquement."
            )
        else:
            st.markdown("Écart net : la fiche peut être classée sans revue manuelle.")
        st.dataframe(
            top[["libelle", "probabilite"]].rename(
                columns={"libelle": "catégorie", "probabilite": "probabilité"}
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "probabilité": st.column_config.ProgressColumn(
                    "probabilité", format="%.1f%%", min_value=0.0, max_value=1.0
                )
            },
        )

    with st.expander("Réponse brute de l'API"):
        st.json(payload)


def _tab_single() -> None:
    """Render the single-product tab."""
    st.markdown(
        "Un titre, une description optionnelle, un appel `POST /predict`. "
        "Les exemples proposés ont été rédigés par l'équipe pour couvrir des familles "
        "de produits variées — ils ne sont pas extraits du jeu d'entraînement, ce qui "
        "évite de démontrer le modèle sur des fiches qu'il a déjà vues. La catégorie "
        "indiquée entre parenthèses est celle qu'on attend, pas une vérité terrain."
    )

    options = ["— saisie libre —"] + [
        f"{sample['designation'][:60]}  ({category_label(sample['expected'])})"
        for sample in SAMPLES
    ]
    choice = st.selectbox("Exemple", options, index=1)

    if choice == options[0]:
        default_designation, default_description, expected = "", "", None
    else:
        sample = SAMPLES[options.index(choice) - 1]
        default_designation = sample["designation"]
        default_description = sample["description"]
        expected = sample["expected"]

    with st.form("single_prediction", border=False):
        designation = st.text_input(
            "Désignation",
            value=default_designation,
            placeholder="Titre de la fiche produit",
        )
        description = st.text_area(
            "Description (optionnelle)",
            value=default_description,
            height=110,
        )
        submitted = st.form_submit_button("Classer ce produit", type="primary")

    if submitted:
        if not designation.strip():
            st.error("La désignation est obligatoire.")
            return
        with st.spinner("Appel de l'API…"):
            try:
                payload = state.api_client().predict(designation, description)
            except ApiError as exc:
                st.error(str(exc))
                return
        st.session_state["last_prediction"] = {"payload": payload, "expected": expected}

    last = st.session_state.get("last_prediction")
    if last:
        st.divider()
        _render_single_result(last["payload"], last["expected"])


def _read_batch_file(uploaded: Any) -> pd.DataFrame:
    """Read an uploaded CSV into a frame, normalising the expected columns."""
    frame = pd.read_csv(uploaded)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if "designation" not in frame.columns:
        raise ValueError("Le fichier doit contenir au moins une colonne `designation`.")
    if "description" not in frame.columns:
        frame["description"] = ""
    return frame[["designation", "description"]].fillna("")


def _tab_batch() -> None:
    """Render the batch tab."""
    st.markdown(
        "`POST /predict/batch` traite toute la liste en une seule passe du modèle. "
        "C'est le mode utilisé pour un import de catalogue : plus rapide qu'une boucle "
        "d'appels unitaires, et une seule authentification."
    )

    uploaded = st.file_uploader("Fichier CSV (`designation`, `description`)", type=["csv"])
    use_sample = st.checkbox("Utiliser le jeu d'exemple fourni", value=uploaded is None)

    frame: pd.DataFrame | None = None
    if uploaded is not None and not use_sample:
        try:
            frame = _read_batch_file(uploaded)
        except (ValueError, pd.errors.ParserError) as exc:
            st.error(str(exc))
            return
    elif use_sample:
        frame = _read_batch_file(io.StringIO(BATCH_SAMPLE_CSV))

    if frame is None:
        st.info("Chargez un CSV ou cochez le jeu d'exemple.")
        return

    if len(frame) > MAX_BATCH_ROWS:
        st.warning(f"Fichier tronqué aux {MAX_BATCH_ROWS} premières lignes pour la démonstration.")
        frame = frame.head(MAX_BATCH_ROWS)

    st.dataframe(frame, hide_index=True, width="stretch", height=240)

    if st.button(f"Classer les {len(frame)} produits", type="primary"):
        with st.spinner("Appel de l'API…"):
            try:
                payload = state.api_client().predict_batch(frame.to_dict("records"))
            except ApiError as exc:
                st.error(str(exc))
                return
        st.session_state["batch_results"] = {"payload": payload, "inputs": frame}

    stored = st.session_state.get("batch_results")
    if not stored:
        return

    st.divider()
    payload = stored["payload"]
    inputs: pd.DataFrame = stored["inputs"]
    results = payload.get("results", [])
    metadata = payload.get("metadata", {})

    if len(results) != len(inputs):
        st.warning("Le nombre de résultats ne correspond pas au nombre de lignes envoyées.")

    output = inputs.copy().head(len(results))
    output["code"] = [str(item.get("label", "")) for item in results]
    output["catégorie"] = [category_name(item.get("label", "")) for item in results]
    output["confiance"] = [item.get("confidence") for item in results]

    total_ms = float(metadata.get("inference_time_ms", 0))
    layout.tiles(
        [
            ("Produits classés", str(len(results)), "en un seul appel"),
            ("Durée totale", f"{total_ms:.0f} ms", "mesurée côté serveur"),
            (
                "Par produit",
                f"{total_ms / max(len(results), 1):.0f} ms",
                "coût unitaire moyen",
            ),
            ("Catégories distinctes", str(output["code"].nunique()), "sur 27 possibles"),
        ]
    )

    st.write("")
    st.dataframe(
        output[["designation", "code", "catégorie", "confiance"]],
        hide_index=True,
        width="stretch",
        column_config={
            "confiance": st.column_config.ProgressColumn(
                "confiance", format="%.1f%%", min_value=0.0, max_value=1.0
            )
        },
    )

    counts = (
        output.groupby(["code", "catégorie"]).size().reset_index(name="produits")
    )
    counts["libelle"] = counts["code"] + " · " + counts["catégorie"]
    counts = counts.sort_values("produits", ascending=False)

    st.markdown("**Répartition des catégories prédites**")
    st.plotly_chart(
        charts.horizontal_bar(
            counts,
            label_column="libelle",
            value_column="produits",
            value_format=".0f",
            color=theme.SERIES_AQUA,
            height=max(220, 38 * len(counts) + 60),
        ),
        width="stretch",
    )

    st.download_button(
        "Télécharger les résultats (CSV)",
        data=output.to_csv(index=False).encode("utf-8-sig"),
        file_name="predictions_rakuten.csv",
        mime="text/csv",
    )


def _tab_contract() -> None:
    """Render the API contract tab."""
    st.markdown(
        "Tout ce que l'interface affiche passe par ces routes. Le même appel depuis "
        "un terminal donne le même résultat — l'interface n'a aucun privilège."
    )

    base_url = st.session_state["api_url"]
    st.markdown("**1. Obtenir un jeton**")
    st.code(
        f"curl -X POST {base_url}/auth/login \\\n"
        "  -d 'username=admin&password=<mot de passe>'",
        language="bash",
    )
    st.markdown("**2. Classer un produit**")
    st.code(
        f"curl -X POST {base_url}/predict \\\n"
        "  -H 'Authorization: Bearer $TOKEN' \\\n"
        "  -H 'Content-Type: application/json' \\\n"
        "  -d '"
        + json.dumps(
            {
                "inputs": {
                    "designation": SAMPLES[0]["designation"],
                    "description": SAMPLES[0]["description"],
                },
                "options": {"return_confidence": True, "return_distribution": True},
            },
            ensure_ascii=False,
        )
        + "'",
        language="bash",
    )

    st.divider()
    st.markdown("**Routes exposées**")
    try:
        schema = state.api_client().openapi()
    except ApiError as exc:
        st.info(f"Schéma OpenAPI indisponible : {exc}")
        return

    rows = [
        {
            "Méthode": method.upper(),
            "Route": path,
            "Résumé": operation.get("summary", ""),
            "Protégée": "oui" if operation.get("security") or "/train" in path else "non",
        }
        for path, operations in schema.get("paths", {}).items()
        for method, operation in operations.items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Documentation interactive complète : {base_url}/docs")


def render() -> None:
    """Render the prediction page."""
    layout.page_header(
        "🎯",
        "Classer un produit",
        "La démonstration en conditions réelles : l'interface envoie du texte à l'API, "
        "l'API rejoue la chaîne de features, le modèle répond une catégorie et sa confiance.",
    )

    tab_single, tab_batch, tab_contract = st.tabs(
        ["Produit unique", "Lot de produits", "Contrat d'API"]
    )

    with tab_single:
        if _require_auth():
            _tab_single()
    with tab_batch:
        if _require_auth():
            _tab_batch()
    with tab_contract:
        _tab_contract()
