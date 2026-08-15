"""
Shared layout pieces: global styles, page headers, stat tiles, status pills
and the sidebar.

Keeping these in one module is what makes the six pages look like one product
rather than six notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from services.streamlit.clients.airflow_client import AirflowClient
from services.streamlit.clients.api_client import ApiClient, ApiError
from services.streamlit.clients.mlflow_client import MlflowClient
from services.streamlit.ui import state, theme

# How long a service status is trusted before being probed again.
STATUS_TTL_SECONDS = 15


@dataclass(frozen=True)
class ServiceStatus:
    """Result of a health probe, as displayed by the sidebar pills."""

    name: str
    up: bool
    detail: str


def inject_css() -> None:
    """Inject the global stylesheet. Called once per rerun from the entrypoint."""
    st.markdown(
        f"""
        <style>
            .block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
            h1, h2, h3 {{ color: {theme.INK_PRIMARY}; letter-spacing: -0.01em; }}

            .page-header {{ margin-bottom: 1.4rem; }}
            .page-header .title {{
                font-size: 1.9rem; font-weight: 700; line-height: 1.2;
                color: {theme.INK_PRIMARY};
            }}
            .page-header .subtitle {{
                font-size: 1rem; color: {theme.INK_SECONDARY}; margin-top: 0.35rem;
                max-width: 62ch;
            }}

            .tile {{
                border: 1px solid rgba(11,11,11,0.10); border-radius: 12px;
                padding: 0.9rem 1rem; background: {theme.SURFACE}; height: 100%;
            }}
            .tile .label {{
                font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em;
                color: {theme.INK_MUTED};
            }}
            .tile .value {{
                font-size: 1.65rem; font-weight: 700; color: {theme.INK_PRIMARY};
                line-height: 1.25; margin-top: 0.15rem;
            }}
            .tile .hint {{ font-size: 0.82rem; color: {theme.INK_SECONDARY}; }}

            .pill {{
                display: inline-flex; align-items: center; gap: 0.4rem;
                padding: 0.25rem 0.6rem; border-radius: 999px;
                font-size: 0.82rem; font-weight: 600; margin: 0.15rem 0;
                border: 1px solid rgba(11,11,11,0.10);
            }}
            .pill .dot {{ width: 8px; height: 8px; border-radius: 50%; }}
            .pill .detail {{ font-weight: 400; color: {theme.INK_SECONDARY}; }}

            .note {{
                border-left: 3px solid {theme.SERIES_BLUE};
                padding: 0.55rem 0.9rem; background: {theme.PLANE};
                border-radius: 0 8px 8px 0; color: {theme.INK_SECONDARY};
                font-size: 0.9rem;
            }}

            .step {{
                border: 1px solid rgba(11,11,11,0.10); border-radius: 10px;
                padding: 0.7rem 0.9rem; background: {theme.SURFACE};
                font-size: 0.88rem; color: {theme.INK_SECONDARY}; height: 100%;
            }}
            .step b {{ color: {theme.INK_PRIMARY}; display: block; margin-bottom: 0.2rem; }}

            div[data-testid="stMetricValue"] {{ font-size: 1.5rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(icon: str, title: str, subtitle: str) -> None:
    """
    Render the title block of a page.

    Args:
        icon: Emoji shown before the title.
        title: Page title.
        subtitle: One or two sentences framing what the page shows.
    """
    st.markdown(
        f"""
        <div class="page-header">
            <div class="title">{icon} {title}</div>
            <div class="subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def tiles(items: list[tuple[str, str, str]]) -> None:
    """
    Render a row of stat tiles.

    Args:
        items: Triples of ``(label, value, hint)``. The hint may be empty.
    """
    columns = st.columns(len(items))
    for column, (label, value, hint) in zip(columns, items):
        with column:
            st.markdown(
                f"""
                <div class="tile">
                    <div class="label">{label}</div>
                    <div class="value">{value}</div>
                    <div class="hint">{hint}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def note(text: str) -> None:
    """Render a discreet explanatory note."""
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def steps(items: list[tuple[str, str]]) -> None:
    """
    Render a row of small cards describing a pipeline stage.

    Args:
        items: Pairs of ``(title, description)``.
    """
    columns = st.columns(len(items))
    for column, (title, description) in zip(columns, items):
        with column:
            st.markdown(
                f'<div class="step"><b>{title}</b>{description}</div>',
                unsafe_allow_html=True,
            )


def pill_html(label: str, up: bool, detail: str = "") -> str:
    """
    Build the HTML of a status pill.

    Args:
        label: Service name.
        up: Whether the service answered.
        detail: Short qualifier shown after the label.

    Returns:
        str: HTML snippet.
    """
    color = theme.STATUS["good"] if up else theme.STATUS["critical"]
    mark = "●" if up else "○"
    suffix = f'<span class="detail">{detail}</span>' if detail else ""
    return (
        f'<span class="pill" title="{label}">'
        f'<span class="dot" style="background:{color}"></span>'
        f"{mark} {label} {suffix}</span>"
    )


def status_pills(statuses: list[ServiceStatus]) -> None:
    """Render a vertical stack of status pills."""
    for status in statuses:
        st.markdown(pill_html(status.name, status.up, status.detail), unsafe_allow_html=True)


@st.cache_data(ttl=STATUS_TTL_SECONDS, show_spinner=False)
def probe_services(
    api_url: str,
    mlflow_url: str,
    airflow_url: str,
    airflow_user: str,
    airflow_password: str,
) -> list[ServiceStatus]:
    """
    Probe the three services and return their status.

    Results are cached for a few seconds so navigating between pages does not
    re-issue three network calls per rerun.

    Args:
        api_url: Root URL of the prediction API.
        mlflow_url: Root URL of the MLflow tracking server.
        airflow_url: Root URL of the Airflow webserver.
        airflow_user: Airflow basic-auth user.
        airflow_password: Airflow basic-auth password.

    Returns:
        list[ServiceStatus]: One entry per service, in display order.
    """
    api = ApiClient(api_url, timeout=state.PROBE_TIMEOUT)
    try:
        health = api.health()
        model_ready, _ = api.ready()
        api_status = ServiceStatus(
            name="API",
            up=True,
            detail=f"v{health.get('version', '?')} · modèle {'prêt' if model_ready else 'absent'}",
        )
    except ApiError:
        api_status = ServiceStatus(name="API", up=False, detail="injoignable")

    mlflow_up = MlflowClient(mlflow_url, timeout=state.PROBE_TIMEOUT).is_up()
    airflow_up = AirflowClient(
        airflow_url,
        airflow_user,
        airflow_password,
        timeout=state.PROBE_TIMEOUT,
    ).is_up()

    return [
        api_status,
        ServiceStatus("MLflow", mlflow_up, "tracking" if mlflow_up else "injoignable"),
        ServiceStatus("Airflow", airflow_up, "scheduler" if airflow_up else "injoignable"),
    ]


def current_statuses() -> list[ServiceStatus]:
    """Return the cached status of the three services for the current URLs."""
    config = state.settings()
    return probe_services(
        st.session_state["api_url"],
        st.session_state["mlflow_url"],
        st.session_state["airflow_url"],
        config.airflow_username,
        config.airflow_password,
    )


def _login_form() -> None:
    """Render the login form and store the token in session state."""
    config = state.settings()
    with st.form("login", border=False):
        username = st.text_input("Utilisateur", value=config.default_username)
        password = st.text_input("Mot de passe", value=config.default_password, type="password")
        submitted = st.form_submit_button("Se connecter", width="stretch")

    if submitted:
        try:
            token = state.api_client().login(username, password)
        except ApiError as exc:
            st.error(str(exc))
            return
        st.session_state["token"] = token
        st.session_state["username"] = username
        st.success(f"Connecté en tant que {username}.")
        st.rerun()


def sidebar() -> None:
    """Render the sidebar: services, authentication and endpoints."""
    with st.sidebar:
        st.markdown(
            f'<div style="font-weight:700;font-size:1.05rem;color:{theme.BRAND}">'
            "Rakuten MLOps</div>"
            f'<div style="color:{theme.INK_MUTED};font-size:0.85rem">'
            "Classification produits · 27 catégories</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("**État des services**")
        status_pills(current_statuses())
        if st.button("Rafraîchir", width="stretch"):
            probe_services.clear()
            st.rerun()

        st.divider()

        st.markdown("**Authentification**")
        if state.is_authenticated():
            st.caption(f"Connecté : `{st.session_state['username']}`")
            if st.button("Se déconnecter", width="stretch"):
                state.logout()
                st.rerun()
        else:
            st.caption("Les routes `/predict` et `/train` exigent un jeton JWT.")
            _login_form()

        st.divider()

        with st.expander("Points d'accès", expanded=False):
            st.session_state["api_url"] = st.text_input("API", st.session_state["api_url"])
            st.session_state["mlflow_url"] = st.text_input("MLflow", st.session_state["mlflow_url"])
            st.session_state["airflow_url"] = st.text_input(
                "Airflow", st.session_state["airflow_url"]
            )

        st.caption("Documentation interactive : `/docs` sur l'API.")
