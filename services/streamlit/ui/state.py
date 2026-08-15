"""
Session state and client factories.

Streamlit reruns the whole script on every interaction, so nothing is cached
across reruns except what lives in ``st.session_state``. Clients are cheap to
rebuild and are therefore recreated on demand, always reading the URLs and the
token currently held in session state.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from services.streamlit.clients.airflow_client import AirflowClient
from services.streamlit.clients.api_client import ApiClient, CallRecord
from services.streamlit.clients.mlflow_client import MlflowClient
from services.streamlit.settings import UISettings, get_ui_settings

# Number of calls kept for the monitoring page.
CALL_LOG_SIZE = 300

# Short timeout used by the sidebar status probes, so a service that is down
# never freezes the interface.
PROBE_TIMEOUT = 3.0


def init_state() -> None:
    """Populate session state with its defaults on the first run."""
    settings = get_ui_settings()
    defaults: dict[str, Any] = {
        "api_url": settings.api_url,
        "mlflow_url": settings.mlflow_url,
        "airflow_url": settings.airflow_url,
        "token": None,
        "username": None,
        "call_log": [],
        "last_prediction": None,
        "batch_results": None,
        "training_job_id": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def settings() -> UISettings:
    """Return the front-end settings."""
    return get_ui_settings()


def record_call(record: CallRecord) -> None:
    """
    Append a call to the session log, keeping only the most recent entries.

    Args:
        record: Measurement produced by the API client.
    """
    log: list[CallRecord] = st.session_state.setdefault("call_log", [])
    log.append(record)
    if len(log) > CALL_LOG_SIZE:
        del log[:-CALL_LOG_SIZE]


def api_client(timeout: float | None = None) -> ApiClient:
    """
    Build an API client bound to the current URL and token.

    Args:
        timeout: Override for the request timeout, in seconds.

    Returns:
        ApiClient: Client instrumented to feed the monitoring page.
    """
    return ApiClient(
        base_url=st.session_state["api_url"],
        token=st.session_state.get("token"),
        timeout=timeout or settings().request_timeout,
        recorder=record_call,
    )


def mlflow_client() -> MlflowClient:
    """Build a MLflow client bound to the current tracking URL."""
    return MlflowClient(base_url=st.session_state["mlflow_url"], timeout=PROBE_TIMEOUT)


def airflow_client() -> AirflowClient:
    """Build an Airflow client bound to the current webserver URL."""
    config = settings()
    return AirflowClient(
        base_url=st.session_state["airflow_url"],
        username=config.airflow_username,
        password=config.airflow_password,
        timeout=PROBE_TIMEOUT,
    )


def is_authenticated() -> bool:
    """Return True when a token has been obtained from the API."""
    return bool(st.session_state.get("token"))


def logout() -> None:
    """Drop the token and the data derived from authenticated calls."""
    st.session_state["token"] = None
    st.session_state["username"] = None
    st.session_state["last_prediction"] = None
    st.session_state["batch_results"] = None


def call_log_frame() -> pd.DataFrame:
    """
    Return the session call log as a DataFrame.

    Returns:
        pd.DataFrame: Columns ``appel``, ``horodatage``, ``methode``, ``route``,
        ``statut``, ``latence_ms`` and ``ok``. Empty when no call was made yet.
    """
    log: list[CallRecord] = st.session_state.get("call_log", [])
    if not log:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "appel": index,
                "horodatage": record.at,
                "methode": record.method,
                "route": record.path,
                "statut": record.status_code,
                "latence_ms": round(record.elapsed_ms, 1),
                "ok": record.ok,
            }
            for index, record in enumerate(log, start=1)
        ]
    )
