"""
Smoke tests: every page must render without raising, including when none of
the upstream services is reachable.

This is the cheapest guard against the failure mode that hurts most during a
demo — a page that crashes because a service is down.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIEWS = ["overview", "data", "predict", "performance", "mlops", "monitoring"]

SCRIPT_TEMPLATE = """
import sys
sys.path.insert(0, {root!r})

from services.streamlit.ui import layout, state
from services.streamlit.views import {module} as view

state.init_state()
layout.inject_css()
view.render()
"""


def _run(module: str) -> AppTest:
    """Render one view in isolation and return the finished AppTest."""
    source = SCRIPT_TEMPLATE.format(root=str(PROJECT_ROOT), module=module)
    app = AppTest.from_string(source, default_timeout=90)
    app.run()
    return app


@pytest.mark.parametrize("module", VIEWS)
def test_page_renders_without_exception(module: str) -> None:
    """Each page renders end to end with no service running."""
    app = _run(module)
    assert not app.exception, [str(exc.value) for exc in app.exception]


def test_entrypoint_runs() -> None:
    """The entrypoint builds its navigation and renders the default page."""
    app = AppTest.from_file(
        str(PROJECT_ROOT / "services" / "streamlit" / "app.py"), default_timeout=90
    )
    app.run()
    assert not app.exception, [str(exc.value) for exc in app.exception]


def test_pages_warn_when_services_are_down() -> None:
    """
    With no API, MLflow or Airflow running, the MLOps page must explain the
    situation rather than fail silently.
    """
    app = _run("mlops")
    messages = [element.value for element in app.warning] + [element.value for element in app.info]
    assert any("Airflow" in message for message in messages)
