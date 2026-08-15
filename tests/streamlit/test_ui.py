"""Tests for the presentation helpers: escaping and chart formatting."""

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from services.streamlit.ui import charts, layout

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CARD_SCRIPT = """
import sys
sys.path.insert(0, {root!r})
from services.streamlit.ui import layout
layout.card({text!r})
"""


def test_untrusted_text_is_escaped_in_cards() -> None:
    """
    Catalogue titles are rendered inside raw HTML, so they must be escaped.

    Real product designations do contain ``&`` — "Ant-Man & The Wasp" ships in
    the demo assets — and nothing guarantees a future one will not contain a tag.
    """
    app = AppTest.from_string(
        CARD_SCRIPT.format(
            root=str(PROJECT_ROOT),
            text="Ant-Man & The Wasp <script>alert(1)</script>",
        ),
        default_timeout=60,
    )
    app.run()

    assert not app.exception
    body = app.markdown[0].value
    assert "&amp;" in body
    assert "&lt;script&gt;" in body
    assert "<script>" not in body


def test_inline_markdown_is_rendered_in_custom_components() -> None:
    """
    Tiles, headers and notes are raw HTML, so their Markdown must be expanded.

    Without this, the landing page printed "codes `prdtypecode`" with its
    backticks visible.
    """
    rendered = layout.inline("codes `prdtypecode` et **gras**")

    assert "<code>prdtypecode</code>" in rendered
    assert "<strong>gras</strong>" in rendered
    assert "`" not in rendered
    assert "**" not in rendered


def test_inline_markdown_still_escapes_html() -> None:
    """Expanding Markdown must not open a hole for tags."""
    rendered = layout.inline("<img src=x onerror=alert(1)> & `ok`")

    assert "&lt;img" in rendered
    assert "&amp;" in rendered
    assert "<img" not in rendered


def test_thousands_are_separated_the_french_way() -> None:
    """A French interface never prints "10,209"."""
    frame = pd.DataFrame({"libelle": ["a"], "produits": [10209]})

    figure = charts.horizontal_bar(
        frame, label_column="libelle", value_column="produits", value_format=",.0f"
    )

    assert figure.data[0].text[0] == "10\u202f209"  # narrow no-break space
    assert figure.layout.separators == ".\u202f"


def test_reference_annotation_has_room_to_breathe() -> None:
    """The macro-average caption sits above the plot and was being clipped."""
    frame = pd.DataFrame({"libelle": ["a", "b"], "f1": [0.9, 0.5]})

    plain = charts.horizontal_bar(frame, label_column="libelle", value_column="f1")
    with_reference = charts.horizontal_bar(
        frame, label_column="libelle", value_column="f1", reference=0.71
    )

    assert with_reference.layout.margin.t > plain.layout.margin.t


def test_status_pills_escape_remote_details() -> None:
    """Health payloads come from other services and are escaped as well."""
    markup = layout.pill_html("API <b>", True, "v1 & ready")

    assert "&lt;b&gt;" in markup
    assert "v1 &amp; ready" in markup


def test_bar_hover_uses_the_column_format() -> None:
    """
    The hover template must carry a valid d3 format.

    Stripping the leading dot turned ``.2f`` into ``2f``, which d3 reads as a
    width rather than a precision and prints six decimals in the tooltip.
    """
    frame = pd.DataFrame({"libelle": ["a", "b"], "valeur": [0.71, 0.42]})

    figure = charts.horizontal_bar(
        frame, label_column="libelle", value_column="valeur", value_format=".2f"
    )
    assert "%{x:.2f}" in figure.data[0].hovertemplate

    percent = charts.horizontal_bar(
        frame, label_column="libelle", value_column="valeur", value_format=".1%"
    )
    assert "%{x:.1%}" in percent.data[0].hovertemplate


def test_vertical_bar_hover_uses_the_column_format() -> None:
    """Same contract on the vertical variant."""
    frame = pd.DataFrame({"etape": ["a", "b"], "dimensions": [12, 384]})

    figure = charts.vertical_bar(
        frame, label_column="etape", value_column="dimensions", value_format=".0f"
    )
    assert "%{y:.0f}" in figure.data[0].hovertemplate


def test_single_series_charts_hide_the_legend() -> None:
    """A lone series is named by the title, so no legend box is drawn."""
    frame = pd.DataFrame({"epoch": [1, 2], "accuracy": [0.5, 0.6]})

    figure = charts.line_chart(frame, x_column="epoch", series=[("accuracy", "entraînement")])
    assert figure.layout.showlegend is False

    frame["val_accuracy"] = [0.4, 0.5]
    paired = charts.line_chart(
        frame,
        x_column="epoch",
        series=[("accuracy", "entraînement"), ("val_accuracy", "validation")],
    )
    assert paired.layout.showlegend is True
