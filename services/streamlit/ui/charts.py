"""
Chart builders.

Every figure in the app goes through one of these functions so the chrome
stays identical everywhere: recessive grid, muted axes, no legend box for a
single series, rounded data ends, hover on by default.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go

from services.streamlit.ui import theme


def _style(fig: go.Figure, height: int, *, show_legend: bool = False) -> go.Figure:
    """
    Apply the shared chart chrome to a figure.

    Args:
        fig: Figure to style, already holding its traces.
        height: Height in pixels.
        show_legend: Whether the legend box is displayed (two series or more).

    Returns:
        go.Figure: The same figure, styled in place.
    """
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=16, t=8, b=8),
        paper_bgcolor=theme.SURFACE,
        plot_bgcolor=theme.SURFACE,
        font=dict(family=theme.FONT_FAMILY, size=13, color=theme.INK_SECONDARY),
        hoverlabel=dict(font_family=theme.FONT_FAMILY, font_size=13),
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            title_text="",
            bgcolor="rgba(0,0,0,0)",
        ),
        bargap=0.25,
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=theme.AXIS,
        tickcolor=theme.AXIS,
        tickfont=dict(color=theme.INK_MUTED),
        title_font=dict(color=theme.INK_MUTED, size=12),
    )
    fig.update_yaxes(
        gridcolor=theme.GRID,
        zeroline=False,
        linecolor="rgba(0,0,0,0)",
        tickfont=dict(color=theme.INK_MUTED),
        title_font=dict(color=theme.INK_MUTED, size=12),
    )
    return fig


def horizontal_bar(
    data: pd.DataFrame,
    *,
    label_column: str,
    value_column: str,
    value_format: str = ".2f",
    value_suffix: str = "",
    color: str | None = None,
    height: int = 620,
    reference: float | None = None,
    reference_label: str = "moyenne",
) -> go.Figure:
    """
    Build a horizontal bar chart for one measure across labelled categories.

    Horizontal bars are used whenever the labels are category names: they stay
    readable without rotating text, which matters with 27 product categories.

    Args:
        data: Source frame, already ordered as it should be displayed.
        label_column: Column holding the category label.
        value_column: Column holding the measure.
        value_format: Python format spec applied to the value labels.
        value_suffix: Text appended to every value label, e.g. ``" %"``.
        color: Bar colour; defaults to the first series slot.
        height: Height in pixels.
        reference: Optional vertical reference line, e.g. the macro average.
        reference_label: Caption of the reference line.

    Returns:
        go.Figure: A styled horizontal bar chart.
    """
    labels = data[label_column].tolist()
    values = data[value_column].tolist()
    texts = [f"{value:{value_format}}{value_suffix}" for value in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=color or theme.SERIES_BLUE, cornerradius=4),
            text=texts,
            textposition="outside",
            textfont=dict(color=theme.INK_SECONDARY, size=12),
            hovertemplate="<b>%{y}</b><br>%{x:" + value_format + "}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor=theme.GRID)

    if reference is not None:
        fig.add_vline(
            x=reference,
            line_width=1,
            line_dash="dash",
            line_color=theme.INK_MUTED,
            annotation_text=f"{reference_label} {reference:{value_format}}{value_suffix}",
            annotation_position="top",
            annotation_font=dict(color=theme.INK_MUTED, size=11),
        )

    return _style(fig, height)


def vertical_bar(
    data: pd.DataFrame,
    *,
    label_column: str,
    value_column: str,
    value_format: str = ".0f",
    color: str | None = None,
    height: int = 300,
) -> go.Figure:
    """
    Build a vertical bar chart, used for short label sets.

    Args:
        data: Source frame, already ordered.
        label_column: Column holding the category label.
        value_column: Column holding the measure.
        value_format: Python format spec applied to the value labels.
        color: Bar colour; defaults to the first series slot.
        height: Height in pixels.

    Returns:
        go.Figure: A styled vertical bar chart.
    """
    fig = go.Figure(
        go.Bar(
            x=data[label_column].tolist(),
            y=data[value_column].tolist(),
            marker=dict(color=color or theme.SERIES_BLUE, cornerradius=4),
            text=[f"{value:{value_format}}" for value in data[value_column]],
            textposition="outside",
            textfont=dict(color=theme.INK_SECONDARY, size=12),
            hovertemplate="<b>%{x}</b><br>%{y:" + value_format + "}<extra></extra>",
        )
    )
    return _style(fig, height)


def line_chart(
    data: pd.DataFrame,
    *,
    x_column: str,
    series: Sequence[tuple[str, str]],
    y_title: str = "",
    x_title: str = "",
    height: int = 280,
) -> go.Figure:
    """
    Build a line chart with one line per series.

    Args:
        data: Source frame in wide format.
        x_column: Column used for the x axis.
        series: Pairs of ``(column, legend label)``, in assignment order.
        y_title: Title of the y axis.
        x_title: Title of the x axis.
        height: Height in pixels.

    Returns:
        go.Figure: A styled line chart with a shared crosshair tooltip.
    """
    fig = go.Figure()
    for index, (column, label) in enumerate(series):
        if column not in data:
            continue
        fig.add_trace(
            go.Scatter(
                x=data[x_column],
                y=data[column],
                name=label,
                mode="lines",
                line=dict(color=theme.SERIES[index % len(theme.SERIES)], width=2),
                hovertemplate=f"{label}: %{{y:.4f}}<extra></extra>",
            )
        )

    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text=y_title)
    return _style(fig, height, show_legend=len(series) > 1)


def latency_chart(data: pd.DataFrame, height: int = 300) -> go.Figure:
    """
    Plot the response time of the API calls made during the session.

    Args:
        data: Frame with ``appel``, ``latence_ms`` and ``route`` columns.
        height: Height in pixels.

    Returns:
        go.Figure: A styled line chart with one marker per call.
    """
    fig = go.Figure(
        go.Scatter(
            x=data["appel"],
            y=data["latence_ms"],
            mode="lines+markers",
            line=dict(color=theme.SERIES_BLUE, width=2),
            marker=dict(size=8, color=theme.SERIES_BLUE, line=dict(color=theme.SURFACE, width=2)),
            customdata=data["route"],
            hovertemplate="Appel %{x} · %{customdata}<br>%{y:.0f} ms<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="appel (ordre chronologique)")
    fig.update_yaxes(title_text="temps de réponse (ms)", rangemode="tozero")
    return _style(fig, height)
