"""
Design tokens shared by every page.

The app is pinned to the light theme (see ``.streamlit/config.toml``) so the
colours below only ever render on a white surface — which is also the surface
they were validated against for colour-blind separation and contrast.

Series colours are assigned in a fixed order and never cycled: a chart that
would need a fourth series is split instead.
"""

from typing import Final

# ── Surfaces and ink ──────────────────────────────────────────────────────────
SURFACE: Final = "#ffffff"
PLANE: Final = "#f9f9f7"
INK_PRIMARY: Final = "#0b0b0b"
INK_SECONDARY: Final = "#52514e"
INK_MUTED: Final = "#898781"
GRID: Final = "#e1e0d9"
AXIS: Final = "#c3c2b7"

# ── Categorical series, in assignment order ──────────────────────────────────
SERIES: Final[tuple[str, ...]] = ("#2a78d6", "#eb6834", "#1baf7a")
SERIES_BLUE: Final = SERIES[0]
SERIES_ORANGE: Final = SERIES[1]
SERIES_AQUA: Final = SERIES[2]

# ── Status colours — reserved, never reused as a series ──────────────────────
STATUS: Final[dict[str, str]] = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

FONT_FAMILY: Final = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# Rakuten accent, used for interface chrome only (never for data marks).
BRAND: Final = "#bf0000"
