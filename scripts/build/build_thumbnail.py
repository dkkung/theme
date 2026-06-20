"""
Generates assets/thumbnail_light.png — the README preview image.

Shows the blues2 palette across all gallery chart types (colorspace and
ΔE sparkline omitted; legends hidden; group labels shortened to A/B/C/…).

Usage (from project root):
    python scripts/build/build_thumbnail.py
"""

from pathlib import Path

import altair as alt
import polars as pl

import scripts.build.build_gallery as _bg
import theme
from theme.layers import mark_violin
from theme.palettes import colors

KEY = "blues2"

# ── Patch group labels: "Group A" → "A" ─────────────────────────────────────

_bg._BOX_CATS    = ["A", "B", "C", "D", "E"]
_bg._VIOLIN_CATS = ["A", "B", "C", "D", "E"]
_bg._AREA_GROUPS = ["A", "B", "C", "D"]
_bg._LINE_GROUPS = ["A", "B", "C", "D"]
_bg._SBAR_GROUPS = ["A", "B", "C", "D", "E"]
_bg._HIST_GROUPS = ["A", "B", "C"]

for _attr in ("_box_df", "_violin_df", "_area_df", "_line_df", "_sbar_df", "_hist_df"):
    df = getattr(_bg, _attr)
    if "group" in df.columns:
        setattr(_bg, _attr, df.with_columns(pl.col("group").str.replace("Group ", "")))

# re-import patched references used in thumbnail-local functions
from scripts.build.build_gallery import (  # noqa: E402
    W,
    _LINE_GROUPS,
    _VIOLIN_CATS,
    _BOX_CATS,
    _SBAR_GROUPS,
    _SBAR_TYPES,
    _area,
    _heatmap,
    _histogram,
    _scatter,
    _seq_heatmap,
    _swatch,
    _box_df,
    _line_df,
    _sbar_df,
    _violin_df,
)


# ── Thumbnail-specific chart overrides (no angled labels, no legends) ────────

def _boxplot_no_angle(key):
    p = colors[key]
    n = len(p)
    box_colors = [p[round(i * (n - 1) / (len(_BOX_CATS) - 1))] for i in range(len(_BOX_CATS))]
    return (
        alt.Chart(_box_df)
        .mark_boxplot()
        .encode(
            x=alt.X("group:N", sort=_BOX_CATS, title=None),
            y=alt.Y("value:Q", title=None),
            color=alt.Color(
                "group:N",
                sort=_BOX_CATS,
                title=None,
                scale=alt.Scale(domain=_BOX_CATS, range=box_colors),
            ),
        )
    )


def _stacked_bar_no_angle(key):
    p = colors[key]
    n = len(p)
    palette = [p[0], p[n - 1]]
    return (
        alt.Chart(_sbar_df)
        .mark_bar()
        .encode(
            x=alt.X("group:N", sort=_SBAR_GROUPS, title=None),
            y=alt.Y("value:Q", title=None, stack="normalize", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("type:N", sort=_SBAR_TYPES, title=None, scale=alt.Scale(range=palette)),
        )
    )


def _violin_no_legend(key):
    p = colors[key]
    n = len(p)
    palette = [p[round(i * (n - 1) / (len(_VIOLIN_CATS) - 1))] for i in range(len(_VIOLIN_CATS))]
    return mark_violin(
        _violin_df, "group", "value", _VIOLIN_CATS, palette=palette, legend=False, angledX=False
    )


def _line_no_legend(key):
    p = colors[key]
    n = len(p)
    palette = [p[round(i * (n - 1) / (len(_LINE_GROUPS) - 1))] for i in range(len(_LINE_GROUPS))]
    return (
        alt.Chart(_line_df)
        .mark_line()
        .encode(
            x=alt.X("time:Q", title=None),
            y=alt.Y("value:Q", title=None),
            color=alt.Color(
                "group:N",
                sort=_LINE_GROUPS,
                title=None,
                scale=alt.Scale(range=palette),
                legend=None,
            ),
        )
    )


# ── Build and save ───────────────────────────────────────────────────────────

def build_thumbnail():
    theme.options(chartWidth=W, chartHeight=W, legend=False)

    row = alt.hconcat(
        _area(KEY),
        _stacked_bar_no_angle(KEY),
        _histogram(KEY),
        _seq_heatmap(KEY),
        _heatmap(KEY),
        _boxplot_no_angle(KEY),
        _violin_no_legend(KEY),
        _scatter(KEY),
        _line_no_legend(KEY),
        spacing=6,
    ).resolve_scale(color="independent", opacity="independent")

    chart = alt.vconcat(
        _swatch(KEY, KEY),
        row,
        spacing=2,
    ).resolve_scale(color="independent", opacity="independent")

    out = Path(__file__).parent.parent.parent / "assets" / "thumbnail_light.png"
    chart.save(str(out), ppi=1200)
    print(f"saved {out}")


if __name__ == "__main__":
    build_thumbnail()
