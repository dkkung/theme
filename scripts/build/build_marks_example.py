"""
Generates docs/marks_example_light.png — the README preview for mark_strip and mark_violin.

Two panels showing the same data with different mark types:
  left  - mark_strip() (jittered points + median tick + error bars)
  right - mark_violin() (KDE + embedded boxplot)

Usage (from project root):
    uv run python scripts/build/build_marks_example.py
"""

from pathlib import Path

import altair as alt
import numpy as np
import polars as pl

import dysonsphere as ds

ROOT = Path(__file__).resolve().parents[2]

CATEGORIES = ["A", "B", "C", "D"]

rng = np.random.default_rng(42)
df = pl.DataFrame(
    {
        "group": (
            ["A"] * 200 + ["B"] * 200 + ["C"] * 200 + ["D"] * 200
        ),
        "value": np.concatenate(
            [
                rng.normal(10, 2, 200),
                rng.normal(14, 2, 200),
                rng.normal(11, 2, 200),
                rng.normal(13, 2, 200),
            ]
        ),
    }
)

ds.theme(viewFill="white", palette="blues2", legend=False)

title_params = dict(orient="top", anchor="start", offset=4)
fontSize = alt.theme.options.get("fontSize", 7)
palette = ds.palette("blues2", n=len(CATEGORIES))

left = ds.mark_strip(df, "group", "value", CATEGORIES, palette=palette, yTitle=None).properties(
    title=alt.TitleParams(
        ["mark_strip(df, xCol, yCol, categories)"], fontSize=fontSize, **title_params
    )
)

right = ds.mark_violin(df, "group", "value", CATEGORIES, palette=palette, yTitle=None).properties(
    title=alt.TitleParams(
        ["mark_violin(df, xCol, yCol, categories)"], fontSize=fontSize, **title_params
    )
)

chart = alt.hconcat(left, right).resolve_scale(y="shared")

ds.save(chart, str(ROOT / "docs" / "marks_example"), background=["light"])
print(f"saved {ROOT / 'docs' / 'marks_example_light.png'}")
