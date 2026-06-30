"""
Generates docs/transforms_example_light.png — the README preview for add_beeswarm and add_jitter.

Two panels showing the same data with different x-offset methods:
  left  - add_beeswarm() (collision-avoiding, deterministic)
  right - add_jitter() (Gaussian random)

Usage (from project root):
    uv run python scripts/build/build_transforms_example.py
"""

from pathlib import Path

import altair as alt
import numpy as np
import polars as pl

import dysonsphere as ds

ROOT = Path(__file__).resolve().parents[2]

CATEGORIES = ["A", "B", "C", "D", "E"]

rng = np.random.default_rng(42)
df = pl.DataFrame(
    {
        "group": (
            ["A"] * 100 + ["B"] * 100 + ["C"] * 100 + ["D"] * 100 + ["E"] * 100
        ),
        "value": np.concatenate(
            [
                rng.normal(10, 2, 100),
                rng.normal(14, 2, 100),
                rng.normal(11, 2, 100),
                rng.normal(13, 2, 100),
                rng.normal(9, 2, 100),
            ]
        ),
    }
)

ds.theme(viewFill="white", palette="blues2", legend=False)

title_params = dict(orient="top", anchor="start", offset=4)
fontSize = alt.theme.options.get("fontSize", 7)
palette = ds.palette("blues2", n=len(CATEGORIES))

x = alt.X("group:N", sort=CATEGORIES, title=None)
y = alt.Y("value:Q", title=None)
color = alt.Color("group:N", sort=CATEGORIES, scale=alt.Scale(range=palette), legend=None)

# -- Left: add_beeswarm() ---------------------------------------------------
bee_df = ds.add_beeswarm(df, yCol="value", groupBy=["group"])

left = (
    alt.Chart(bee_df)
    .mark_circle()
    .encode(x=x, y=y, xOffset=alt.XOffset("beeswarm_x:Q"), color=color)
    .properties(
        title=alt.TitleParams(["add_beeswarm(df, ...)"], fontSize=fontSize, **title_params)
    )
)

# -- Right: add_jitter() ----------------------------------------------------
jit_df = ds.add_jitter(df, spread=4.0)

right = (
    alt.Chart(jit_df)
    .mark_point()
    .encode(x=x, y=y, xOffset=alt.XOffset("jitter_x:Q"), color=color)
    .properties(
        title=alt.TitleParams(
            ["add_jitter(df, spread=4.0)"], fontSize=fontSize, **title_params
        )
    )
)

chart = alt.hconcat(left, right)

ds.save(chart, str(ROOT / "docs" / "transforms_example"), background=["light"])
print(f"saved {ROOT / 'docs' / 'transforms_example_light.png'}")
