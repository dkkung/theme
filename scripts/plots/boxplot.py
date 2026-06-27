import altair as alt
import numpy as np
import polars as pl

import dysonsphere as ds

rng = np.random.default_rng(42)

df = pl.DataFrame(
    {
        "group": (
            ["Control"] * 200
            + ["Drug A"] * 200
            + ["Drug B"] * 200
            + ["Drug C"] * 200
            + ["Drug D"] * 200
            + ["Drug E"] * 200
        ),
        "value": np.concatenate(
            [
                rng.normal(10, 2, 200),
                rng.normal(14, 2, 200),
                rng.normal(11, 2, 200),
                rng.normal(13, 2, 200),
                rng.normal(9, 2, 200),
                rng.normal(10, 2, 200),
            ]
        ),
    }
)

CATEGORIES = ["Control", "Drug A", "Drug B", "Drug C", "Drug D", "Drug E"]

ds.theme()

df = ds.add_beeswarm(
    df,
    yCol="value",
    groupBy=["group"],
)

# palette = ds.palette("greys", n=1, start=2)
# palette = ds.palette("mpl_viridis", n=len(CATEGORIES), start=4)
palette = ds.palette("blues", n=6, start=0)

base = alt.Chart(df).encode(
    x=alt.X(
        "group:N",
        sort=CATEGORIES,
        # title="Treatment",
        axis=alt.Axis(labelAngle=-45, labelAlign="right"),
    ),
    y=alt.Y("value:Q", title="Response (AU)"),
)

boxplot = base.mark_boxplot().encode(
    color=alt.Color("group:N", sort=CATEGORIES, scale=alt.Scale(range=palette), legend=None),
)

points = base.mark_circle().encode(
    xOffset=alt.XOffset("beeswarm_x:Q"),
)

ann = ds.add_pvalue(
    df,
    "group",
    "value",
    pairs=[("Control", "Drug A"), ("Control", "Drug B"), ("Drug A", "Drug B")],
    test="mannwhitneyu",
    categories=CATEGORIES,
    bracketStyle="line",
)

shade = ds.add_shade(
    CATEGORIES, "group", palette=ds.palette("blues", n=2, start=0, end=2), repeat=2, opacity=1.0
)

chart = shade + points + boxplot  # + ann

groups = {
    # "n =": [152, 214, 187, 203, 198, 222],
    "Condition A": [False, False, False, False, False, True],
    "Condition B": [False, False, False, False, True, True],
    "Condition C": [False, False, False, True, True, True],
}

plot = ds.add_multilabel(
    chart,
    groups,
    categories=CATEGORIES,
    rowStyles=["symbol", "symbol", "symbol"],
)

ds.save(plot, "boxplot")
print("saved boxplot")
