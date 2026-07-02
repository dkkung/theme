from typing import Any

import polars as pl


def count_n(df: pl.DataFrame, xCol: str, categories: list[str]) -> list[int]:
    """
    Count the number of rows in ``df`` belonging to each category.

    Parameters
    ----------
    df:
        A ``polars.DataFrame`` or ``pandas.DataFrame``.
    xCol:
        Column name used for grouping (the x-axis column).
    categories:
        Ordered list of category labels; the returned counts follow this order.
        Categories with no matching rows return 0.

    Returns
    -------
    list[int]
        Per-category row counts in the same order as ``categories``.

    Examples
    --------
    ::

        counts = ds.count_n(df, "group", ["Control", "Group A", "Group B"])
        # [12, 15, 11]
    """
    df = ensure_polars(df)
    return [len(df.filter(pl.col(xCol) == cat)) for cat in categories]


def ensure_polars(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert a pandas DataFrame to Polars, or pass a Polars DataFrame through unchanged.

    Accepts either a ``polars.DataFrame`` or a ``pandas.DataFrame`` without
    requiring pandas as a hard dependency — the check is done via the module
    name only.  If ``df`` is neither, a ``TypeError`` is raised.

    Parameters
    ----------
    df:
        A ``polars.DataFrame`` or ``pandas.DataFrame``.

    Returns
    -------
    polars.DataFrame
        The original DataFrame if already Polars, otherwise the result of
        ``polars.from_pandas(df)``.

    Examples
    --------
    ::

        import pandas as pd
        pdf = pd.DataFrame({"group": ["A", "B"], "value": [1.0, 2.0]})
        pldf = ds.ensure_polars(pdf)  # returns a polars.DataFrame
    """
    if isinstance(df, pl.DataFrame):
        return df
    if type(df).__module__.startswith("pandas"):
        return pl.from_pandas(df)
    raise TypeError(f"Expected a polars.DataFrame or pandas.DataFrame, got {type(df).__name__}.")


# ── Internal-data sentinel ───────────────────────────────────────────────────
# dysonsphere's composite marks / annotations generate their own small "sidecar" data
# (bracket coords, mean/error bars, KDE curves, labels, …).  Altair inlines each of those
# as a separate named dataset in the saved spec, alongside the user's dataframe.  To let
# export.read(what="data") return only the USER's frame(s), every internal data source is
# tagged with this sentinel column; read() treats any dataset carrying it as internal.
#
# DISCIPLINE: any NEW code that builds a dysonsphere-generated data source for a chart layer
# MUST route it through `_internal_data(...)` (i.e. `alt.Chart(_internal_data(rows_or_df))`).
# Miss one, and that sidecar leaks as a phantom "user" dataframe on read.  See CLAUDE.md.
_INTERNAL_COL = "__dysonsphere__"


def _internal_data(data: "list[dict] | pl.DataFrame | Any") -> "Any":
    """Tag dysonsphere-generated (non-user) chart data with the internal sentinel column.

    Accepts a list of record dicts (returned as an ``alt.Data``) or a polars/pandas
    DataFrame (returned as a polars DataFrame with the sentinel column added).  Pass the
    result straight to ``alt.Chart(...)``.
    """
    import altair as alt

    if isinstance(data, list):
        return alt.Data(values=[{**dict(row), _INTERNAL_COL: 1} for row in data])
    return ensure_polars(data).with_columns(pl.lit(1).alias(_INTERNAL_COL))
