"""
analysis.py
-----------
Aggregation and visualisation utilities for topic-modelled multilingual corpora.

Covers: volume over time, theme/country distributions, cross-tabulation heatmaps,
top-N selection and platform/language breakdowns.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ── Volume over time ──────────────────────────────────────────────────────────

def volume_over_time(
    df: pd.DataFrame,
    date_col: str,
    group_col: Optional[str] = None,
    top_n: int = 10,
    freq: str = "ME",
    title: str = "Volume over time",
) -> go.Figure:
    """
    Monthly post volume, optionally broken down by a grouping column (theme, country, etc.).

    Args:
        df: DataFrame with at least a date column.
        date_col: Name of the datetime column.
        group_col: Optional column to colour-split the lines (e.g. 'theme', 'country').
        top_n: Keep only the top-N groups by total volume (ignored if group_col is None).
        freq: Pandas resample frequency string. Default 'ME' (month-end).
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    if group_col:
        top_groups = (
            df[group_col].value_counts().head(top_n).index.tolist()
        )
        df = df[df[group_col].isin(top_groups)]
        agg = (
            df.groupby([pd.Grouper(key=date_col, freq=freq), group_col])
            .size()
            .reset_index(name="count")
        )
        fig = px.line(agg, x=date_col, y="count", color=group_col, title=title)
    else:
        agg = df.resample(freq, on=date_col).size().reset_index(name="count")
        fig = px.line(agg, x=date_col, y="count", title=title)

    fig.update_layout(xaxis_title="Date", yaxis_title="Posts")
    return fig


# ── Distribution helpers ───────────────────────────────────────────────────────

def top_n_distribution(
    df: pd.DataFrame,
    col: str,
    top_n: int = 15,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Horizontal bar chart of the top-N values in a categorical column.

    Args:
        df: Input DataFrame.
        col: Column to aggregate (e.g. 'theme', 'country', 'language').
        top_n: Number of top categories to show.
        title: Chart title (defaults to column name).

    Returns:
        Plotly Figure.
    """
    counts = df[col].value_counts().head(top_n).reset_index()
    counts.columns = [col, "count"]
    fig = px.bar(
        counts,
        x="count",
        y=col,
        orientation="h",
        title=title or f"Top {top_n} {col}",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def platform_language_breakdown(
    df: pd.DataFrame,
    platform_col: str = "platform",
    language_col: str = "language",
) -> pd.DataFrame:
    """
    Cross-tabulation of platform × language post counts.

    Returns:
        DataFrame pivot table (platforms as rows, languages as columns).
    """
    return pd.crosstab(df[platform_col], df[language_col], margins=True)


# ── Heatmap ───────────────────────────────────────────────────────────────────

class Heatmap:
    """
    Cross-tabulation heatmap for two categorical dimensions (e.g. country × theme).

    Usage::

        hm = Heatmap(df, row_col="country", col_col="theme")
        pivot = hm.get_pivot_table()
        fig = hm.plot(title="Country × Theme distribution")
    """

    def __init__(
        self,
        df: pd.DataFrame,
        row_col: str,
        col_col: str,
        value_col: Optional[str] = None,
        top_n_rows: int = 20,
        top_n_cols: int = 20,
    ):
        self.df = df.copy()
        self.row_col = row_col
        self.col_col = col_col
        self.value_col = value_col
        self.top_n_rows = top_n_rows
        self.top_n_cols = top_n_cols

    def get_pivot_table(self) -> pd.DataFrame:
        top_rows = self.df[self.row_col].value_counts().head(self.top_n_rows).index
        top_cols = self.df[self.col_col].value_counts().head(self.top_n_cols).index
        subset = self.df[
            self.df[self.row_col].isin(top_rows) & self.df[self.col_col].isin(top_cols)
        ]
        if self.value_col:
            pivot = subset.pivot_table(
                index=self.row_col, columns=self.col_col,
                values=self.value_col, aggfunc="sum", fill_value=0,
            )
        else:
            pivot = pd.crosstab(subset[self.row_col], subset[self.col_col])
        return pivot

    def plot(self, title: str = "Heatmap", colorscale: str = "Blues") -> go.Figure:
        pivot = self.get_pivot_table()
        fig = px.imshow(
            pivot,
            title=title,
            color_continuous_scale=colorscale,
            aspect="auto",
        )
        fig.update_layout(
            xaxis_title=self.col_col,
            yaxis_title=self.row_col,
        )
        return fig


# ── Account coordination ──────────────────────────────────────────────────────

def top_accounts_by_country(
    df: pd.DataFrame,
    account_col: str,
    country_col: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Cross-tabulation of top-N accounts against countries.

    Useful for identifying accounts driving narrative in specific geographies.

    Args:
        df: Input DataFrame.
        account_col: Column containing account identifiers.
        country_col: Column containing country labels.
        top_n: Number of most active accounts to include.

    Returns:
        Pivot DataFrame (accounts × countries).
    """
    top_accounts = df[account_col].value_counts().head(top_n).index.tolist()
    subset = df[df[account_col].isin(top_accounts)]
    return pd.crosstab(subset[account_col], subset[country_col])
