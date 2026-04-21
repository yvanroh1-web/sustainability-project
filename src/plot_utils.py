"""
plot_utils.py
=============
Reusable plotting utilities for the SAAM project.

All figures follow a consistent visual style and are saved at 200 dpi
by default for report-quality output.

Functions
---------
plot_cumulative_returns(series_dict, title, save_path)
    Plot and optionally save cumulative growth-of-$1 series.
plot_carbon_metric(metric_dict, ylabel, title, save_path)
    Bar / line chart for an annual carbon metric across portfolios.
plot_weights_bar(weights_series, top_n, title, save_path)
    Horizontal bar chart of top-N holdings for a given year.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Consistent colour palette
COLOURS = {
    "VW"       : "#2563EB",
    "MV"       : "#DC2626",
    "MV_05"    : "#16A34A",
    "VW_05"    : "#D97706",
    "VW_NZ"    : "#7C3AED",
    "default"  : "#374151",
}


def _style_ax(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.3, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_cumulative_returns(
    series_dict: dict[str, pd.Series],
    title: str = "Cumulative Portfolio Performance",
    save_path: str | None = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Plot cumulative growth-of-$1 series for multiple portfolios.

    Parameters
    ----------
    series_dict : dict  label → monthly return pd.Series
    title       : str
    save_path   : str or None  if given, saves the figure
    figsize     : tuple

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for label, ret in series_dict.items():
        colour = COLOURS.get(label, COLOURS["default"])
        cum    = (1 + ret).cumprod()
        ax.plot(cum.index, cum.values, colour=colour, linewidth=1.8, label=label, alpha=0.9)
        ax.annotate(
            f"{label}: ${cum.iloc[-1]:.2f}",
            xy=(cum.index[-1], cum.iloc[-1]),
            xytext=(10, 0), textcoords="offset points",
            fontsize=9, color=colour,
        )

    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_ylabel("Cumulative Return (Growth of $1)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    _style_ax(ax)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_carbon_metric(
    metric_dict: dict[str, list | np.ndarray],
    years: list[int],
    ylabel: str,
    title: str = "",
    save_path: str | None = None,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Line chart for an annual carbon metric across multiple portfolios.

    Parameters
    ----------
    metric_dict : dict  label → list/array of annual values (aligned with years)
    years       : list of int
    ylabel      : str
    title       : str
    save_path   : str or None
    figsize     : tuple

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for label, values in metric_dict.items():
        colour = COLOURS.get(label, COLOURS["default"])
        ax.plot(years, values, marker="o", markersize=5, linewidth=1.8,
                colour=colour, label=label)

    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    _style_ax(ax)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_weights_bar(
    weights_series: pd.Series,
    top_n: int = 15,
    title: str = "Top Portfolio Holdings",
    save_path: str | None = None,
    name_map: dict | None = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Horizontal bar chart of the top-N holdings.

    Parameters
    ----------
    weights_series : pd.Series  (index = ISIN, values = weights)
    top_n          : int
    title          : str
    save_path      : str or None
    name_map       : dict  ISIN → firm name for readable labels
    figsize        : tuple

    Returns
    -------
    matplotlib Figure
    """
    top = weights_series.nlargest(top_n).sort_values()
    labels = (
        [f"{name_map.get(isin, isin)} ({isin})" for isin in top.index]
        if name_map else list(top.index)
    )

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(labels, top.values * 100, colour=COLOURS["MV"], alpha=0.8)
    ax.bar_label(bars, fmt="%.2f%%", fontsize=8, padding=3)
    ax.set_xlabel("Portfolio Weight (%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    _style_ax(ax)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig
