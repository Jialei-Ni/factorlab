"""
portfolio_pipeline/equity_plots.py
----------------------------------
Generate portfolio performance visualizations.

Outputs:
    equity_curve.png
    drawdown_curve.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from portfolio_pipeline.performance import (
    cumulative_return_series,
    drawdown_series,
    annual_return_series,
    monthly_return_series,
)


def save_equity_curve_plot(
    returns: pd.Series,
    output_dir: Path,
) -> Path:
    """
    Save cumulative return curve.

    Parameters
    ----------
    returns
        Daily portfolio return series.

    output_dir
        Portfolio run directory.

    Returns
    -------
    Path to saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    equity = cumulative_return_series(returns)

    fig, ax = plt.subplots(figsize=(10, 6))

    equity.plot(ax=ax)

    ax.set_title("Portfolio Equity Curve")
    ax.set_ylabel("Portfolio Value")
    ax.grid(True)

    fig.tight_layout()

    path = output_dir / "equity_curve.png"

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return path


def save_drawdown_plot(
    returns: pd.Series,
    output_dir: Path,
) -> Path:
    """
    Save drawdown curve.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    drawdown = drawdown_series(returns)

    fig, ax = plt.subplots(figsize=(10, 6))

    drawdown.plot(ax=ax)

    ax.set_title("Portfolio Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(True)

    fig.tight_layout()

    path = output_dir / "drawdown_curve.png"

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return path


def save_performance_plots(
    returns: pd.Series,
    output_dir: Path,
) -> None:
    save_equity_curve_plot(returns, output_dir)
    save_drawdown_plot(returns, output_dir)
    save_annual_returns_plot(returns, output_dir)
    save_monthly_returns_heatmap(returns, output_dir)


def save_annual_returns_plot(
    returns: pd.Series,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    annual = annual_return_series(returns)

    fig, ax = plt.subplots(figsize=(10, 5))
    annual.plot(kind="bar", ax=ax)

    ax.set_title("Annual Returns")
    ax.set_ylabel("Return")
    ax.grid(True, axis="y")

    fig.tight_layout()

    path = output_dir / "annual_returns.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return path


def save_monthly_returns_heatmap(
    returns: pd.Series,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    r = returns.dropna()

    monthly = (1 + r).groupby(pd.Grouper(freq="M")).agg(np.prod) - 1

    df = monthly.to_frame("ret")
    df["year"] = df.index.year
    df["month"] = df.index.month

    pivot = df.pivot(index="year", columns="month", values="ret").sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_title("Monthly Returns Heatmap")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")

    ax.set_xticks(range(12))
    ax.set_xticklabels(range(1, 13))

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # ---- ADD ANNOTATIONS (percentage, 1 decimal) ----
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(
                    j,
                    i,
                    f"{val * 100:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )

    fig.colorbar(im, ax=ax)

    fig.tight_layout()

    path = output_dir / "monthly_returns_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return path


def export_performance_series(
    returns: pd.Series,
    output_dir: Path,
) -> None:
    equity = cumulative_return_series(returns)
    drawdown = drawdown_series(returns)
    annual = annual_return_series(returns)

    equity.to_csv(output_dir / "equity_curve.csv", header=["equity"])
    drawdown.to_csv(output_dir / "drawdown_curve.csv", header=["drawdown"])
    annual.to_csv(output_dir / "annual_returns.csv", header=["return"])
