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

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from portfolio_pipeline.performance import (
    cumulative_return_series,
    drawdown_series,
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
    """
    Generate all performance plots.
    """
    save_equity_curve_plot(
        returns,
        output_dir,
    )

    save_drawdown_plot(
        returns,
        output_dir,
    )


def export_performance_series(
    returns: pd.Series,
    output_dir: Path,
) -> None:
    equity = cumulative_return_series(returns)
    drawdown = drawdown_series(returns)

    equity.to_csv(
        output_dir / "equity_curve.csv",
        header=["equity"]
    )

    drawdown.to_csv(
        output_dir / "drawdown_curve.csv",
        header=["drawdown"]
    )
