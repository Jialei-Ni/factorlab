from __future__ import annotations

import pandas as pd
from pathlib import Path


def save_portfolio_reports(
    run_dir: Path,
    holdings: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    performance_summary: pd.DataFrame,
    exposures: pd.DataFrame,
) -> None:
    holdings_path = run_dir / "holdings.csv"
    returns_path = run_dir / "portfolio_returns.csv"
    summary_path = run_dir / "performance_summary.csv"
    exposures_path = run_dir / "factor_exposures.csv"

    holdings.to_csv(holdings_path, index=False)
    portfolio_returns.to_csv(returns_path, index=True, index_label="date")
    performance_summary.to_csv(summary_path, index=True, index_label="metric")
    exposures.to_csv(exposures_path, index=True)
