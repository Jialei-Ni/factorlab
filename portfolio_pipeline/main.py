from __future__ import annotations

import argparse
import sys
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.paths import portfolio_output_root
from portfolio_pipeline.config import DEFAULT_CONFIG
from portfolio_pipeline.signal_loader import load_signals
from portfolio_pipeline.signal_combiner import combine_signals
from portfolio_pipeline.portfolio_constructor import construct_weights
from portfolio_pipeline.backtester import apply_transaction_costs, build_holdings
from portfolio_pipeline.performance import summarize_performance, performance_summary_frame
from portfolio_pipeline.metadata import save_metadata, save_config_snapshot
from portfolio_pipeline.reports import save_portfolio_reports
from portfolio_pipeline.universe import UNIVERSE
from portfolio_pipeline.equity_plots import save_performance_plots, export_performance_series


def _run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portfolio backtesting pipeline")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--universe-name", default=None, help="Universe name")
    return parser.parse_args()


def _build_output_directory(
    universe_name: str,
    start: str,
    end: str,
    run_id: str,
) -> Path:
    return portfolio_output_root(universe_name, start, end) / run_id


def _signal_names(config: dict[str, object]) -> list[str]:
    return list(config["signals"].keys())


def _signal_weights(config: dict[str, object]) -> dict[str, float]:
    return {name: float(weight) for name, weight in config["signals"].items()}


def _asset_universe(prices: pd.DataFrame) -> pd.Index:
    return prices.columns


def main() -> None:
    args = _parse_args()
    cfg = DEFAULT_CONFIG.copy()
    if args.start:
        cfg["start_date"] = args.start
    if args.end:
        cfg["end_date"] = args.end
    if args.universe_name:
        cfg["universe_name"] = args.universe_name

    universe_name = cfg["universe_name"]
    start = cfg["start_date"]
    end = cfg["end_date"]
    run_id = _run_id()
    output_dir = _build_output_directory(universe_name, start, end, run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Portfolio pipeline")
    print(f"  universe: {universe_name}")
    print(f"  period  : {start} → {end}")
    print(f"  run_dir : {output_dir}")

    prices, volume, ohlcv, signals = load_signals(
        UNIVERSE,
        _signal_names(cfg),
        universe_name,
        start,
        end,
        cache_dir=cfg["cache_dir"],
        min_data_fraction=cfg["min_data_fraction"],
    )

    exposures = combine_signals(signals, _signal_weights(cfg))

    trading_dates = prices.index
    asset_universe = _asset_universe(prices)
    weights = construct_weights(
        exposures=exposures,
        portfolio_type=cfg["portfolio_type"],
        long_quantile=cfg["long_quantile"],
        short_quantile=cfg["short_quantile"],
        holding_days=int(cfg["holding_days"]),
        rebalance_frequency=int(cfg["rebalance_frequency"]),
        trading_dates=trading_dates,
        asset_universe=asset_universe,
    )

    returns = prices.pct_change()
    portfolio_returns = apply_transaction_costs(
        weights,
        returns,
        commission_bps=float(cfg["commission_bps"]),
        slippage_bps=float(cfg["slippage_bps"]),
    )

    holdings = build_holdings(weights)
    summary = summarize_performance(portfolio_returns["portfolio_return"], weights)
    summary_frame = performance_summary_frame(summary)

    save_metadata(output_dir / "metadata.json", universe_name, start, end, run_id)
    save_config_snapshot(output_dir / "config_snapshot.json", cfg)
    save_portfolio_reports(output_dir, holdings, portfolio_returns, summary_frame, exposures)
    returns_series = portfolio_returns["portfolio_return"]
    save_performance_plots(returns_series, output_dir)
    export_performance_series(returns_series, output_dir)

    print("✓ Portfolio backtest complete.")


if __name__ == "__main__":
    main()
