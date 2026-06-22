from __future__ import annotations

from pathlib import Path
import pandas as pd

from alphalens_pipeline.data.fetcher import load_universe
from alphalens_pipeline.data.cleaner import clean_prices
from alphalens_pipeline.factors import build_factor
from alphalens_pipeline.pipeline import run_pipeline
from config.paths import signal_cache_root


def _signal_path(
    signal_name: str,
    universe_name: str,
    start: str,
    end: str,
) -> Path:
    return signal_cache_root(universe_name, start, end) / f"{signal_name}.parquet"


def _ensure_cache_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_signals(
    universe: list[str],
    signal_names: list[str],
    universe_name: str,
    start: str,
    end: str,
    cache_dir: str,
    min_data_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    prices_raw, volume_raw, ohlcv = load_universe(
        universe,
        start,
        end,
        cache_dir=cache_dir,
    )

    prices = clean_prices(prices_raw, min_data_fraction=min_data_fraction)

    signals: dict[str, pd.DataFrame] = {}
    for signal_name in signal_names:
        path = _signal_path(signal_name, universe_name, start, end)
        _ensure_cache_directory(path)

        if path.exists():
            signal_df = pd.read_parquet(path)
        else:
            factor_cfg = {"factor": signal_name}
            factor_engine = build_factor(factor_cfg)
            factor_series = run_pipeline(prices, volume_raw, factor_engine, ohlcv=ohlcv)
            signal_df = factor_series.unstack().reindex(
                index=prices.index,
                columns=prices.columns,
            )
            signal_df.index.name = "date"
            signal_df.columns.name = None
            signal_df.to_parquet(path)

        signals[signal_name] = signal_df

    return prices, volume_raw, ohlcv, signals
