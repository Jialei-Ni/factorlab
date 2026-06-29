"""
pipeline/builder.py
-------------------
Orchestrates data → factor computation and formats the factor Series
so it is ready for Alphalens.
"""

import pandas as pd
from alphalens_pipeline.factors.base import Factor
from alphalens_pipeline.factors.stockstats_factor import StockStatsFactor

import os

def run_pipeline(
    prices: pd.DataFrame,
    volume: pd.DataFrame | None,
    factor_engine: Factor,
    ohlcv: dict[str, pd.DataFrame] | None = None,
    output_dir: str | None = None,
) -> pd.Series:
    """
    Compute the factor and ensure the output matches Alphalens expectations:
      - Series with MultiIndex (date, asset)
      - named "factor"
      - no NaNs / infinities

    Parameters
    ----------
    prices        : cleaned price DataFrame (date × ticker)
    volume        : volume DataFrame, may be None
    factor_engine : instantiated Factor subclass
    ohlcv         : dict[ticker → OHLCV DataFrame] from fetcher;
                    forwarded to StockStatsFactor (and ignored by others)
    output_dir    : output directory of factor data
    """
    # StockStatsFactor needs ohlcv; pass it when available
    if isinstance(factor_engine, StockStatsFactor):
        factor = factor_engine.compute(prices, volume, ohlcv=ohlcv)
    else:
        factor = factor_engine.compute(prices, volume)

    factor.index.names = ["date", "asset"]
    factor.name = "factor"
    factor = factor.replace([float("inf"), float("-inf")], pd.NA).dropna()

    print(f"  Factor computed: {len(factor):,} (date, asset) observations.")
    save_factor_values(factor, output_dir)
    return factor


def save_factor_values(factor: pd.Series, output_dir: str | None = None) -> None:
    """
    Save raw factor values for downstream research (correlation, clustering).
    """
    if output_dir is None:
        # TODO: remove this print after finishing debugging
        print("No output directory designated. Skipping saving factor values.")
        return

    import os

    os.makedirs(output_dir, exist_ok=True)

    df = factor.copy()
    df = df.reset_index()
    df.columns = ["date", "asset", "factor"]

    path = os.path.join(output_dir, "factor_values.parquet")
    df.to_parquet(path)

    print(f"  Saved factor values: {path}")