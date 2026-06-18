"""
pipeline/builder.py
-------------------
Orchestrates data → factor computation and formats the factor Series
so it is ready for Alphalens.
"""

import pandas as pd
from factors.base import Factor
from factors.stockstats_factor import StockStatsFactor


def run_pipeline(
    prices: pd.DataFrame,
    volume: pd.DataFrame | None,
    factor_engine: Factor,
    ohlcv: dict[str, pd.DataFrame] | None = None,
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
    return factor
