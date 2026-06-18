"""
data/cleaner.py
---------------
Light-touch cleaning that keeps the time series usable for Alphalens
without introducing look-ahead bias.
"""

import pandas as pd


def clean_prices(
    prices: pd.DataFrame,
    min_data_fraction: float = 0.85,
) -> pd.DataFrame:
    """
    1. Drop tickers with too many missing values.
    2. Forward-fill remaining small gaps (e.g. halted trading days).
    3. Sort index chronologically.

    Parameters
    ----------
    prices              : raw price DataFrame (date × ticker)
    min_data_fraction   : drop ticker if it has < this fraction of valid rows
    """
    n_rows = len(prices)
    threshold = int(n_rows * min_data_fraction)

    # Drop sparse tickers first
    before = prices.shape[1]
    prices = prices.dropna(axis=1, thresh=threshold)
    after  = prices.shape[1]
    if before != after:
        print(f"  Cleaner: dropped {before - after} sparse tickers "
              f"(kept {after}/{before}).")

    # Forward-fill small gaps (max 5 consecutive days)
    prices = prices.ffill(limit=5)

    # Ensure chronological order
    prices = prices.sort_index()

    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns (t / t-1 − 1)."""
    return prices.pct_change()
