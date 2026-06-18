"""
factors/momentum.py
-------------------
Simple price momentum: return over the last `window` trading days.

A high score = strong recent uptrend (long signal).
"""

import pandas as pd
from .base import Factor


class MomentumFactor(Factor):
    """
    Cross-sectional momentum factor.

    Score = pct_change over `window` days, z-scored cross-sectionally
    so it is comparable across tickers and time.
    """

    def __init__(self, window: int = 20):
        self.window = window

    def compute(self, prices: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.Series:
        raw = prices.pct_change(self.window)

        # Cross-sectional z-score (rank within each day)
        zscore = raw.apply(lambda row: (row - row.mean()) / row.std(), axis=1)

        return self._stack(zscore)
