"""
factors/mean_reversion.py
-------------------------
Short-term mean reversion: negative of the recent return.

Hypothesis: stocks that sold off hard over the last `window` days
tend to bounce back (contrarian signal).
"""

import pandas as pd
from .base import Factor


class MeanReversionFactor(Factor):
    """
    Short-term mean-reversion (contrarian) factor.

    Score = −pct_change(window), z-scored cross-sectionally.
    """

    def __init__(self, window: int = 5):
        self.window = window

    def compute(self, prices: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.Series:
        raw = -prices.pct_change(self.window)   # negative: losers get high score

        # Cross-sectional z-score
        zscore = raw.apply(lambda row: (row - row.mean()) / row.std(), axis=1)

        return self._stack(zscore)
