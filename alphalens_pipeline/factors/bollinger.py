"""
factors/bollinger.py
--------------------
Bollinger band z-score: (price - rolling_mean) / rolling_std.

A high score = price is far above its recent average (mean-reversion → short signal,
or momentum continuation → long signal, depending on your hypothesis).
"""

import pandas as pd
from .base import Factor


class BollingerFactor(Factor):
    """
    Bollinger z-score factor.

    Score = (Close − SMA_window) / STD_window
    """

    def __init__(self, window: int = 20):
        self.window = window

    def compute(self, prices: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.Series:
        ma  = prices.rolling(self.window, min_periods=self.window // 2).mean()
        std = prices.rolling(self.window, min_periods=self.window // 2).std()

        z = (prices - ma) / std.replace(0, float("nan")) 

        return self._stack(z)
