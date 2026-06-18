"""
factors/volume.py
-----------------
Relative volume factor: today's volume / rolling average volume.

High score = unusual trading activity (can signal breakout or reversal).
"""

import pandas as pd
from .base import Factor


class VolumeFactor(Factor):
    """
    Relative-volume factor.

    Score = Volume / SMA_window(Volume)
    Requires volume data — raises ValueError if not supplied.
    """

    def __init__(self, window: int = 20):
        self.window = window

    def compute(self, prices: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.Series:
        if volume is None or volume.empty:
            raise ValueError("VolumeFactor requires a non-empty volume DataFrame.")

        avg_vol = volume.rolling(self.window, min_periods=self.window // 2).mean()
        rvol    = volume / avg_vol.replace(0, float("nan"))

        # Align to the same tickers as prices
        rvol = rvol.reindex(columns=prices.columns)

        return self._stack(rvol)
