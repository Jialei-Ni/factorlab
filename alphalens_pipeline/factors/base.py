"""
factors/base.py
---------------
Abstract base class for all factor calculators.

Every factor must implement `compute(prices, volume)` and return a
pandas Series with a MultiIndex of (date, asset).
"""

from abc import ABC, abstractmethod
import pandas as pd


class Factor(ABC):
    """
    Interface for factor calculators.

    Subclass this and implement `compute` to add a new factor.
    The pipeline will call compute() and pass the result to Alphalens.
    """

    @abstractmethod
    def compute(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame | None = None,
    ) -> pd.Series:
        """
        Parameters
        ----------
        prices  : DataFrame (date × ticker) of adjusted close prices
        volume  : DataFrame (date × ticker) of trading volume, may be None

        Returns
        -------
        factor  : Series with MultiIndex (date, asset), named "factor"
        """
        ...

    def _stack(self, df: pd.DataFrame) -> pd.Series:
        """Utility: wide DataFrame → long (date, asset) Series."""
        s = df.stack(future_stack=True)
        s.index.names = ["date", "asset"]
        s.name = "factor"
        return s.dropna()
