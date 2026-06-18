"""
factors/stockstats_factor.py
----------------------------
Generic StockStats wrapper factor.

Computes any indicator supported by the stockstats library and exposes it
through the common Factor interface.  Rapid experimentation: swap the
indicator name in config rather than writing new code.

Supported examples
------------------
Close-only (no extra data needed):
  rsi_14        14-period RSI
  macd          MACD line
  macds         MACD signal line
  macdh         MACD histogram
  boll          Bollinger mid-band (SMA)
  boll_ub       Bollinger upper band
  boll_lb       Bollinger lower band
  close_5_sma   5-period SMA of close
  close_10_ema  10-period EMA of close

High + Low required (fetcher always provides these):
  cci           Commodity Channel Index
  atr           Average True Range
  kdjk / kdjd / kdjj  KDJ oscillator lines

Volume required (fetcher always provides this):
  vr            Volume Ratio
  mfi           Money Flow Index

Full stockstats docs: https://github.com/jealousleopard/stockstats
"""

from __future__ import annotations

import warnings
import pandas as pd
from stockstats import StockDataFrame

from .base import Factor


# Indicators that need high + low (beyond close alone).
# If the indicator is NOT in this set we try close-only first.
_NEEDS_HIGH_LOW = {
    "cci", "atr", "atr_14",
    "kdjk", "kdjd", "kdjj",
    "wr", "wr_14",
    "dmi", "pdi", "mdi", "dx",
}

# Indicators that additionally need volume.
_NEEDS_VOLUME = {"vr", "mfi", "mfi_14", "obv"}


class StockStatsFactor(Factor):
    """
    Wraps any stockstats indicator as a cross-sectional factor.

    Parameters
    ----------
    indicator : str
        Any column name recognised by stockstats, e.g. ``"rsi_14"``,
        ``"macd"``, ``"cci"``, ``"atr"``, ``"boll_ub"``.
    """

    def __init__(self, indicator: str):
        self.indicator = indicator.lower().strip()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _needs_high_low(self) -> bool:
        base = self.indicator.split("_")[0]
        return self.indicator in _NEEDS_HIGH_LOW or base in _NEEDS_HIGH_LOW

    def _needs_volume(self) -> bool:
        base = self.indicator.split("_")[0]
        return self.indicator in _NEEDS_VOLUME or base in _NEEDS_VOLUME

    def _build_input_df(
        self,
        ticker: str,
        prices: pd.DataFrame,
        volume: pd.DataFrame | None,
        ohlcv: dict[str, pd.DataFrame] | None,
    ) -> pd.DataFrame:
        """
        Assemble the per-ticker DataFrame that stockstats needs.

        Priority:
          1. Use the full OHLCV dict when available (always preferred).
          2. Fall back to prices (close) + volume columns.
          3. Fall back to close-only (works for most indicators).
        """
        if ohlcv and ticker in ohlcv:
            df = ohlcv[ticker].copy()
            # Normalise column names
            df.columns = [c.lower() for c in df.columns]
            return df

        # Fallback: synthesise from what we have
        df = pd.DataFrame({"close": prices[ticker]})
        if volume is not None and ticker in volume.columns:
            df["volume"] = volume[ticker]
        return df

    # ── Factor interface ──────────────────────────────────────────────────────

    def compute(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame | None = None,
        ohlcv: dict[str, pd.DataFrame] | None = None,
    ) -> pd.Series:
        """
        Parameters
        ----------
        prices  : DataFrame (date × ticker) of adjusted close prices
        volume  : DataFrame (date × ticker) — passed through from pipeline
        ohlcv   : dict[ticker → OHLCV DataFrame] — from fetcher; provides
                  open/high/low for indicators that require them
        """
        out: list[pd.Series] = []

        for ticker in prices.columns:
            df = self._build_input_df(ticker, prices, volume, ohlcv)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ss = StockDataFrame.retype(df)
                try:
                    series = ss[self.indicator].copy()
                except KeyError:
                    # stockstats raises KeyError for unknown indicators
                    raise ValueError(
                        f"StockStatsFactor: indicator '{self.indicator}' is not "
                        f"recognised by stockstats.  "
                        f"See https://github.com/jealous/stockstats for "
                        f"the full list of supported column names."
                    )

            series.name = ticker
            out.append(series)

        if not out:
            raise RuntimeError("StockStatsFactor: no tickers produced output.")

        factor_df = pd.concat(out, axis=1)
        return self._stack(factor_df)
