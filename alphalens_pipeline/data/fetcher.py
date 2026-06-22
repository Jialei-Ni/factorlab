"""
data/fetcher.py
---------------
Fetches full OHLCV data from yfinance with local parquet caching.

Cache key  = (ticker, start_date, end_date)
Cache file = data/cache/<ticker>_<start>_<end>.parquet
             stores: open, high, low, close, volume (all adjusted)

Re-run always uses the cache unless you delete the file.
"""

import os
import warnings
import yfinance as yf
import pandas as pd

warnings.filterwarnings("ignore")

OHLCV_COLS = ["open", "high", "low", "close", "volume"]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _cache_path(cache_dir: str, ticker: str, start: str, end: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{ticker}_{start}_{end}.parquet")


def _fetch_single(ticker: str, start: str, end: str, cache_dir: str) -> pd.DataFrame:
    """Download one ticker (full OHLCV), using cache when available."""
    path = _cache_path(cache_dir, ticker, start, end)

    if os.path.exists(path):
        return pd.read_parquet(path)

    print(f"  Downloading {ticker} …")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if df.empty:
        print(f"  WARNING: no data for {ticker}, skipping.")
        return pd.DataFrame()

    # Flatten MultiIndex columns that yfinance sometimes returns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]

    # Keep only OHLCV; drop anything else (e.g. "dividends")
    keep = [c for c in OHLCV_COLS if c in df.columns]
    df = df[keep]
    df.index = pd.to_datetime(df.index)

    df.to_parquet(path)
    return df


# ── Public API ────────────────────────────────────────────────────────────────

from config.paths import CACHE_DIR


def load_universe(
    universe: list[str],
    start: str,
    end: str,
    cache_dir: str = str(CACHE_DIR),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Fetch full OHLCV for every ticker in *universe*.

    Returns
    -------
    prices : DataFrame  (index=date, columns=ticker)  — adjusted close
    volume : DataFrame  (index=date, columns=ticker)  — share volume
    ohlcv  : dict[ticker -> DataFrame(open,high,low,close,volume)]
             Used by indicators that need more than just close (CCI, ATR, MFI …)
    """
    prices_map = {}
    volume_map = {}
    ohlcv: dict[str, pd.DataFrame] = {}

    print(f"Loading {len(universe)} tickers ({start} → {end}) …")

    for ticker in universe:
        df = _fetch_single(ticker, start, end, cache_dir)
        if df.empty:
            continue

        prices_map[ticker] = df["close"]
        if "volume" in df.columns:
            volume_map[ticker] = df["volume"]
        ohlcv[ticker] = df

    prices_df = pd.DataFrame(prices_map)
    volume_df = pd.DataFrame(volume_map) if volume_map else pd.DataFrame()

    prices_df.index = pd.to_datetime(prices_df.index)
    if not volume_df.empty:
        volume_df.index = pd.to_datetime(volume_df.index)

    print(f"  Loaded {prices_df.shape[1]} tickers, {len(prices_df)} trading days.")
    return prices_df, volume_df, ohlcv
