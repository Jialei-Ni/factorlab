"""
Central configuration for the alphalens research pipeline.

Factor options
--------------
  "momentum"       Cross-sectional price momentum
  "bollinger"      Bollinger band z-score
  "volume"         Relative volume
  "mean_reversion" Short-term contrarian
  "stockstats"     Any stockstats indicator — set stockstats_indicator below

stockstats_indicator examples
------------------------------
  Close-only  : rsi_14, macd, macds, macdh, boll, boll_ub, boll_lb,
                close_5_sma, close_10_ema
  High + Low  : cci, atr, kdjk, kdjd, kdjj, wr_14
  Volume      : vr, mfi
"""

CONFIG = {
    # ── Date range ──────────────────────────────────────────────
    "start_date": "2022-01-01",
    "end_date":   "2024-01-01",

    # ── Factor selection ─────────────────────────────────────────
    "factor": "stockstats",

    # ── StockStats indicator (used when factor == "stockstats") ──
    "stockstats_indicator": "rsi_14",

    # ── Other factor parameters ──────────────────────────────────
    "momentum_window":       20,
    "bollinger_window":      20,
    "volume_window":         20,
    "mean_reversion_window":  5,

    # ── Data cleaning ────────────────────────────────────────────
    "min_data_fraction": 0.85,

    # ── Alphalens settings ───────────────────────────────────────
    "quantiles": 5,
    "periods":   (1, 5, 10),

    # ── Paths ────────────────────────────────────────────────────
    "cache_dir":  "data/cache",
    "output_dir": "output",
}
