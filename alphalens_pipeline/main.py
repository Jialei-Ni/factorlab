"""
main.py  —  Alphalens research pipeline entry point

Usage
-----
  python main.py                                   # config.py defaults
  python main.py --factor momentum
  python main.py --factor stockstats --indicator rsi_14
  python main.py --factor stockstats --indicator cci
  python main.py --factor stockstats --indicator macd --start 2021-01-01 --end 2023-06-01

Output hierarchy
----------------
  <base_output_dir>/
    <start>_<end>/          e.g. 2022-01-01_2024-01-01/
      <factor_slug>/        e.g. momentum_20/  macd/  rsi_14/  cci/  volume_20/
        returns_tear_sheet.png
        ic_tear_sheet.png
        ...
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from alphalens_pipeline.config import CONFIG
from config.paths import CACHE_DIR, FACTOR_OUTPUT_DIR
from universe import UNIVERSE

from data.fetcher import load_universe
from data.cleaner import clean_prices, compute_returns
from factors import build_factor
from pipeline import run_pipeline, run_alphalens


# ── Output-path construction ──────────────────────────────────────────────────

def _factor_slug(cfg: dict) -> str:
    """
    Return a short filesystem-safe name for the factor + its key parameter.

    Rules:
      - Native factors (momentum, bollinger, volume, mean_reversion):
          <factor_name>_<window>   e.g. "momentum_20", "volume_20"
      - StockStats factors: the indicator name as-is, which already encodes
          the parameter (e.g. "rsi_14", "macd", "cci", "boll_ub", "kdjk").
    """
    key = cfg["factor"]
    if key == "stockstats":
        return cfg.get("stockstats_indicator", "rsi_14")

    window_key = f"{key}_window"
    window = cfg.get(window_key)
    return f"{key}_{window}" if window is not None else key


def _resolve_path(value: str | Path | None, default: Path) -> Path:
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def resolve_output_dir(cfg: dict) -> str:
    """
    Build:  <base_output_dir>/<start>_<end>/<factor_slug>
    """
    default_base = FACTOR_OUTPUT_DIR
    base = _resolve_path(cfg.get("output_dir"), default_base)
    period = f"{cfg['start_date']}_{cfg['end_date']}"
    slug   = _factor_slug(cfg)
    return str(base / period / slug)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Alphalens research pipeline")
    p.add_argument("--factor",    default=None,
                   help="Factor key: momentum | bollinger | volume | "
                        "mean_reversion | stockstats")
    p.add_argument("--indicator", default=None,
                   help="StockStats indicator (only with --factor stockstats), "
                        "e.g. rsi_14, macd, cci, atr, boll_ub, kdjk, mfi")
    p.add_argument("--start",     default=None, help="Start date YYYY-MM-DD")
    p.add_argument("--end",       default=None, help="End date YYYY-MM-DD")
    return p.parse_args()


# ── Pipeline ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = CONFIG.copy()

    if args.factor:    cfg["factor"]              = args.factor
    if args.indicator: cfg["stockstats_indicator"] = args.indicator
    if args.start:     cfg["start_date"]           = args.start
    if args.end:       cfg["end_date"]             = args.end

    start, end   = cfg["start_date"], cfg["end_date"]
    output_dir   = resolve_output_dir(cfg)
    factor_label = (_factor_slug(cfg) if cfg["factor"] != "stockstats"
                    else cfg.get("stockstats_indicator", "rsi_14"))

    print("=" * 60)
    print(f"  Alphalens Pipeline")
    print(f"  Factor   : {factor_label}")
    print(f"  Universe : {len(UNIVERSE)} tickers")
    print(f"  Period   : {start} → {end}")
    print(f"  Output   : {output_dir}")
    print("=" * 60)

    # ── 1 & 2: Fetch + clean ──────────────────────────────────────────────────
    print("\n[1/5] Fetching data …")
    prices_raw, volume_raw, ohlcv = load_universe(
        UNIVERSE, start, end, cache_dir=cfg["cache_dir"]
    )

    print("\n[2/5] Cleaning prices …")
    prices = clean_prices(prices_raw, min_data_fraction=cfg["min_data_fraction"])

    # ── 3: Returns ────────────────────────────────────────────────────────────
    print("\n[3/5] Computing daily returns …")
    returns = compute_returns(prices)
    print(f"  Returns shape: {returns.shape}")

    # ── 4: Factor ─────────────────────────────────────────────────────────────
    print(f"\n[4/5] Computing factor: {factor_label} …")
    factor_engine = build_factor(cfg)
    factor        = run_pipeline(prices, volume_raw, factor_engine, ohlcv=ohlcv)

    # ── 5: Alphalens ──────────────────────────────────────────────────────────
    print("\n[5/5] Running Alphalens …")
    factor_data = run_alphalens(
        factor,
        prices,
        quantiles  = cfg["quantiles"],
        periods    = cfg["periods"],
        output_dir = output_dir,
    )

    print("\n✓ Pipeline complete.")
    return factor_data


if __name__ == "__main__":
    main()