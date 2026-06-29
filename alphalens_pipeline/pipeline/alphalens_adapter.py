"""
pipeline/alphalens_adapter.py
------------------------------
Wraps alphalens to:
  1. Build the canonical factor_data object.
  2. Generate tear-sheet plots and save them to output/.
  3. Save summary stats as CSV.

Root cause of blank tear sheets:
  alphalens.tears.GridFigure.close() calls plt.close(fig) at the end of
  every tear-sheet function, destroying the figure before we can save it.
  We intercept close() with a monkey-patch, grab the figure, save it,
  then let the original close() run.
"""

import os
import warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import alphalens as al
import alphalens.tears as _tears

warnings.filterwarnings("ignore")


# ── Core fix: intercept GridFigure.close() ────────────────────────────────────

def _save_tearsheet(tear_fn, path: str, *args, **kwargs):
    """
    Call an alphalens tear-sheet function and save the figure it produces.

    alphalens destroys the figure via GridFigure.close() at the very end of
    each tear-sheet function.  We temporarily replace close() with a version
    that stashes the figure reference first, save it, then hand off to the
    original close().
    """
    captured = []
    _orig_close = _tears.GridFigure.close

    def _patched_close(self):
        if self.fig is not None:
            captured.append(self.fig)
        _orig_close(self)

    _tears.GridFigure.close = _patched_close
    try:
        tear_fn(*args, **kwargs)
    finally:
        _tears.GridFigure.close = _orig_close   # always restore

    if not captured:
        print(f"  WARNING: no figure captured for {os.path.basename(path)}")
        return

    fig = captured[0]
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    size_kb = os.path.getsize(path) // 1024
    print(f"  Saved: {os.path.basename(path)}  ({size_kb} KB)")


# ── Public API ────────────────────────────────────────────────────────────────

from config.paths import FACTOR_OUTPUT_DIR


def run_alphalens(
    factor: pd.Series,
    prices: pd.DataFrame,
    quantiles: int = 5,
    periods: tuple = (1, 5, 10),
    output_dir: str = str(FACTOR_OUTPUT_DIR),
    save_factor_values_flag: bool = True,
) -> pd.DataFrame:
    """
    Run the full Alphalens evaluation suite and save results.

    Parameters
    ----------
    factor      : Series with MultiIndex (date, asset)
    prices      : DataFrame (date × ticker) of adjusted close prices
    quantiles   : number of quantile buckets
    periods     : forward-return horizons in trading days
    output_dir  : directory where plots and CSVs are saved

    Returns
    -------
    factor_data : cleaned factor DataFrame (for further custom analysis)
    """
    os.makedirs(output_dir, exist_ok=True)
    if save_factor_values_flag:
        # already saved upstream; this is a safeguard hook
        pass

    # ── 1. Build Alphalens factor_data ────────────────────────────────────────
    print("\nBuilding Alphalens factor_data …")
    factor_data = al.utils.get_clean_factor_and_forward_returns(
        factor,
        prices,
        quantiles=quantiles,
        periods=periods,
        max_loss=0.35,
    )
    print(f"  factor_data shape: {factor_data.shape}")

    # ── 2. Returns tear sheet ─────────────────────────────────────────────────
    print("Generating returns tear sheet …")
    _save_tearsheet(
        al.tears.create_returns_tear_sheet,
        os.path.join(output_dir, "returns_tear_sheet.png"),
        factor_data, long_short=True, by_group=False,
    )

    # ── 3. IC tear sheet ──────────────────────────────────────────────────────
    print("Generating IC tear sheet …")
    _save_tearsheet(
        al.tears.create_information_tear_sheet,
        os.path.join(output_dir, "ic_tear_sheet.png"),
        factor_data,
    )

    # ── 4. Turnover tear sheet ────────────────────────────────────────────────
    print("Generating turnover tear sheet …")
    _save_tearsheet(
        al.tears.create_turnover_tear_sheet,
        os.path.join(output_dir, "turnover_tear_sheet.png"),
        factor_data,
    )

    # ── 5. Summary statistics CSV ─────────────────────────────────────────────
    _save_summary_stats(factor_data, output_dir)

    print(f"\nAll outputs saved to \'{output_dir}\'")
    return factor_data


def _save_summary_stats(factor_data: pd.DataFrame, output_dir: str) -> None:
    try:
        ic = al.performance.factor_information_coefficient(factor_data)
        ic.to_csv(os.path.join(output_dir, "ic_by_date.csv"))
        ic.describe().to_csv(os.path.join(output_dir, "ic_summary.csv"))
        print("  Saved: ic_by_date.csv, ic_summary.csv")
    except Exception as e:
        print(f"  WARNING: IC CSV export failed — {e}")

    try:
        mean_ret, _ = al.performance.mean_return_by_quantile(factor_data)
        mean_ret.to_csv(os.path.join(output_dir, "mean_return_by_quantile.csv"))
        print("  Saved: mean_return_by_quantile.csv")
    except Exception as e:
        print(f"  WARNING: quantile return CSV export failed — {e}")
