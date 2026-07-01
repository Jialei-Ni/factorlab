# portfolio_pipeline/signal_diagnostics.py

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run_signal_diagnostics(
    signals: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    output_dir: Path,
    composite: pd.DataFrame | None = None,
    horizons: tuple[int, ...] = (1, 5, 10),
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    # normalize horizons
    if isinstance(horizons, str):
        horizons = tuple(int(x) for x in horizons.split(",") if x.strip())
    else:
        horizons = tuple(int(x) for x in horizons)

    # keep everything vectorized
    all_signals = dict(signals)

    if composite is not None:
        all_signals["__composite__"] = composite

    rows = []

    for horizon in horizons:

        forward_returns = prices.shift(-horizon).div(prices).sub(1.0)

        # vectorized ranking (same as before)
        forward_rank = forward_returns.rank(axis=1, method="average")

        for signal_name, signal in all_signals.items():

            signal_rank = signal.rank(axis=1, method="average")

            # vectorized IC time series
            ic = signal_rank.corrwith(
                forward_rank,
                axis=1,
                method="pearson",
            )

            ic = ic.dropna()

            if ic.empty:
                continue

            std = ic.std()

            rows.append(
                {
                    "signal": signal_name,
                    "horizon": horizon,
                    "mean_ic": ic.mean(),
                    "std_ic": std,
                    "ic_ir": ic.mean() / (std + 1e-12),
                    "hit_rate": (ic > 0).mean(),
                    "observations": len(ic),
                }
            )

    pd.DataFrame(rows).to_csv(output_dir / "ic_summary.csv", index=False)

    print(f"Saved IC summary → {output_dir / 'ic_summary.csv'}")


def _compute_ic_timeseries(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> pd.Series:
    """
    Vectorized IC time series (same logic as your existing code).
    """
    signal_rank = signal.rank(axis=1, method="average")
    forward_rank = forward_returns.rank(axis=1, method="average")

    ic = signal_rank.corrwith(forward_rank, axis=1, method="pearson")
    return ic.dropna()


def _style_ic_axis(ax):
    ax.set_ylim(-0.20, 0.20)
    ax.set_yticks([x / 100 for x in range(-20, 21, 5)])
    ax.grid(True)


def _compute_rolling_ic(
    ic_df: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """
    Rolling mean IC.
    """
    return ic_df.rolling(window).mean()


def _compute_rolling_ic_zscore(
    ic_df: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """
    Rolling z-score of IC.
    """
    mean = ic_df.rolling(window).mean()
    std = ic_df.rolling(window).std()

    return (ic_df - mean) / (std + 1e-12)


def _compute_ic_autocorrelation(
    ic_df: pd.DataFrame,
    max_lag: int = 60,
) -> pd.DataFrame:
    """
    Lag autocorrelation of each IC series.
    """
    rows = []

    for signal in ic_df.columns:

        s = ic_df[signal].dropna()

        for lag in range(max_lag + 1):
            rows.append(
                {
                    "signal": signal,
                    "lag": lag,
                    "autocorrelation": s.autocorr(lag),
                }
            )

    return pd.DataFrame(rows)


def _compute_sign_persistence(
    ic_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Statistics of consecutive positive/negative IC runs.
    """
    rows = []

    for signal in ic_df.columns:

        s = ic_df[signal].dropna()

        if s.empty:
            continue

        signs = s > 0

        runs = []

        current = signs.iloc[0]
        length = 1

        for value in signs.iloc[1:]:

            if value == current:
                length += 1
            else:
                runs.append((current, length))
                current = value
                length = 1

        runs.append((current, length))

        positive = [l for sign, l in runs if sign]
        negative = [l for sign, l in runs if not sign]

        rows.append(
            {
                "signal": signal,
                "positive_mean": np.mean(positive) if positive else np.nan,
                "positive_max": np.max(positive) if positive else np.nan,
                "negative_mean": np.mean(negative) if negative else np.nan,
                "negative_max": np.max(negative) if negative else np.nan,
                "flip_count": len(runs) - 1,
            }
        )

    return pd.DataFrame(rows)
    

def run_rolling_ic_diagnostics(
    signals: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    output_dir: Path,
    horizon: int = 5,
    rolling_window: int = 60,
    composite: pd.DataFrame | None = None,
    signal_weights: dict[str, float] | None = None,
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    forward_returns = prices.shift(-horizon).div(prices).sub(1.0)

    signals = dict(signals)

    if composite is not None:
        signals["__composite__"] = composite

    ic_series_map: dict[str, pd.Series] = {}

    rows = []

    # Compute IC time series

    for name, signal in signals.items():

        ic_ts = _compute_ic_timeseries(signal, forward_returns)

        ic_series_map[name] = ic_ts

        rows.append(
            {
                "signal": name,
                "horizon": horizon,
                "mean_ic": ic_ts.mean(),
                "std_ic": ic_ts.std(),
                "ic_ir": ic_ts.mean() / (ic_ts.std() + 1e-12),
                "hit_rate": (ic_ts > 0).mean(),
                "observations": len(ic_ts),
            }
        )

    ic_df = pd.DataFrame(ic_series_map)

    # Composite weighted IC

    weighted_ic_series = None

    if signal_weights is not None:

        aligned_weights = {
            k: v
            for k, v in signal_weights.items()
            if k in ic_df.columns
        }

        total = sum(abs(v) for v in aligned_weights.values()) or 1.0

        aligned_weights = {
            k: v / total
            for k, v in aligned_weights.items()
        }

        weighted_ic_series = sum(
            ic_df[k].fillna(0.0) * w
            for k, w in aligned_weights.items()
        )

        ic_df["__weighted__"] = weighted_ic_series

    # Additional diagnostics

    rolling_ic = _compute_rolling_ic(
        ic_df,
        rolling_window,
    )

    rolling_zscore = _compute_rolling_ic_zscore(
        ic_df,
        rolling_window,
    )

    autocorrelation = _compute_ic_autocorrelation(
        ic_df,
        max_lag=60,
    )

    sign_persistence = _compute_sign_persistence(
        ic_df,
    )

    # Export tables

    pd.DataFrame(rows).to_csv(
        output_dir / "rolling_ic_summary.csv",
        index=False,
    )

    ic_df.to_csv(
        output_dir / "rolling_ic_timeseries.csv"
    )

    rolling_ic.to_csv(
        output_dir / "rolling_ic.csv"
    )

    rolling_zscore.to_csv(
        output_dir / "rolling_ic_zscore.csv"
    )

    autocorrelation.to_csv(
        output_dir / "ic_autocorrelation.csv",
        index=False,
    )

    sign_persistence.to_csv(
        output_dir / "sign_persistence.csv",
        index=False,
    )

    # 5. plot rolling IC of individual signals
    fig, ax = plt.subplots(figsize=(12, 6))

    for name in ic_df.columns:
        if name in ["__composite__", "__weighted__"]:
            continue

        rolling = ic_df[name].rolling(rolling_window).mean()
        ax.plot(rolling, label=name)

    ax.set_title(f"Rolling IC (Individual Signals, window={rolling_window})")
    ax.set_ylabel("IC (rolling mean)")
    _style_ic_axis(ax)

    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(
        output_dir / "rolling_ic_signals.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 6. plot rolling IC of composite + weighted signals
    fig, ax = plt.subplots(figsize=(12, 6))
    if "__composite__" in ic_df.columns:
        rolling = ic_df["__composite__"].rolling(rolling_window).mean()
        ax.plot(rolling, label="composite", linewidth=2)

    if weighted_ic_series is not None:
        ax.plot(
            weighted_ic_series.rolling(rolling_window).mean(),
            label="weighted",
            linewidth=2,
            linestyle="--",
        )

    ax.set_title(f"Rolling IC (Composite vs Weighted, window={rolling_window})")
    ax.set_ylabel("IC (rolling mean)")
    _style_ic_axis(ax)

    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(
        output_dir / "rolling_ic_composite.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    for name in ic_df.columns:
        rolling = ic_df[name].rolling(rolling_window).mean()
        ax.plot(rolling, label=name)

    if weighted_ic_series is not None:
        ax.plot(
            weighted_ic_series.rolling(rolling_window).mean(),
            label="__weighted__",
            linewidth=2,
            linestyle="--",
        )

    ax.set_title(f"Rolling IC (window={rolling_window})")
    ax.set_ylabel("IC (rolling mean)")
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "rolling_ic_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved rolling IC diagnostics → {output_dir}")