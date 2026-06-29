from __future__ import annotations

from pathlib import Path
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

    # ---- normalize horizons ----
    if isinstance(horizons, str):
        horizons = tuple(int(x) for x in horizons.split(",") if x.strip())
    else:
        horizons = tuple(int(x) for x in horizons)

    # ---- keep everything vectorized ----
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

    # forward returns once (shared)
    forward_returns = prices.shift(-horizon).div(prices).sub(1.0)

    signals = dict(signals)

    if composite is not None:
        signals["__composite__"] = composite

    ic_series_map: dict[str, pd.Series] = {}

    rows = []

    # 1. compute IC time series
    for name, signal in signals.items():
        ic_ts = _compute_ic_timeseries(signal, forward_returns)
        ic_series_map[name] = ic_ts

        rolling_ic = ic_ts.rolling(rolling_window).mean()

        rows.append(
            {
                "signal": name,
                "horizon": horizon,
                "mean_ic": ic_ts.mean(),
                "rolling_mean_ic": rolling_ic.mean(),
                "rolling_std_ic": rolling_ic.std(),
                "stability": ic_ts.rolling(rolling_window).std().mean(),
            }
        )

    # 2. weighted IC (FIXED + aligned)
    weighted_ic_series = None

    if signal_weights is not None:

        aligned_weights = {
            k: v for k, v in signal_weights.items() if k in ic_series_map
        }

        total = sum(abs(w) for w in aligned_weights.values()) or 1.0
        aligned_weights = {k: w / total for k, w in aligned_weights.items()}

        aligned = []
        for k, w in aligned_weights.items():
            aligned.append(ic_series_map[k].rename(k) * w)

        if aligned:
            ic_mat = pd.concat(aligned, axis=1).fillna(0.0)
            weighted_ic_series = ic_mat.sum(axis=1)

            r = weighted_ic_series.rolling(rolling_window).mean()

            rows.append(
                {
                    "signal": "__weighted__",
                    "horizon": horizon,
                    "mean_ic": weighted_ic_series.mean(),
                    "rolling_mean_ic": r.mean(),
                    "rolling_std_ic": r.std(),
                    "stability": r.std(),
                }
            )

            weighted_ic_series.to_csv(
                output_dir / "weighted_ic_timeseries.csv",
                header=["weighted_ic"],
            )

    # 3. save summary
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "rolling_ic_summary.csv", index=False)

    # 4. export full time series
    ic_df = pd.DataFrame(ic_series_map)
    ic_df.to_csv(output_dir / "rolling_ic_timeseries.csv")

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