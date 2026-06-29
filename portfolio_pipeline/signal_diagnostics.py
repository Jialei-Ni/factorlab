from __future__ import annotations

from pathlib import Path

import pandas as pd


def run_signal_diagnostics(
    signals: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    output_dir: Path,
    horizons: tuple[int, ...] = (1, 5, 10),
) -> None:
    """
    Compute cross-sectional Spearman Information Coefficient (IC)
    for each signal.

    Parameters
    ----------
    signals
        Mapping from signal name to signal DataFrame
        (index=date, columns=ticker).

    prices
        Cleaned price DataFrame.

    output_dir
        Portfolio output directory.

    horizons
        Forward return horizons (trading days).

    Outputs
    -------
    ic_summary.csv
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for horizon in horizons:

        # Forward returns
        forward_returns = prices.shift(-horizon).div(prices).sub(1.0)

        # Spearman = Pearson on ranks
        forward_rank = forward_returns.rank(
            axis=1,
            method="average",
            na_option="keep",
        )

        for signal_name, signal in signals.items():

            signal_rank = signal.rank(
                axis=1,
                method="average",
                na_option="keep",
            )

            # Cross-sectional correlation for each date
            ic = signal_rank.corrwith(
                forward_rank,
                axis=1,
                method="pearson",
            ).dropna()

            std = ic.std()

            rows.append(
                {
                    "signal": signal_name,
                    "horizon": horizon,
                    "mean_ic": ic.mean(),
                    "std_ic": std,
                    "ic_ir": (
                        ic.mean() / std
                        if pd.notna(std) and std > 0
                        else float("nan")
                    ),
                    "hit_rate": (ic > 0).mean(),
                    "observations": len(ic),
                }
            )

    summary = pd.DataFrame(rows)

    summary.to_csv(
        output_dir / "ic_summary.csv",
        index=False,
    )

    print(f"Saved IC summary to {output_dir / 'ic_summary.csv'}")