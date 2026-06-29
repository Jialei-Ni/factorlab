# tools/capm_analysis.py

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm

from alphalens_pipeline.data.fetcher import load_universe
from alphalens_pipeline.data.cleaner import clean_prices


START_DEFAULT = "2015-01-01"
END_DEFAULT = "2025-12-31"


def load_series(path: Path, column: str | None = None) -> pd.Series:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    if isinstance(df, pd.DataFrame):
        if column:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in file.")
            s = df[column]
        else:
            if df.shape[1] != 1:
                raise ValueError("File has multiple columns; specify --column.")
            s = df.iloc[:, 0]
    else:
        s = df

    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def compute_factor_model(
    factors: dict[str, pd.Series],
    target_returns: pd.Series,
) -> dict:

    aligned = pd.concat(
        [target_returns.rename("target"), *[
            f.rename(name) for name, f in factors.items()
        ]],
        axis=1,
        join="inner",
    ).dropna()

    if len(aligned) < 3:
        raise ValueError(
            "Not enough overlapping observations to perform regression."
        )

    y = aligned["target"]
    X = aligned.drop(columns=["target"])
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    params = model.params
    tvals = model.tvalues
    pvals = model.pvalues

    result = {
        "n": int(model.nobs),
        "r2": model.rsquared,
        "alpha_daily": params["const"],
        "alpha_annual": params["const"] * 252,
        "alpha_t": tvals["const"],
        "alpha_p": pvals["const"],
    }

    for name in factors.keys():
        result[f"{name}_beta"] = params[name]
        result[f"{name}_t"] = tvals[name]
        result[f"{name}_p"] = pvals[name]

    return result


def run_analysis(
    series_path: Path,
    output_path: Path | None = None,
    start: str = START_DEFAULT,
    end: str = END_DEFAULT,
    column: str | None = None,
) -> dict:
    # Market: S&P 500
    market_prices, _, _ = load_universe(
        universe=["^GSPC"],
        start=start,
        end=end,
    )
    market_prices = clean_prices(market_prices)

    market_returns = market_prices["^GSPC"].pct_change().dropna()

    # Risk-free: 13-week Treasury Bill yield
    rf_prices, _, _ = load_universe(
        universe=["^IRX"],
        start=start,
        end=end,
    )
    rf_prices = clean_prices(rf_prices)

    rf_daily = rf_prices["^IRX"] / 100.0 / 252.0

    # Target series
    target_prices = load_series(series_path, column)
    target_returns = target_prices.pct_change().dropna()

    factors = {
        "mkt_rf": (market_returns - rf_daily).rename("mkt_rf"),
    }

    result = compute_factor_model(
        factors=factors,
        target_returns=target_returns,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([result]).to_csv(output_path, index=False)

    return result


def main():

    parser = argparse.ArgumentParser(
        description="Factor regression analysis (CAPM / multi-factor)"
    )

    parser.add_argument("--start", type=str, default=START_DEFAULT)
    parser.add_argument("--end", type=str, default=END_DEFAULT)

    parser.add_argument(
        "--series",
        type=str,
        required=True,
        help="Path to target series",
    )

    parser.add_argument(
        "--column",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (must include filename).",
    )

    args = parser.parse_args()

    result = run_analysis(
        series_path=Path(args.series),
        output_path=Path(args.output) if args.output else None,
        start=args.start,
        end=args.end,
        column=args.column,
    )

    print(f"Observations: {result['n']:,}\n")

    print(f"Alpha/day:    {result['alpha_daily']:.8f}")
    print(f"Alpha/year:   {result['alpha_annual']:.6f}")
    print(f"Alpha t-stat: {result['alpha_t']:.6f}")
    print(f"Alpha p-value:{result['alpha_p']:.6g}\n")

    for k, v in result.items():
        if "beta" in k:
            print(f"{k}: {v:.6f}")

    print(f"\nR²: {result['r2']:.6f}")

    if args.output is not None:
        print(f"\nSaved analysis to {args.output}")


if __name__ == "__main__":
    main()