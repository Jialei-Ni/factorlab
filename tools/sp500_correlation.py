from pathlib import Path
import argparse

import pandas as pd

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


def compute_correlation(spx: pd.Series, target: pd.Series) -> tuple[int, float]:
    aligned = pd.concat(
        [spx.rename("sp500"), target.rename("target")],
        axis=1,
        join="inner",
    ).dropna()

    corr = aligned["sp500"].corr(aligned["target"])
    return len(aligned), corr


def main():
    parser = argparse.ArgumentParser(description="SPX correlation checker")

    parser.add_argument("--start", type=str, default=START_DEFAULT)
    parser.add_argument("--end", type=str, default=END_DEFAULT)
    parser.add_argument("--series", type=str, required=True, help="Series path")
    parser.add_argument("--column", type=str, default=None)

    args = parser.parse_args()

    series_path = Path(args.series)

    prices, _, _ = load_universe(
        universe=["^GSPC"],
        start=args.start,
        end=args.end,
    )

    spx = clean_prices(prices)["^GSPC"]

    target = load_series(series_path, args.column)

    n, corr = compute_correlation(spx, target)

    print(f"Observations: {n:,}")
    print(f"Correlation:  {corr:.6f}")


if __name__ == "__main__":
    main()