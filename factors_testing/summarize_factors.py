from __future__ import annotations

import argparse
import json
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import t as student_t


DEFAULT_START = "2015-01-01"
DEFAULT_END = "2025-12-31"


def safe_spearman(values):
    try:
        stat, _ = spearmanr([1, 2, 3, 4, 5], values)
        if pd.isna(stat):
            return np.nan
        return abs(float(stat))
    except Exception:
        return np.nan


def compute_ic(series):
    series = series.dropna()
    n = len(series)

    if n < 3:
        return dict(mean=np.nan, std=np.nan, raic=np.nan, t=np.nan)

    mean = series.mean()
    std = series.std()

    if std == 0:
        return dict(mean=mean, std=std, raic=np.nan, t=np.nan)

    raic = mean / std
    t_stat = mean / (std / np.sqrt(n))
    p = 2 * (1 - student_t.cdf(abs(t_stat), df=n - 1))

    return dict(mean=mean, std=std, raic=raic, t=t_stat, p=p)


def load_config(path: Path):
    if path.exists():
        return json.loads(path.read_text())
    return {
        "weights": {"ic": 0.35, "t": 0.25, "raic": 0.20, "spread": 0.10, "mono": 0.10},
        "horizon_weights": {"1D": 0.1, "5D": 0.3, "10D": 0.6},
        "thresholds": {"ic_abs": 0.02, "t_abs": 2, "raic_abs": 0.10, "mono_min": 0.70},
    }


def zscore(s):
    std = s.std()
    if std == 0:
        return pd.Series(0, index=s.index)
    return (s - s.mean()) / std


def process_factor(dir_path: Path, cfg: dict):

    row = {"factor": dir_path.name, "status": "success"}

    ic_file = dir_path / "ic_by_date.csv"
    q_file = dir_path / "mean_return_by_quantile.csv"

    if not dir_path.exists():
        return {"factor": dir_path.name, "status": "missing_folder"}
    
    if not ic_file.exists() or not q_file.exists():
        return {"factor": dir_path.name, "status": "missing_files"}

    ic = pd.read_csv(ic_file)
    q = pd.read_csv(q_file).sort_values("factor_quantile")

    horizons = cfg["horizon_weights"]

    for h, w in horizons.items():

        if h in ic.columns:
            m = compute_ic(ic[h])
            row[f"ic_{h}"] = m["mean"]
            row[f"t_{h}"] = m["t"]
            row[f"raic_{h}"] = m["raic"]

        if h in q.columns:
            vals = q[h].astype(float).to_numpy()
            if len(vals) == 5:
                row[f"spread_{h}"] = vals[-1] - vals[0]
                row[f"mono_{h}"] = safe_spearman(vals)

    return row


def load_factor_list(path: Path) -> list[str]:
    factors = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            factors.append(line)

    return factors


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--config", default="ranking_config.json")
    ap.add_argument("--factor-file", default="factors_testing/factor_list.txt")

    args = ap.parse_args()

    cfg = load_config(Path(args.config))

    root = Path("output/factors") / f"{args.start}_{args.end}"

    factor_file = Path(args.factor_file)

    factors = load_factor_list(factor_file)

    rows = []

    for factor in factors:

        factor_dir = root / factor

        rows.append(
            process_factor(
                factor_dir,
                cfg,
            )
        )

    df = pd.DataFrame(rows)

    summary_dir = Path("summary")
    summary_dir.mkdir(exist_ok=True)

    df["generated_at"] = datetime.now(UTC).isoformat()
    # df["generated_at"] = datetime.utcnow()
    df["start"] = args.start
    df["end"] = args.end

    df.to_csv(summary_dir / "factor_metrics.csv", index=False)

    ok = df[df["status"] == "success"].copy()

    if ok.empty:
        return

    w = cfg["weights"]

    ok["score"] = (
        w["ic"] * zscore(ok["ic_10D"].abs())
        + w["t"] * zscore(ok["t_10D"].abs())
        + w["raic"] * zscore(ok["raic_10D"].abs())
        + w["spread"] * zscore(ok["spread_10D"].abs())
        + w["mono"] * zscore(ok["mono_10D"].abs())
    )

    ok.sort_values("score", ascending=False).to_csv(
        summary_dir / "factor_ranking.csv",
        index=False,
    )

    t = cfg["thresholds"]

    top = ok[
        (ok["ic_10D"].abs() > t["ic_abs"])
        & (ok["t_10D"].abs() > t["t_abs"])
        & (ok["raic_10D"].abs() > t["raic_abs"])
        & (ok["mono_10D"] > t["mono_min"])
    ]

    top.to_csv(summary_dir / "top_factors.csv", index=False)


if __name__ == "__main__":
    main()