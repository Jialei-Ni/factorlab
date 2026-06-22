"""
Generate failed_factor_list.txt by comparing factor_list.txt
against factor_ranking.csv.

Expected structure:

Factor Testing Using Alphalens/
│
├── factors_testing/
│   └── factor_list.txt
│
└── summary/
    ├── factor_ranking.csv
    └── failed_factor_list.txt
"""

from pathlib import Path

import pandas as pd

from config.paths import FACTOR_LIST_PATH, SUMMARY_DIR


FACTOR_LIST_FILE = FACTOR_LIST_PATH
RANKING_FILE = SUMMARY_DIR / "factor_ranking.csv"
OUTPUT_FILE = SUMMARY_DIR / "failed_factor_list.txt"


def load_requested_factors(path: Path) -> set[str]:
    factors = set()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            factors.add(line)

    return factors


def load_successful_factors(path: Path) -> set[str]:
    df = pd.read_csv(path)

    if "factor" not in df.columns:
        raise ValueError(
            "factor_ranking.csv does not contain a 'factor' column."
        )

    return set(
        df["factor"]
        .dropna()
        .astype(str)
        .str.strip()
    )


def main():
    requested = load_requested_factors(
        FACTOR_LIST_FILE
    )

    successful = load_successful_factors(
        RANKING_FILE
    )

    failed = sorted(
        requested - successful
    )

    OUTPUT_FILE.write_text(
        "\n".join(failed),
        encoding="utf-8",
    )

    print(
        f"Requested: {len(requested)}\n"
        f"Successful: {len(successful)}\n"
        f"Failed: {len(failed)}\n"
        f"Output: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()  