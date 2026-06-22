"""
Run Alphalens tests for a list of indicators with logging + resume support.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from config.paths import ALPHALENS_PIPELINE_DIR, FACTOR_LIST_PATH, FACTOR_OUTPUT_DIR, LOG_DIR, ROOT
from pathlib import Path


DEFAULT_START = "2015-01-01"
DEFAULT_END = "2025-12-31"


def load_factors(path: Path) -> list[str]:
    factors = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            factors.append(line)
    return factors


def configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--factor-file", default=str(FACTOR_LIST_PATH))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    factor_file = Path(args.factor_file)

    if not factor_file.exists():
        raise FileNotFoundError(factor_file)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    configure_logging(LOG_DIR / "run_all_factors.log")

    factors = load_factors(factor_file)

    logging.info("Loaded %d factors", len(factors))

    failures = []

    output_root = FACTOR_OUTPUT_DIR / f"{args.start}_{args.end}"

    for idx, factor in enumerate(factors, start=1):

        factor_output_dir = output_root / factor

        if args.resume and factor_output_dir.exists():
            logging.info("[%d/%d] SKIP %s (already exists)", idx, len(factors), factor)
            continue

        logging.info("[%d/%d] RUN %s", idx, len(factors), factor)

        start_time = time.perf_counter()

        cmd = [
            sys.executable,
            str(ROOT / ALPHALENS_PIPELINE_DIR / "main.py"),
            "--factor", "stockstats",
            "--indicator", factor,
            "--start", args.start,
            "--end", args.end,
        ]

        factor_log_file = LOG_DIR / f"{factor}.log"

        try:
            with open(factor_log_file, "w", encoding="utf-8") as logf:
                result = subprocess.run(
                    cmd,
                    cwd=ROOT,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )

            elapsed = time.perf_counter() - start_time

            # if result.returncode == 0:
            #     logging.info("SUCCESS %s (%.2fs)", factor, elapsed)
            # else:
            #     logging.error("FAILED %s rc=%s", factor, result.returncode)
            #     failures.append(factor)
            
        except Exception as e:
            logging.exception("EXCEPTION %s", factor)
            # failures.append(factor)

        # Check whether the factor was successful
        # by checking if the required output files exist
        factor_output_dir = output_root / factor

        required_files = [
            factor_output_dir / "ic_by_date.csv",
            factor_output_dir / "ic_summary.csv",
            factor_output_dir / "mean_return_by_quantile.csv",
        ]

        is_success = (
            factor_output_dir.exists()
            and all(f.exists() for f in required_files)
        )

        if is_success:
            logging.info("SUCCESS %s", factor)
        else:
            logging.error("FAILED %s (missing outputs)", factor)
            failures.append(factor)

    (LOG_DIR / "failed_factors.txt").write_text(
        "\n".join(failures),
        encoding="utf-8",
    )

    logging.info(
        "DONE | success=%d failed=%d",
        len(factors) - len(failures),
        len(failures),
    )


if __name__ == "__main__":
    main()