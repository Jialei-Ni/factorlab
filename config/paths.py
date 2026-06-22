from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = Path(__file__).resolve().parent / "paths.json"


def _load_overrides() -> dict[str, Any]:
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {CONFIG_FILE}: {exc}") from exc


_OVERRIDES = _load_overrides()


def _resolve(key: str, default: str) -> Path:
    value = _OVERRIDES.get(key)
    path_value = Path(value) if value is not None else Path(default)
    if path_value.is_absolute():
        return path_value
    return ROOT / path_value


DATA_DIR = _resolve("DATA_DIR", "data")
CACHE_DIR = _resolve("CACHE_DIR", "data/cache")
OUTPUT_DIR = _resolve("OUTPUT_DIR", "output")
FACTOR_OUTPUT_DIR = OUTPUT_DIR / "factors"
PORTFOLIO_OUTPUT_DIR = OUTPUT_DIR / "portfolios"
SIGNAL_CACHE_DIR = CACHE_DIR / "signals"
SUMMARY_DIR = _resolve("SUMMARY_DIR", "summary")
LOG_DIR = _resolve("LOG_DIR", "logs")
FACTOR_LIST_PATH = _resolve("FACTOR_LIST_PATH", "factors_testing/factor_list.txt")
FACTORS_TESTING_DIR = _resolve("FACTORS_TESTING_DIR", "factors_testing")
ALPHALENS_PIPELINE_DIR = _resolve("ALPHALENS_PIPELINE_DIR", "alphalens_pipeline")


def ensure_directories() -> None:
    for directory in (DATA_DIR, CACHE_DIR, OUTPUT_DIR, FACTOR_OUTPUT_DIR,
                      PORTFOLIO_OUTPUT_DIR, SIGNAL_CACHE_DIR, SUMMARY_DIR, LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def alphalens_output_root(start: str, end: str) -> Path:
    return FACTOR_OUTPUT_DIR / f"{start}_{end}"


def portfolio_output_root(
    universe_name: str,
    start: str,
    end: str,
) -> Path:
    return PORTFOLIO_OUTPUT_DIR / universe_name / f"{start}_{end}"


def signal_cache_root(
    universe_name: str,
    start: str,
    end: str,
) -> Path:
    return CACHE_DIR / "signals" / universe_name / f"{start}_{end}"
