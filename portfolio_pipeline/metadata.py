from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path


def save_metadata(
    path: Path,
    universe_name: str,
    start_date: str,
    end_date: str,
    run_id: str,
    status: str = "completed",
) -> None:
    metadata = {
        "universe": universe_name,
        "start_date": start_date,
        "end_date": end_date,
        "generated_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "status": status,
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_config_snapshot(path: Path, config: dict) -> None:
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
