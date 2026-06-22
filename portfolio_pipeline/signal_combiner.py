from __future__ import annotations

import pandas as pd


def _rank_normalize(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.rank(axis=1, pct=True, method="average", na_option="keep")
    return ranked * 2 - 1


def combine_signals(
    signals: dict[str, pd.DataFrame],
    weights: dict[str, float],
) -> pd.DataFrame:
    if not weights:
        raise ValueError("No signal weights provided for combination.")

    normalized: dict[str, pd.DataFrame] = {}
    for signal_name, weight in weights.items():
        if signal_name not in signals:
            raise ValueError(
                f"Signal '{signal_name}' is not available for combination. "
                f"Available signals: {list(signals.keys())}"
            )
        normalized[signal_name] = _rank_normalize(signals[signal_name])

    composite = None
    for signal_name, weight in weights.items():
        composite = (
            normalized[signal_name] * weight
            if composite is None
            else composite + normalized[signal_name] * weight
        )

    composite_stack = composite.stack(dropna=False).rename("composite_score")
    exposures = pd.DataFrame(index=composite_stack.index)
    exposures["composite_score"] = composite_stack

    for signal_name, frame in normalized.items():
        exposures[signal_name] = frame.stack(dropna=False)

    exposures = exposures.sort_index()
    return exposures
