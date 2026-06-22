from __future__ import annotations

import pandas as pd


def _normalize_long_only(weights: pd.DataFrame) -> pd.DataFrame:
    positive = weights.clip(lower=0)
    scale = positive.sum(axis=1).replace(0, 1.0)
    return positive.div(scale, axis=0)


def _normalize_long_short(weights: pd.DataFrame) -> pd.DataFrame:
    longs = weights.clip(lower=0)
    shorts = weights.clip(upper=0)

    long_scale = (0.5 / longs.sum(axis=1)).replace([float("inf"), 0], 0.0)
    short_scale = (0.5 / (-shorts.sum(axis=1))).replace([float("inf"), 0], 0.0)

    longs = longs.mul(long_scale, axis=0)
    shorts = shorts.mul(short_scale, axis=0)
    return longs + shorts


def _quantile_count(n_assets: int, quantile: float) -> int:
    count = int(round(n_assets * quantile))
    return max(1, min(count, n_assets))


def construct_weights(
    exposures: pd.DataFrame,
    portfolio_type: str,
    long_quantile: float,
    short_quantile: float,
    holding_days: int,
    rebalance_frequency: int,
    trading_dates: pd.DatetimeIndex,
    asset_universe: pd.Index,
) -> pd.DataFrame:
    composite = exposures["composite_score"].unstack()
    weights = pd.DataFrame(0.0, index=trading_dates, columns=asset_universe)

    for position, signal_date in enumerate(trading_dates[::rebalance_frequency]):
        if signal_date not in composite.index:
            continue

        next_index = trading_dates.get_indexer([signal_date])[0] + 1
        if next_index >= len(trading_dates):
            continue

        execution_dates = trading_dates[next_index : next_index + holding_days]
        candidate_scores = composite.loc[signal_date].dropna()
        if candidate_scores.empty:
            continue

        n_assets = len(candidate_scores)
        long_count = _quantile_count(n_assets, long_quantile)
        short_count = _quantile_count(n_assets, short_quantile)

        if portfolio_type == "long_only":
            selected = candidate_scores.nlargest(long_count)
            daily_weights = pd.Series(0.0, index=asset_universe)
            if not selected.empty:
                daily_weights[selected.index] = 1.0 / len(selected)

        elif portfolio_type == "long_short":
            longs = candidate_scores.nlargest(long_count)
            shorts = candidate_scores.nsmallest(short_count)
            daily_weights = pd.Series(0.0, index=asset_universe)
            if not longs.empty:
                daily_weights[longs.index] = 0.5 / len(longs)
            if not shorts.empty:
                daily_weights[shorts.index] = -0.5 / len(shorts)
        else:
            raise ValueError(
                f"Unsupported portfolio_type '{portfolio_type}'. "
                "Use 'long_only' or 'long_short'."
            )

        for execution_date in execution_dates:
            weights.loc[execution_date] += daily_weights

    if portfolio_type == "long_only":
        weights = _normalize_long_only(weights)
    else:
        weights = _normalize_long_short(weights)

    return weights.fillna(0.0)
