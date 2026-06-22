from __future__ import annotations

import pandas as pd


def apply_transaction_costs(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    commission_bps: float,
    slippage_bps: float,
) -> pd.DataFrame:
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    cost_pct = turnover * ((commission_bps + slippage_bps) / 10000)
    gross_returns = (weights * returns).sum(axis=1)
    net_returns = gross_returns - cost_pct
    return net_returns.to_frame(name="portfolio_return")


def build_holdings(
    weights: pd.DataFrame,
) -> pd.DataFrame:
    return weights.stack().reset_index().rename(
        columns={0: "portfolio_weight"}
    )
