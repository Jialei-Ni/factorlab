from __future__ import annotations

import pandas as pd
import numpy as np

import matplotlib as plt


def cumulative_return(returns: pd.Series) -> float:
    return (1 + returns).cumprod().iloc[-1] - 1


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    total_days = len(returns.dropna())
    if total_days == 0:
        return 0.0
    cumulative = cumulative_return(returns)
    return (1 + cumulative) ** (periods_per_year / total_days) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    return returns.std(ddof=1) * (periods_per_year ** 0.5)


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    vol = annualized_volatility(returns, periods_per_year)
    if vol == 0:
        return 0.0
    return annualized_return(returns, periods_per_year) / vol


def max_drawdown(returns: pd.Series) -> float:
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return float(drawdown.min())


def turnover( weights: pd.DataFrame ) -> float:
    daily_turnover = weights.diff().abs().sum(axis=1)
    return float(daily_turnover.mean())


def summarize_performance(
    portfolio_returns: pd.Series,
    weights: pd.DataFrame,
) -> dict[str, float]:
    returns = portfolio_returns.dropna()
    return {
        "cumulative_return": cumulative_return(returns),
        "annualized_return": annualized_return(returns),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "turnover": turnover(weights),
    }

def performance_summary_frame(summary: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame.from_dict(summary, orient="index", columns=["value"]) 


def cumulative_return_series(returns: pd.Series) -> pd.Series:
    return (1 + returns.fillna(0)).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    equity = cumulative_return_series(returns)
    peak = equity.cummax()
    return (equity - peak) / peak

def annual_return_series(returns: pd.Series) -> pd.Series:
    r = returns.dropna()
    return (1 + r).groupby(r.index.year).agg(np.prod) - 1

def monthly_return_series(returns: pd.Series) -> pd.Series:
    r = returns.dropna()
    return (1 + r).groupby(pd.Grouper(freq="M")).agg(np.prod) - 1