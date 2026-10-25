"""Headline statistics of an equity curve, shared by the simulator and the dashboard.

One definition, so ``tradingagents simulate`` and the dashboard show the same
numbers for the same run. Conventions:

- Given the cash the account started with, returns are measured from it: the
  first session's return is against that cash, so costs paid on the first
  session show, and the drawdown's peak is never below it.
- Returns are simple daily returns; annualization uses 252 sessions, a zero
  risk-free rate and the sample standard deviation.
"""

from __future__ import annotations

import math

import pandas as pd

TRADING_DAYS = 252


def curve_returns(curve: pd.Series, starting_value: float | None = None) -> pd.Series:
    """Daily returns of ``curve``, with day zero against ``starting_value`` when given."""
    returns = curve.astype(float).pct_change().dropna()
    if starting_value is None or curve.empty:
        return returns
    first = pd.Series([float(curve.iloc[0]) / starting_value - 1], index=curve.index[:1])
    return pd.concat([first, returns])


def curve_drawdown(curve: pd.Series, starting_value: float | None = None) -> pd.Series:
    """Fractional distance below the running peak, 0 at a new high; the peak starts at ``starting_value``."""
    peak = curve.cummax()
    if starting_value is not None:
        peak = peak.clip(lower=starting_value)
    return curve / peak - 1


def curve_stats(curve: pd.Series, starting_value: float | None = None) -> dict:
    """Total return, CAGR, volatility, Sharpe and maximum drawdown of ``curve``.

    Without ``starting_value`` the curve's first value is the base.
    """
    base = float(starting_value if starting_value is not None else curve.iloc[0])
    returns = curve_returns(curve, starting_value)
    total = float(curve.iloc[-1]) / base - 1
    days = max((curve.index[-1] - curve.index[0]).days, 1)
    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    return {
        "total_return": total,
        "cagr": (1 + total) ** (365.25 / days) - 1 if total > -1 else -1.0,
        "volatility": std * math.sqrt(TRADING_DAYS),
        "sharpe": float(returns.mean()) / std * math.sqrt(TRADING_DAYS) if std > 0 else 0.0,
        "max_drawdown": float(curve_drawdown(curve, starting_value).min()),
    }
