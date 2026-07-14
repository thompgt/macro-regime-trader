"""Performance analytics computed from an equity curve.

Win rate is defined per-bar (fraction of steps with a non-negative equity
change) rather than per-round-trip-trade, since the broker doesn't track
FIFO cost basis across partial rebalances -- this is the standard choice
for a continuously-rebalanced exposure strategy rather than discrete
buy/sell pairs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def periodic_returns(equity_curve: pd.Series) -> pd.Series:
    return equity_curve.pct_change().dropna()


def total_return(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return 0.0
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)


def sharpe_ratio(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    returns = periodic_returns(equity_curve)
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return 0.0
    return float(returns.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1.0
    return float(drawdown.min())


def win_rate(equity_curve: pd.Series) -> float:
    returns = periodic_returns(equity_curve)
    if returns.empty:
        return 0.0
    return float((returns > 0).mean())


def compute_metrics(equity_curve: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    return {
        "total_return": total_return(equity_curve),
        "sharpe_ratio": sharpe_ratio(equity_curve, periods_per_year),
        "max_drawdown": max_drawdown(equity_curve),
        "win_rate": win_rate(equity_curve),
    }
