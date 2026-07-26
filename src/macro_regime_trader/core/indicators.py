"""Causal technical indicators shared by the regime engine and strategy layer.

Every function here returns a series aligned to the input index where row ``t``
depends only on data up to and including ``t``. Keeping them in one module
means the regime classifier and the position sizer measure volatility the same
way, rather than each rolling their own and silently disagreeing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def realized_volatility(
    close: pd.Series, window: int, trading_days_per_year: int = 252
) -> pd.Series:
    """Annualized rolling standard deviation of log returns."""
    log_returns = np.log(close).diff()
    return log_returns.rolling(window=window, min_periods=window).std(ddof=0) * np.sqrt(
        trading_days_per_year
    )


def average_true_range(ohlcv: pd.DataFrame, window: int) -> pd.Series:
    """Average True Range over ``window`` bars.

    True range at row ``t`` uses ``close`` at ``t - 1`` together with
    ``high``/``low`` at ``t``, which is standard and causal.
    """
    high = ohlcv["high"]
    low = ohlcv["low"]
    prev_close = ohlcv["close"].shift(1)

    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(window=window, min_periods=window).mean()


def donchian_channels(ohlcv: pd.DataFrame, window: int) -> tuple[pd.Series, pd.Series]:
    """Rolling Donchian upper/lower channels, shifted to avoid lookahead.

    Shifting by one bar before rolling ensures the channel for row ``t`` only
    reflects information available at the close of row ``t - 1``, so comparing
    row ``t``'s price against it is a genuine breakout test.
    """
    upper = ohlcv["high"].shift(1).rolling(window=window, min_periods=window).max()
    lower = ohlcv["low"].shift(1).rolling(window=window, min_periods=window).min()
    return upper, lower
