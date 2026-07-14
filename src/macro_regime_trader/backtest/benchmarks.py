"""Reference benchmarks to compare strategy performance against."""

from __future__ import annotations

import pandas as pd

from macro_regime_trader.config import Settings, get_settings


def buy_and_hold_equity(ohlcv: pd.DataFrame, settings: Settings | None = None) -> pd.Series:
    settings = settings or get_settings()
    close = ohlcv["close"]
    shares = settings.starting_balance * (1 - settings.slippage_pct) / close.iloc[0]
    return (shares * close).rename("buy_and_hold")


def dma_crossover_equity(ohlcv: pd.DataFrame, settings: Settings | None = None) -> pd.Series:
    """Long-only strategy: fully invested when close > trailing DMA, flat otherwise.

    Uses the prior bar's DMA/position (``.shift(1)``) to decide today's exposure,
    avoiding lookahead bias.
    """
    settings = settings or get_settings()
    close = ohlcv["close"]
    dma = close.rolling(settings.benchmark_dma_window).mean()
    invested = (close.shift(1) > dma.shift(1)).fillna(False)

    cash = settings.starting_balance
    position_qty = 0.0
    equity = []
    for price, is_invested in zip(close, invested):
        target_value = (cash + position_qty * price) * (1.0 if is_invested else 0.0)
        current_value = position_qty * price
        delta_value = target_value - current_value
        if abs(delta_value) > 1e-9:
            fill_price = price * (1 + settings.slippage_pct if delta_value > 0 else 1 - settings.slippage_pct)
            delta_qty = delta_value / fill_price
            cash -= delta_qty * fill_price
            position_qty += delta_qty
        equity.append(cash + position_qty * price)

    return pd.Series(equity, index=ohlcv.index, name="dma_crossover")
