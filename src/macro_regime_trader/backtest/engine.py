"""Backtest engine: wires MacroRegimeEngine -> StrategyManager -> RiskManager -> MockBroker
together into a single sequential simulation, plus walk-forward windowing for
out-of-sample reporting.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from macro_regime_trader.config import Settings, get_settings
from macro_regime_trader.core.macro_engine import MacroRegimeEngine
from macro_regime_trader.core.risk_manager import RiskManager
from macro_regime_trader.core.strategies import StrategyManager
from macro_regime_trader.simulation.mock_broker import MockBroker


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    regimes: pd.Series
    ledger: pd.DataFrame


def run_backtest(ohlcv: pd.DataFrame, settings: Settings | None = None) -> BacktestResult:
    """Run the full sequential simulation over every bar in ``ohlcv``."""
    settings = settings or get_settings()

    engine = MacroRegimeEngine(settings)
    strategy = StrategyManager(settings)
    risk = RiskManager(settings)
    broker = MockBroker(settings)

    regimes = engine.classify(ohlcv)
    signals = strategy.generate_signals(ohlcv, regimes)

    for (timestamp, row), signal in zip(ohlcv.iterrows(), signals, strict=True):
        price = float(row["close"])
        decision = risk.validate(signal, current_equity=broker.total_equity(price))
        stop = signal.stop_price if decision.approved else None
        broker.step(timestamp, price, decision.adjusted_exposure, stop)

    equity_curve = pd.Series(broker.equity_curve, index=ohlcv.index, name="strategy")
    return BacktestResult(equity_curve=equity_curve, regimes=regimes, ledger=broker.ledger)


def walk_forward_windows(
    n_bars: int, train_window: int, test_window: int
) -> list[tuple[slice, slice]]:
    """Rolling (train_slice, test_slice) index-position pairs.

    The train segment only serves as indicator warmup (EMA/Donchian/ATR history) --
    none of this engine's parameters are fitted -- so each test window is the true
    out-of-sample segment for reporting.
    """
    windows = []
    start = 0
    while start + train_window + test_window <= n_bars:
        train_slice = slice(start, start + train_window)
        test_slice = slice(start + train_window, start + train_window + test_window)
        windows.append((train_slice, test_slice))
        start += test_window
    return windows


def run_walk_forward_backtest(
    ohlcv: pd.DataFrame, settings: Settings | None = None
) -> list[BacktestResult]:
    """Run one independent backtest per walk-forward window, each warmed up on its
    own train segment and evaluated on its OOS test segment.
    """
    settings = settings or get_settings()
    windows = walk_forward_windows(len(ohlcv), settings.train_window, settings.test_window)

    results = []
    for train_slice, test_slice in windows:
        warmup_and_test = ohlcv.iloc[train_slice.start : test_slice.stop]
        full_result = run_backtest(warmup_and_test, settings)
        oos_index = ohlcv.index[test_slice]
        results.append(
            BacktestResult(
                equity_curve=full_result.equity_curve.loc[oos_index],
                regimes=full_result.regimes.loc[oos_index],
                ledger=full_result.ledger[full_result.ledger["timestamp"].isin(oos_index)],
            )
        )
    return results
