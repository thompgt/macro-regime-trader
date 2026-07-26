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
    exposure: pd.Series


def run_backtest(ohlcv: pd.DataFrame, settings: Settings | None = None) -> BacktestResult:
    """Run the full sequential simulation over every bar in ``ohlcv``.

    Execution is deliberately lagged by one bar: the signal computed from the
    close of bar ``t - 1`` is what trades at the close of bar ``t``. Acting on
    the same close that produced the signal would be a (mild but real)
    lookahead, and it flatters stop-loss behaviour in particular.
    """
    settings = settings or get_settings()

    engine = MacroRegimeEngine(settings)
    strategy = StrategyManager(settings)
    risk = RiskManager(settings)
    broker = MockBroker(settings)

    regimes = engine.classify(ohlcv)
    signals = strategy.generate_signals(ohlcv, regimes)

    # signals[i] is derived from bar i's close, so it is only actionable from
    # bar i + 1 onward. Bar 0 therefore trades on no signal at all (flat).
    lagged_signals = [None, *signals[:-1]]

    exposures: list[float] = []
    cooldown_remaining = 0
    session_date = None

    for (timestamp, row), signal in zip(ohlcv.iterrows(), lagged_signals, strict=True):
        price = float(row["close"])

        # The circuit breaker measures loss within a session; tell the risk
        # manager when the calendar session rolls over so its baseline tracks.
        bar_date = timestamp.date() if hasattr(timestamp, "date") else timestamp
        if bar_date != session_date:
            session_date = bar_date
            risk.reset_session(broker.total_equity(price))

        if signal is None:
            exposures.append(0.0)
            broker.step(timestamp, price, 0.0, None)
            continue

        decision = risk.validate(signal, current_equity=broker.total_equity(price))
        exposure = decision.adjusted_exposure if decision.approved else 0.0
        stop = signal.stop_price if decision.approved else None

        # After a forced stop-out, stand down for a few bars rather than
        # immediately buying back into the move that just stopped us out.
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            exposure = 0.0
            stop = None

        fill = broker.step(timestamp, price, exposure, stop)
        if fill.reason == "stop_loss":
            cooldown_remaining = settings.reentry_cooldown_bars

        # Record the exposure actually held after the bar, not the target that
        # was requested: the broker's no-trade band may have deferred the trade.
        equity_after = broker.total_equity(price)
        exposures.append(broker.position_qty * price / equity_after if equity_after else 0.0)

    equity_curve = pd.Series(broker.equity_curve, index=ohlcv.index, name="strategy")
    return BacktestResult(
        equity_curve=equity_curve,
        regimes=regimes,
        ledger=broker.ledger,
        exposure=pd.Series(exposures, index=ohlcv.index, name="exposure"),
    )


def rebase(equity: pd.Series, starting_balance: float) -> pd.Series:
    """Rescale an equity curve so it starts at ``starting_balance``."""
    if equity.empty or equity.iloc[0] == 0:
        return equity
    return equity / equity.iloc[0] * starting_balance


def run_backtest_from(
    ohlcv: pd.DataFrame,
    evaluation_start: object,
    settings: Settings | None = None,
) -> BacktestResult:
    """Backtest over all of ``ohlcv`` but report only from ``evaluation_start``.

    Bars before ``evaluation_start`` are consumed purely as indicator warmup.
    This keeps the strategy/benchmark comparison like-for-like: otherwise the
    strategy is unclassified and therefore flat for its first ~100 bars while
    buy-and-hold is fully invested from bar one, which penalizes the strategy
    for a data-availability artifact rather than for its signal.

    The returned equity curve is rebased to ``settings.starting_balance`` at
    ``evaluation_start`` so it is directly comparable to a benchmark curve
    computed over the same window.
    """
    settings = settings or get_settings()
    full = run_backtest(ohlcv, settings)

    index = full.equity_curve.loc[evaluation_start:].index
    if index.empty:
        raise ValueError(
            f"No bars at or after evaluation_start={evaluation_start!r}; nothing to evaluate."
        )

    return BacktestResult(
        equity_curve=rebase(full.equity_curve.loc[index], settings.starting_balance).rename(
            "strategy"
        ),
        regimes=full.regimes.loc[index],
        ledger=full.ledger[full.ledger["timestamp"].isin(index)],
        exposure=full.exposure.loc[index],
    )


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
                exposure=full_result.exposure.loc[oos_index],
            )
        )
    return results
