"""Tests for :mod:`macro_regime_trader.core.strategies`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macro_regime_trader.config import get_settings
from macro_regime_trader.core.strategies import StrategyManager
from macro_regime_trader.types import Regime


def _make_ohlcv(n: int = 60, start_price: float = 100.0, seed: int = 7) -> pd.DataFrame:
    """Synthetic, mildly-trending OHLCV frame with enough bars for warmup."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")

    drift = np.linspace(0, 5, n)
    noise = rng.normal(0, 0.5, n)
    close = start_price + drift + noise
    close = np.maximum(close, 1.0)

    high = close + rng.uniform(0.1, 0.6, n)
    low = close - rng.uniform(0.1, 0.6, n)
    open_ = close + rng.normal(0, 0.2, n)
    volume = rng.uniform(1_000, 2_000, n)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture
def manager() -> StrategyManager:
    return StrategyManager(get_settings())


def test_exposure_ordering(manager: StrategyManager) -> None:
    ohlcv = _make_ohlcv()

    bull = manager.generate_signal(ohlcv, Regime.SUSTAINED_BULL)
    volatile = manager.generate_signal(ohlcv, Regime.VOLATILE_DISTRIBUTION)
    compressed = manager.generate_signal(ohlcv, Regime.COMPRESSED_LIQUIDITY)
    bear = manager.generate_signal(ohlcv, Regime.STRUCTURAL_BEAR)

    assert bear.target_exposure == 0.0
    assert bull.target_exposure > volatile.target_exposure
    assert bull.target_exposure > compressed.target_exposure
    assert volatile.target_exposure > bear.target_exposure
    assert compressed.target_exposure > bear.target_exposure


def test_breakout_increases_bull_exposure(manager: StrategyManager) -> None:
    ohlcv = _make_ohlcv()

    non_breakout = manager.generate_signal(ohlcv, Regime.SUSTAINED_BULL)
    assert "no_breakout" in non_breakout.reason

    # Force the final bar to break decisively above the Donchian upper channel.
    breakout_ohlcv = ohlcv.copy()
    window = get_settings().donchian_window
    recent_high = breakout_ohlcv["high"].iloc[-(window + 1) : -1].max()
    breakout_close = recent_high + 10.0
    breakout_ohlcv.iloc[-1, breakout_ohlcv.columns.get_loc("close")] = breakout_close
    breakout_ohlcv.iloc[-1, breakout_ohlcv.columns.get_loc("high")] = breakout_close + 0.5
    breakout_ohlcv.iloc[-1, breakout_ohlcv.columns.get_loc("low")] = breakout_close - 0.5

    breakout_signal = manager.generate_signal(breakout_ohlcv, Regime.SUSTAINED_BULL)

    assert "donchian_breakout" in breakout_signal.reason
    assert breakout_signal.target_exposure > non_breakout.target_exposure


def test_stop_price_set_for_nonzero_exposure(manager: StrategyManager) -> None:
    ohlcv = _make_ohlcv()

    bull = manager.generate_signal(ohlcv, Regime.SUSTAINED_BULL)
    volatile = manager.generate_signal(ohlcv, Regime.VOLATILE_DISTRIBUTION)
    compressed = manager.generate_signal(ohlcv, Regime.COMPRESSED_LIQUIDITY)

    latest_close = float(ohlcv["close"].iloc[-1])

    for signal in (bull, volatile, compressed):
        assert signal.target_exposure > 0.0
        assert signal.stop_price is not None
        assert signal.stop_price < latest_close


def test_stop_price_none_for_zero_exposure(manager: StrategyManager) -> None:
    ohlcv = _make_ohlcv()
    bear = manager.generate_signal(ohlcv, Regime.STRUCTURAL_BEAR)

    assert bear.target_exposure == 0.0
    assert bear.stop_price is None


def test_generate_signals_aligned_length(manager: StrategyManager) -> None:
    ohlcv = _make_ohlcv(n=80)

    regime_cycle = [
        Regime.SUSTAINED_BULL,
        Regime.VOLATILE_DISTRIBUTION,
        Regime.STRUCTURAL_BEAR,
        Regime.COMPRESSED_LIQUIDITY,
    ]
    regimes = pd.Series(
        [regime_cycle[i % len(regime_cycle)].value for i in range(len(ohlcv))],
        index=ohlcv.index,
    )

    signals = manager.generate_signals(ohlcv, regimes)

    assert len(signals) == len(ohlcv)
    assert [s.timestamp for s in signals] == list(ohlcv.index)


def test_generate_signals_handles_missing_regime(manager: StrategyManager) -> None:
    ohlcv = _make_ohlcv(n=30)
    regimes = pd.Series([None] * len(ohlcv), index=ohlcv.index, dtype=object)

    signals = manager.generate_signals(ohlcv, regimes)

    assert len(signals) == len(ohlcv)
    assert all(s.target_exposure == 0.0 for s in signals)
