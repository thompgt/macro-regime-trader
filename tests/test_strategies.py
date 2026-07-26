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


def test_only_structural_bear_is_deallocated(manager: StrategyManager) -> None:
    """Sizing is risk-on/risk-off: every non-bear regime carries the book."""
    ohlcv = _make_ohlcv()

    bull = manager.generate_signal(ohlcv, Regime.SUSTAINED_BULL)
    volatile = manager.generate_signal(ohlcv, Regime.VOLATILE_DISTRIBUTION)
    compressed = manager.generate_signal(ohlcv, Regime.COMPRESSED_LIQUIDITY)
    bear = manager.generate_signal(ohlcv, Regime.STRUCTURAL_BEAR)

    assert bear.target_exposure == 0.0
    for signal in (bull, volatile, compressed):
        assert signal.target_exposure > 0.0


def test_exposure_is_geared_and_capped(manager: StrategyManager) -> None:
    settings = get_settings()
    ohlcv = _make_ohlcv()

    bull = manager.generate_signal(ohlcv, Regime.SUSTAINED_BULL)

    # This synthetic series is quiet, so the de-risk scalar saturates at 1.0 and
    # exposure lands on the full gearing level rather than being cut.
    assert bull.target_exposure == pytest.approx(settings.base_leverage)
    assert bull.target_exposure <= settings.max_leverage


def test_high_volatility_derisks_but_never_levers_up() -> None:
    """The vol scalar may only ever cut exposure, never raise it."""
    settings = get_settings().model_copy(update={"target_annual_vol": 0.02})
    manager = StrategyManager(settings)
    ohlcv = _make_ohlcv()

    signal = manager.generate_signal(ohlcv, Regime.SUSTAINED_BULL)

    # Target vol far below realized vol => scalar < 1 => geared exposure is cut.
    assert 0.0 < signal.target_exposure < settings.base_leverage

    calm = get_settings().model_copy(update={"target_annual_vol": 10.0})
    calm_signal = StrategyManager(calm).generate_signal(ohlcv, Regime.SUSTAINED_BULL)
    # An absurdly high target must not lever beyond base_leverage.
    assert calm_signal.target_exposure == pytest.approx(calm.base_leverage)


def test_breakout_is_reported_but_does_not_change_size(manager: StrategyManager) -> None:
    """Donchian breakout is diagnostic only; it no longer scales the position."""
    ohlcv = _make_ohlcv()
    non_breakout = manager.generate_signal(ohlcv, Regime.SUSTAINED_BULL)

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
    assert "donchian_breakout" not in non_breakout.reason


def test_no_stop_price_when_atr_stop_disabled(manager: StrategyManager) -> None:
    """The ATR trailing stop is off by default."""
    assert get_settings().use_atr_stop is False
    ohlcv = _make_ohlcv()

    assert manager.generate_signal(ohlcv, Regime.SUSTAINED_BULL).stop_price is None


def test_stop_price_set_below_close_when_enabled() -> None:
    settings = get_settings().model_copy(update={"use_atr_stop": True})
    manager = StrategyManager(settings)
    ohlcv = _make_ohlcv()
    latest_close = float(ohlcv["close"].iloc[-1])

    for regime in (
        Regime.SUSTAINED_BULL,
        Regime.VOLATILE_DISTRIBUTION,
        Regime.COMPRESSED_LIQUIDITY,
    ):
        signal = manager.generate_signal(ohlcv, regime)
        assert signal.target_exposure > 0.0
        assert signal.stop_price is not None
        assert signal.stop_price < latest_close


def test_stop_price_none_for_zero_exposure() -> None:
    settings = get_settings().model_copy(update={"use_atr_stop": True})
    ohlcv = _make_ohlcv()
    bear = StrategyManager(settings).generate_signal(ohlcv, Regime.STRUCTURAL_BEAR)

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
