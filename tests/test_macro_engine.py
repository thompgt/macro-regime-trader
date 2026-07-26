from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macro_regime_trader.config import Settings
from macro_regime_trader.core.macro_engine import MacroRegimeEngine
from macro_regime_trader.types import Regime

N_BARS = 150
SETTINGS = Settings(ema_fast=20, ema_slow=50, volume_zscore_window=20)


def _make_ohlcv(close: np.ndarray, volume: np.ndarray) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=len(close), freq="D")
    high = close * 1.001
    low = close * 0.999
    open_ = close
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def _uptrend_rising_volume() -> pd.DataFrame:
    close = 100.0 + np.arange(N_BARS) * 1.5
    volume = 1_000 + np.arange(N_BARS) * 50
    return _make_ohlcv(close, volume)


def _downtrend_rising_volume() -> pd.DataFrame:
    close = 300.0 - np.arange(N_BARS) * 1.5
    volume = 1_000 + np.arange(N_BARS) * 50
    return _make_ohlcv(close, volume)


def _quiet_drift_declining_volume() -> pd.DataFrame:
    """Compressing volatility on a near-flat drift, with fading participation.

    Volatility is *ranked against its own history*, so a perfectly constant
    series has no distribution to rank within. This fixture instead decays the
    noise amplitude over time, which is what "compressed liquidity" actually
    describes: the late bars are quiet relative to the earlier ones.
    """
    rng = np.random.default_rng(42)
    drift = 100.0 * (1.0 + 0.00005 * np.arange(N_BARS))
    decaying_noise = rng.normal(0, 1.0, size=N_BARS) * np.linspace(0.5, 0.01, N_BARS)
    close = drift + decaying_noise
    volume = np.maximum(5_000 - np.arange(N_BARS) * 30, 200)
    return _make_ohlcv(close, volume)


@pytest.fixture
def engine() -> MacroRegimeEngine:
    return MacroRegimeEngine(settings=SETTINGS)


def test_uptrend_rising_volume_is_sustained_bull(engine: MacroRegimeEngine) -> None:
    ohlcv = _uptrend_rising_volume()
    assert engine.classify_latest(ohlcv) == Regime.SUSTAINED_BULL


def test_downtrend_rising_volume_is_structural_bear(engine: MacroRegimeEngine) -> None:
    ohlcv = _downtrend_rising_volume()
    assert engine.classify_latest(ohlcv) == Regime.STRUCTURAL_BEAR


def test_quiet_drift_declining_volume_is_compressed_liquidity(
    engine: MacroRegimeEngine,
) -> None:
    ohlcv = _quiet_drift_declining_volume()
    assert engine.classify_latest(ohlcv) == Regime.COMPRESSED_LIQUIDITY


def test_sustained_downtrend_overrides_fast_reentry(engine: MacroRegimeEngine) -> None:
    """A bounce that does not lift a short MA must not clear the bear call."""
    ohlcv = _downtrend_rising_volume()
    regimes = engine.classify(ohlcv).dropna()
    # A monotone decline offers no rising short MA to reclaim, so every
    # classified bar stays risk-off.
    assert (regimes == Regime.STRUCTURAL_BEAR.value).all()


def test_fast_reentry_clears_bear_while_below_long_trend(engine: MacroRegimeEngine) -> None:
    """Reclaiming a rising short MA lifts risk-off even below the long trend."""
    # Long decline, then a sharp V-shaped rally that recovers only part of the
    # fall -- price is still far below its 200-bar mean at the final bar.
    decline = 300.0 - np.arange(200) * 1.0
    rally = decline[-1] + np.arange(1, 13) * 1.5
    close = np.concatenate([decline, rally])
    volume = np.full(len(close), 1_000.0)
    ohlcv = _make_ohlcv(close, volume)

    features = engine.features(ohlcv)
    assert features["trend_gap"].iloc[-1] < 0.0, "expected price still below long trend"
    assert bool(features["reclaim"].iloc[-1]) is True

    assert engine.classify_latest(ohlcv) != Regime.STRUCTURAL_BEAR


def test_classify_returns_series_aligned_to_index_with_warmup_nans(
    engine: MacroRegimeEngine,
) -> None:
    ohlcv = _uptrend_rising_volume()
    result = engine.classify(ohlcv)

    assert isinstance(result, pd.Series)
    assert len(result) == len(ohlcv)
    assert result.index.equals(ohlcv.index)

    # Readiness is gated by the slowest input: the long-term trend MA (half its
    # window), the slow EMA, and the volatility rank's own minimum history.
    warmup = max(
        SETTINGS.ema_slow,
        SETTINGS.trend_window // 2,
        SETTINGS.realized_vol_window + 60,
    )
    early = result.iloc[: SETTINGS.ema_slow - 1]
    assert early.isna().all() or early.apply(lambda v: v is None).all()

    later = result.iloc[warmup + 5 :]
    assert later.notna().all()
    valid_values = {r.value for r in Regime}
    assert later.isin(valid_values).all()


def test_classify_raises_on_missing_columns(engine: MacroRegimeEngine) -> None:
    ohlcv = _uptrend_rising_volume().drop(columns=["volume"])
    with pytest.raises(ValueError):
        engine.classify(ohlcv)


def test_classify_latest_raises_when_insufficient_history(engine: MacroRegimeEngine) -> None:
    short_ohlcv = _uptrend_rising_volume().iloc[:10]
    with pytest.raises(ValueError):
        engine.classify_latest(short_ohlcv)


def test_default_constructor_uses_get_settings() -> None:
    engine = MacroRegimeEngine()
    assert engine.settings.ema_fast > 0
    ohlcv = _uptrend_rising_volume()
    regime = engine.classify_latest(ohlcv)
    assert isinstance(regime, Regime)
