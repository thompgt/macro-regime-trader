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


def _flat_declining_volume() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100.0 + rng.normal(0, 0.01, size=N_BARS).cumsum() * 0.0
    close = 100.0 + np.zeros(N_BARS)
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


def test_flat_declining_volume_is_compressed_liquidity(engine: MacroRegimeEngine) -> None:
    ohlcv = _flat_declining_volume()
    assert engine.classify_latest(ohlcv) == Regime.COMPRESSED_LIQUIDITY


def test_classify_returns_series_aligned_to_index_with_warmup_nans(
    engine: MacroRegimeEngine,
) -> None:
    ohlcv = _uptrend_rising_volume()
    result = engine.classify(ohlcv)

    assert isinstance(result, pd.Series)
    assert len(result) == len(ohlcv)
    assert result.index.equals(ohlcv.index)

    warmup = max(SETTINGS.ema_slow, SETTINGS.volume_zscore_window)
    early = result.iloc[: warmup - 1]
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
