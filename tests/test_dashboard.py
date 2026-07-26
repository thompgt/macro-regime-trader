"""Tests for the dashboard's pure helper functions.

Only the data-shaping helpers are covered -- the Streamlit ``main()`` render
path needs a script runner and a network fetch, so it stays out of the suite.
These are the pieces that can silently produce a misleading chart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macro_regime_trader.backtest.engine import BacktestResult
from macro_regime_trader.config import Settings
from macro_regime_trader.dashboard import app
from macro_regime_trader.types import Regime


def _ohlcv(n: int = 300, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2021-01-01", periods=n, freq="B")
    close = 100 * np.cumprod(1 + rng.normal(0.0006, 0.01, n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.004,
            "low": close * 0.996,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=index,
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(ema_fast=5, ema_slow=10, atr_window=5, donchian_window=10)


def test_every_regime_has_a_label_and_color():
    """A regime the engine can emit must never render as an unlabeled swatch."""
    for regime in Regime:
        assert regime.value in app.REGIME_LABELS
        assert regime.value in app.REGIME_COLORS


def test_series_colors_cover_every_comparison_column(settings):
    """Chart color lists are built by indexing SERIES_COLORS per column."""
    ohlcv = _ohlcv()
    result = BacktestResult(
        equity_curve=pd.Series(100_000.0, index=ohlcv.index, name="strategy"),
        regimes=pd.Series([None] * len(ohlcv), index=ohlcv.index),
        ledger=pd.DataFrame(),
        exposure=pd.Series(0.0, index=ohlcv.index),
    )
    comparison = app.build_equity_comparison(result, ohlcv, settings)
    for column in comparison.columns:
        assert column in app.SERIES_COLORS


def test_drawdown_frame_is_non_positive_and_starts_at_zero():
    equity = pd.DataFrame(
        {
            "strategy": [100.0, 120.0, 90.0, 150.0],
            "buy_and_hold": [100.0, 80.0, 95.0, 99.0],
        }
    )
    drawdowns = app.drawdown_frame(equity)

    assert (drawdowns <= 1e-12).all().all()
    assert drawdowns.iloc[0].eq(0.0).all()
    # Peak 120 -> trough 90 is -25%.
    assert drawdowns["strategy"].min() == pytest.approx(-0.25)
    assert drawdowns["buy_and_hold"].min() == pytest.approx(-0.20)


def test_metrics_table_renames_and_keeps_every_model():
    rows = {
        "strategy": {
            "total_return": 1.0,
            "sharpe_ratio": 0.9,
            "max_drawdown": -0.2,
            "win_rate": 0.55,
        },
        "buy_and_hold": {
            "total_return": 0.8,
            "sharpe_ratio": 0.7,
            "max_drawdown": -0.3,
            "win_rate": 0.52,
        },
    }
    table = app.metrics_table(rows)

    assert list(table.index) == ["strategy", "buy_and_hold"]
    assert list(table.columns) == ["Total return", "Sharpe", "Max drawdown", "Win rate"]


def test_latest_regime_ignores_trailing_warmup_nulls():
    index = pd.date_range("2024-01-01", periods=4, freq="B")
    regimes = pd.Series(
        [None, Regime.SUSTAINED_BULL.value, Regime.STRUCTURAL_BEAR.value, None],
        index=index,
        dtype=object,
    )
    assert app.latest_regime(regimes) == Regime.STRUCTURAL_BEAR.value


def test_latest_regime_returns_none_when_never_classified():
    regimes = pd.Series([None, None], index=pd.date_range("2024-01-01", periods=2), dtype=object)
    assert app.latest_regime(regimes) is None
