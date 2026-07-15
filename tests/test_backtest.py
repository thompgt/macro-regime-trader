import numpy as np
import pandas as pd
import pytest

from macro_regime_trader.backtest.analytics import (
    compute_metrics,
    max_drawdown,
    sharpe_ratio,
    total_return,
    win_rate,
)
from macro_regime_trader.backtest.benchmarks import buy_and_hold_equity, dma_crossover_equity
from macro_regime_trader.backtest.engine import (
    run_backtest,
    run_walk_forward_backtest,
    walk_forward_windows,
)
from macro_regime_trader.config import Settings


def _synthetic_ohlcv(n: int = 300, seed: int = 0, drift: float = 0.001) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    returns = rng.normal(loc=drift, scale=0.01, size=n)
    close = 100 * np.cumprod(1 + returns)
    high = close * 1.005
    low = close * 0.995
    open_ = close * (1 + rng.normal(0, 0.001, size=n))
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(
        train_window=60, test_window=30, ema_fast=5, ema_slow=10, donchian_window=10, atr_window=5
    )


def test_run_backtest_produces_full_length_equity_curve(settings):
    ohlcv = _synthetic_ohlcv(120, seed=1)
    result = run_backtest(ohlcv, settings)
    assert len(result.equity_curve) == len(ohlcv)
    assert result.equity_curve.index.equals(ohlcv.index)
    assert (result.equity_curve > 0).all()


def test_walk_forward_windows_are_contiguous_and_non_overlapping():
    windows = walk_forward_windows(n_bars=200, train_window=60, test_window=30)
    assert len(windows) > 0
    for i in range(1, len(windows)):
        prev_test = windows[i - 1][1]
        curr_train = windows[i][0]
        assert curr_train.start == prev_test.start + (curr_train.start - prev_test.start)
    for train_slice, test_slice in windows:
        assert train_slice.stop == test_slice.start


def test_run_walk_forward_backtest_covers_oos_segments_only(settings):
    ohlcv = _synthetic_ohlcv(200, seed=2)
    results = run_walk_forward_backtest(ohlcv, settings)
    assert len(results) > 0
    total_oos_bars = sum(len(r.equity_curve) for r in results)
    assert total_oos_bars <= len(ohlcv)
    assert all(len(r.equity_curve) == settings.test_window for r in results)


def test_analytics_functions_on_known_equity_curve():
    equity = pd.Series([100.0, 110.0, 121.0, 108.9, 130.0])
    assert total_return(equity) == pytest.approx(0.30)
    assert max_drawdown(equity) < 0
    assert 0.0 <= win_rate(equity) <= 1.0
    assert isinstance(sharpe_ratio(equity), float)

    metrics = compute_metrics(equity)
    assert set(metrics) == {"total_return", "sharpe_ratio", "max_drawdown", "win_rate"}


def test_analytics_handle_degenerate_curves():
    flat = pd.Series([100.0, 100.0, 100.0])
    assert total_return(flat) == pytest.approx(0.0)
    assert sharpe_ratio(flat) == 0.0
    assert max_drawdown(flat) == pytest.approx(0.0)

    empty = pd.Series([], dtype=float)
    assert total_return(empty) == 0.0
    assert max_drawdown(empty) == 0.0
    assert win_rate(empty) == 0.0


def test_buy_and_hold_equity_matches_price_return(settings):
    ohlcv = _synthetic_ohlcv(50, seed=3)
    equity = buy_and_hold_equity(ohlcv, settings)
    price_return = ohlcv["close"].iloc[-1] / ohlcv["close"].iloc[0] - 1.0
    equity_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    assert equity_return == pytest.approx(price_return, rel=1e-3)


def test_dma_crossover_equity_is_full_length_and_positive(settings):
    ohlcv = _synthetic_ohlcv(120, seed=4)
    equity = dma_crossover_equity(ohlcv, settings)
    assert len(equity) == len(ohlcv)
    assert (equity > 0).all()
