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
    BacktestResult,
    chain_walk_forward_equity,
    rebase,
    run_backtest,
    run_backtest_from,
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


def test_chain_walk_forward_equity_compounds_across_windows():
    """Windows must be linked by return, not concatenated by level."""
    index_a = pd.date_range("2020-01-01", periods=3, freq="B")
    index_b = pd.date_range("2020-01-06", periods=3, freq="B")
    # Two independent windows, each restarting at 100 and each gaining 10%.
    results = [
        BacktestResult(
            equity_curve=pd.Series([100.0, 105.0, 110.0], index=index_a),
            regimes=pd.Series([None] * 3, index=index_a),
            ledger=pd.DataFrame(),
            exposure=pd.Series([0.0] * 3, index=index_a),
        ),
        BacktestResult(
            equity_curve=pd.Series([100.0, 105.0, 110.0], index=index_b),
            regimes=pd.Series([None] * 3, index=index_b),
            ledger=pd.DataFrame(),
            exposure=pd.Series([0.0] * 3, index=index_b),
        ),
    ]

    chained = chain_walk_forward_equity(results, starting_balance=100.0)

    assert len(chained) == 6
    assert chained.iloc[0] == pytest.approx(100.0)
    # 1.10 * 1.10 -- compounded, not a jump back down to 100 at the seam.
    assert chained.iloc[-1] == pytest.approx(121.0)
    assert chained.is_monotonic_increasing


def test_chain_walk_forward_equity_handles_empty_input():
    assert chain_walk_forward_equity([], starting_balance=100.0).empty


def test_walk_forward_chaining_avoids_seam_discontinuities(settings):
    """The chained curve must not contain artificial jumps at window seams."""
    ohlcv = _synthetic_ohlcv(400, seed=11)
    results = run_walk_forward_backtest(ohlcv, settings)
    assert len(results) > 2

    chained = chain_walk_forward_equity(results, settings.starting_balance)
    naive = pd.concat([r.equity_curve for r in results]).sort_index()
    naive = naive[~naive.index.duplicated(keep="first")]

    # The naive concatenation resets to the starting balance at every seam,
    # producing far larger single-bar moves than the strategy ever takes.
    assert chained.pct_change().abs().max() < naive.pct_change().abs().max()


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


def test_execution_is_lagged_one_bar(settings):
    """The first bar must trade flat: its signal is not yet actionable."""
    ohlcv = _synthetic_ohlcv(120, seed=5)
    result = run_backtest(ohlcv, settings)

    assert result.exposure.iloc[0] == 0.0
    # Flat, so equity moves only by the one bar of yield credited on idle cash.
    one_bar_of_yield = settings.starting_balance * (
        settings.cash_yield_rate / settings.trading_days_per_year
    )
    assert result.equity_curve.iloc[0] == pytest.approx(
        settings.starting_balance + one_bar_of_yield
    )


def test_run_backtest_from_rebases_and_restricts_window(settings):
    ohlcv = _synthetic_ohlcv(300, seed=6)
    evaluation_start = ohlcv.index[150]

    result = run_backtest_from(ohlcv, evaluation_start, settings)

    assert result.equity_curve.index[0] == evaluation_start
    assert len(result.equity_curve) == 150
    # Rebased so it is directly comparable to a benchmark over the same window.
    assert result.equity_curve.iloc[0] == pytest.approx(settings.starting_balance)
    assert result.regimes.index.equals(result.equity_curve.index)
    assert result.exposure.index.equals(result.equity_curve.index)


def test_run_backtest_from_uses_warmup_to_avoid_starting_flat(settings):
    """Warmup bars should leave the strategy already positioned on bar one."""
    ohlcv = _synthetic_ohlcv(300, seed=7)

    cold = run_backtest(ohlcv, settings)
    warm = run_backtest_from(ohlcv, ohlcv.index[150], settings)

    assert cold.exposure.iloc[0] == 0.0
    assert warm.exposure.iloc[0] > 0.0


def test_run_backtest_from_raises_when_window_empty(settings):
    ohlcv = _synthetic_ohlcv(50, seed=8)
    with pytest.raises(ValueError, match="evaluation_start"):
        run_backtest_from(ohlcv, pd.Timestamp("2099-01-01"), settings)


def test_rebase_scales_curve_to_starting_balance():
    equity = pd.Series([250.0, 275.0, 300.0])
    rebased = rebase(equity, 100.0)
    assert rebased.iloc[0] == pytest.approx(100.0)
    # Shape is preserved; only the level changes.
    assert (rebased / rebased.iloc[0]).tolist() == pytest.approx((equity / equity.iloc[0]).tolist())


def test_no_trade_band_suppresses_churn_without_changing_stance():
    """A wide band should cut trade count sharply, not the exposure level."""
    ohlcv = _synthetic_ohlcv(400, seed=9)
    base = dict(ema_fast=5, ema_slow=10, donchian_window=10, atr_window=5)

    tight = run_backtest(ohlcv, Settings(rebalance_band=0.0, **base))
    wide = run_backtest(ohlcv, Settings(rebalance_band=0.25, **base))

    tight_trades = int((tight.ledger["side"] != "hold").sum())
    wide_trades = int((wide.ledger["side"] != "hold").sum())

    assert wide_trades < tight_trades / 2
    assert wide.exposure.mean() == pytest.approx(tight.exposure.mean(), abs=0.15)


def test_exposure_series_reports_actual_not_target():
    """Exposure must reflect what was held, including deferred rebalances."""
    ohlcv = _synthetic_ohlcv(200, seed=10)
    result = run_backtest(
        ohlcv, Settings(rebalance_band=0.5, ema_fast=5, ema_slow=10, atr_window=5)
    )
    # With a very wide band the held exposure drifts away from any single
    # target level, so the series must not be piecewise-constant.
    held = result.exposure[result.exposure > 0]
    assert held.nunique() > 5
