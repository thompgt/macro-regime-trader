"""Streamlit dashboard for the macro-regime detection + adaptive allocation simulator.

Run with:

    streamlit run src/macro_regime_trader/dashboard/app.py

This app is a research/backtesting visualization tool. Nothing here places
live orders or constitutes trading advice -- it simply replays historical
OHLCV data through the engine's regime classifier, strategy manager, risk
manager, and mock broker, then compares the resulting equity curve against
simple benchmarks.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from macro_regime_trader.backtest.analytics import compute_metrics
from macro_regime_trader.backtest.benchmarks import buy_and_hold_equity, dma_crossover_equity
from macro_regime_trader.backtest.engine import BacktestResult, run_backtest_from
from macro_regime_trader.config import get_settings
from macro_regime_trader.data.yfinance_provider import YFinanceProvider

REGIME_LABELS = {
    "sustained_bull": "Sustained Bull",
    "volatile_distribution": "Volatile Distribution",
    "structural_bear": "Structural Bear",
    "compressed_liquidity": "Compressed Liquidity",
}

# Categorical palette validated for colorblind separation (worst all-pairs CVD
# deltaE 13.0). `sustained_bull` is deliberately blue rather than green: a
# green/red bull/bear pair is the most important distinction on this page and it
# fails protanopia separation, so hue must not carry that contrast alone.
REGIME_COLORS = {
    "sustained_bull": "#2a78d6",
    "volatile_distribution": "#eda100",
    "compressed_liquidity": "#4a3aa7",
    "structural_bear": "#e34948",
}

SERIES_COLORS = {
    "strategy": "#2a78d6",
    "buy_and_hold": "#eb6834",
    "dma_crossover": "#1baf7a",
}


def fetch_and_run(
    ticker: str,
    start: str,
    end: str | None,
    interval: str,
) -> tuple[pd.DataFrame, BacktestResult]:
    """Fetch OHLCV data and run the backtest over an indicator-warmed window.

    Extra history is fetched *before* ``start`` and used only to warm up
    indicators, so the reported window compares a fully-warmed strategy against
    a fully-invested benchmark. Without it the strategy is unclassified and flat
    for its first ~100 bars and looks worse than it is.
    """
    settings = get_settings()
    provider = YFinanceProvider(cache_dir=settings.data_cache_dir)

    evaluation_start = pd.Timestamp(start)
    fetch_start = evaluation_start - pd.Timedelta(days=settings.warmup_calendar_days)
    ohlcv = provider.get_ohlcv(
        ticker, start=fetch_start.strftime("%Y-%m-%d"), end=end, interval=interval
    )
    if ohlcv.empty:
        return ohlcv, None  # type: ignore[return-value]

    result = run_backtest_from(ohlcv, evaluation_start, settings)
    return ohlcv.loc[result.equity_curve.index], result


def build_equity_comparison(result: BacktestResult, ohlcv: pd.DataFrame, settings) -> pd.DataFrame:
    """Combine strategy equity with benchmark curves into one DataFrame for charting."""
    bh = buy_and_hold_equity(ohlcv, settings)
    dma = dma_crossover_equity(ohlcv, settings)
    combined = pd.DataFrame(
        {
            "strategy": result.equity_curve,
            "buy_and_hold": bh,
            "dma_crossover": dma,
        }
    )
    return combined


def latest_regime(regimes: pd.Series) -> str | None:
    """Return the most recent non-null regime label, or None if unavailable."""
    valid = regimes.dropna()
    if valid.empty:
        return None
    last = valid.iloc[-1]
    return last.value if hasattr(last, "value") else str(last)


def render_metrics(label: str, metrics: dict[str, float]) -> None:
    cols = st.columns(4)
    cols[0].metric(f"{label} Total Return", f"{metrics['total_return'] * 100:.2f}%")
    cols[1].metric(f"{label} Sharpe", f"{metrics['sharpe_ratio']:.2f}")
    cols[2].metric(f"{label} Max Drawdown", f"{metrics['max_drawdown'] * 100:.2f}%")
    cols[3].metric(f"{label} Win Rate", f"{metrics['win_rate'] * 100:.1f}%")


def render_headline_metrics(strategy: dict[str, float], benchmark: dict[str, float]) -> None:
    """Strategy metrics with the buy-and-hold gap shown as the delta.

    The whole question this page answers is "did it beat buy and hold?", so the
    comparison belongs in the primary tiles rather than behind an expander.
    Drawdown deltas are inverted (`delta_color`) because a shallower drawdown is
    an improvement even though the number is larger.
    """
    cols = st.columns(4)
    cols[0].metric(
        "Total Return",
        f"{strategy['total_return'] * 100:.2f}%",
        f"{(strategy['total_return'] - benchmark['total_return']) * 100:+.2f} pts vs B&H",
    )
    cols[1].metric(
        "Sharpe",
        f"{strategy['sharpe_ratio']:.2f}",
        f"{strategy['sharpe_ratio'] - benchmark['sharpe_ratio']:+.2f} vs B&H",
    )
    cols[2].metric(
        "Max Drawdown",
        f"{strategy['max_drawdown'] * 100:.2f}%",
        f"{(strategy['max_drawdown'] - benchmark['max_drawdown']) * 100:+.2f} pts vs B&H",
    )
    cols[3].metric(
        "Win Rate",
        f"{strategy['win_rate'] * 100:.1f}%",
        f"{(strategy['win_rate'] - benchmark['win_rate']) * 100:+.1f} pts vs B&H",
    )


def drawdown_frame(equity_comparison: pd.DataFrame) -> pd.DataFrame:
    """Drawdown-from-running-peak for each equity curve."""
    return equity_comparison.apply(lambda col: col / col.cummax() - 1.0)


def metrics_table(rows: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Side-by-side metrics for every model, as a readable table."""
    frame = pd.DataFrame(rows).T
    frame.index.name = "model"
    return frame.rename(
        columns={
            "total_return": "Total return",
            "sharpe_ratio": "Sharpe",
            "max_drawdown": "Max drawdown",
            "win_rate": "Win rate",
        }
    ).round(4)


def main() -> None:
    st.set_page_config(page_title="Macro Regime Trader", layout="wide")

    st.title("Macro Regime Trader")
    st.caption(
        "A macro-regime detection and adaptive allocation *simulation*. "
        "Results are derived from historical backtests only and do not "
        "constitute live trading, investment advice, or a recommendation "
        "to buy or sell any security."
    )

    with st.sidebar:
        st.header("Backtest Settings")
        ticker = st.text_input("Ticker", value="SPY")
        start_date = st.date_input("Start date", value=dt.date.today() - dt.timedelta(days=5 * 365))
        use_end_date = st.checkbox("Specify end date", value=False)
        end_date = None
        if use_end_date:
            end_date = st.date_input("End date", value=dt.date.today())
        interval = st.selectbox("Interval", options=["1d", "1wk", "1mo"], index=0)
        run_clicked = st.button("Run Backtest", type="primary")

    if not run_clicked:
        st.info("Configure a ticker and date range in the sidebar, then click **Run Backtest**.")
        return

    start_str = start_date.isoformat()
    end_str = end_date.isoformat() if end_date is not None else None

    try:
        with st.spinner(f"Fetching data for {ticker} and running backtest..."):
            ohlcv, result = fetch_and_run(ticker, start_str, end_str, interval)
            settings = get_settings()
            equity_comparison = build_equity_comparison(result, ohlcv, settings)
    except ValueError as exc:
        st.error(f"Could not run backtest: {exc}")
        return
    except Exception as exc:  # noqa: BLE001 - surface any provider/engine failure to the user
        st.error(f"Unexpected error while running the backtest: {exc}")
        return

    if ohlcv.empty:
        st.error(f"No data returned for ticker {ticker!r} in the given range.")
        return

    strategy_metrics = compute_metrics(result.equity_curve)
    bh_metrics = compute_metrics(equity_comparison["buy_and_hold"].dropna())
    dma_metrics = compute_metrics(equity_comparison["dma_crossover"].dropna())

    beats = (
        strategy_metrics["total_return"] > bh_metrics["total_return"],
        strategy_metrics["sharpe_ratio"] > bh_metrics["sharpe_ratio"],
        strategy_metrics["max_drawdown"] > bh_metrics["max_drawdown"],
    )

    st.subheader("Strategy vs. Buy & Hold")
    st.caption(
        f"Evaluated {len(result.equity_curve):,} bars, "
        f"{result.equity_curve.index[0].date()} to {result.equity_curve.index[-1].date()} - "
        f"mean exposure {result.exposure.mean():.2f}x, "
        f"{int((result.ledger['side'] != 'hold').sum())} trades. "
        "Indicators were warmed on history before the start date, so both models "
        "are live from the first bar shown."
    )
    render_headline_metrics(strategy_metrics, bh_metrics)

    won = sum(beats)
    summary = f"Beats buy & hold on {won} of 3 measures (return, Sharpe, drawdown)."
    if won == 3:
        st.success(summary)
    elif won == 0:
        st.error(summary)
    else:
        st.warning(summary)

    st.subheader("Equity Curve: Strategy vs. Benchmarks")
    st.line_chart(equity_comparison, color=[SERIES_COLORS[c] for c in equity_comparison.columns])

    st.subheader("Drawdown")
    st.caption("Decline from each model's own running peak - closer to zero is better.")
    drawdowns = drawdown_frame(equity_comparison)
    st.line_chart(drawdowns, color=[SERIES_COLORS[c] for c in drawdowns.columns])

    st.subheader("Exposure")
    st.caption(
        "Fraction of equity held in the asset. Above 1.0x is geared; 0.0x is "
        "fully in cash during a risk-off regime."
    )
    st.area_chart(result.exposure, color=SERIES_COLORS["strategy"])

    with st.expander("Full metrics table (all models)"):
        st.dataframe(
            metrics_table(
                {
                    "strategy": strategy_metrics,
                    "buy_and_hold": bh_metrics,
                    "dma_crossover": dma_metrics,
                }
            ),
            use_container_width=True,
        )

    st.subheader("Regime")
    current_regime = latest_regime(result.regimes)
    if current_regime is not None:
        color = REGIME_COLORS.get(current_regime, "#888888")
        label = REGIME_LABELS.get(current_regime, current_regime)
        st.markdown(
            f"**Current detected regime:** "
            f"<span style='background-color:{color};color:white;padding:4px 10px;"
            f"border-radius:6px;font-weight:600'>{label}</span>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("No regime classified yet (insufficient warmup history).")

    regime_counts = result.regimes.dropna().value_counts()
    if not regime_counts.empty:
        share = (regime_counts / regime_counts.sum() * 100).round(1)
        st.caption("Share of bars spent in each regime")
        st.dataframe(
            pd.DataFrame(
                {
                    "regime": [REGIME_LABELS.get(str(k), str(k)) for k in share.index],
                    "bars": regime_counts.to_numpy(),
                    "share_%": share.to_numpy(),
                }
            ).set_index("regime"),
            use_container_width=True,
        )

    st.subheader("Trade Log")
    ledger = result.ledger
    if ledger.empty:
        st.write("No trades were executed during this backtest.")
    else:
        ledger_sorted = ledger.sort_values("timestamp", ascending=False)
        if len(ledger_sorted) > 25:
            with st.expander(f"Show all {len(ledger_sorted)} trades (most recent first)"):
                st.dataframe(ledger_sorted, use_container_width=True)
        else:
            st.dataframe(ledger_sorted, use_container_width=True)


if __name__ == "__main__":
    main()
