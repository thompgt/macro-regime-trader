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
from macro_regime_trader.backtest.engine import BacktestResult, run_backtest
from macro_regime_trader.config import get_settings
from macro_regime_trader.data.yfinance_provider import YFinanceProvider

REGIME_ORDER = [
    "structural_bear",
    "compressed_liquidity",
    "volatile_distribution",
    "sustained_bull",
]

REGIME_LABELS = {
    "sustained_bull": "Sustained Bull",
    "volatile_distribution": "Volatile Distribution",
    "structural_bear": "Structural Bear",
    "compressed_liquidity": "Compressed Liquidity",
}

REGIME_COLORS = {
    "sustained_bull": "#1a9e46",
    "volatile_distribution": "#e0a30f",
    "structural_bear": "#c0392b",
    "compressed_liquidity": "#3477a6",
}


def fetch_and_run(
    ticker: str,
    start: str,
    end: str | None,
    interval: str,
) -> tuple[pd.DataFrame, BacktestResult]:
    """Fetch OHLCV data and run the full backtest. Raises on bad input/no data."""
    settings = get_settings()
    provider = YFinanceProvider(cache_dir=settings.data_cache_dir)
    ohlcv = provider.get_ohlcv(ticker, start=start, end=end, interval=interval)
    result = run_backtest(ohlcv, settings)
    return ohlcv, result


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


def regime_series_to_numeric(regimes: pd.Series) -> pd.Series:
    """Map Regime string values (or None during warmup) to an integer code for plotting."""
    mapping = {name: idx for idx, name in enumerate(REGIME_ORDER)}

    def _to_code(value: object) -> float | None:
        if value is None:
            return None
        key = value.value if hasattr(value, "value") else str(value)
        return mapping.get(key)

    return regimes.map(_to_code).rename("regime_code")


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

    st.subheader("Strategy Performance")
    render_metrics("Strategy", strategy_metrics)

    with st.expander("Show buy & hold / DMA-crossover benchmark metrics"):
        render_metrics("Buy & Hold", bh_metrics)
        render_metrics("DMA Crossover", dma_metrics)

    st.subheader("Equity Curve: Strategy vs. Benchmarks")
    st.line_chart(equity_comparison)

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

    regime_codes = regime_series_to_numeric(result.regimes)
    if regime_codes.notna().any():
        st.caption(
            "Regime timeline (0=Structural Bear, 1=Compressed Liquidity, "
            "2=Volatile Distribution, 3=Sustained Bull)"
        )
        st.area_chart(regime_codes)

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
