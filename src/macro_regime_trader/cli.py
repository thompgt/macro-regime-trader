"""Command-line entry points: `mrt backtest`, `mrt dashboard`."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click
import pandas as pd

from macro_regime_trader.backtest.analytics import compute_metrics
from macro_regime_trader.backtest.benchmarks import buy_and_hold_equity, dma_crossover_equity
from macro_regime_trader.backtest.engine import run_backtest, run_walk_forward_backtest
from macro_regime_trader.config import get_settings
from macro_regime_trader.data.yfinance_provider import YFinanceProvider
from macro_regime_trader.logging_config import configure_logging, get_logger

logger = get_logger("cli")


def _format_metrics_table(rows: dict[str, dict[str, float]]) -> str:
    columns = ["total_return", "sharpe_ratio", "max_drawdown", "win_rate"]
    header = f"{'model':<16}" + "".join(f"{c:>16}" for c in columns)
    lines = [header, "-" * len(header)]
    for name, metrics in rows.items():
        line = f"{name:<16}" + "".join(f"{metrics[c]:>16.4f}" for c in columns)
        lines.append(line)
    return "\n".join(lines)


@click.group()
def main() -> None:
    """Macro Regime Trader command-line interface."""
    settings = get_settings()
    configure_logging(settings.log_level)


@main.command()
@click.option("--ticker", default="SPY", show_default=True, help="Ticker symbol to backtest.")
@click.option("--start", default="2015-01-01", show_default=True, help="Start date (YYYY-MM-DD).")
@click.option("--end", default=None, help="End date (YYYY-MM-DD). Defaults to latest available.")
@click.option("--interval", default="1d", show_default=True, help="Bar interval.")
@click.option(
    "--walk-forward/--full-sample",
    default=False,
    help="Run rolling walk-forward evaluation instead of one full-sample backtest.",
)
def backtest(ticker: str, start: str, end: str | None, interval: str, walk_forward: bool) -> None:
    """Fetch real market data and run the strategy against buy-and-hold and DMA benchmarks."""
    settings = get_settings()
    provider = YFinanceProvider(cache_dir=settings.data_cache_dir)

    click.echo(f"Fetching {ticker} {interval} bars from {start} to {end or 'latest'}...")
    ohlcv = provider.get_ohlcv(ticker, start=start, end=end, interval=interval)
    click.echo(f"Loaded {len(ohlcv)} bars.\n")

    if walk_forward:
        window_results = run_walk_forward_backtest(ohlcv, settings)
        if not window_results:
            click.echo(
                "Not enough bars for even one walk-forward window "
                f"(need >= {settings.train_window + settings.test_window})."
            )
            sys.exit(1)
        strategy_equity = pd.concat([r.equity_curve for r in window_results]).sort_index()
        strategy_equity = strategy_equity[~strategy_equity.index.duplicated(keep="first")]
    else:
        strategy_equity = run_backtest(ohlcv, settings).equity_curve

    benchmark_index = strategy_equity.index
    bench_ohlcv = ohlcv.loc[benchmark_index]
    rows = {
        "strategy": compute_metrics(strategy_equity),
        "buy_and_hold": compute_metrics(buy_and_hold_equity(bench_ohlcv, settings)),
        "dma_crossover": compute_metrics(dma_crossover_equity(bench_ohlcv, settings)),
    }
    click.echo(_format_metrics_table(rows))


@main.command()
def dashboard() -> None:
    """Launch the Streamlit dashboard."""
    app_path = Path(__file__).parent / "dashboard" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=True)


if __name__ == "__main__":
    main()
