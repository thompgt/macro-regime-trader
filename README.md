# Macro Regime Trader

[![CI](https://github.com/thompgt/macro-regime-trader/actions/workflows/ci.yml/badge.svg)](https://github.com/thompgt/macro-regime-trader/actions/workflows/ci.yml)

A quantitative trading **simulation** engine that detects macroeconomic/market
regimes from real historical price data, adaptively sizes exposure per
regime, and enforces strict capital-preservation risk limits. It trades
against a local paper-trading broker only — there is no live order routing
and no broker API keys are required.

**Not investment advice.** This is a research/education project.

## Quickstart

```bash
git clone https://github.com/thompgt/macro-regime-trader.git
cd macro-regime-trader
pip install -e ".[dev]"

# Run the test suite
pytest -q

# Backtest a real ticker against buy-and-hold and a 200-DMA benchmark
mrt backtest --ticker SPY --start 2015-01-01

# Launch the interactive dashboard
mrt dashboard

# Or open the live walkthrough notebook
jupyter notebook notebooks/demo.ipynb
```

Example `mrt backtest` output:

```
model               total_return    sharpe_ratio    max_drawdown        win_rate
--------------------------------------------------------------------------------
strategy                  0.1296          0.5763         -0.0527          0.4674
buy_and_hold              0.6726          0.7380         -0.2450          0.5379
dma_crossover             0.5478          0.9594         -0.1194          0.4101
```

(The strategy trades off upside for a much smaller drawdown — that trade-off
is the point of the risk layer, not a bug.)

## How it works

Real OHLCV data flows through four stages, in order, on every bar:

```
YFinanceProvider  →  MacroRegimeEngine  →  StrategyManager  →  RiskManager  →  MockBroker
   (real data)          (classify)          (size signal)      (veto/clamp)     (execute)
```

1. **`MacroRegimeEngine`** (`core/macro_engine.py`) — classifies each bar into
   one of four regimes from EMA momentum crossover + rolling volume z-score:
   `sustained_bull`, `volatile_distribution`, `structural_bear`,
   `compressed_liquidity`.
2. **`StrategyManager`** (`core/strategies.py`) — turns the regime into a
   target exposure (0-100%), using Donchian-channel breakout confirmation for
   bull entries and an ATR-based trailing stop that tightens in
   riskier regimes.
3. **`RiskManager`** (`core/risk_manager.py`) — the only gate before
   execution. Halts trading for a cooldown period on a >2.5% intraday
   drawdown (circuit breaker), and permanently locks trading (writes
   `TRADING_LOCKED.json`) on a >12% peak-to-trough drawdown (kill switch).
4. **`MockBroker`** (`simulation/mock_broker.py`) — a stateful, long-only
   paper broker: $100,000 starting balance, 0.04% slippage per trade,
   ratcheting trailing stops, full ledger + equity curve.

`backtest/engine.py` wires all four together for a full-sample run, or a
rolling walk-forward evaluation (`--walk-forward`) that only reports
out-of-sample windows. `backtest/analytics.py` and `backtest/benchmarks.py`
compute Sharpe/drawdown/win-rate and compare against buy-and-hold and a
200-day moving-average crossover.

All market data comes from Yahoo Finance via `yfinance`
(`data/yfinance_provider.py`), cached to local parquet so repeat backtests
don't re-hit the network. Every numeric threshold (EMA windows, risk limits,
slippage, starting balance, walk-forward window sizes, ...) is centralized in
`config.py` and overridable via environment variables or a `.env` file — see
`.env.example`.

## CLI

```bash
mrt backtest --ticker SPY --start 2015-01-01 [--end YYYY-MM-DD] [--interval 1d] [--walk-forward]
mrt dashboard [--host 0.0.0.0] [--port 8501]
```

## Dashboard & notebook

- `mrt dashboard` launches a Streamlit app: current regime badge, overlaid
  equity curves (strategy vs. buy-and-hold vs. DMA crossover), and the trade
  ledger.
- `notebooks/demo.ipynb` is the same walkthrough as a live, re-runnable
  notebook — fetch real data, detect regimes, backtest, compare benchmarks,
  and run a walk-forward evaluation, with plots at each step.

## Docker

```bash
docker build -t macro-regime-trader .
docker run --rm -p 8501:8501 macro-regime-trader          # dashboard
docker run --rm --entrypoint mrt macro-regime-trader backtest --ticker SPY --start 2020-01-01
```

## Project layout

```
src/macro_regime_trader/
  config.py            # centralized, env-overridable settings
  types.py             # shared Regime / Signal / RiskDecision / Fill contracts
  data/                 # DataProvider protocol + real YFinanceProvider
  core/                 # macro_engine, strategies, risk_manager
  simulation/           # mock_broker
  backtest/             # engine, analytics, benchmarks
  dashboard/            # Streamlit app
  cli.py                # `mrt` entry point
tests/                  # one test file per module, no network calls
notebooks/demo.ipynb    # live, executable demo
```

## Development

```bash
pip install -e ".[dev]"
pytest -q               # 36 tests, no network required
ruff format . && ruff check .
mypy src
```

CI (`.github/workflows/ci.yml`) runs lint, format-check, type-check, and
tests on Python 3.11/3.12 for every push and PR.

See `workplan.md` for the build milestone history and `CLAUDE.md` for the
working conventions (commit frequently, keep contracts in `types.py`, etc.)
used while building this project with Claude Code.
