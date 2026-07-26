# Workplan: Macro Regime Trader

Milestone tracker for turning this repo into a production-ready quant simulation
engine with real market data and a live Jupyter demo. Update checkboxes as work lands.

## Phase 1 — Scaffold
- [x] `git init`, GitHub remote created and pushed
- [x] `pyproject.toml`, `.gitignore`, `.env.example`
- [x] `config.py` (pydantic-settings), `logging_config.py`
- [x] `types.py` — shared `Regime`, `Signal`, `RiskDecision`, `Fill` contracts
- [x] `data/provider.py` — `DataProvider` protocol

## Phase 2 — Core modules (parallelized)
- [x] `core/macro_engine.py` — `MacroRegimeEngine`: EMA crossover + volume z-score → `Regime`
- [x] `core/strategies.py` — `StrategyManager`: exposure sizing, Donchian breakout, ATR stop → `Signal`
- [x] `core/risk_manager.py` — `RiskManager`: circuit breaker, kill switch, `TRADING_LOCKED.json` → `RiskDecision`
- [x] `simulation/mock_broker.py` — `MockBroker`: ledger, slippage, trailing stop execution → `Fill`
- [x] `data/yfinance_provider.py` — `YFinanceProvider` implementing `DataProvider`, with on-disk parquet cache
- [x] Matching unit tests for each module above (29/29 passing)

## Phase 3 — Integration
- [x] `backtest/engine.py` — walk-forward loop wiring engine → strategy → risk → broker
- [x] `backtest/analytics.py` — Sharpe, max drawdown, win rate, total return
- [x] `backtest/benchmarks.py` — buy-and-hold, 200-DMA comparison
- [x] `cli.py` — `mrt backtest`, `mrt dashboard` entry points (verified end-to-end against real SPY data via yfinance)

## Phase 4 — Dashboard + Demo (parallelized)
- [x] `dashboard/app.py` — Streamlit: live regime state, equity curve, trade log
- [x] `notebooks/demo.ipynb` — real-data walkthrough: fetch → detect regime → backtest → plot (executed end-to-end, no errors, real SPY data)

## Phase 5 — Production hardening
- [x] `.github/workflows/ci.yml` — lint (ruff), type-check (mypy), pytest on push/PR
- [x] `Dockerfile` (verified: builds and runs `mrt backtest` end-to-end in-container)
- [x] `README.md` rewrite: architecture, quickstart, notebook link
- [x] `.claude/skills/run-tests`, `run-backtest`, `run-dashboard`
- [x] `.claude/settings.json` — auto-format hook on `.py` edits (pipe-tested; needs `/hooks` reload or restart to take effect this session)
- [x] Full verification: `pytest` (36/36), `ruff` (clean), `mypy` (clean, 20 files), `mrt backtest` (real SPY data), notebook execution (0 errors), dashboard smoke test (HTTP 200)

## Phase 6 — Make the strategy actually work
- [x] Diagnose why the strategy bought once and sat flat: `RiskManager`'s session
      baseline never moved, so the circuit breaker re-tripped every bar after the
      first 2.5% dip (2739 of 2906 bars vetoed to zero exposure)
- [x] Rebuild `MacroRegimeEngine` on trend + volatility rank; volume demoted to a
      confirming input (the old volume gate pushed steady uptrends into the
      residual bucket at 0.3 exposure)
- [x] Add the fast re-entry filter — risk-off needs a broken long-term trend AND
      no reclaim of a rising short MA. This is the source of the Sharpe edge
      (0.99 vs 0.81 unlevered); a 200-day filter alone bleeds through V-recoveries
- [x] `core/indicators.py` — shared causal indicators so the regime engine and
      the sizer measure volatility the same way
- [x] Rewrite sizing: `regime weight x vol de-risk x gearing`. De-risk clipped at
      1.0 (cuts only; levering into calm markets is short vol-of-vol)
- [x] Move the anti-churn no-trade band to the broker, where actual drift is
      known (trade count 2580 -> 91 over a decade of SPY)
- [x] Financing on borrowed/idle cash so gearing is never free
- [x] One-bar execution lag; retire the ATR trailing stop by default
- [x] Recalibrate the kill switch (0.12 -> 0.50): a permanent lock firing on
      ordinary declines silently bricked whole runs
- [x] `run_backtest_from` — indicator warmup before the reported window, so the
      strategy isn't compared flat against a fully-invested benchmark
- [x] `chain_walk_forward_equity` — compound OOS windows instead of concatenating
      equity levels (the seams were dominating the reported return: 0.07 vs 4.08)
- [x] Validate across SPY/QQQ/IWM/EFA/EEM over 2005-2015 and 2015-2026 rather
      than tuning to one sample (19 of 30 comparisons won; failures documented)
- [x] `scripts/generate_readme_charts.py` + regenerated figures, CVD-validated
      palette, drawdown and exposure panels
- [x] Dashboard reworked around the benchmark comparison; `tests/test_dashboard.py`
- [x] Notebook rebuilt and executed (27 cells, 0 errors); README/.env.example
      brought back in sync with `config.py`
- [x] Verification: `pytest` (59), `ruff` (clean), `mypy` (clean, 21 files),
      `mrt backtest` full-sample and `--walk-forward`, dashboard HTTP 200

**Result on SPY 2015-2026, net of slippage and financing:** strategy total return
3.4576 / Sharpe 0.8655 / max drawdown -0.2353 versus buy-and-hold 3.3546 / 0.8129
/ -0.3372. Walk-forward over 51 out-of-sample windows: 4.0768 / 0.8808 / -0.2439
versus 3.9411 / 0.8467 / -0.3372.

## Conventions
- Commit after every module/file group lands; push immediately after each commit.
- All modules import shared contracts from `macro_regime_trader.types` — do not redefine
  `Regime`/`Signal`/`RiskDecision`/`Fill` locally.
- Config values (thresholds, windows, balances) come from `macro_regime_trader.config.get_settings()`,
  never hardcoded in module bodies.
