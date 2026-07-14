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
- [ ] `core/macro_engine.py` — `MacroRegimeEngine`: EMA crossover + volume z-score → `Regime`
- [ ] `core/strategies.py` — `StrategyManager`: exposure sizing, Donchian breakout, ATR stop → `Signal`
- [ ] `core/risk_manager.py` — `RiskManager`: circuit breaker, kill switch, `TRADING_LOCKED.json` → `RiskDecision`
- [ ] `simulation/mock_broker.py` — `MockBroker`: ledger, slippage, trailing stop execution → `Fill`
- [ ] `data/yfinance_provider.py` — `YFinanceProvider` implementing `DataProvider`, with on-disk parquet cache
- [ ] Matching unit tests for each module above

## Phase 3 — Integration
- [ ] `backtest/engine.py` — walk-forward loop wiring engine → strategy → risk → broker
- [ ] `backtest/analytics.py` — Sharpe, max drawdown, win rate, total return
- [ ] `backtest/benchmarks.py` — buy-and-hold, 200-DMA comparison
- [ ] `cli.py` — `mrt backtest`, `mrt dashboard` entry points

## Phase 4 — Dashboard + Demo (parallelized)
- [ ] `dashboard/app.py` — Streamlit: live regime state, equity curve, trade log
- [ ] `notebooks/demo.ipynb` — real-data walkthrough: fetch → detect regime → backtest → plot

## Phase 5 — Production hardening
- [ ] `.github/workflows/ci.yml` — lint (ruff), type-check (mypy), pytest on push/PR
- [ ] `Dockerfile`
- [ ] `README.md` rewrite: architecture, quickstart, notebook link
- [ ] `.claude/skills/run-tests`, `run-backtest`, `run-dashboard`
- [ ] `.claude/settings.json` — auto-format hook on `.py` edits
- [ ] Full verification: `pytest`, `mrt backtest`, notebook execution, dashboard smoke test

## Conventions
- Commit after every module/file group lands; push immediately after each commit.
- All modules import shared contracts from `macro_regime_trader.types` — do not redefine
  `Regime`/`Signal`/`RiskDecision`/`Fill` locally.
- Config values (thresholds, windows, balances) come from `macro_regime_trader.config.get_settings()`,
  never hardcoded in module bodies.
