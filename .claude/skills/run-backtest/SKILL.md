---
name: run-backtest
description: Run the mrt backtest CLI against real market data and return a compact metrics table. Use whenever you need to verify strategy behavior end-to-end (not just unit tests) in the macro_regime_trader repo.
---

Run (defaults to SPY since 2015, full-sample):

```
cd C:\Users\thoma\macro_regime_trader && python -m macro_regime_trader.cli backtest --ticker SPY --start 2015-01-01 2>&1 | tail -15
```

Adjust `--ticker`, `--start`, `--end`, `--interval`, or add `--walk-forward` for
rolling out-of-sample evaluation, per the arguments the user cares about.

The command already prints a compact strategy vs. buy-and-hold vs. DMA-crossover
table — relay that table directly, don't re-summarize it into prose unless asked.

Data is cached under `data_cache/` as parquet after the first fetch, so repeat
runs for the same ticker/date range don't hit the network again.
