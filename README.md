# Macro Regime Trader

A high-fidelity quantitative trading simulation engine focused on macroeconomic regime detection, adaptive allocation, and strict capital preservation.

## Core Components

### 1. Macro Regime Engine (`core/macro_engine.py`)
Classifies market structures into 4 distinct phases using an EMA crossover matrix and rolling volume Z-scores:
*   **Sustained Bull**: Positive momentum, expanding volume.
*   **Volatile Distribution**: High variance, flattening momentum.
*   **Structural Bear**: Negative momentum, accelerating volume.
*   **Compressed Liquidity**: Low volatility, declining volume.

### 2. Strategy Manager (`core/strategies.py`)
Executes regime-specific allocation and signal logic:
*   **Dynamic Exposure**: Varies from 0% (Bear) to 100% (Bull).
*   **Breakout Signals**: Uses Donchian Channels for structural entry points.
*   **Adaptive Risk**: ATR-based trailing stops that tighten in volatile regimes.

### 3. Risk Manager (`core/risk_manager.py`)
Strict capital preservation layer with absolute veto authority:
*   **Intra-day Circuit Breaker**: Halts trading for 48 steps if session equity drops > 2.5%.
*   **Hard Kill Switch**: Permanent lock if peak-to-trough drawdown exceeds 12%.
*   **System Lock**: Generates `TRADING_LOCKED.json` upon catastrophic failure.

### 4. Local Simulator (`simulation/mock_broker.py`)
Stateful brokerage simulation:
*   Virtual $100,000 starting balance.
*   **0.04% slippage** per trade for realistic performance.
*   Dynamic trailing stop-loss processing.

### 5. Backtest Framework (`backtest/`)
*   **Walk-Forward Optimization**: 180-period training / 60-period OOS verification windows.
*   **Analytics**: Sharpe Ratio, Max Drawdown, Win Rate, and Total Returns.
*   **Benchmarking**: Comparison against Buy-and-Hold and 200-period DMA models.

## Execution

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Tests**:
   ```bash
   python tests/test_macro_engine.py
   python tests/test_strategies.py
   python tests/test_mock_broker.py
   python tests/test_backtest.py
   ```

3. **Launch Dashboard**:
   ```bash
   streamlit run dashboard/app.py
   ```

## Workflow
The engine follows a strict sequential process for every data step:
1. `MacroRegimeEngine` detects the current state.
2. `StrategyManager` generates potential signals and allocation targets.
3. `RiskManager` validates signals against preservation rules.
4. `LocalSimulator` executes approved trades and manages ledger state.
5. `Analytics` tracks performance vs. benchmarks.
