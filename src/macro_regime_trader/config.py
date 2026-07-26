"""Centralized, environment-overridable configuration for the whole engine."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MRT_", env_file=".env", extra="ignore")

    # Data
    data_cache_dir: str = "data_cache"
    log_level: str = "INFO"

    # Risk manager
    starting_balance: float = 100_000.0
    slippage_pct: float = 0.0004
    # Single-session loss that trips a temporary halt, and how many bars the
    # halt lasts. Sized for daily bars: a >4% one-day loss is a genuine tail
    # event on broad equity indices, and the halt sits out the aftershock.
    circuit_breaker_drawdown_pct: float = 0.04
    circuit_breaker_halt_steps: int = 10
    kill_switch_drawdown_pct: float = 0.35
    lock_file_path: str = "TRADING_LOCKED.json"
    # Financing: annualized rate charged on borrowed cash when exposure > 1x,
    # and credited on idle cash. Keeps leverage from being free.
    margin_interest_rate: float = 0.05
    cash_yield_rate: float = 0.02
    trading_days_per_year: int = 252

    # Regime engine
    ema_fast: int = 20
    ema_slow: int = 50
    volume_zscore_window: int = 20
    # Long-term trend filter: the primary bull/bear discriminator.
    trend_window: int = 200
    # Realized-volatility window and the percentile (of its own trailing
    # history) above which volatility counts as "elevated".
    realized_vol_window: int = 20
    realized_vol_lookback: int = 252
    high_vol_percentile: float = 0.80
    low_vol_percentile: float = 0.35

    # Strategy manager
    donchian_window: int = 20
    atr_window: int = 14
    atr_stop_multiplier: float = 4.0
    # Volatility targeting: exposure scales as target_vol / realized_vol so the
    # strategy carries roughly constant risk instead of constant notional.
    target_annual_vol: float = 0.16
    max_leverage: float = 1.6
    # Minimum change in target exposure before the broker is asked to trade.
    # Suppresses churn (and its slippage) from small vol-scalar wiggles.
    rebalance_band: float = 0.08
    # Bars to stay flat after a stop-out before re-entry is allowed.
    reentry_cooldown_bars: int = 5

    # Backtest
    train_window: int = 180
    test_window: int = 60
    benchmark_dma_window: int = 200


def get_settings() -> Settings:
    return Settings()
