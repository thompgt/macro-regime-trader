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
    circuit_breaker_drawdown_pct: float = 0.025
    circuit_breaker_halt_steps: int = 48
    kill_switch_drawdown_pct: float = 0.12
    lock_file_path: str = "TRADING_LOCKED.json"

    # Regime engine
    ema_fast: int = 20
    ema_slow: int = 50
    volume_zscore_window: int = 20

    # Strategy manager
    donchian_window: int = 20
    atr_window: int = 14
    atr_stop_multiplier: float = 2.5

    # Backtest
    train_window: int = 180
    test_window: int = 60
    benchmark_dma_window: int = 200


def get_settings() -> Settings:
    return Settings()
