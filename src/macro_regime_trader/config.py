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
    # Permanent-shutdown threshold. Must sit well beyond the strategy's own
    # expected worst drawdown: because the lock is permanent, a threshold set
    # near normal behaviour turns a survivable drawdown into a dead account for
    # the rest of the run. At 0.35 this fired during ordinary 2008/2020-scale
    # declines and silently pinned exposure to zero for years afterward.
    kill_switch_drawdown_pct: float = 0.50
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
    # Fast re-entry filter. A pure long-term trend filter exits after a
    # drawdown is underway and only re-enters long after the recovery, which
    # is what made this strategy lag through V-shaped rebounds (2018Q4, 2020,
    # 2022). Reclaiming a *rising* short MA re-arms risk far sooner.
    reclaim_window: int = 20
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
    # Gearing applied to the risk-on book. The regime gate earns a materially
    # better Sharpe than buy-and-hold but at lower volatility, so unlevered it
    # trails on raw return; gearing expresses that edge at comparable risk.
    # Only ever applied when the gate is on and after vol de-risking.
    base_leverage: float = 1.5
    # EWM span used to smooth the volatility scalar. Raw 20-day realized vol is
    # noisy enough that sizing off it directly churns the book daily and
    # de-levers into the bottom of a selloff; smoothing damps both. 1 disables.
    vol_scalar_smoothing: int = 10
    # How far actual exposure may drift from target (in units of equity) before
    # the broker rebalances. Applied at the broker because only it knows the
    # drift. Wide on purpose: continuously rebalancing a geared position back to
    # a fixed fraction incurs volatility drag and pays slippage every bar for no
    # change in stance. Raising this from 0.08 to 0.25 cut trade count from ~2600
    # to ~90 over a decade of SPY and improved return, Sharpe and drawdown.
    rebalance_band: float = 0.25
    # Bars to stay flat after a stop-out before re-entry is allowed.
    reentry_cooldown_bars: int = 5

    # Backtest
    train_window: int = 180
    test_window: int = 60
    benchmark_dma_window: int = 200
    # Calendar days of history fetched *before* the requested start date, used
    # purely to warm up indicators. Without it the strategy sits flat for its
    # first ~100 bars while buy-and-hold is invested from bar one, which
    # understates the strategy for reasons that have nothing to do with its
    # signal. ~500 calendar days covers trend_window=200 trading bars.
    warmup_calendar_days: int = 520
    # Toggle for the ATR trailing stop. Off by default: the regime gate, the
    # volatility de-risk scalar, the circuit breaker and the kill switch already
    # cover downside, and stacking a ratchet-only trailing stop on top of them
    # measurably cost both return and Sharpe (it sells dips and re-enters
    # higher). Kept available because it is useful for higher-volatility assets.
    use_atr_stop: bool = False


def get_settings() -> Settings:
    return Settings()
