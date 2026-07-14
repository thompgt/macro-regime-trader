"""Regime-conditioned allocation and signal logic.

Translates a classified :class:`~macro_regime_trader.types.Regime` plus recent
OHLCV history into a :class:`~macro_regime_trader.types.Signal`: a target
exposure fraction and an ATR-based protective stop. This module only expresses
*intent* -- the risk manager downstream is responsible for approving,
shrinking, or vetoing it.
"""

from __future__ import annotations

import pandas as pd

from macro_regime_trader.config import Settings, get_settings
from macro_regime_trader.types import Regime, Signal

_REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")

# Base target exposures per regime, before any breakout scaling is applied.
_BASE_EXPOSURE: dict[Regime, float] = {
    Regime.SUSTAINED_BULL: 0.6,
    Regime.VOLATILE_DISTRIBUTION: 0.3,
    Regime.STRUCTURAL_BEAR: 0.0,
    Regime.COMPRESSED_LIQUIDITY: 0.15,
}

# Full exposure a bull-regime breakout bar is allowed to reach.
_BULL_BREAKOUT_EXPOSURE = 1.0

# Regimes in which the trailing stop is tightened (smaller ATR multiplier)
# because elevated realized volatility/downside risk warrants less room
# before a position is cut.
_TIGHTENED_STOP_REGIMES = (Regime.VOLATILE_DISTRIBUTION, Regime.STRUCTURAL_BEAR)
_TIGHTENED_STOP_FACTOR = 0.6  # multiply settings.atr_stop_multiplier by this


class StrategyManager:
    """Maps (regime, OHLCV) pairs to allocation/stop :class:`Signal` objects."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def _donchian_channels(self, ohlcv: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Rolling Donchian upper/lower channels, shifted to avoid lookahead.

        Shifting by one bar before rolling ensures the channel for row ``t``
        only reflects information available at the close of row ``t - 1``.
        """
        window = self.settings.donchian_window
        upper = ohlcv["high"].shift(1).rolling(window=window, min_periods=window).max()
        lower = ohlcv["low"].shift(1).rolling(window=window, min_periods=window).min()
        return upper, lower

    def _atr(self, ohlcv: pd.DataFrame) -> pd.Series:
        """Wilder-style Average True Range over ``settings.atr_window``.

        True range at row ``t`` uses ``close`` at ``t - 1`` (via ``.shift(1)``)
        together with ``high``/``low`` at ``t``, which is standard and causal:
        nothing from beyond bar ``t`` is used.
        """
        high = ohlcv["high"]
        low = ohlcv["low"]
        prev_close = ohlcv["close"].shift(1)

        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        window = self.settings.atr_window
        return tr.rolling(window=window, min_periods=window).mean()

    def generate_signal(
        self,
        ohlcv: pd.DataFrame,
        regime: Regime,
        current_position_exposure: float = 0.0,
    ) -> Signal:
        """Generate a :class:`Signal` for the most recent bar in ``ohlcv``.

        ``current_position_exposure`` is accepted for forward compatibility
        (e.g. hysteresis/anti-churn logic) but does not currently affect the
        computed target exposure.
        """
        missing = [c for c in _REQUIRED_COLUMNS if c not in ohlcv.columns]
        if missing:
            raise ValueError(f"ohlcv is missing required columns: {missing}")
        if ohlcv.empty:
            raise ValueError("ohlcv must contain at least one row")

        upper, lower = self._donchian_channels(ohlcv)
        atr = self._atr(ohlcv)

        timestamp = ohlcv.index[-1]
        close = float(ohlcv["close"].iloc[-1])
        upper_latest = upper.iloc[-1]
        atr_latest = atr.iloc[-1]

        base_exposure = _BASE_EXPOSURE[regime]
        breakout = bool(pd.notna(upper_latest) and close > upper_latest)

        if regime == Regime.SUSTAINED_BULL:
            target_exposure = _BULL_BREAKOUT_EXPOSURE if breakout else base_exposure
            reason = f"{regime.value} + {'donchian_breakout' if breakout else 'no_breakout'}"
        else:
            target_exposure = base_exposure
            reason = regime.value

        stop_price: float | None = None
        if target_exposure > 0.0 and pd.notna(atr_latest):
            multiplier = self.settings.atr_stop_multiplier
            if regime in _TIGHTENED_STOP_REGIMES:
                multiplier *= _TIGHTENED_STOP_FACTOR
                reason += " (tightened_stop)"
            stop_price = close - multiplier * float(atr_latest)

        return Signal(
            timestamp=timestamp,
            regime=regime,
            target_exposure=target_exposure,
            stop_price=stop_price,
            reason=reason,
        )

    def generate_signals(self, ohlcv: pd.DataFrame, regimes: pd.Series) -> list[Signal]:
        """Generate one :class:`Signal` per row, aligned to ``ohlcv``'s index.

        Donchian channels and ATR are computed once (vectorized) up front;
        the per-row regime -> exposure/stop mapping is then applied in a
        plain loop since it is cheap and clarity matters more than
        micro-optimizing this simulation-time path.
        """
        missing = [c for c in _REQUIRED_COLUMNS if c not in ohlcv.columns]
        if missing:
            raise ValueError(f"ohlcv is missing required columns: {missing}")

        regimes = regimes.reindex(ohlcv.index)

        upper, _ = self._donchian_channels(ohlcv)
        atr = self._atr(ohlcv)
        close = ohlcv["close"]

        signals: list[Signal] = []
        for idx in ohlcv.index:
            regime_value = regimes.loc[idx]
            close_i = float(close.loc[idx])

            if regime_value is None or (isinstance(regime_value, float) and pd.isna(regime_value)):
                # No regime classification available (e.g. warmup window);
                # stay flat rather than guessing, but still emit one Signal
                # per row so callers can rely on index-aligned output.
                signals.append(
                    Signal(
                        timestamp=idx,
                        regime=Regime.STRUCTURAL_BEAR,
                        target_exposure=0.0,
                        stop_price=None,
                        reason="unclassified_regime",
                    )
                )
                continue
            regime = Regime(regime_value)

            upper_i = upper.loc[idx]
            atr_i = atr.loc[idx]

            base_exposure = _BASE_EXPOSURE[regime]
            breakout = bool(pd.notna(upper_i) and close_i > upper_i)

            if regime == Regime.SUSTAINED_BULL:
                target_exposure = _BULL_BREAKOUT_EXPOSURE if breakout else base_exposure
                reason = f"{regime.value} + {'donchian_breakout' if breakout else 'no_breakout'}"
            else:
                target_exposure = base_exposure
                reason = regime.value

            stop_price: float | None = None
            if target_exposure > 0.0 and pd.notna(atr_i):
                multiplier = self.settings.atr_stop_multiplier
                if regime in _TIGHTENED_STOP_REGIMES:
                    multiplier *= _TIGHTENED_STOP_FACTOR
                    reason += " (tightened_stop)"
                stop_price = close_i - multiplier * float(atr_i)

            signals.append(
                Signal(
                    timestamp=idx,
                    regime=regime,
                    target_exposure=target_exposure,
                    stop_price=stop_price,
                    reason=reason,
                )
            )

        return signals
