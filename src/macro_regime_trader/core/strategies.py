"""Regime-conditioned allocation and signal logic.

Translates a classified :class:`~macro_regime_trader.types.Regime` plus recent
OHLCV history into a :class:`~macro_regime_trader.types.Signal`: a target
exposure fraction and an ATR-based protective stop. This module only expresses
*intent* -- the risk manager downstream is responsible for approving,
shrinking, or vetoing it.

Sizing is the product of three parts:

1. A **regime base weight** -- essentially a risk-on/risk-off decision, since
   only ``STRUCTURAL_BEAR`` is de-allocated.
2. A **volatility de-risk scalar**, ``min(1, target_annual_vol / realized_vol)``.
   It only ever cuts: when volatility runs above target the position shrinks
   automatically, ahead of any stop being hit. It never levers up, because
   scaling into calm markets is short vol-of-vol and measurably hurt
   risk-adjusted returns.
3. **Gearing** (``Settings.base_leverage``), applied last and capped by
   ``Settings.max_leverage``.

Why gearing at all: the regime gate earns a distinctly better Sharpe than
buy-and-hold, but it does so at lower volatility (it sits in cash during
bears), so unlevered it trails the benchmark on raw return. Gearing is how that
risk-adjusted edge is expressed as an absolute return at comparable risk -- it
is a consequence of the edge, not a substitute for one. The de-risk scalar is
what makes carrying it defensible.

Constant-notional sizing is why the previous version could not keep up: it
capped exposure at 0.6 in its best regime, so its average allocation was
roughly a third of the index and no amount of correct regime calls could close
that gap.
"""

from __future__ import annotations

import pandas as pd

from macro_regime_trader.config import Settings, get_settings
from macro_regime_trader.core.indicators import (
    average_true_range,
    donchian_channels,
    realized_volatility,
)
from macro_regime_trader.types import Regime, Signal

_REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")

# Base risk weight per regime, before de-risking and gearing are applied.
#
# This is effectively a risk-on/risk-off switch: only STRUCTURAL_BEAR is
# de-allocated. Hand-set intermediate weights per regime were tested against
# flat risk-on weighting across five tickers and two decades and lost, because
# the regime label already encodes *direction* while realized volatility
# encodes *how much that direction is worth paying for* -- and the vol de-risk
# scalar measures the latter directly instead of proxying it through a label.
# The regime taxonomy therefore drives the entry/exit decision and reporting,
# and volatility drives the sizing.
_BASE_WEIGHT: dict[Regime, float] = {
    Regime.SUSTAINED_BULL: 1.0,
    Regime.COMPRESSED_LIQUIDITY: 1.0,
    Regime.VOLATILE_DISTRIBUTION: 1.0,
    Regime.STRUCTURAL_BEAR: 0.0,
}

# Regimes in which the trailing stop is tightened (smaller ATR multiplier)
# because elevated realized volatility/downside risk warrants less room
# before a position is cut.
_TIGHTENED_STOP_REGIMES = (Regime.VOLATILE_DISTRIBUTION, Regime.STRUCTURAL_BEAR)
_TIGHTENED_STOP_FACTOR = 0.7  # multiply settings.atr_stop_multiplier by this


class StrategyManager:
    """Maps (regime, OHLCV) pairs to allocation/stop :class:`Signal` objects."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    # -- indicator plumbing ------------------------------------------------

    def indicators(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Vectorized causal inputs to sizing: Donchian upper, ATR, vol scalar."""
        missing = [c for c in _REQUIRED_COLUMNS if c not in ohlcv.columns]
        if missing:
            raise ValueError(f"ohlcv is missing required columns: {missing}")

        settings = self.settings
        upper, lower = donchian_channels(ohlcv, settings.donchian_window)
        atr = average_true_range(ohlcv, settings.atr_window)
        realized_vol = realized_volatility(
            ohlcv["close"], settings.realized_vol_window, settings.trading_days_per_year
        )

        # Guard against a zero/absent vol estimate producing an infinite scalar.
        vol_scalar = settings.target_annual_vol / realized_vol.where(realized_vol > 0)
        if settings.vol_scalar_smoothing > 1:
            vol_scalar = vol_scalar.ewm(
                span=settings.vol_scalar_smoothing, min_periods=1, adjust=False
            ).mean()
        # Clipped at 1.0 so volatility can only ever *cut* the position, never
        # add to it. Letting it lever up in calm markets measurably hurt
        # risk-adjusted returns: quiet periods are precisely the ones that end
        # in a volatility spike, so scaling up into them is short vol-of-vol.
        # The level of risk is set by `base_leverage` instead.
        vol_scalar = vol_scalar.clip(upper=1.0)

        return pd.DataFrame(
            {
                "donchian_upper": upper,
                "donchian_lower": lower,
                "atr": atr,
                "realized_vol": realized_vol,
                "vol_scalar": vol_scalar,
            },
            index=ohlcv.index,
        )

    # -- sizing ------------------------------------------------------------

    def _target_exposure(
        self,
        regime: Regime,
        close: float,
        donchian_upper: float,
        vol_scalar: float,
    ) -> tuple[float, str]:
        """Regime base weight, de-risked by volatility, then geared.

        ``donchian_upper`` is recorded in the reason string for diagnostics but
        no longer scales the position. Breakout-based sizing was retired after
        validation: it is largely redundant with the regime engine's fast
        re-entry filter (both answer "has upward momentum resumed?"), and adding
        it on top only increased turnover.
        """
        breakout = bool(pd.notna(donchian_upper) and close > donchian_upper)

        weight = _BASE_WEIGHT[regime]
        reason = f"{regime.value}{' + donchian_breakout' if breakout else ''}"

        if weight <= 0.0:
            return 0.0, reason

        # A missing vol estimate (warmup) falls back to unscaled sizing rather
        # than silently zeroing the position.
        scalar = float(vol_scalar) if pd.notna(vol_scalar) else 1.0
        exposure = weight * scalar * self.settings.base_leverage
        exposure = min(exposure, self.settings.max_leverage)
        return max(0.0, exposure), f"{reason} (vol_scalar={scalar:.2f})"

    def _stop_price(self, regime: Regime, close: float, atr: float) -> tuple[float | None, str]:
        if not self.settings.use_atr_stop or pd.isna(atr):
            return None, ""
        multiplier = self.settings.atr_stop_multiplier
        suffix = ""
        if regime in _TIGHTENED_STOP_REGIMES:
            multiplier *= _TIGHTENED_STOP_FACTOR
            suffix = " (tightened_stop)"
        return close - multiplier * float(atr), suffix

    # -- public API --------------------------------------------------------

    def generate_signal(
        self,
        ohlcv: pd.DataFrame,
        regime: Regime,
        current_position_exposure: float = 0.0,
    ) -> Signal:
        """Generate a :class:`Signal` for the most recent bar in ``ohlcv``.

        ``current_position_exposure`` is accepted so callers can pass the live
        position, but sizing is deliberately stateless: the anti-churn no-trade
        band is applied at the broker, which is the only layer that knows how
        far actual exposure has drifted from target.
        """
        if ohlcv.empty:
            raise ValueError("ohlcv must contain at least one row")

        indicators = self.indicators(ohlcv)
        row = indicators.iloc[-1]
        close = float(ohlcv["close"].iloc[-1])

        target, reason = self._target_exposure(
            regime, close, row["donchian_upper"], row["vol_scalar"]
        )

        stop_price: float | None = None
        if target > 0.0:
            stop_price, suffix = self._stop_price(regime, close, row["atr"])
            reason += suffix

        return Signal(
            timestamp=ohlcv.index[-1],
            regime=regime,
            target_exposure=target,
            stop_price=stop_price,
            reason=reason,
        )

    def generate_signals(self, ohlcv: pd.DataFrame, regimes: pd.Series) -> list[Signal]:
        """Generate one :class:`Signal` per row, aligned to ``ohlcv``'s index.

        Indicators are computed once (vectorized) up front; the per-row
        regime -> exposure/stop mapping then runs in a plain loop since it is
        cheap and clarity matters more than micro-optimizing this path.
        """
        indicators = self.indicators(ohlcv)
        regimes = regimes.reindex(ohlcv.index)
        close = ohlcv["close"]

        signals: list[Signal] = []

        for idx in ohlcv.index:
            regime_value = regimes.loc[idx]

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
            row = indicators.loc[idx]
            close_i = float(close.loc[idx])

            target, reason = self._target_exposure(
                regime, close_i, row["donchian_upper"], row["vol_scalar"]
            )

            stop_price: float | None = None
            if target > 0.0:
                stop_price, suffix = self._stop_price(regime, close_i, row["atr"])
                reason += suffix

            signals.append(
                Signal(
                    timestamp=idx,
                    regime=regime,
                    target_exposure=target,
                    stop_price=stop_price,
                    reason=reason,
                )
            )

        return signals
