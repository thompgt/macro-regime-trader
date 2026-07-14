"""Classifies market structure into one of four macro regimes.

Combines an EMA crossover matrix (momentum) with rolling volume Z-scores
(participation) to distinguish trending, choppy, and quiet market states.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from macro_regime_trader.config import Settings, get_settings
from macro_regime_trader.types import Regime

_REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


class MacroRegimeEngine:
    """Rule-based classifier mapping OHLCV history to :class:`Regime` phases."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def classify(self, ohlcv: pd.DataFrame) -> pd.Series:
        """Classify every bar in ``ohlcv`` for which enough lookback exists.

        Rows within the warmup window (before the slower EMA/rolling window
        have enough observations) are ``None`` rather than raising, since
        callers typically want an index-aligned series to concatenate with
        other frames. All computations at row ``t`` use only data up to and
        including ``t`` (``pandas.Series.ewm``/``rolling`` are inherently
        causal), so there is no lookahead bias.
        """
        missing = [c for c in _REQUIRED_COLUMNS if c not in ohlcv.columns]
        if missing:
            raise ValueError(f"ohlcv is missing required columns: {missing}")

        settings = self.settings
        close = ohlcv["close"]
        volume = ohlcv["volume"]

        ema_fast = close.ewm(span=settings.ema_fast, min_periods=settings.ema_fast, adjust=False).mean()
        ema_slow = close.ewm(span=settings.ema_slow, min_periods=settings.ema_slow, adjust=False).mean()

        momentum = (ema_fast - ema_slow) / ema_slow
        momentum_slope = momentum.diff()

        vol_window = settings.volume_zscore_window
        volume_mean = volume.rolling(window=vol_window, min_periods=vol_window).mean()
        volume_std = volume.rolling(window=vol_window, min_periods=vol_window).std(ddof=0)
        volume_zscore = (volume - volume_mean) / volume_std.replace(0.0, np.nan)

        regimes = pd.Series(index=ohlcv.index, dtype=object)

        # `momentum_slope` needs one extra observation beyond the slow EMA's
        # own warmup, so gate readiness on it even though it isn't used to
        # branch the classification below (only to describe "flattening").
        ready = ema_slow.notna() & volume_zscore.notna() & momentum_slope.notna()

        momentum_eps = 0.01  # spread magnitude below which momentum reads as "flat"
        participation_eps = 0.25  # volume z-score magnitude below which volume reads as "average"

        momentum_positive = momentum > momentum_eps
        momentum_negative = momentum < -momentum_eps
        momentum_quiet = momentum.abs() <= momentum_eps

        volume_expanding = volume_zscore > participation_eps
        volume_contracting = volume_zscore < -participation_eps

        sustained_bull = momentum_positive & volume_expanding
        structural_bear = momentum_negative & volume_expanding
        compressed_liquidity = momentum_quiet & volume_contracting
        # Volatile distribution ("high variance, flattening momentum") is the
        # residual phase covering everything that is neither a clean directional
        # trend with rising participation, nor quiet with fading participation:
        # e.g. choppy momentum with elevated volume, or a weakening trend whose
        # momentum has flattened (small `momentum_slope`) even though volume
        # hasn't fully contracted.
        volatile_distribution = ~sustained_bull & ~structural_bear & ~compressed_liquidity

        regimes[ready & sustained_bull] = Regime.SUSTAINED_BULL.value
        regimes[ready & structural_bear] = Regime.STRUCTURAL_BEAR.value
        regimes[ready & compressed_liquidity] = Regime.COMPRESSED_LIQUIDITY.value
        regimes[ready & volatile_distribution] = Regime.VOLATILE_DISTRIBUTION.value

        regimes[~ready] = None
        return regimes

    def classify_latest(self, ohlcv: pd.DataFrame) -> Regime:
        regimes = self.classify(ohlcv)
        if regimes.empty or regimes.iloc[-1] is None or pd.isna(regimes.iloc[-1]):
            raise ValueError(
                "Not enough history to classify the latest bar; "
                f"need at least {max(self.settings.ema_slow, self.settings.volume_zscore_window)} bars."
            )
        return Regime(regimes.iloc[-1])
