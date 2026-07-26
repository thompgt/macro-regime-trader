"""Classifies market structure into one of four macro regimes.

The classifier is built on two largely independent axes:

- **Trend** -- price relative to a long-term moving average, plus the sign of
  a fast/slow EMA spread. This is the primary bull/bear discriminator, and it
  is what actually separates regimes that deserve capital from ones that do
  not.
- **Volatility state** -- realized volatility ranked against its own trailing
  history. This separates a calm advance from a violent one, and quiet
  consolidation from high-variance chop.

Volume participation (a rolling Z-score) is a *confirmation* input only: it
refines the quiet/compressed classification but never vetoes a trend call.
An earlier version of this module required expanding volume for a bar to
count as bullish, which pushed the majority of a steady grind higher into the
residual "volatile distribution" bucket and structurally under-allocated the
strategy. Trend leads, volume corroborates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from macro_regime_trader.config import Settings, get_settings
from macro_regime_trader.core.indicators import realized_volatility
from macro_regime_trader.types import Regime

_REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")

# Minimum observations before a rolling percentile rank is meaningful. Well
# below `realized_vol_lookback` so the warmup window stays usable on shorter
# histories; the rank simply ranks against whatever history exists so far.
_MIN_RANK_OBSERVATIONS = 60

# Spread magnitude below which the EMA spread reads as "flat" rather than
# directional. Small, because the long-term trend filter carries the
# directional call and this only suppresses noise around the crossover.
_MOMENTUM_EPS = 0.002

# Volume Z-score above which participation counts as expanding.
_PARTICIPATION_EPS = 0.25


class MacroRegimeEngine:
    """Rule-based classifier mapping OHLCV history to :class:`Regime` phases."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Compute the causal feature frame the classification is derived from.

        Every column at row ``t`` uses only data up to and including ``t``
        (``ewm``/``rolling`` are inherently causal), so there is no lookahead
        bias. Exposed publicly because the dashboard and notebook plot these
        features alongside the regime timeline.
        """
        missing = [c for c in _REQUIRED_COLUMNS if c not in ohlcv.columns]
        if missing:
            raise ValueError(f"ohlcv is missing required columns: {missing}")

        settings = self.settings
        close = ohlcv["close"]
        volume = ohlcv["volume"]

        ema_fast = close.ewm(
            span=settings.ema_fast, min_periods=settings.ema_fast, adjust=False
        ).mean()
        ema_slow = close.ewm(
            span=settings.ema_slow, min_periods=settings.ema_slow, adjust=False
        ).mean()
        momentum = (ema_fast - ema_slow) / ema_slow

        # Long-term trend filter. `min_periods` is relaxed to half the window so
        # a backtest over a shorter history still classifies instead of sitting
        # in warmup for 200 bars.
        trend_window = settings.trend_window
        trend_ma = close.rolling(window=trend_window, min_periods=max(2, trend_window // 2)).mean()
        trend_gap = close / trend_ma - 1.0

        # Fast re-entry filter: a short MA that price has reclaimed *and* that
        # is itself turning up. Requiring the slope rules out the "falling
        # knife" case where price pops above a still-declining average.
        reclaim_ma = close.rolling(
            window=settings.reclaim_window, min_periods=max(2, settings.reclaim_window // 2)
        ).mean()
        reclaim = (close > reclaim_ma) & (reclaim_ma.diff() > 0)

        # Realized volatility, annualized, then ranked against its own trailing
        # distribution. Ranking rather than thresholding in absolute terms keeps
        # the classifier portable across assets of very different volatility.
        realized_vol = realized_volatility(
            close, settings.realized_vol_window, settings.trading_days_per_year
        )
        vol_rank = realized_vol.rolling(
            window=settings.realized_vol_lookback,
            min_periods=min(_MIN_RANK_OBSERVATIONS, settings.realized_vol_lookback),
        ).rank(pct=True)

        vol_window = settings.volume_zscore_window
        volume_mean = volume.rolling(window=vol_window, min_periods=vol_window).mean()
        volume_std = volume.rolling(window=vol_window, min_periods=vol_window).std(ddof=0)
        volume_zscore = (volume - volume_mean) / volume_std.replace(0.0, np.nan)

        return pd.DataFrame(
            {
                "ema_fast": ema_fast,
                "ema_slow": ema_slow,
                "momentum": momentum,
                "trend_ma": trend_ma,
                "trend_gap": trend_gap,
                "reclaim_ma": reclaim_ma,
                "reclaim": reclaim,
                "realized_vol": realized_vol,
                "vol_rank": vol_rank,
                "volume_zscore": volume_zscore,
            },
            index=ohlcv.index,
        )

    def classify(self, ohlcv: pd.DataFrame) -> pd.Series:
        """Classify every bar in ``ohlcv`` for which enough lookback exists.

        Rows still inside the warmup window are ``None`` rather than raising,
        since callers typically want an index-aligned series to concatenate
        with other frames.

        Precedence, applied in order:

        1. ``STRUCTURAL_BEAR`` -- below the long-term trend *and* price has not
           reclaimed a rising short MA. This is the only risk-off state, and it
           deliberately takes two failing conditions to reach: requiring the
           long-term trend alone would keep the book flat through the whole of
           a V-shaped recovery.
        2. ``SUSTAINED_BULL`` -- above the long-term trend, EMA spread
           positive, and volatility not in its top decile-ish tail. A calm
           confirmed uptrend.
        3. ``COMPRESSED_LIQUIDITY`` -- quiet volatility with non-expanding
           participation. Low risk, low opportunity.
        4. ``VOLATILE_DISTRIBUTION`` -- the residual: high-variance chop, or a
           fast re-entry that the long-term trend has not yet confirmed.
        """
        features = self.features(ohlcv)

        momentum = features["momentum"]
        trend_gap = features["trend_gap"]
        vol_rank = features["vol_rank"]
        volume_zscore = features["volume_zscore"]

        regimes = pd.Series(index=ohlcv.index, dtype=object)
        ready = momentum.notna() & trend_gap.notna() & vol_rank.notna()

        above_trend = trend_gap > 0.0
        reclaim = features["reclaim"].fillna(False).astype(bool)
        momentum_up = momentum > _MOMENTUM_EPS
        momentum_down = momentum < -_MOMENTUM_EPS

        vol_elevated = vol_rank > self.settings.high_vol_percentile
        vol_quiet = vol_rank < self.settings.low_vol_percentile
        volume_expanding = volume_zscore > _PARTICIPATION_EPS

        # Risk is off only when the long-term trend is broken AND price has not
        # reclaimed a rising short MA. Either condition alone keeps the book on.
        structural_bear = ~above_trend & ~reclaim
        sustained_bull = above_trend & momentum_up & ~vol_elevated & ~structural_bear
        compressed_liquidity = (
            vol_quiet
            & ~volume_expanding.fillna(False)
            & ~momentum_down
            & ~structural_bear
            & ~sustained_bull
        )
        volatile_distribution = ~structural_bear & ~sustained_bull & ~compressed_liquidity

        regimes[ready & structural_bear] = Regime.STRUCTURAL_BEAR.value
        regimes[ready & sustained_bull] = Regime.SUSTAINED_BULL.value
        regimes[ready & compressed_liquidity] = Regime.COMPRESSED_LIQUIDITY.value
        regimes[ready & volatile_distribution] = Regime.VOLATILE_DISTRIBUTION.value

        regimes[~ready] = None
        return regimes

    def classify_latest(self, ohlcv: pd.DataFrame) -> Regime:
        regimes = self.classify(ohlcv)
        if regimes.empty or regimes.iloc[-1] is None or pd.isna(regimes.iloc[-1]):
            settings = self.settings
            needed = max(
                settings.ema_slow,
                settings.realized_vol_window + _MIN_RANK_OBSERVATIONS,
                max(2, settings.trend_window // 2),
            )
            raise ValueError(
                f"Not enough history to classify the latest bar; need at least {needed} bars."
            )
        return Regime(regimes.iloc[-1])
