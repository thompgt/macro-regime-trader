"""YFinance-backed implementation of the DataProvider protocol.

Fetches historical OHLCV data via the ``yfinance`` package and transparently
caches results to disk as parquet files so repeated backtests / regime
computations over the same (ticker, start, end, interval) window never hit
the network twice.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import yfinance as yf

from macro_regime_trader.config import get_settings

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]

Downloader = Callable[..., pd.DataFrame]


class YFinanceProvider:
    """DataProvider implementation backed by ``yfinance``, with an on-disk parquet cache.

    Caching behavior
    -----------------
    ``get_ohlcv`` derives a deterministic cache file path from the
    ``(ticker, start, end, interval)`` arguments:
    ``{cache_dir}/{ticker}_{start}_{end or 'latest'}_{interval}.parquet``.

    If that file already exists it is loaded from disk and returned directly,
    with *no* network call made. Otherwise the injected ``downloader`` is
    invoked, the result is normalized to the required OHLCV schema, written
    to that cache path (creating ``cache_dir`` if necessary), and returned.
    """

    def __init__(
        self,
        cache_dir: str | os.PathLike[str] | None = None,
        downloader: Downloader | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir if cache_dir is not None else get_settings().data_cache_dir)
        self._downloader: Downloader = downloader if downloader is not None else yf.download

    def _cache_path(self, ticker: str, start: str, end: str | None, interval: str) -> Path:
        end_part = end if end is not None else "latest"
        filename = f"{ticker}_{start}_{end_part}_{interval}.parquet"
        return self.cache_dir / filename

    @staticmethod
    def _normalize(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        df = df.copy()

        # Flatten yfinance's MultiIndex columns (e.g. ('Close', 'AAPL')) for a single ticker.
        if isinstance(df.columns, pd.MultiIndex):
            # Try to find the level that holds field names like Open/High/Low/Close/Volume.
            level0 = [str(v) for v in df.columns.get_level_values(0)]
            if any(v.lower() in REQUIRED_COLUMNS for v in level0):
                df.columns = df.columns.get_level_values(0)
            else:
                df.columns = df.columns.get_level_values(-1)

        df.columns = [str(c).lower() for c in df.columns]

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Downloaded data for {ticker!r} is missing required columns: {missing}"
            )
        df = df[REQUIRED_COLUMNS]

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = "date"

        df = df.dropna(how="all")
        df = df.sort_index()
        df = df.astype(float)

        return df

    def get_ohlcv(
        self,
        ticker: str,
        start: str,
        end: str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Return an OHLCV DataFrame for ``ticker``, using an on-disk parquet cache.

        See class docstring for the caching behavior in detail: a cache hit
        avoids calling the downloader entirely.
        """
        cache_path = self._cache_path(ticker, start, end, interval)

        if cache_path.exists():
            return pd.read_parquet(cache_path)

        raw = self._downloader(
            ticker,
            start=start,
            end=end,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )

        if raw is None or len(raw) == 0:
            raise ValueError(
                f"No data returned for ticker={ticker!r}, start={start!r}, "
                f"end={end!r}, interval={interval!r}"
            )

        normalized = self._normalize(raw, ticker)

        if normalized.empty:
            raise ValueError(
                f"No usable data after normalization for ticker={ticker!r}, "
                f"start={start!r}, end={end!r}, interval={interval!r}"
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        normalized.to_parquet(cache_path)

        return normalized
