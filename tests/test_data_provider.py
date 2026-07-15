"""Tests for YFinanceProvider caching and normalization behavior.

No real network calls are made: a fake downloader stands in for
``yfinance.download`` and returns small synthetic DataFrames shaped like
yfinance's real output (including its MultiIndex-columns variant).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macro_regime_trader.data.yfinance_provider import YFinanceProvider


def _simple_frame() -> pd.DataFrame:
    index = pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-01"])
    return pd.DataFrame(
        {
            "Open": [102.0, 101.0, 100.0],
            "High": [103.0, 102.0, 101.0],
            "Low": [99.0, 98.0, 97.0],
            "Close": [101.5, 100.5, 99.5],
            "Volume": [1000, 1100, 1200],
        },
        index=index,
    )


def _multiindex_frame(ticker: str = "AAPL") -> pd.DataFrame:
    df = _simple_frame()
    df.columns = pd.MultiIndex.from_product([df.columns, [ticker]])
    return df


class FakeDownloader:
    def __init__(self, frame_factory) -> None:
        self.frame_factory = frame_factory
        self.call_count = 0

    def __call__(
        self, ticker, start=None, end=None, interval="1d", progress=False, auto_adjust=True
    ):
        self.call_count += 1
        return self.frame_factory()


def test_first_call_writes_cache_and_returns_normalized_frame(tmp_path):
    downloader = FakeDownloader(_simple_frame)
    provider = YFinanceProvider(cache_dir=tmp_path, downloader=downloader)

    result = provider.get_ohlcv("AAPL", start="2024-01-01", end="2024-01-05", interval="1d")

    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.tz is None
    assert result.index.is_monotonic_increasing
    assert result.dtypes.apply(lambda d: d.type == np.float64).all()
    assert downloader.call_count == 1

    cache_files = list(tmp_path.glob("*.parquet"))
    assert len(cache_files) == 1


def test_second_call_uses_cache_without_calling_downloader(tmp_path):
    downloader = FakeDownloader(_simple_frame)
    provider = YFinanceProvider(cache_dir=tmp_path, downloader=downloader)

    first = provider.get_ohlcv("AAPL", start="2024-01-01", end="2024-01-05", interval="1d")
    second = provider.get_ohlcv("AAPL", start="2024-01-01", end="2024-01-05", interval="1d")

    assert downloader.call_count == 1
    pd.testing.assert_frame_equal(first, second)


def test_empty_or_none_downloader_result_raises_value_error(tmp_path):
    provider_none = YFinanceProvider(cache_dir=tmp_path, downloader=FakeDownloader(lambda: None))
    with pytest.raises(ValueError):
        provider_none.get_ohlcv("BADTICKER", start="2024-01-01", end="2024-01-05")

    provider_empty = YFinanceProvider(
        cache_dir=tmp_path / "other", downloader=FakeDownloader(lambda: pd.DataFrame())
    )
    with pytest.raises(ValueError):
        provider_empty.get_ohlcv("BADTICKER", start="2024-01-01", end="2024-01-05")


def test_multiindex_columns_are_flattened_and_normalized(tmp_path):
    downloader = FakeDownloader(_multiindex_frame)
    provider = YFinanceProvider(cache_dir=tmp_path, downloader=downloader)

    result = provider.get_ohlcv("AAPL", start="2024-01-01", end="2024-01-05", interval="1d")

    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert result.index.is_monotonic_increasing
    assert result.loc["2024-01-01", "close"] == 99.5
    assert result.loc["2024-01-03", "close"] == 101.5
