"""Data provider abstraction. Concrete providers (e.g. YFinanceProvider) implement this
protocol so the rest of the engine never depends on a specific market-data vendor.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class DataProvider(Protocol):
    def get_ohlcv(
        self,
        ticker: str,
        start: str,
        end: str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Return an OHLCV DataFrame indexed by tz-naive DatetimeIndex, sorted ascending,
        with float columns: open, high, low, close, volume.
        """
        ...
