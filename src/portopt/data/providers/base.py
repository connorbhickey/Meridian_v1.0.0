"""Abstract base class for data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

from portopt.data.models import Asset


class BaseDataProvider(ABC):
    """Interface for fetching market data from external sources."""

    @abstractmethod
    def get_prices(
        self, symbol: str, start: date, end: date | None = None
    ) -> pd.DataFrame:
        """Fetch OHLCV price data for a symbol.

        Returns a DataFrame with DatetimeIndex and columns:
        Open, High, Low, Close, Volume, Adj Close
        """

    @abstractmethod
    def get_asset_info(self, symbol: str) -> Asset:
        """Fetch basic asset metadata."""

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Fetch the latest price for a symbol."""

    def get_multiple_prices(
        self, symbols: list[str], start: date, end: date | None = None
    ) -> dict[str, pd.DataFrame]:
        """Fetch prices for multiple symbols. Default: sequential calls."""
        results = {}
        for sym in symbols:
            try:
                results[sym] = self.get_prices(sym, start, end)
            except Exception:
                results[sym] = pd.DataFrame()
        return results
