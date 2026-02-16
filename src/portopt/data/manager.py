"""DataManager — cache-then-fetch coordinator for all market data."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from portopt.data.cache import CacheDB
from portopt.data.models import Asset
from portopt.data.providers.base import BaseDataProvider
from portopt.data.providers.yfinance_provider import YFinanceProvider
from portopt.data.providers.alphavantage_provider import AlphaVantageProvider

logger = logging.getLogger(__name__)


class DataManager:
    """Coordinates data retrieval: checks cache first, then fetches from providers."""

    def __init__(self):
        self.cache = CacheDB()
        self._primary: BaseDataProvider = YFinanceProvider()
        self._fallback: BaseDataProvider | None = None
        av = AlphaVantageProvider()
        if av.available:
            self._fallback = av

    def get_prices(
        self, symbol: str, start: date, end: date | None = None
    ) -> pd.DataFrame:
        """Get price data, using cache when available and fetching deltas."""
        end = end or date.today()

        # Check cache
        cached = self.cache.get_prices(symbol, start, end)
        latest_cached = self.cache.get_latest_date(symbol)

        # If cache covers the full range, return it
        if not cached.empty and latest_cached and latest_cached >= end - timedelta(days=3):
            return cached

        # Fetch from provider (from day after last cached, or from start)
        fetch_start = (latest_cached + timedelta(days=1)) if latest_cached else start
        df = self._fetch_with_fallback(symbol, fetch_start, end)

        # Store in cache
        if not df.empty:
            self.cache.store_prices(symbol, df)

        # Return combined data
        if not cached.empty and not df.empty:
            combined = pd.concat([cached, df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
            return combined
        return df if not df.empty else cached

    def get_multiple_prices(
        self, symbols: list[str], start: date, end: date | None = None
    ) -> dict[str, pd.DataFrame]:
        """Fetch prices for multiple symbols efficiently."""
        end = end or date.today()
        results = {}
        to_fetch = []

        # Check cache first
        for sym in symbols:
            cached = self.cache.get_prices(sym, start, end)
            latest = self.cache.get_latest_date(sym)
            if not cached.empty and latest and latest >= end - timedelta(days=3):
                results[sym] = cached
            else:
                to_fetch.append(sym)

        # Batch fetch missing symbols
        if to_fetch:
            fetched = self._primary.get_multiple_prices(to_fetch, start, end)
            for sym, df in fetched.items():
                if not df.empty:
                    self.cache.store_prices(sym, df)
                    existing = results.get(sym, pd.DataFrame())
                    if not existing.empty:
                        combined = pd.concat([existing, df])
                        combined = combined[~combined.index.duplicated(keep="last")]
                        combined.sort_index(inplace=True)
                        results[sym] = combined
                    else:
                        results[sym] = df
                elif sym not in results:
                    results[sym] = pd.DataFrame()

        return results

    def get_returns(
        self, symbols: list[str], start: date, end: date | None = None
    ) -> pd.DataFrame:
        """Get log returns for multiple symbols as a DataFrame (columns=symbols)."""
        prices = self.get_close_prices(symbols, start, end)
        if prices.empty:
            return pd.DataFrame()
        return np.log(prices / prices.shift(1)).dropna()

    def get_close_prices(
        self, symbols: list[str], start: date, end: date | None = None
    ) -> pd.DataFrame:
        """Get adjusted close prices for multiple symbols as a DataFrame."""
        all_prices = self.get_multiple_prices(symbols, start, end)
        close_data = {}
        for sym, df in all_prices.items():
            if not df.empty:
                col = "Adj Close" if "Adj Close" in df.columns else "Close"
                close_data[sym] = df[col]
        if not close_data:
            return pd.DataFrame()
        result = pd.DataFrame(close_data)
        result.sort_index(inplace=True)
        return result.dropna(how="all")

    def get_covariance(
        self, symbols: list[str], start: date, end: date | None = None
    ) -> pd.DataFrame:
        """Compute sample covariance matrix from returns."""
        returns = self.get_returns(symbols, start, end)
        if returns.empty:
            return pd.DataFrame()
        return returns.cov()

    def get_asset_info(self, symbol: str) -> Asset:
        """Get asset info, checking cache first."""
        cached = self.cache.get_asset(symbol)
        if cached:
            return Asset(
                symbol=cached["symbol"],
                name=cached.get("name", ""),
                sector=cached.get("sector", ""),
                exchange=cached.get("exchange", ""),
                currency=cached.get("currency", "USD"),
            )
        try:
            asset = self._primary.get_asset_info(symbol)
            self.cache.store_asset(
                symbol=asset.symbol, name=asset.name,
                asset_type=asset.asset_type.name, sector=asset.sector,
                exchange=asset.exchange, currency=asset.currency,
            )
            return asset
        except Exception as e:
            logger.warning("Failed to get asset info for %s: %s", symbol, e)
            return Asset(symbol=symbol)

    def get_current_price(self, symbol: str) -> float:
        """Get the latest price for a symbol."""
        return self._fetch_with_fallback_price(symbol)

    def _fetch_with_fallback(
        self, symbol: str, start: date, end: date
    ) -> pd.DataFrame:
        """Try primary provider, fall back if it fails."""
        try:
            df = self._primary.get_prices(symbol, start, end)
            if not df.empty:
                return df
        except Exception as e:
            logger.warning("Primary provider failed for %s: %s", symbol, e)

        if self._fallback:
            try:
                return self._fallback.get_prices(symbol, start, end)
            except Exception as e:
                logger.warning("Fallback provider also failed for %s: %s", symbol, e)

        return pd.DataFrame()

    def _fetch_with_fallback_price(self, symbol: str) -> float:
        try:
            return self._primary.get_current_price(symbol)
        except Exception:
            pass
        if self._fallback:
            try:
                return self._fallback.get_current_price(symbol)
            except Exception:
                pass
        return 0.0

    def close(self):
        self.cache.close()
