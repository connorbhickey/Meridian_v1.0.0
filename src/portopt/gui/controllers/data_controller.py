"""Controller for market data operations (prices, returns, covariance)."""

from __future__ import annotations

import logging
from datetime import date, timedelta

from PySide6.QtCore import QObject, Signal

from portopt.data.manager import DataManager
from portopt.utils.threading import run_in_thread

logger = logging.getLogger(__name__)


class DataController(QObject):
    """Manages market data fetching with background threading."""

    prices_loaded = Signal(dict)           # {symbol: DataFrame}
    returns_loaded = Signal(object)        # DataFrame
    covariance_loaded = Signal(object)     # DataFrame
    asset_info_loaded = Signal(object)     # Asset
    current_price_loaded = Signal(str, float)  # symbol, price
    error = Signal(str)
    status_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_manager = DataManager()
        self._worker = None

    def fetch_prices(self, symbols: list[str], start: date, end: date | None = None):
        """Fetch prices for multiple symbols in background."""
        self.status_changed.emit(f"Fetching prices for {len(symbols)} symbols...")
        self._worker = run_in_thread(
            self.data_manager.get_multiple_prices, symbols, start, end,
            on_result=self._on_prices,
            on_error=self._on_error,
        )

    def _on_prices(self, result):
        loaded = sum(1 for df in result.values() if not df.empty)
        self.status_changed.emit(f"Loaded prices for {loaded}/{len(result)} symbols")
        self.prices_loaded.emit(result)

    def fetch_returns(self, symbols: list[str], start: date, end: date | None = None):
        """Fetch log returns in background."""
        self.status_changed.emit("Computing returns...")
        self._worker = run_in_thread(
            self.data_manager.get_returns, symbols, start, end,
            on_result=lambda r: (self.returns_loaded.emit(r), self.status_changed.emit("Returns ready")),
            on_error=self._on_error,
        )

    def fetch_covariance(self, symbols: list[str], start: date, end: date | None = None):
        """Fetch covariance matrix in background."""
        self.status_changed.emit("Computing covariance...")
        self._worker = run_in_thread(
            self.data_manager.get_covariance, symbols, start, end,
            on_result=lambda r: (self.covariance_loaded.emit(r), self.status_changed.emit("Covariance ready")),
            on_error=self._on_error,
        )

    def fetch_asset_info(self, symbol: str):
        """Fetch asset metadata in background."""
        self._worker = run_in_thread(
            self.data_manager.get_asset_info, symbol,
            on_result=self.asset_info_loaded.emit,
            on_error=self._on_error,
        )

    def fetch_current_price(self, symbol: str):
        """Fetch latest price for a single symbol."""
        self._worker = run_in_thread(
            lambda: (symbol, self.data_manager.get_current_price(symbol)),
            on_result=lambda r: self.current_price_loaded.emit(r[0], r[1]),
            on_error=self._on_error,
        )

    def _on_error(self, msg: str):
        logger.error("Data error: %s", msg)
        self.status_changed.emit("Data error")
        self.error.emit(msg)

    def get_cache_size(self) -> float:
        return self.data_manager.cache.get_cache_size_mb()

    def close(self):
        self.data_manager.close()
