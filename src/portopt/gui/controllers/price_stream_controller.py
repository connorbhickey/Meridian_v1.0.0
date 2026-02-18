"""Real-time price streaming controller using QTimer polling."""

from __future__ import annotations

import logging
from datetime import datetime

from PySide6.QtCore import QObject, QTimer, Signal

from portopt.utils.threading import run_in_thread

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_MS = 30_000  # 30 seconds


class PriceStreamController(QObject):
    """Polls current prices at a configurable interval.

    Emits price updates for the portfolio panel, watchlist, and status bar.
    """

    prices_updated = Signal(dict)           # {symbol: float}
    portfolio_value_updated = Signal(float)  # total portfolio value
    error = Signal(str)
    status_changed = Signal(str)

    def __init__(self, data_controller, parent=None):
        super().__init__(parent)
        self._data_controller = data_controller
        self._symbols: list[str] = []
        self._holdings: dict[str, float] = {}  # {symbol: quantity}
        self._last_prices: dict[str, float] = {}
        self._running = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._interval_ms = DEFAULT_INTERVAL_MS

    @property
    def is_streaming(self) -> bool:
        return self._running

    def set_symbols(self, symbols: list[str]):
        """Set the symbols to track."""
        self._symbols = list(symbols)

    def set_holdings(self, holdings: dict[str, float]):
        """Set holdings for portfolio value computation.

        Args:
            holdings: {symbol: quantity}
        """
        self._holdings = dict(holdings)
        if not self._symbols:
            self._symbols = list(holdings.keys())

    def set_interval(self, seconds: int):
        """Change polling interval."""
        self._interval_ms = max(5000, seconds * 1000)  # Min 5s
        if self._running:
            self._timer.setInterval(self._interval_ms)

    def start(self):
        """Start streaming."""
        if not self._symbols:
            return
        self._running = True
        self._timer.start(self._interval_ms)
        self.status_changed.emit(f"Price streaming started ({self._interval_ms // 1000}s)")
        # Do an immediate poll
        self._poll()

    def stop(self):
        """Stop streaming."""
        self._running = False
        self._timer.stop()
        self.status_changed.emit("Price streaming stopped")

    def _poll(self):
        """Fetch current prices for all tracked symbols."""
        if not self._symbols:
            return

        dm = self._data_controller.data_manager

        def _fetch():
            prices = {}
            for sym in self._symbols:
                try:
                    p = dm.get_current_price(sym)
                    if p > 0:
                        prices[sym] = p
                except Exception:
                    pass
            return prices

        def _on_result(prices: dict):
            self._last_prices.update(prices)
            self.prices_updated.emit(prices)

            # Compute portfolio value if holdings are set
            if self._holdings:
                total = sum(
                    self._last_prices.get(sym, 0) * qty
                    for sym, qty in self._holdings.items()
                )
                if total > 0:
                    self.portfolio_value_updated.emit(total)

        def _on_error(msg):
            logger.warning("Price poll failed: %s", msg)

        run_in_thread(_fetch, on_result=_on_result, on_error=_on_error)
