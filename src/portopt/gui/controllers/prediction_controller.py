"""Controller for stock prediction — wires GUI to prediction engine."""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal

from portopt.engine.prediction.data_provider import fetch_prediction_data
from portopt.engine.prediction.ensemble import PredictionResult, run_prediction
from portopt.utils.threading import run_in_thread

logger = logging.getLogger(__name__)


class PredictionController(QObject):
    """Manages stock prediction workflows with background threading."""

    # Signals
    prediction_complete = Signal(object)  # PredictionResult
    error = Signal(str)
    status_changed = Signal(str)
    running_changed = Signal(bool)

    def __init__(self, data_controller=None, parent=None):
        super().__init__(parent)
        self.data_controller = data_controller
        self._worker = None
        self._running = False
        self._last_result: PredictionResult | None = None

    @property
    def last_result(self) -> PredictionResult | None:
        return self._last_result

    def run_prediction(self, symbol: str, horizon_days: int = 252):
        """Run stock prediction for the given symbol.

        Fetches data and runs the 25-method ensemble in a background thread.
        """
        if self._running:
            self.error.emit("Prediction already running. Please wait.")
            return

        if not symbol or not symbol.strip():
            self.error.emit("No symbol provided.")
            return

        self._running = True
        self.running_changed.emit(True)
        self.status_changed.emit(f"Running prediction for {symbol.upper()}...")

        self._worker = run_in_thread(
            self._do_prediction, symbol.strip().upper(), horizon_days,
            on_result=self._on_prediction_done,
            on_error=self._on_error,
        )

    def _do_prediction(self, symbol: str, horizon_days: int) -> PredictionResult:
        """Run prediction in background thread."""
        # Get FRED provider if available
        fred_provider = None
        if self.data_controller:
            dm = getattr(self.data_controller, "data_manager", None)
            if dm:
                fred_provider = getattr(dm, "_fred_provider", None)

        # Phase 1: Fetch data (~2-3s)
        logger.info("Fetching prediction data for %s...", symbol)
        data = fetch_prediction_data(
            symbol, horizon_days=horizon_days,
            fred_provider=fred_provider,
        )

        # Phase 2: Run ensemble (~1-2s)
        logger.info("Running 25-method ensemble for %s...", symbol)
        is_etf = data.get("isETF", False)
        result = run_prediction(data, is_etf=is_etf)

        return result

    def _on_prediction_done(self, result: PredictionResult):
        """Handle prediction results on main thread."""
        self._running = False
        self.running_changed.emit(False)
        self._last_result = result

        direction = "▲" if result.ensemble_return_pct > 0 else "▼"
        self.status_changed.emit(
            f"Prediction complete: {result.symbol} | "
            f"${result.ensemble_point:.2f} ({direction}{abs(result.ensemble_return_pct):.1f}%) | "
            f"Confidence: {result.signals.get('kelly', {}).get('label', 'N/A')}"
        )
        self.prediction_complete.emit(result)

    def _on_error(self, msg: str):
        self._running = False
        self.running_changed.emit(False)
        summary = msg.split("\n")[0] if "\n" in msg else msg
        logger.error("Prediction error: %s", msg)
        self.status_changed.emit("Prediction failed")
        self.error.emit(f"Prediction error: {summary}")
