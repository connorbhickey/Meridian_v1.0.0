"""Controller for Monte Carlo simulation — wires GUI to engine layer."""

from __future__ import annotations

import logging

import pandas as pd
from PySide6.QtCore import QObject, Signal

from portopt.constants import CovEstimator, ReturnEstimator
from portopt.data.models import MonteCarloConfig, MonteCarloResult
from portopt.engine.monte_carlo import run_monte_carlo
from portopt.engine.returns import estimate_returns
from portopt.engine.risk import estimate_covariance
from portopt.utils.threading import run_in_thread

logger = logging.getLogger(__name__)


class MonteCarloController(QObject):
    """Manages Monte Carlo simulation with background threading."""

    simulation_complete = Signal(object)   # MonteCarloResult
    error = Signal(str)
    status_changed = Signal(str)
    progress = Signal(str)
    running_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._running = False
        self._prices: pd.DataFrame | None = None
        self._weights: dict[str, float] | None = None

    # ── Data Management ────────────────────────────────────────────────

    def set_prices(self, prices: pd.DataFrame):
        """Set price data (close prices, same as used for optimization)."""
        self._prices = prices

    def set_weights(self, weights: dict[str, float]):
        """Set portfolio weights from last optimization."""
        self._weights = weights

    @property
    def is_ready(self) -> bool:
        """True if we have both prices and weights."""
        return self._prices is not None and self._weights is not None

    # ── Run Simulation ─────────────────────────────────────────────────

    def run_simulation(self, config: MonteCarloConfig):
        """Launch Monte Carlo simulation in background thread."""
        if self._running:
            self.error.emit("Simulation already running. Please wait.")
            return

        if self._prices is None or self._prices.empty:
            self.error.emit("No price data available. Run optimization first.")
            return

        if not self._weights:
            self.error.emit("No portfolio weights. Run optimization first.")
            return

        self._running = True
        self.running_changed.emit(True)
        self.status_changed.emit("Running Monte Carlo simulation...")

        self._worker = run_in_thread(
            self._do_simulation, config,
            on_result=self._on_simulation_done,
            on_error=self._on_error,
        )

    def _do_simulation(self, config: MonteCarloConfig) -> MonteCarloResult:
        """Run simulation in background thread."""
        prices = self._prices
        weights = self._weights

        self.progress.emit("Estimating returns...")
        mu = estimate_returns(prices, method=ReturnEstimator.HISTORICAL_MEAN)

        self.progress.emit("Estimating covariance...")
        cov = estimate_covariance(prices, method=CovEstimator.SAMPLE)

        # Historical returns for bootstrap
        historical_returns = prices.pct_change().dropna()

        def _progress(msg: str):
            self.progress.emit(msg)

        return run_monte_carlo(
            weights=weights,
            mu=mu,
            cov=cov,
            config=config,
            historical_returns=historical_returns,
            progress_cb=_progress,
        )

    def _on_simulation_done(self, result: MonteCarloResult):
        """Handle simulation results on main thread."""
        self._running = False
        self.running_changed.emit(False)
        self.status_changed.emit(
            f"Monte Carlo complete: {result.n_sims} sims | "
            f"Median terminal=${result.metadata.get('median_terminal', 0):,.0f} | "
            f"P(shortfall)={result.shortfall_probability:.1%}"
        )
        self.simulation_complete.emit(result)

    def _on_error(self, msg: str):
        self._running = False
        self.running_changed.emit(False)
        summary = msg.split("\n")[0] if "\n" in msg else msg
        logger.error("Monte Carlo error: %s", msg)
        self.status_changed.emit("Monte Carlo failed")
        self.error.emit(f"Monte Carlo error: {summary}")
