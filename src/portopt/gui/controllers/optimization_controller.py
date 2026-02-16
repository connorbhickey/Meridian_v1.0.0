"""Controller for portfolio optimization — wires GUI to engine layer."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, Signal

from portopt.constants import (
    CovEstimator, LinkageMethod, OptMethod, ReturnEstimator, RiskMeasure,
)
from portopt.data.models import OptimizationResult
from portopt.engine.constraints import PortfolioConstraints
from portopt.engine.optimization.mean_variance import MeanVarianceOptimizer
from portopt.engine.optimization.hrp import hrp_optimize
from portopt.engine.optimization.herc import herc_optimize
from portopt.engine.optimization.black_litterman import BlackLittermanModel, BLView
from portopt.engine.optimization.tic import theory_implied_correlation
from portopt.engine.returns import estimate_returns
from portopt.engine.risk import estimate_covariance
from portopt.engine.network.mst import compute_mst
from portopt.utils.threading import run_in_thread

logger = logging.getLogger(__name__)


class OptimizationController(QObject):
    """Manages optimization workflows with background threading."""

    # Signals
    optimization_complete = Signal(object)     # OptimizationResult
    frontier_complete = Signal(object, object)  # risks array, returns array
    frontier_assets = Signal(list, object, object)  # symbols, risks, returns
    metrics_ready = Signal(dict)               # metrics dict
    risk_metrics_ready = Signal(dict)          # risk-specific metrics
    correlation_ready = Signal(object, list)   # corr matrix, labels
    mst_ready = Signal(list, list, dict)       # nodes, edges, sectors
    dendrogram_ready = Signal(object, list)    # linkage_matrix, labels
    error = Signal(str)
    status_changed = Signal(str)
    progress = Signal(str)
    running_changed = Signal(bool)

    def __init__(self, data_controller, parent=None):
        super().__init__(parent)
        self.data_controller = data_controller
        self._worker = None
        self._running = False
        self._prices: pd.DataFrame | None = None
        self._symbols: list[str] = []
        self._bl_views: list[BLView] = []
        self._last_result: OptimizationResult | None = None

    # ── Data Management ───────────────────────────────────────────────

    def set_prices(self, prices: pd.DataFrame):
        """Set the price data for optimization."""
        self._prices = prices
        self._symbols = list(prices.columns)

    def set_symbols(self, symbols: list[str]):
        """Set target symbols — will fetch prices before optimizing."""
        self._symbols = list(symbols)

    def set_bl_views(self, views: list[BLView]):
        """Set Black-Litterman views."""
        self._bl_views = views

    @property
    def last_result(self) -> OptimizationResult | None:
        return self._last_result

    # ── Run Optimization ──────────────────────────────────────────────

    def run_optimization(self, config: dict):
        """Run portfolio optimization with the given configuration.

        config keys:
            method: OptMethod enum value
            cov_estimator: CovEstimator enum value
            return_estimator: ReturnEstimator enum value
            linkage: LinkageMethod enum value (for HRP/HERC/TIC)
            risk_measure: RiskMeasure enum value (for HERC)
            risk_free_rate: float
            risk_aversion: float
            target_return: float | None
            target_risk: float | None
            long_only: bool
            min_weight: float
            max_weight: float
        """
        if self._running:
            self.error.emit("Optimization already running. Please wait.")
            return

        if self._prices is None or self._prices.empty:
            self.error.emit("No price data available. Load prices first.")
            return

        self._running = True
        self.running_changed.emit(True)
        self.status_changed.emit("Running optimization...")
        self.progress.emit("Estimating returns and covariance...")

        self._worker = run_in_thread(
            self._do_optimization, config,
            on_result=self._on_optimization_done,
            on_error=self._on_error,
        )

    def _do_optimization(self, config: dict) -> dict:
        """Run optimization in background thread. Returns dict of results."""
        prices = self._prices
        method = config.get("method", OptMethod.MAX_SHARPE)
        cov_est = config.get("cov_estimator", CovEstimator.SAMPLE)
        ret_est = config.get("return_estimator", ReturnEstimator.HISTORICAL_MEAN)
        linkage_method = config.get("linkage", LinkageMethod.SINGLE)
        risk_measure = config.get("risk_measure", RiskMeasure.VARIANCE)
        risk_free = config.get("risk_free_rate", 0.04)
        risk_aversion = config.get("risk_aversion", 1.0)
        target_ret = config.get("target_return")
        target_risk = config.get("target_risk")
        long_only = config.get("long_only", True)
        min_weight = config.get("min_weight", 0.0)
        max_weight = config.get("max_weight", 1.0)

        # Estimate covariance and returns
        self.progress.emit("Estimating covariance matrix...")
        cov = estimate_covariance(prices, method=cov_est)
        self.progress.emit("Estimating expected returns...")
        mu = estimate_returns(prices, method=ret_est, risk_free_rate=risk_free)

        # Build constraints
        constraints = PortfolioConstraints(
            long_only=long_only,
            min_weight=min_weight,
            max_weight=max_weight,
            risk_aversion=risk_aversion,
            target_return=target_ret,
            target_risk=target_risk,
        )

        symbols = list(prices.columns)
        result = None
        linkage_matrix = None

        # Dispatch to correct optimizer
        self.progress.emit(f"Running {method.name} optimizer...")
        if method == OptMethod.HRP:
            returns_df = prices.pct_change().dropna()
            result = hrp_optimize(
                covariance=cov,
                linkage_method=linkage_method,
                risk_measure=risk_measure,
                returns=returns_df,
                long_only=long_only,
            )
            linkage_matrix = result.metadata.get("linkage_matrix")

        elif method == OptMethod.HERC:
            returns_df = prices.pct_change().dropna()
            result = herc_optimize(
                covariance=cov,
                linkage_method=linkage_method,
                risk_measure=risk_measure,
                returns=returns_df,
            )
            linkage_matrix = result.metadata.get("linkage_matrix")

        elif method == OptMethod.BLACK_LITTERMAN:
            bl = BlackLittermanModel(
                covariance=cov,
                risk_aversion=risk_aversion,
                risk_free_rate=risk_free,
            )
            if self._bl_views:
                posterior_mu = bl.posterior_returns(self._bl_views)
            else:
                posterior_mu = bl.equilibrium_returns()

            optimizer = MeanVarianceOptimizer(
                expected_returns=posterior_mu,
                covariance=cov,
                constraints=constraints,
                method=OptMethod.MAX_SHARPE,
            )
            result = optimizer.optimize()
            result.method = "black_litterman"
            result.metadata["bl_views"] = len(self._bl_views)

        elif method == OptMethod.TIC:
            tic_corr = theory_implied_correlation(
                covariance=cov,
                linkage_method=linkage_method,
            )
            # Reconstruct covariance from TIC correlation
            from portopt.engine.risk import corr_to_cov
            std = np.sqrt(np.diag(cov.values))
            tic_cov = pd.DataFrame(
                corr_to_cov(tic_corr.values, std),
                index=cov.index, columns=cov.columns,
            )
            optimizer = MeanVarianceOptimizer(
                expected_returns=mu,
                covariance=tic_cov,
                constraints=constraints,
                method=OptMethod.MAX_SHARPE,
            )
            result = optimizer.optimize()
            result.method = "tic"

        else:
            # MVO methods
            optimizer = MeanVarianceOptimizer(
                expected_returns=mu,
                covariance=cov,
                constraints=constraints,
                method=method,
            )
            result = optimizer.optimize()

        # Compute efficient frontier for MVO-like methods
        self.progress.emit("Computing efficient frontier...")
        frontier_risks = None
        frontier_returns = None
        if method not in (OptMethod.HRP, OptMethod.HERC):
            try:
                frontier_opt = MeanVarianceOptimizer(
                    expected_returns=mu, covariance=cov,
                    constraints=constraints, method=OptMethod.MAX_SHARPE,
                )
                frontier_points = frontier_opt.efficient_frontier(n_points=50)
                frontier_risks = np.array([p.volatility for p in frontier_points])
                frontier_returns = np.array([p.expected_return for p in frontier_points])
            except Exception as e:
                logger.warning("Frontier computation failed: %s", e)

        # Per-asset risk/return
        self.progress.emit("Computing correlation, MST & risk metrics...")
        asset_vols = np.sqrt(np.diag(cov.values))
        asset_mus = mu.values

        # Correlation matrix
        from portopt.engine.risk import cov_to_corr
        corr = cov_to_corr(cov)

        # MST
        mst_data = None
        try:
            mst = compute_mst(cov)
            edges = [(u, v, d["weight"]) for u, v, d in mst.edges(data=True)]
            mst_data = {"nodes": symbols, "edges": edges}
        except Exception as e:
            logger.warning("MST computation failed: %s", e)

        return {
            "result": result,
            "frontier_risks": frontier_risks,
            "frontier_returns": frontier_returns,
            "symbols": symbols,
            "asset_vols": asset_vols,
            "asset_mus": asset_mus,
            "correlation": corr,
            "linkage_matrix": linkage_matrix,
            "mst": mst_data,
            "risk_free_rate": risk_free,
            "_mu": mu,
            "_cov": cov,
        }

    def _on_optimization_done(self, output: dict):
        """Handle optimization results on main thread."""
        self._running = False
        self.running_changed.emit(False)
        self.progress.emit("Optimization complete ✓")
        result: OptimizationResult = output["result"]
        self._last_result = result
        self._last_mu = output.get("_mu")
        self._last_cov = output.get("_cov")

        self.optimization_complete.emit(result)
        self.status_changed.emit(
            f"Optimization complete: {result.method} | "
            f"Return={result.expected_return:.2%} | Vol={result.volatility:.2%} | "
            f"Sharpe={result.sharpe_ratio:.3f}"
        )

        # Frontier
        if output.get("frontier_risks") is not None:
            self.frontier_complete.emit(output["frontier_risks"], output["frontier_returns"])

        # Per-asset
        symbols = output["symbols"]
        self.frontier_assets.emit(symbols, output["asset_vols"], output["asset_mus"])

        # Correlation
        corr = output["correlation"]
        self.correlation_ready.emit(corr.values, list(corr.index))

        # MST
        if output.get("mst"):
            self.mst_ready.emit(
                output["mst"]["nodes"],
                output["mst"]["edges"],
                {},
            )

        # Dendrogram
        if output.get("linkage_matrix") is not None:
            self.dendrogram_ready.emit(output["linkage_matrix"], symbols)

        # Risk metrics from the result
        risk_metrics = {
            "var_95": -1.645 * result.volatility / np.sqrt(252),
            "cvar_95": -2.063 * result.volatility / np.sqrt(252),
            "max_drawdown": 0.0,
            "annual_volatility": result.volatility,
            "downside_vol": result.volatility * 0.7,  # approximate
            "beta": 1.0,
        }
        self.risk_metrics_ready.emit(risk_metrics)

    def _on_error(self, msg: str):
        self._running = False
        self.running_changed.emit(False)
        logger.error("Optimization error: %s", msg)
        self.status_changed.emit("Optimization failed")
        self.error.emit(f"Optimization error: {msg}")

    # ── Fetch Prices + Optimize ───────────────────────────────────────

    def fetch_and_optimize(self, symbols: list[str], config: dict,
                           lookback_days: int = 756):
        """Fetch prices for symbols then run optimization."""
        self._symbols = symbols
        self.status_changed.emit(f"Fetching prices for {len(symbols)} symbols...")

        start = date.today() - timedelta(days=lookback_days)
        self._pending_config = config

        self._worker = run_in_thread(
            self.data_controller.data_manager.get_multiple_prices,
            symbols, start,
            on_result=self._on_prices_for_opt,
            on_error=self._on_error,
        )

    def _on_prices_for_opt(self, prices_dict: dict):
        """Prices fetched — build DataFrame and run optimization."""
        # Build close-price DataFrame
        frames = {}
        for sym, df in prices_dict.items():
            if not df.empty and "Close" in df.columns:
                frames[sym] = df["Close"]
            elif not df.empty and "Adj Close" in df.columns:
                frames[sym] = df["Adj Close"]

        if not frames:
            self.error.emit("No price data retrieved for any symbol")
            return

        self._prices = pd.DataFrame(frames).dropna()
        self._symbols = list(self._prices.columns)
        self.status_changed.emit(f"Prices loaded: {len(self._symbols)} symbols, {len(self._prices)} days")

        # Now run optimization with pending config
        config = getattr(self, "_pending_config", {})
        self.run_optimization(config)
