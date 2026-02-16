"""BacktestEngine: high-level orchestrator for backtesting workflows.

Provides a unified interface that ties together:
- Data preparation
- Optimization method selection
- Walk-forward analysis
- Performance metrics computation
- Attribution analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from portopt.backtest.attribution import (
    BrinsonAttribution,
    FactorAttribution,
    brinson_attribution,
    contribution_analysis,
    cumulative_contribution,
    factor_attribution,
)
from portopt.backtest.costs import BaseCostModel, ZeroCost, create_cost_model
from portopt.backtest.rebalancer import RebalanceSchedule
from portopt.backtest.results import SingleRunResult, WalkForwardResult
from portopt.backtest.runner import OptimizerFn, make_optimizer_fn, run_backtest
from portopt.backtest.validation import (
    CVResult,
    PBOResult,
    probability_of_backtest_overfitting,
    time_series_cv,
    train_test_split,
)
from portopt.backtest.walk_forward import WalkForwardConfig, run_walk_forward
from portopt.constants import CostModel, CovEstimator, RebalanceFreq, ReturnEstimator
from portopt.engine.constraints import PortfolioConstraints
from portopt.engine.metrics import compute_all_metrics


@dataclass
class BacktestConfig:
    """Full configuration for a backtest run.

    Attributes:
        method: Optimization method name.
        rebalance_freq: How often to rebalance.
        cost_model_type: Transaction cost model type.
        cost_params: Parameters for the cost model.
        cov_estimator: Covariance estimation method.
        return_estimator: Return estimation method.
        constraints: Portfolio constraints.
        lookback: Lookback window in days (None = all available).
        risk_free_rate: Annual risk-free rate.
        initial_value: Starting portfolio value.
        drift_threshold: Drift-based rebalance threshold (None = disabled).
        walk_forward: If not None, run walk-forward analysis with this config.
    """
    method: str = "max_sharpe"
    rebalance_freq: RebalanceFreq = RebalanceFreq.MONTHLY
    cost_model_type: str = "zero"
    cost_params: dict = field(default_factory=dict)
    cov_estimator: CovEstimator = CovEstimator.SAMPLE
    return_estimator: ReturnEstimator = ReturnEstimator.HISTORICAL_MEAN
    constraints: PortfolioConstraints | None = None
    lookback: int | None = None
    risk_free_rate: float = 0.04
    initial_value: float = 1_000_000.0
    drift_threshold: float | None = None
    walk_forward: WalkForwardConfig | None = None


@dataclass
class BacktestOutput:
    """Complete output from a backtest run."""
    config: BacktestConfig
    result: SingleRunResult | None = None
    walk_forward_result: WalkForwardResult | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    benchmark_metrics: dict[str, float] = field(default_factory=dict)
    attribution: BrinsonAttribution | None = None
    factor_attribution: FactorAttribution | None = None
    contribution: pd.DataFrame | None = None
    cv_result: CVResult | None = None
    pbo_result: PBOResult | None = None


class BacktestEngine:
    """High-level backtesting orchestrator.

    Usage:
        engine = BacktestEngine(prices, config)
        output = engine.run()
        output = engine.run_with_benchmark(benchmark_prices)
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        config: BacktestConfig | None = None,
    ):
        self.prices = prices
        self.config = config or BacktestConfig()
        self._optimizer_fn: OptimizerFn | None = None
        self._cost_model: BaseCostModel | None = None
        self._schedule: RebalanceSchedule | None = None

    def _build_optimizer(self) -> OptimizerFn:
        """Create optimizer function from config."""
        if self._optimizer_fn is not None:
            return self._optimizer_fn

        self._optimizer_fn = make_optimizer_fn(
            method=self.config.method,
            constraints=self.config.constraints,
            cov_estimator=self.config.cov_estimator,
            return_estimator=self.config.return_estimator,
            lookback=self.config.lookback,
            risk_free_rate=self.config.risk_free_rate,
        )
        return self._optimizer_fn

    def _build_cost_model(self) -> BaseCostModel:
        """Create cost model from config."""
        if self._cost_model is not None:
            return self._cost_model

        self._cost_model = create_cost_model(
            self.config.cost_model_type,
            **self.config.cost_params,
        )
        return self._cost_model

    def _build_schedule(self) -> RebalanceSchedule:
        """Create rebalance schedule from config."""
        if self._schedule is not None:
            return self._schedule

        self._schedule = RebalanceSchedule(
            frequency=self.config.rebalance_freq,
            drift_threshold=self.config.drift_threshold,
        )
        return self._schedule

    def run(self) -> BacktestOutput:
        """Run the backtest and compute metrics.

        If walk_forward config is set, runs walk-forward analysis.
        Otherwise runs a single backtest.
        """
        optimizer_fn = self._build_optimizer()
        cost_model = self._build_cost_model()
        schedule = self._build_schedule()

        output = BacktestOutput(config=self.config)

        if self.config.walk_forward is not None:
            # Walk-forward mode
            wf_result = run_walk_forward(
                prices=self.prices,
                optimizer_fn=optimizer_fn,
                config=self.config.walk_forward,
                schedule=schedule,
                cost_model=cost_model,
                initial_value=self.config.initial_value,
            )
            output.walk_forward_result = wf_result

            # Compute metrics on aggregate OOS returns
            if len(wf_result.aggregate_returns) > 0:
                output.metrics = compute_all_metrics(
                    wf_result.aggregate_returns,
                    risk_free_rate=self.config.risk_free_rate,
                )
        else:
            # Single run mode
            result = run_backtest(
                prices=self.prices,
                optimizer_fn=optimizer_fn,
                schedule=schedule,
                cost_model=cost_model,
                initial_value=self.config.initial_value,
            )
            output.result = result

            # Compute metrics
            if len(result.daily_returns) > 0:
                output.metrics = compute_all_metrics(
                    result.daily_returns,
                    risk_free_rate=self.config.risk_free_rate,
                )

            # Contribution analysis
            if result.weights_history:
                asset_returns = self.prices.pct_change().iloc[1:]
                output.contribution = contribution_analysis(
                    result.weights_history, asset_returns,
                )

        return output

    def run_with_benchmark(
        self,
        benchmark_prices: pd.Series,
        sector_map: dict[str, str] | None = None,
    ) -> BacktestOutput:
        """Run backtest and compare against benchmark.

        Args:
            benchmark_prices: Benchmark price series.
            sector_map: {symbol: sector} for Brinson attribution.

        Returns:
            BacktestOutput with benchmark metrics and attribution.
        """
        output = self.run()

        # Compute benchmark returns
        bench_returns = benchmark_prices.pct_change().dropna().values

        # Add benchmark metrics
        if output.result is not None and len(output.result.daily_returns) > 0:
            output.metrics = compute_all_metrics(
                output.result.daily_returns,
                benchmark_returns=bench_returns,
                risk_free_rate=self.config.risk_free_rate,
            )
            output.benchmark_metrics = compute_all_metrics(
                bench_returns,
                risk_free_rate=self.config.risk_free_rate,
            )

        return output

    def run_cross_validation(
        self,
        n_folds: int = 5,
        purge_days: int = 5,
        embargo_days: int = 5,
    ) -> CVResult:
        """Run time-series cross-validation.

        Args:
            n_folds: Number of CV folds.
            purge_days: Days to purge around train/test boundary.
            embargo_days: Days of embargo after test set.

        Returns:
            CVResult with per-fold and aggregate scores.
        """
        optimizer_fn = self._build_optimizer()
        cost_model = self._build_cost_model()
        schedule = self._build_schedule()

        return time_series_cv(
            prices=self.prices,
            optimizer_fn=optimizer_fn,
            n_folds=n_folds,
            purge_days=purge_days,
            embargo_days=embargo_days,
            schedule=schedule,
            cost_model=cost_model,
        )
