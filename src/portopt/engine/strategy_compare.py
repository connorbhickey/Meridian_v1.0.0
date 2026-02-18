"""Strategy comparison engine — run multiple optimizations and compare statistically.

Supports:
- Running multiple optimization methods on the same universe
- Parameter sweeps (e.g., vary risk aversion or lookback)
- Bootstrap paired t-test for statistical significance
- Aggregated comparison results for the ComparisonPanel
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from portopt.constants import CovEstimator, OptMethod, ReturnEstimator
from portopt.data.models import OptimizationResult
from portopt.engine.constraints import PortfolioConstraints
from portopt.engine.metrics import compute_all_metrics
from portopt.engine.returns import estimate_returns
from portopt.engine.risk import estimate_covariance

logger = logging.getLogger(__name__)


# ── Result dataclasses ───────────────────────────────────────────────


@dataclass
class StrategyResult:
    """Result of a single strategy evaluation."""
    name: str
    opt_result: OptimizationResult
    backtest_returns: np.ndarray | None = None
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class PairwiseTest:
    """Result of a pairwise statistical test between two strategies."""
    strategy_a: str
    strategy_b: str
    test_statistic: float
    p_value: float
    mean_diff: float
    significant: bool  # at 5% level


@dataclass
class ComparisonResult:
    """Aggregated result from comparing multiple strategies."""
    strategies: list[StrategyResult]
    pairwise_tests: list[PairwiseTest] = field(default_factory=list)
    best_sharpe: str = ""
    best_return: str = ""
    lowest_vol: str = ""
    lowest_drawdown: str = ""


# ── Core comparison functions ────────────────────────────────────────


def compare_methods(
    prices: pd.DataFrame,
    methods: list[str] | None = None,
    constraints: PortfolioConstraints | None = None,
    cov_estimator: CovEstimator = CovEstimator.SAMPLE,
    return_estimator: ReturnEstimator = ReturnEstimator.HISTORICAL_MEAN,
    risk_free_rate: float = 0.04,
) -> ComparisonResult:
    """Run multiple optimization methods on the same data and compare.

    Args:
        prices: Price DataFrame (index=dates, columns=symbols).
        methods: List of method names. Defaults to common MVO methods.
        constraints: Portfolio constraints (shared across all methods).
        cov_estimator: Covariance estimation method.
        return_estimator: Return estimation method.
        risk_free_rate: Annual risk-free rate.

    Returns:
        ComparisonResult with all strategy results and pairwise tests.
    """
    if methods is None:
        methods = [
            "max_sharpe", "min_volatility", "max_diversification",
            "inverse_variance", "hrp",
        ]

    if constraints is None:
        constraints = PortfolioConstraints()

    # Estimate returns and covariance
    mu = estimate_returns(prices, return_estimator)
    cov = estimate_covariance(prices, cov_estimator)
    daily_returns = prices.pct_change().dropna()

    method_map = {
        "inverse_variance": OptMethod.INVERSE_VARIANCE,
        "min_volatility": OptMethod.MIN_VOLATILITY,
        "max_sharpe": OptMethod.MAX_SHARPE,
        "efficient_risk": OptMethod.EFFICIENT_RISK,
        "efficient_return": OptMethod.EFFICIENT_RETURN,
        "max_quadratic_utility": OptMethod.MAX_QUADRATIC_UTILITY,
        "max_diversification": OptMethod.MAX_DIVERSIFICATION,
        "max_decorrelation": OptMethod.MAX_DECORRELATION,
        "hrp": OptMethod.HRP,
        "herc": OptMethod.HERC,
    }

    strategies: list[StrategyResult] = []

    for method_name in methods:
        opt_method = method_map.get(method_name.lower())
        if opt_method is None:
            logger.warning("Unknown method '%s', skipping", method_name)
            continue

        try:
            opt_result = _run_single_optimization(
                mu, cov, constraints, opt_method, risk_free_rate,
            )
            # Compute historical portfolio returns using these weights
            weights_arr = np.array([
                opt_result.weights.get(s, 0.0) for s in prices.columns
            ])
            port_returns = daily_returns.values @ weights_arr

            metrics = compute_all_metrics(
                port_returns, risk_free_rate=risk_free_rate,
            )

            strategies.append(StrategyResult(
                name=method_name,
                opt_result=opt_result,
                backtest_returns=port_returns,
                metrics=metrics,
            ))
        except Exception as e:
            logger.warning("Method '%s' failed: %s", method_name, e)

    # Pairwise tests
    pairwise = _run_pairwise_tests(strategies)

    # Find best strategies
    result = ComparisonResult(
        strategies=strategies,
        pairwise_tests=pairwise,
    )
    if strategies:
        result.best_sharpe = max(
            strategies, key=lambda s: s.metrics.get("sharpe_ratio", -999)
        ).name
        result.best_return = max(
            strategies, key=lambda s: s.metrics.get("annualized_return", -999)
        ).name
        result.lowest_vol = min(
            strategies, key=lambda s: s.metrics.get("annualized_volatility", 999)
        ).name
        result.lowest_drawdown = max(
            strategies, key=lambda s: s.metrics.get("max_drawdown", -999)
        ).name

    return result


def parameter_sweep(
    prices: pd.DataFrame,
    method: str = "max_quadratic_utility",
    param_name: str = "risk_aversion",
    param_values: list[float] | None = None,
    constraints: PortfolioConstraints | None = None,
    cov_estimator: CovEstimator = CovEstimator.SAMPLE,
    return_estimator: ReturnEstimator = ReturnEstimator.HISTORICAL_MEAN,
    risk_free_rate: float = 0.04,
) -> list[StrategyResult]:
    """Sweep a single parameter across a range and collect results.

    Args:
        prices: Price DataFrame.
        method: Optimization method name.
        param_name: Parameter to sweep ('risk_aversion', 'lookback', 'max_weight').
        param_values: Values to test. Defaults based on param_name.
        constraints: Base constraints (will be cloned and modified).
        risk_free_rate: Annual risk-free rate.

    Returns:
        List of StrategyResult for each parameter value.
    """
    if constraints is None:
        constraints = PortfolioConstraints()

    if param_values is None:
        param_values = _default_sweep_values(param_name)

    mu = estimate_returns(prices, return_estimator)
    cov = estimate_covariance(prices, cov_estimator)
    daily_returns = prices.pct_change().dropna()

    method_map = {
        "inverse_variance": OptMethod.INVERSE_VARIANCE,
        "min_volatility": OptMethod.MIN_VOLATILITY,
        "max_sharpe": OptMethod.MAX_SHARPE,
        "max_quadratic_utility": OptMethod.MAX_QUADRATIC_UTILITY,
        "max_diversification": OptMethod.MAX_DIVERSIFICATION,
        "max_decorrelation": OptMethod.MAX_DECORRELATION,
        "hrp": OptMethod.HRP,
        "herc": OptMethod.HERC,
    }

    opt_method = method_map.get(method.lower())
    if opt_method is None:
        raise ValueError(f"Unknown method: {method}")

    results: list[StrategyResult] = []
    for val in param_values:
        try:
            # Clone constraints with modified parameter
            c = _modify_constraints(constraints, param_name, val)
            label = f"{method}({param_name}={val})"

            opt_result = _run_single_optimization(
                mu, cov, c, opt_method, risk_free_rate,
            )
            weights_arr = np.array([
                opt_result.weights.get(s, 0.0) for s in prices.columns
            ])
            port_returns = daily_returns.values @ weights_arr
            metrics = compute_all_metrics(
                port_returns, risk_free_rate=risk_free_rate,
            )
            results.append(StrategyResult(
                name=label,
                opt_result=opt_result,
                backtest_returns=port_returns,
                metrics=metrics,
            ))
        except Exception as e:
            logger.warning("Sweep %s=%s failed: %s", param_name, val, e)

    return results


# ── Statistical tests ────────────────────────────────────────────────


def bootstrap_paired_test(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
) -> PairwiseTest:
    """Bootstrap paired test for mean return difference.

    Tests H0: mean(returns_a) == mean(returns_b)
    using circular block bootstrap.

    Args:
        returns_a: Daily returns of strategy A.
        returns_b: Daily returns of strategy B.
        n_bootstrap: Number of bootstrap replications.
        alpha: Significance level.

    Returns:
        PairwiseTest with test statistic, p-value, and significance.
    """
    n = min(len(returns_a), len(returns_b))
    diff = returns_a[:n] - returns_b[:n]
    observed_mean = float(np.mean(diff))

    # Block bootstrap (block size ~ sqrt(n))
    block_size = max(1, int(np.sqrt(n)))
    boot_means = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        n_blocks = int(np.ceil(n / block_size))
        starts = np.random.randint(0, n, size=n_blocks)
        indices = np.concatenate([
            np.arange(s, s + block_size) % n for s in starts
        ])[:n]
        boot_means[i] = np.mean(diff[indices])

    # Two-sided p-value
    centered = boot_means - np.mean(boot_means)
    p_value = float(np.mean(np.abs(centered) >= np.abs(observed_mean)))

    # Test statistic (t-like)
    boot_std = np.std(boot_means, ddof=1)
    t_stat = observed_mean / boot_std if boot_std > 0 else 0.0

    return PairwiseTest(
        strategy_a="",  # filled by caller
        strategy_b="",
        test_statistic=float(t_stat),
        p_value=p_value,
        mean_diff=observed_mean,
        significant=p_value < alpha,
    )


# ── Internal helpers ─────────────────────────────────────────────────


def _run_single_optimization(
    mu: pd.Series,
    cov: pd.DataFrame,
    constraints: PortfolioConstraints,
    method: OptMethod,
    risk_free_rate: float,
) -> OptimizationResult:
    """Run a single optimization and return the result."""
    if method in (OptMethod.HRP, OptMethod.HERC):
        if method == OptMethod.HRP:
            from portopt.engine.optimization.hrp import hrp_optimize
            return hrp_optimize(cov)
        else:
            from portopt.engine.optimization.herc import herc_optimize
            return herc_optimize(cov)
    else:
        from portopt.engine.optimization.mean_variance import MeanVarianceOptimizer
        opt = MeanVarianceOptimizer(mu, cov, constraints, method)
        return opt.optimize()


def _run_pairwise_tests(
    strategies: list[StrategyResult],
    n_bootstrap: int = 5000,
) -> list[PairwiseTest]:
    """Run pairwise bootstrap tests between all strategy pairs."""
    tests = []
    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            a = strategies[i]
            b = strategies[j]
            if a.backtest_returns is None or b.backtest_returns is None:
                continue
            test = bootstrap_paired_test(
                a.backtest_returns, b.backtest_returns,
                n_bootstrap=n_bootstrap,
            )
            test.strategy_a = a.name
            test.strategy_b = b.name
            tests.append(test)
    return tests


def _default_sweep_values(param_name: str) -> list[float]:
    """Return sensible default sweep values for a parameter."""
    defaults = {
        "risk_aversion": [0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
        "lookback": [60, 120, 252, 504],
        "max_weight": [0.1, 0.2, 0.3, 0.5, 1.0],
        "min_weight": [0.0, 0.01, 0.02, 0.05],
    }
    return defaults.get(param_name, [1.0, 2.0, 5.0, 10.0])


def _modify_constraints(
    base: PortfolioConstraints,
    param_name: str,
    value: float,
) -> PortfolioConstraints:
    """Clone constraints with one parameter modified."""
    import copy
    c = copy.deepcopy(base)
    if param_name == "risk_aversion":
        c.risk_aversion = value
    elif param_name == "max_weight":
        c.max_weight = value
    elif param_name == "min_weight":
        c.min_weight = value
    elif param_name == "lookback":
        pass  # lookback is handled externally
    else:
        logger.warning("Unknown sweep parameter: %s", param_name)
    return c
