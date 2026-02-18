"""Backtest runner: step through time, rebalance, apply costs, track values.

This is the core execution loop. Given prices, an optimization method,
a rebalance schedule, and a cost model, it produces a SingleRunResult
with daily portfolio values, returns, trades, and rebalance events.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Callable

import numpy as np
import pandas as pd

from portopt.backtest.costs import BaseCostModel, ZeroCost
from portopt.backtest.rebalancer import (
    RebalanceSchedule,
    check_drift_rebalance,
    compute_turnover,
    compute_weight_drift,
    generate_rebalance_dates,
)
from portopt.backtest.results import RebalanceEvent, SingleRunResult, Trade
from portopt.constants import CovEstimator, RebalanceFreq, ReturnEstimator
from portopt.engine.constraints import PortfolioConstraints

logger = logging.getLogger(__name__)


# Type alias for the optimizer callback
OptimizerFn = Callable[[pd.DataFrame, pd.DataFrame | None], dict[str, float]]


def run_backtest(
    prices: pd.DataFrame,
    optimizer_fn: OptimizerFn,
    schedule: RebalanceSchedule | None = None,
    cost_model: BaseCostModel | None = None,
    initial_value: float = 1_000_000.0,
    benchmark_prices: pd.Series | None = None,
) -> SingleRunResult:
    """Execute a single backtest run.

    Args:
        prices: Price DataFrame (index=DatetimeIndex, columns=symbols).
        optimizer_fn: Callable(prices_so_far, covariance) -> {symbol: weight}.
                      Called at each rebalance date with data available up to that point.
        schedule: Rebalance timing configuration. Defaults to monthly.
        cost_model: Transaction cost model. Defaults to zero cost.
        initial_value: Starting portfolio value.
        benchmark_prices: Optional benchmark price series for tracking.

    Returns:
        SingleRunResult with full backtest history.
    """
    if schedule is None:
        schedule = RebalanceSchedule()
    if cost_model is None:
        cost_model = ZeroCost()

    symbols = list(prices.columns)
    dates_idx = prices.index
    n_days = len(dates_idx)

    if n_days < 2:
        raise ValueError("Need at least 2 days of price data for backtesting")

    # Compute daily simple returns for each asset
    asset_returns = prices.pct_change().iloc[1:]  # Drop first NaN row
    return_dates = asset_returns.index

    # Generate scheduled rebalance dates
    rebalance_dates_list = generate_rebalance_dates(return_dates, schedule)
    rebalance_set = set(rebalance_dates_list)

    # State tracking
    portfolio_values = np.zeros(len(return_dates))
    daily_returns = np.zeros(len(return_dates))
    current_weights: dict[str, float] = {}
    target_weights: dict[str, float] = {}
    all_trades: list[Trade] = []
    all_rebalance_events: list[RebalanceEvent] = []
    weights_history: list[dict[str, float]] = []
    total_costs = 0.0
    current_value = initial_value
    last_rebalance_date: date | None = None

    for i, dt in enumerate(return_dates):
        dt_date = dt.date() if hasattr(dt, 'date') else dt

        # Check if we should rebalance
        should_rebalance = False

        # Scheduled rebalance
        if dt_date in rebalance_set:
            should_rebalance = True

        # Drift-based rebalance
        if (
            not should_rebalance
            and schedule.drift_threshold is not None
            and current_weights
            and target_weights
        ):
            if check_drift_rebalance(current_weights, target_weights, schedule.drift_threshold):
                # Check min interval
                if last_rebalance_date is None:
                    should_rebalance = True
                elif (dt_date - last_rebalance_date).days >= schedule.min_rebalance_interval:
                    should_rebalance = True

        # Execute rebalance
        if should_rebalance:
            # Get prices up to current date for optimization
            available_prices = prices.loc[:dt]

            try:
                new_weights = optimizer_fn(available_prices, None)
            except (np.linalg.LinAlgError, RuntimeError, ValueError) as e:
                logger.warning("Optimization failed on %s: %s — using fallback weights", dt_date, e)
                new_weights = current_weights.copy() if current_weights else {
                    s: 1.0 / len(symbols) for s in symbols
                }
            except Exception as e:
                logger.error("Unexpected optimization error on %s: %s — using fallback weights", dt_date, e)
                new_weights = current_weights.copy() if current_weights else {
                    s: 1.0 / len(symbols) for s in symbols
                }

            # Ensure all symbols present
            for s in symbols:
                new_weights.setdefault(s, 0.0)

            weights_before = current_weights.copy()

            # Compute trades and costs
            weight_changes = {
                s: new_weights.get(s, 0.0) - current_weights.get(s, 0.0)
                for s in symbols
            }

            current_prices = {s: float(prices.loc[dt, s]) for s in symbols}
            rebalance_cost = cost_model.compute_total_cost(
                weight_changes, current_value, current_prices,
            )

            turnover = compute_turnover(current_weights, new_weights)

            # Record trades
            trades = []
            for s in symbols:
                dw = weight_changes.get(s, 0.0)
                if abs(dw) < 1e-10:
                    continue
                trade = Trade(
                    date=dt_date,
                    symbol=s,
                    side="BUY" if dw > 0 else "SELL",
                    quantity=dw,
                    price=current_prices.get(s, 0.0),
                    cost=cost_model.compute_cost(s, dw, current_value, current_prices.get(s, 0.0)),
                    weight_before=current_weights.get(s, 0.0),
                    weight_after=new_weights.get(s, 0.0),
                )
                trades.append(trade)
                all_trades.append(trade)

            # Record rebalance event
            event = RebalanceEvent(
                date=dt_date,
                weights_before=weights_before,
                weights_after=new_weights.copy(),
                turnover=turnover,
                total_cost=rebalance_cost,
                trades=trades,
            )
            all_rebalance_events.append(event)

            # Apply cost and update state
            current_value -= rebalance_cost
            total_costs += rebalance_cost
            current_weights = new_weights.copy()
            target_weights = new_weights.copy()
            last_rebalance_date = dt_date

        # Compute daily portfolio return from asset returns
        if current_weights:
            day_returns = asset_returns.iloc[i]
            port_return = sum(
                current_weights.get(s, 0.0) * float(day_returns.get(s, 0.0))
                for s in symbols
            )

            # Drift weights based on differential returns
            day_returns_dict = {s: float(day_returns.get(s, 0.0)) for s in symbols}
            current_weights = compute_weight_drift(current_weights, day_returns_dict)
        else:
            port_return = 0.0

        # Update portfolio value
        current_value *= (1 + port_return)
        portfolio_values[i] = current_value
        daily_returns[i] = port_return
        weights_history.append(current_weights.copy())

    return SingleRunResult(
        portfolio_values=portfolio_values,
        daily_returns=daily_returns,
        dates=[d.date() if hasattr(d, 'date') else d for d in return_dates],
        weights_history=weights_history,
        rebalance_events=all_rebalance_events,
        trades=all_trades,
        total_costs=total_costs,
        initial_value=initial_value,
        final_value=current_value,
    )


def make_optimizer_fn(
    method: str = "max_sharpe",
    constraints: PortfolioConstraints | None = None,
    cov_estimator: CovEstimator = CovEstimator.SAMPLE,
    return_estimator: ReturnEstimator = ReturnEstimator.HISTORICAL_MEAN,
    lookback: int | None = None,
    risk_free_rate: float = 0.04,
) -> OptimizerFn:
    """Create an optimizer callback for use with run_backtest.

    Args:
        method: Optimization method name (maps to OptMethod or special methods).
        constraints: Portfolio constraints.
        cov_estimator: Covariance estimation method.
        return_estimator: Return estimation method.
        lookback: Lookback window in days (None = use all available data).
        risk_free_rate: Annual risk-free rate.

    Returns:
        Callable compatible with run_backtest's optimizer_fn parameter.
    """
    from portopt.constants import OptMethod
    from portopt.engine.constraints import PortfolioConstraints
    from portopt.engine.returns import estimate_returns
    from portopt.engine.risk import estimate_covariance

    if constraints is None:
        constraints = PortfolioConstraints()

    # Map string method names to OptMethod
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

    opt_method = method_map.get(method.lower())
    if opt_method is None:
        raise ValueError(f"Unknown optimization method: {method}")

    def optimizer_fn(prices: pd.DataFrame, _cov=None) -> dict[str, float]:
        # Apply lookback window
        if lookback is not None and len(prices) > lookback:
            prices = prices.iloc[-lookback:]

        if len(prices) < 30:
            # Too little data — equal weight
            symbols = list(prices.columns)
            return {s: 1.0 / len(symbols) for s in symbols}

        # Estimate returns and covariance
        mu = estimate_returns(prices, return_estimator)
        cov = estimate_covariance(prices, cov_estimator)

        # Run optimization
        if opt_method in (OptMethod.HRP, OptMethod.HERC):
            from portopt.engine.optimization.hrp import hrp_optimize
            from portopt.engine.optimization.herc import herc_optimize

            if opt_method == OptMethod.HRP:
                result = hrp_optimize(cov)
            else:
                result = herc_optimize(cov)
            return result.weights
        else:
            from portopt.engine.optimization.mean_variance import MeanVarianceOptimizer
            opt = MeanVarianceOptimizer(mu, cov, constraints, opt_method)
            result = opt.optimize()
            return result.weights

    return optimizer_fn
