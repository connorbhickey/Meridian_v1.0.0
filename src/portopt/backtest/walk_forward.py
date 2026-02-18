"""Walk-forward analysis with rolling and anchored windows.

Splits the data into train/test windows, optimizes on each training set,
then evaluates out-of-sample on the test set. Aggregates all OOS periods
into a single equity curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from portopt.backtest.costs import BaseCostModel, ZeroCost
from portopt.backtest.rebalancer import RebalanceSchedule
from portopt.backtest.results import SingleRunResult, WalkForwardResult, WalkForwardWindow
from portopt.backtest.runner import OptimizerFn, run_backtest


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis.

    Attributes:
        train_window: Training window size in trading days.
        test_window: Test (OOS) window size in trading days.
        anchored: If True, training window starts from the beginning (expanding).
                  If False, rolling window of fixed size.
        step: Step size between windows in trading days.
              If None, equals test_window (non-overlapping OOS).
        min_train_size: Minimum training window size (for anchored mode).
    """
    train_window: int = 252
    test_window: int = 63
    anchored: bool = False
    step: int | None = None
    min_train_size: int = 126


def generate_windows(
    dates: pd.DatetimeIndex,
    config: WalkForwardConfig,
) -> list[WalkForwardWindow]:
    """Generate train/test window splits.

    Args:
        dates: Full date index of the dataset.
        config: Walk-forward configuration.

    Returns:
        List of WalkForwardWindow with date boundaries.
    """
    n = len(dates)
    step = config.step if config.step is not None else config.test_window
    windows = []
    window_id = 0

    idx = config.train_window
    while idx + config.test_window <= n:
        if config.anchored:
            train_start_idx = 0
        else:
            train_start_idx = idx - config.train_window

        train_end_idx = idx - 1
        test_start_idx = idx
        test_end_idx = min(idx + config.test_window - 1, n - 1)

        # Check minimum training size
        train_size = train_end_idx - train_start_idx + 1
        if train_size < config.min_train_size:
            idx += step
            continue

        def to_date(i):
            return dates[i].date() if hasattr(dates[i], 'date') else dates[i]

        window = WalkForwardWindow(
            window_id=window_id,
            train_start=to_date(train_start_idx),
            train_end=to_date(train_end_idx),
            test_start=to_date(test_start_idx),
            test_end=to_date(test_end_idx),
        )
        windows.append(window)
        window_id += 1
        idx += step

    return windows


def run_walk_forward(
    prices: pd.DataFrame,
    optimizer_fn: OptimizerFn,
    config: WalkForwardConfig | None = None,
    schedule: RebalanceSchedule | None = None,
    cost_model: BaseCostModel | None = None,
    initial_value: float = 1_000_000.0,
) -> WalkForwardResult:
    """Run walk-forward analysis.

    Args:
        prices: Full price DataFrame.
        optimizer_fn: Optimizer callback (see runner.py).
        config: Walk-forward window configuration.
        schedule: Rebalance schedule for each OOS window.
        cost_model: Transaction cost model.
        initial_value: Starting portfolio value.

    Returns:
        WalkForwardResult with per-window and aggregate results.
    """
    if config is None:
        config = WalkForwardConfig()
    if schedule is None:
        schedule = RebalanceSchedule()
    if cost_model is None:
        cost_model = ZeroCost()

    # Generate windows
    windows = generate_windows(prices.index, config)
    if not windows:
        raise ValueError(
            f"No valid windows generated. Need at least {config.train_window + config.test_window} "
            f"trading days, got {len(prices)}."
        )

    # Run each window
    all_oos_returns = []
    all_oos_dates = []
    total_costs = 0.0
    current_value = initial_value

    for window in windows:
        # Extract train and test data
        train_mask = (prices.index >= pd.Timestamp(window.train_start)) & \
                     (prices.index <= pd.Timestamp(window.train_end))
        test_mask = (prices.index >= pd.Timestamp(window.test_start)) & \
                    (prices.index <= pd.Timestamp(window.test_end))

        train_prices = prices.loc[train_mask]
        test_prices = prices.loc[test_mask]

        if len(train_prices) < config.min_train_size or len(test_prices) < 2:
            continue

        # Optimize on training data
        try:
            optimal_weights = optimizer_fn(train_prices, None)
            window.optimal_weights = optimal_weights
        except Exception:
            # Fallback to equal weight
            symbols = list(prices.columns)
            optimal_weights = {s: 1.0 / len(symbols) for s in symbols}
            window.optimal_weights = optimal_weights

        # Create a fixed-weight optimizer for OOS
        fixed_weights = optimal_weights.copy()

        def fixed_optimizer_fn(p, _cov=None, _w=fixed_weights):
            return _w.copy()

        # Run backtest on test period
        try:
            test_result = run_backtest(
                prices=test_prices,
                optimizer_fn=fixed_optimizer_fn,
                schedule=schedule,
                cost_model=cost_model,
                initial_value=current_value,
            )
            window.test_result = test_result

            # Collect OOS returns and dates
            all_oos_returns.extend(test_result.daily_returns.tolist())
            all_oos_dates.extend(test_result.dates)
            total_costs += test_result.total_costs
            current_value = test_result.final_value
        except Exception:
            continue

    # Build aggregate results
    if all_oos_returns:
        agg_returns = np.array(all_oos_returns)
        agg_values = initial_value * np.cumprod(1 + agg_returns)
    else:
        agg_returns = np.array([])
        agg_values = np.array([])

    return WalkForwardResult(
        windows=windows,
        aggregate_returns=agg_returns,
        aggregate_dates=all_oos_dates,
        aggregate_values=agg_values,
        total_costs=total_costs,
        config={
            "train_window": config.train_window,
            "test_window": config.test_window,
            "anchored": config.anchored,
            "step": config.step,
        },
    )
