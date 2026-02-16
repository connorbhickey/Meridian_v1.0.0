"""Rebalancing logic: frequency-based and drift-threshold triggers.

Determines WHEN to rebalance. The actual weight computation is done
by the optimization engine; the runner applies the rebalance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from portopt.constants import RebalanceFreq


@dataclass
class RebalanceSchedule:
    """Configuration for rebalance timing.

    Attributes:
        frequency: Calendar frequency (daily, weekly, monthly, quarterly, yearly).
        drift_threshold: Rebalance if any weight drifts more than this from target.
                         None = no drift-based rebalancing.
        min_rebalance_interval: Minimum days between rebalances (prevents excessive trading).
        initial_rebalance: Whether to rebalance on the first day.
    """
    frequency: RebalanceFreq = RebalanceFreq.MONTHLY
    drift_threshold: float | None = None
    min_rebalance_interval: int = 1
    initial_rebalance: bool = True


def generate_rebalance_dates(
    dates: pd.DatetimeIndex,
    schedule: RebalanceSchedule,
) -> list[date]:
    """Generate rebalance dates from a date index and schedule.

    Args:
        dates: Full date index of the backtest period.
        schedule: Rebalance configuration.

    Returns:
        Sorted list of dates on which to rebalance.
    """
    if len(dates) == 0:
        return []

    freq = schedule.frequency
    rebalance_dates = set()

    if schedule.initial_rebalance:
        rebalance_dates.add(dates[0].date() if hasattr(dates[0], 'date') else dates[0])

    if freq == RebalanceFreq.DAILY:
        for d in dates:
            dt = d.date() if hasattr(d, 'date') else d
            rebalance_dates.add(dt)
    else:
        # Group by period and take first trading day of each new period
        period_map = {
            RebalanceFreq.WEEKLY: "W",
            RebalanceFreq.MONTHLY: "MS",
            RebalanceFreq.QUARTERLY: "QS",
            RebalanceFreq.YEARLY: "YS",
        }
        freq_str = period_map.get(freq, "MS")

        # Find first trading day at or after each period start
        period_starts = pd.date_range(
            start=dates[0],
            end=dates[-1],
            freq=freq_str,
        )

        for ps in period_starts:
            # Find first trading day >= period start
            mask = dates >= ps
            if mask.any():
                first_day = dates[mask][0]
                dt = first_day.date() if hasattr(first_day, 'date') else first_day
                rebalance_dates.add(dt)

    # Filter by min_rebalance_interval
    sorted_dates = sorted(rebalance_dates)
    if schedule.min_rebalance_interval > 1 and len(sorted_dates) > 1:
        filtered = [sorted_dates[0]]
        for d in sorted_dates[1:]:
            days_since = (d - filtered[-1]).days
            if days_since >= schedule.min_rebalance_interval:
                filtered.append(d)
        sorted_dates = filtered

    return sorted_dates


def check_drift_rebalance(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    threshold: float,
) -> bool:
    """Check if any weight has drifted beyond the threshold.

    Args:
        current_weights: Current portfolio weights.
        target_weights: Target portfolio weights.
        threshold: Maximum allowed absolute drift per asset.

    Returns:
        True if a rebalance is triggered.
    """
    all_symbols = set(list(current_weights.keys()) + list(target_weights.keys()))
    for sym in all_symbols:
        cur = current_weights.get(sym, 0.0)
        tgt = target_weights.get(sym, 0.0)
        if abs(cur - tgt) > threshold:
            return True
    return False


def compute_weight_drift(
    initial_weights: dict[str, float],
    asset_returns: dict[str, float],
) -> dict[str, float]:
    """Compute how weights drift after one period of returns.

    Args:
        initial_weights: Starting weights.
        asset_returns: {symbol: simple_return} for the period.

    Returns:
        New weights after drift (due to differential returns).
    """
    new_values = {}
    for sym, w in initial_weights.items():
        ret = asset_returns.get(sym, 0.0)
        new_values[sym] = w * (1 + ret)

    total = sum(new_values.values())
    if total == 0:
        return initial_weights.copy()

    return {sym: v / total for sym, v in new_values.items()}


def compute_turnover(
    weights_before: dict[str, float],
    weights_after: dict[str, float],
) -> float:
    """Compute total turnover as sum of absolute weight changes.

    Args:
        weights_before: Weights before rebalance.
        weights_after: Weights after rebalance.

    Returns:
        Total turnover (0 to 2 for long-only, higher for short portfolios).
    """
    all_symbols = set(list(weights_before.keys()) + list(weights_after.keys()))
    return sum(
        abs(weights_after.get(s, 0.0) - weights_before.get(s, 0.0))
        for s in all_symbols
    )
