"""Backtest result dataclasses — detailed results for single runs and walk-forward."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd


@dataclass
class Trade:
    """A single trade executed during backtesting."""
    date: date
    symbol: str
    side: str              # "BUY" or "SELL"
    quantity: float        # Positive for buy, negative for sell (in weight terms)
    price: float
    cost: float = 0.0      # Transaction cost incurred
    weight_before: float = 0.0
    weight_after: float = 0.0

    @property
    def notional(self) -> float:
        return abs(self.quantity) * self.price


@dataclass
class RebalanceEvent:
    """Record of a single rebalance."""
    date: date
    weights_before: dict[str, float]
    weights_after: dict[str, float]
    turnover: float
    total_cost: float
    trades: list[Trade] = field(default_factory=list)
    method: str = ""


@dataclass
class SingleRunResult:
    """Result from a single backtest run (one optimization window)."""
    portfolio_values: np.ndarray        # Daily portfolio values
    daily_returns: np.ndarray           # Daily portfolio returns
    dates: list[date] = field(default_factory=list)
    weights_history: list[dict[str, float]] = field(default_factory=list)
    rebalance_events: list[RebalanceEvent] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    total_costs: float = 0.0
    initial_value: float = 1.0
    final_value: float = 1.0
    method: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def n_periods(self) -> int:
        return len(self.portfolio_values)

    @property
    def n_rebalances(self) -> int:
        return len(self.rebalance_events)

    @property
    def total_return(self) -> float:
        if self.initial_value == 0:
            return 0.0
        return self.final_value / self.initial_value - 1

    @property
    def total_turnover(self) -> float:
        return sum(e.turnover for e in self.rebalance_events)

    def to_series(self) -> pd.Series:
        """Convert daily returns to a pandas Series with date index."""
        if self.dates and len(self.dates) == len(self.daily_returns):
            return pd.Series(self.daily_returns, index=pd.DatetimeIndex(self.dates))
        return pd.Series(self.daily_returns)

    def equity_curve(self) -> pd.Series:
        """Portfolio value series."""
        if self.dates and len(self.dates) == len(self.portfolio_values):
            return pd.Series(self.portfolio_values, index=pd.DatetimeIndex(self.dates))
        return pd.Series(self.portfolio_values)


@dataclass
class WalkForwardWindow:
    """A single in-sample / out-of-sample window."""
    window_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_result: SingleRunResult | None = None
    test_result: SingleRunResult | None = None
    optimal_weights: dict[str, float] = field(default_factory=dict)
    method: str = ""

    @property
    def test_return(self) -> float:
        if self.test_result is None:
            return 0.0
        return self.test_result.total_return


@dataclass
class WalkForwardResult:
    """Aggregated result from walk-forward analysis."""
    windows: list[WalkForwardWindow] = field(default_factory=list)
    aggregate_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    aggregate_dates: list[date] = field(default_factory=list)
    aggregate_values: np.ndarray = field(default_factory=lambda: np.array([]))
    total_costs: float = 0.0
    method: str = ""
    config: dict = field(default_factory=dict)

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    @property
    def total_return(self) -> float:
        if len(self.aggregate_values) == 0:
            return 0.0
        return float(self.aggregate_values[-1] / self.aggregate_values[0] - 1)

    @property
    def oos_returns(self) -> list[float]:
        """Out-of-sample returns per window."""
        return [w.test_return for w in self.windows]

    def aggregate_equity_curve(self) -> pd.Series:
        """Aggregate OOS equity curve."""
        if self.aggregate_dates and len(self.aggregate_dates) == len(self.aggregate_values):
            return pd.Series(
                self.aggregate_values,
                index=pd.DatetimeIndex(self.aggregate_dates),
            )
        return pd.Series(self.aggregate_values)

    def aggregate_return_series(self) -> pd.Series:
        """Aggregate OOS return series."""
        if self.aggregate_dates and len(self.aggregate_dates) == len(self.aggregate_returns):
            return pd.Series(
                self.aggregate_returns,
                index=pd.DatetimeIndex(self.aggregate_dates),
            )
        return pd.Series(self.aggregate_returns)
