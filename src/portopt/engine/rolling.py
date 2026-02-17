"""Rolling window analytics — Sharpe, Sortino, volatility, drawdown, beta, correlation.

Zero GUI knowledge. All inputs/outputs are pandas/numpy objects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portopt.engine.metrics import (
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)


def rolling_sharpe(
    returns: pd.Series,
    window: int = 63,
    risk_free_rate: float = 0.04,
    frequency: int = 252,
) -> pd.Series:
    """Rolling annualized Sharpe ratio.

    Parameters
    ----------
    returns : daily return series
    window : rolling window size in trading days
    risk_free_rate : annual risk-free rate
    frequency : trading days per year
    """
    if returns.empty or len(returns) < window:
        return pd.Series(dtype=float)

    daily_rf = risk_free_rate / frequency

    def _sharpe(arr: np.ndarray) -> float:
        return sharpe_ratio(arr, daily_rf, frequency)

    return returns.rolling(window).apply(_sharpe, raw=True)


def rolling_sortino(
    returns: pd.Series,
    window: int = 63,
    risk_free_rate: float = 0.04,
    frequency: int = 252,
) -> pd.Series:
    """Rolling annualized Sortino ratio."""
    if returns.empty or len(returns) < window:
        return pd.Series(dtype=float)

    daily_rf = risk_free_rate / frequency

    def _sortino(arr: np.ndarray) -> float:
        return sortino_ratio(arr, daily_rf, frequency)

    return returns.rolling(window).apply(_sortino, raw=True)


def rolling_volatility(
    returns: pd.Series,
    window: int = 63,
    frequency: int = 252,
) -> pd.Series:
    """Rolling annualized volatility."""
    if returns.empty or len(returns) < window:
        return pd.Series(dtype=float)

    def _vol(arr: np.ndarray) -> float:
        return annualized_volatility(arr, frequency)

    return returns.rolling(window).apply(_vol, raw=True)


def rolling_max_drawdown(
    returns: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Rolling maximum drawdown (negative values)."""
    if returns.empty or len(returns) < window:
        return pd.Series(dtype=float)

    return returns.rolling(window).apply(max_drawdown, raw=True)


def rolling_beta(
    returns: pd.Series,
    benchmark: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Rolling beta of returns vs benchmark.

    Uses covariance / variance formula over the rolling window.
    """
    if returns.empty or benchmark.empty:
        return pd.Series(dtype=float)

    # Align series
    aligned = pd.DataFrame({"r": returns, "b": benchmark}).dropna()
    if len(aligned) < window:
        return pd.Series(dtype=float)

    cov = aligned["r"].rolling(window).cov(aligned["b"])
    var = aligned["b"].rolling(window).var()
    beta = cov / var
    beta = beta.replace([np.inf, -np.inf], np.nan)
    return beta


def rolling_correlation(
    returns_a: pd.Series,
    returns_b: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Rolling Pearson correlation between two return series."""
    if returns_a.empty or returns_b.empty:
        return pd.Series(dtype=float)

    # Align series
    aligned = pd.DataFrame({"a": returns_a, "b": returns_b}).dropna()
    if len(aligned) < window:
        return pd.Series(dtype=float)

    return aligned["a"].rolling(window).corr(aligned["b"])
