"""Return estimators: historical mean, CAPM, exponentially-weighted."""

from __future__ import annotations

import numpy as np
import pandas as pd

from portopt.constants import ReturnEstimator


def estimate_returns(
    prices: pd.DataFrame,
    method: ReturnEstimator = ReturnEstimator.HISTORICAL_MEAN,
    **kwargs,
) -> pd.Series:
    """Estimate expected asset returns.

    Args:
        prices: DataFrame of close prices (index=date, columns=symbols).
        method: Which estimator to use.
        **kwargs: Method-specific parameters.

    Returns:
        Series of annualized expected returns indexed by symbol.
    """
    estimators = {
        ReturnEstimator.HISTORICAL_MEAN: _historical_mean,
        ReturnEstimator.CAPM: _capm,
        ReturnEstimator.EXPONENTIAL: _exponential,
    }
    fn = estimators[method]
    return fn(prices, **kwargs)


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from a price DataFrame."""
    return np.log(prices / prices.shift(1)).dropna()


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple (arithmetic) returns from a price DataFrame."""
    return prices.pct_change().dropna()


# ── Private estimators ────────────────────────────────────────────────


def _historical_mean(
    prices: pd.DataFrame,
    frequency: int = 252,
    log: bool = True,
    **_kwargs,
) -> pd.Series:
    """Annualized historical mean returns."""
    rets = log_returns(prices) if log else simple_returns(prices)
    return rets.mean() * frequency


def _capm(
    prices: pd.DataFrame,
    market_prices: pd.Series | None = None,
    risk_free_rate: float = 0.04,
    frequency: int = 252,
    **_kwargs,
) -> pd.Series:
    """CAPM-implied expected returns: E[r_i] = r_f + beta_i * (E[r_m] - r_f).

    If market_prices is None, uses the equal-weighted portfolio of all assets.
    """
    rets = log_returns(prices)

    if market_prices is not None:
        mkt_ret = log_returns(market_prices.to_frame()).iloc[:, 0]
    else:
        mkt_ret = rets.mean(axis=1)

    # Align indices
    common = rets.index.intersection(mkt_ret.index)
    rets = rets.loc[common]
    mkt_ret = mkt_ret.loc[common]

    mkt_excess = mkt_ret - risk_free_rate / frequency
    mkt_var = mkt_excess.var()

    betas = pd.Series(index=rets.columns, dtype=float)
    for col in rets.columns:
        cov = np.cov(rets[col].values, mkt_excess.values)[0, 1]
        betas[col] = cov / mkt_var if mkt_var > 0 else 1.0

    market_premium = mkt_ret.mean() * frequency - risk_free_rate
    return risk_free_rate + betas * market_premium


def _exponential(
    prices: pd.DataFrame,
    span: int = 60,
    frequency: int = 252,
    **_kwargs,
) -> pd.Series:
    """Exponentially-weighted mean returns (recent data weighted more)."""
    rets = log_returns(prices)
    ewm_mean = rets.ewm(span=span).mean().iloc[-1]
    return ewm_mean * frequency
