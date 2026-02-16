"""Shared test fixtures for the entire test suite."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest


# ── Deterministic Seed ─────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _seed_rng():
    """Fix random seed for reproducibility."""
    np.random.seed(42)


# ── Price Data ─────────────────────────────────────────────────────────

def _make_prices(
    symbols: list[str],
    n_days: int = 504,
    annual_returns: list[float] | None = None,
    annual_vols: list[float] | None = None,
) -> pd.DataFrame:
    """Generate synthetic daily close prices via geometric Brownian motion."""
    n = len(symbols)
    if annual_returns is None:
        annual_returns = np.linspace(0.05, 0.15, n).tolist()
    if annual_vols is None:
        annual_vols = np.linspace(0.10, 0.30, n).tolist()

    dt = 1 / 252
    dates = pd.bdate_range(end=date.today(), periods=n_days)
    data = {}

    for i, sym in enumerate(symbols):
        mu = annual_returns[i]
        sigma = annual_vols[i]
        log_rets = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn(n_days)
        prices = 100.0 * np.exp(np.cumsum(log_rets))
        data[sym] = prices

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def symbols_3() -> list[str]:
    return ["AAPL", "MSFT", "GOOG"]


@pytest.fixture
def symbols_5() -> list[str]:
    return ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]


@pytest.fixture
def symbols_10() -> list[str]:
    return ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA",
            "META", "NVDA", "JPM", "V", "JNJ"]


@pytest.fixture
def prices_3(symbols_3) -> pd.DataFrame:
    """3-asset price DataFrame, 504 trading days (~2 years)."""
    return _make_prices(symbols_3)


@pytest.fixture
def prices_5(symbols_5) -> pd.DataFrame:
    """5-asset price DataFrame."""
    return _make_prices(symbols_5)


@pytest.fixture
def prices_10(symbols_10) -> pd.DataFrame:
    """10-asset price DataFrame."""
    return _make_prices(symbols_10)


@pytest.fixture
def returns_5(prices_5) -> pd.DataFrame:
    """Simple returns from 5-asset prices."""
    return prices_5.pct_change().dropna()


# ── Covariance / Returns Estimates ─────────────────────────────────────

@pytest.fixture
def sample_cov(prices_5) -> pd.DataFrame:
    """Sample covariance matrix (annualized)."""
    from portopt.engine.risk import estimate_covariance
    from portopt.constants import CovEstimator
    return estimate_covariance(prices_5, method=CovEstimator.SAMPLE)


@pytest.fixture
def expected_returns(prices_5) -> pd.Series:
    """Historical mean expected returns (annualized)."""
    from portopt.engine.returns import estimate_returns
    from portopt.constants import ReturnEstimator
    return estimate_returns(prices_5, method=ReturnEstimator.HISTORICAL_MEAN)


# ── Helpers ────────────────────────────────────────────────────────────

def assert_valid_weights(weights: dict[str, float], symbols: list[str], tol: float = 1e-4):
    """Assert weights are valid: correct symbols, sum to ~1, within bounds."""
    assert set(weights.keys()) == set(symbols), f"Weight keys mismatch: {set(weights.keys())} vs {set(symbols)}"
    total = sum(weights.values())
    assert abs(total - 1.0) < tol, f"Weights sum to {total}, expected ~1.0"
    for sym, w in weights.items():
        assert w >= -tol, f"Weight for {sym} is {w} (negative)"


def assert_positive_definite(matrix: np.ndarray | pd.DataFrame):
    """Assert matrix is positive (semi-)definite."""
    if isinstance(matrix, pd.DataFrame):
        matrix = matrix.values
    eigvals = np.linalg.eigvalsh(matrix)
    assert np.all(eigvals >= -1e-8), f"Not PSD: min eigenvalue = {eigvals.min()}"
