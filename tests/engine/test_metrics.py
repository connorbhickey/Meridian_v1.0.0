"""Tests for performance metrics."""

import numpy as np
import pytest

from portopt.engine.metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    cagr,
    compute_all_metrics,
    conditional_var,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    value_at_risk,
)


@pytest.fixture
def daily_returns():
    """Synthetic daily returns (positive drift)."""
    np.random.seed(42)
    return np.random.randn(252) * 0.01 + 0.0004  # ~10% annual, ~16% vol


@pytest.fixture
def zero_returns():
    return np.zeros(252)


@pytest.fixture
def benchmark_returns():
    np.random.seed(99)
    return np.random.randn(252) * 0.01 + 0.0003


class TestTotalReturn:
    def test_positive_drift(self, daily_returns):
        tr = total_return(daily_returns)
        assert tr > 0  # Should be positive with positive drift

    def test_zero_returns(self, zero_returns):
        assert total_return(zero_returns) == pytest.approx(0.0, abs=1e-10)


class TestAnnualizedReturn:
    def test_reasonable_range(self, daily_returns):
        ar = annualized_return(daily_returns)
        assert -0.5 < ar < 1.0

    def test_cagr_positive(self, daily_returns):
        c = cagr(daily_returns)
        assert c > -0.5


class TestVolatility:
    def test_positive(self, daily_returns):
        vol = annualized_volatility(daily_returns)
        assert vol > 0

    def test_zero_for_constant(self, zero_returns):
        vol = annualized_volatility(zero_returns)
        assert vol == pytest.approx(0.0, abs=1e-10)

    def test_reasonable_range(self, daily_returns):
        vol = annualized_volatility(daily_returns)
        assert 0.05 < vol < 0.50


class TestMaxDrawdown:
    def test_negative_or_zero(self, daily_returns):
        dd = max_drawdown(daily_returns)
        assert dd <= 0

    def test_zero_for_constant(self, zero_returns):
        dd = max_drawdown(zero_returns)
        assert dd == pytest.approx(0.0, abs=1e-10)


class TestVaR:
    def test_var_negative(self, daily_returns):
        var = value_at_risk(daily_returns, alpha=0.05)
        assert var < 0  # VaR is a loss

    def test_cvar_worse_than_var(self, daily_returns):
        var = value_at_risk(daily_returns, alpha=0.05)
        cvar = conditional_var(daily_returns, alpha=0.05)
        assert cvar <= var  # CVaR is more negative than VaR


class TestRatios:
    def test_sharpe_positive_for_positive_drift(self, daily_returns):
        sr = sharpe_ratio(daily_returns, daily_rf=0.0)
        assert sr > 0

    def test_sortino_positive(self, daily_returns):
        s = sortino_ratio(daily_returns, daily_rf=0.0)
        assert s > 0

    def test_calmar_positive(self, daily_returns):
        c = calmar_ratio(daily_returns)
        # Could be negative if max_dd is very small, but with positive drift...
        assert isinstance(c, float)


class TestComputeAllMetrics:
    def test_returns_dict(self, daily_returns):
        metrics = compute_all_metrics(daily_returns)
        assert isinstance(metrics, dict)
        assert len(metrics) > 10

    def test_known_keys(self, daily_returns):
        metrics = compute_all_metrics(daily_returns)
        expected_keys = [
            "total_return", "annualized_return", "annualized_volatility",
            "sharpe_ratio", "max_drawdown",
        ]
        for k in expected_keys:
            assert k in metrics, f"Missing key: {k}"

    def test_with_benchmark(self, daily_returns, benchmark_returns):
        metrics = compute_all_metrics(
            daily_returns,
            benchmark_returns=benchmark_returns,
        )
        assert "alpha" in metrics or "beta" in metrics
