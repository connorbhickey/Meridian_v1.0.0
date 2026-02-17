"""Tests for rolling window analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portopt.engine.rolling import (
    rolling_beta,
    rolling_correlation,
    rolling_max_drawdown,
    rolling_sharpe,
    rolling_sortino,
    rolling_volatility,
)

from tests.conftest import _make_prices


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def daily_returns() -> pd.Series:
    """Single-asset daily returns, 504 trading days."""
    prices = _make_prices(["TEST"], n_days=504)
    return prices["TEST"].pct_change().dropna()


@pytest.fixture
def two_returns() -> tuple[pd.Series, pd.Series]:
    """Two-asset daily returns for correlation/beta tests."""
    prices = _make_prices(["A", "B"], n_days=504)
    rets = prices.pct_change().dropna()
    return rets["A"], rets["B"]


# ── Rolling Sharpe ──────────────────────────────────────────────────────

class TestRollingSharpe:
    def test_output_length(self, daily_returns):
        result = rolling_sharpe(daily_returns, window=63)
        non_nan = result.dropna()
        expected_len = len(daily_returns) - 63 + 1
        assert len(non_nan) == expected_len

    def test_reasonable_range(self, daily_returns):
        result = rolling_sharpe(daily_returns, window=63).dropna()
        assert result.max() < 10.0, "Sharpe too high"
        assert result.min() > -10.0, "Sharpe too low"

    def test_short_window(self, daily_returns):
        result = rolling_sharpe(daily_returns, window=21)
        assert len(result.dropna()) > 0


# ── Rolling Sortino ─────────────────────────────────────────────────────

class TestRollingSortino:
    def test_output_length(self, daily_returns):
        result = rolling_sortino(daily_returns, window=63)
        non_nan = result.dropna()
        expected_len = len(daily_returns) - 63 + 1
        assert len(non_nan) == expected_len

    def test_sortino_not_all_zero(self, daily_returns):
        result = rolling_sortino(daily_returns, window=63).dropna()
        assert not np.allclose(result.values, 0)


# ── Rolling Volatility ─────────────────────────────────────────────────

class TestRollingVolatility:
    def test_all_positive(self, daily_returns):
        result = rolling_volatility(daily_returns, window=63).dropna()
        assert (result >= 0).all(), "Volatility should be non-negative"

    def test_output_length(self, daily_returns):
        result = rolling_volatility(daily_returns, window=63)
        non_nan = result.dropna()
        expected_len = len(daily_returns) - 63 + 1
        assert len(non_nan) == expected_len


# ── Rolling Max Drawdown ───────────────────────────────────────────────

class TestRollingMaxDrawdown:
    def test_all_non_positive(self, daily_returns):
        result = rolling_max_drawdown(daily_returns, window=63).dropna()
        assert (result <= 0).all(), "Max drawdown should be <= 0"

    def test_output_length(self, daily_returns):
        result = rolling_max_drawdown(daily_returns, window=63)
        non_nan = result.dropna()
        expected_len = len(daily_returns) - 63 + 1
        assert len(non_nan) == expected_len


# ── Rolling Beta ───────────────────────────────────────────────────────

class TestRollingBeta:
    def test_self_beta_near_one(self, daily_returns):
        result = rolling_beta(daily_returns, daily_returns, window=63).dropna()
        assert np.allclose(result.values, 1.0, atol=1e-6), \
            "Beta of series vs itself should be ~1.0"

    def test_output_length(self, two_returns):
        a, b = two_returns
        result = rolling_beta(a, b, window=63).dropna()
        assert len(result) > 0


# ── Rolling Correlation ────────────────────────────────────────────────

class TestRollingCorrelation:
    def test_self_correlation_one(self, daily_returns):
        result = rolling_correlation(daily_returns, daily_returns, window=63).dropna()
        assert np.allclose(result.values, 1.0, atol=1e-6), \
            "Correlation of series with itself should be ~1.0"

    def test_range(self, two_returns):
        a, b = two_returns
        result = rolling_correlation(a, b, window=63).dropna()
        assert (result >= -1.0 - 1e-10).all(), "Correlation below -1"
        assert (result <= 1.0 + 1e-10).all(), "Correlation above 1"

    def test_output_length(self, two_returns):
        a, b = two_returns
        result = rolling_correlation(a, b, window=63).dropna()
        assert len(result) > 0


# ── Edge Cases ─────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_series(self):
        empty = pd.Series(dtype=float)
        assert len(rolling_sharpe(empty, window=63)) == 0
        assert len(rolling_volatility(empty, window=63)) == 0
        assert len(rolling_max_drawdown(empty, window=63)) == 0

    def test_series_shorter_than_window(self):
        short = pd.Series(np.random.randn(10))
        assert len(rolling_sharpe(short, window=63)) == 0
        assert len(rolling_beta(short, short, window=63)) == 0

    def test_mismatched_lengths(self):
        a = pd.Series(np.random.randn(200), index=pd.date_range("2020-01-01", periods=200))
        b = pd.Series(np.random.randn(150), index=pd.date_range("2020-01-01", periods=150))
        result = rolling_correlation(a, b, window=63)
        # Should handle gracefully (align on common dates)
        assert isinstance(result, pd.Series)
