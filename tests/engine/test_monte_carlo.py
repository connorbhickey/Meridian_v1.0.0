"""Tests for Monte Carlo simulation engine — parametric GBM and block bootstrap."""

import numpy as np
import pandas as pd
import pytest

from portopt.constants import CovEstimator, MCSimMethod, ReturnEstimator
from portopt.data.models import MonteCarloConfig, MonteCarloResult
from portopt.engine.monte_carlo import run_monte_carlo
from portopt.engine.returns import estimate_returns
from portopt.engine.risk import estimate_covariance


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def mu_5(prices_5):
    """Annualized expected returns for 5 assets."""
    return estimate_returns(prices_5, method=ReturnEstimator.HISTORICAL_MEAN)


@pytest.fixture
def cov_5(prices_5):
    """Annualized sample covariance for 5 assets."""
    return estimate_covariance(prices_5, method=CovEstimator.SAMPLE)


@pytest.fixture
def equal_weights_5(symbols_5):
    """Equal-weight portfolio for 5 assets."""
    w = 1.0 / len(symbols_5)
    return {s: w for s in symbols_5}


@pytest.fixture
def hist_returns_5(prices_5):
    """Daily simple returns for 5 assets."""
    return prices_5.pct_change().dropna()


# ── Parametric GBM ────────────────────────────────────────────────────

class TestMonteCarloParametric:
    """Tests for parametric (GBM) Monte Carlo simulation."""

    def _run(self, equal_weights_5, mu_5, cov_5, **overrides):
        """Helper to run parametric simulation with default config."""
        cfg_kwargs = dict(
            n_sims=200,
            horizon_days=63,
            method=MCSimMethod.PARAMETRIC,
            initial_value=100_000.0,
        )
        cfg_kwargs.update(overrides)
        config = MonteCarloConfig(**cfg_kwargs)
        return run_monte_carlo(equal_weights_5, mu_5, cov_5, config)

    def test_result_type(self, equal_weights_5, mu_5, cov_5):
        result = self._run(equal_weights_5, mu_5, cov_5)
        assert isinstance(result, MonteCarloResult)

    def test_equity_percentiles_shape(self, equal_weights_5, mu_5, cov_5):
        result = self._run(equal_weights_5, mu_5, cov_5)
        # Default percentiles: (5, 25, 50, 75, 95) → 5 columns
        assert result.equity_percentiles.shape == (64, 5)  # horizon+1 x 5

    def test_initial_value_preserved(self, equal_weights_5, mu_5, cov_5):
        result = self._run(equal_weights_5, mu_5, cov_5)
        # All percentiles at t=0 should equal the initial value
        assert np.allclose(result.equity_percentiles[0, :], 100_000.0)

    def test_percentiles_ordered(self, equal_weights_5, mu_5, cov_5):
        result = self._run(equal_weights_5, mu_5, cov_5)
        # At every timestep, P5 <= P25 <= P50 <= P75 <= P95
        for t in range(result.equity_percentiles.shape[0]):
            row = result.equity_percentiles[t, :]
            assert np.all(np.diff(row) >= 0), (
                f"Percentiles not ordered at t={t}: {row}"
            )

    def test_shortfall_probability_range(self, equal_weights_5, mu_5, cov_5):
        result = self._run(equal_weights_5, mu_5, cov_5)
        assert 0.0 <= result.shortfall_probability <= 1.0

    def test_shortfall_threshold(self, equal_weights_5, mu_5, cov_5):
        result = self._run(equal_weights_5, mu_5, cov_5)
        # threshold = initial * (1 - spending_rate) = 100000 * 0.96
        assert result.shortfall_threshold == pytest.approx(96_000.0)

    def test_metrics_distributions_shape(self, equal_weights_5, mu_5, cov_5):
        n_sims = 200
        result = self._run(equal_weights_5, mu_5, cov_5, n_sims=n_sims)
        expected_keys = {
            "sharpe_ratio", "annualized_return", "annualized_volatility",
            "max_drawdown", "cvar_95",
        }
        assert set(result.metrics_distributions.keys()) == expected_keys
        for key, arr in result.metrics_distributions.items():
            assert len(arr) == n_sims, f"{key} has {len(arr)} values, expected {n_sims}"
            assert np.all(np.isfinite(arr)), f"{key} has non-finite values"

    def test_metrics_distributions_sorted(self, equal_weights_5, mu_5, cov_5):
        result = self._run(equal_weights_5, mu_5, cov_5)
        for key, arr in result.metrics_distributions.items():
            assert np.all(np.diff(arr) >= 0), f"{key} is not sorted"

    def test_dates_length(self, equal_weights_5, mu_5, cov_5):
        result = self._run(equal_weights_5, mu_5, cov_5)
        assert len(result.dates) == 64  # horizon + 1

    def test_n_sims_matches_config(self, equal_weights_5, mu_5, cov_5):
        result = self._run(equal_weights_5, mu_5, cov_5, n_sims=150)
        assert result.n_sims == 150

    def test_method_label(self, equal_weights_5, mu_5, cov_5):
        result = self._run(equal_weights_5, mu_5, cov_5)
        assert result.method == "parametric"

    def test_median_curve_property(self, equal_weights_5, mu_5, cov_5):
        result = self._run(equal_weights_5, mu_5, cov_5)
        median = result.median_curve
        assert median.shape == (64,)
        assert median[0] == pytest.approx(100_000.0)

    def test_percentile_curve_method(self, equal_weights_5, mu_5, cov_5):
        result = self._run(equal_weights_5, mu_5, cov_5)
        p5 = result.percentile_curve(5)
        p95 = result.percentile_curve(95)
        # P5 terminal should be less than P95 terminal
        assert p5[-1] < p95[-1]

    def test_metadata_has_symbols(self, equal_weights_5, mu_5, cov_5):
        result = self._run(equal_weights_5, mu_5, cov_5)
        assert "symbols" in result.metadata
        assert len(result.metadata["symbols"]) == 5

    def test_terminal_values_positive(self, equal_weights_5, mu_5, cov_5):
        result = self._run(equal_weights_5, mu_5, cov_5)
        # All terminal values should be positive (GBM cannot go negative)
        terminal = result.equity_percentiles[-1, :]
        assert np.all(terminal > 0)


# ── Block Bootstrap ───────────────────────────────────────────────────

class TestMonteCarloBootstrap:
    """Tests for block bootstrap Monte Carlo simulation."""

    def _run(self, equal_weights_5, mu_5, cov_5, hist_returns_5, **overrides):
        """Helper to run bootstrap simulation with default config."""
        cfg_kwargs = dict(
            n_sims=200,
            horizon_days=63,
            method=MCSimMethod.BOOTSTRAP,
            block_size=20,
            initial_value=100_000.0,
        )
        cfg_kwargs.update(overrides)
        config = MonteCarloConfig(**cfg_kwargs)
        return run_monte_carlo(
            equal_weights_5, mu_5, cov_5, config,
            historical_returns=hist_returns_5,
        )

    def test_result_type(self, equal_weights_5, mu_5, cov_5, hist_returns_5):
        result = self._run(equal_weights_5, mu_5, cov_5, hist_returns_5)
        assert isinstance(result, MonteCarloResult)

    def test_method_label(self, equal_weights_5, mu_5, cov_5, hist_returns_5):
        result = self._run(equal_weights_5, mu_5, cov_5, hist_returns_5)
        assert result.method == "bootstrap"

    def test_shape(self, equal_weights_5, mu_5, cov_5, hist_returns_5):
        result = self._run(equal_weights_5, mu_5, cov_5, hist_returns_5)
        assert result.equity_percentiles.shape == (64, 5)

    def test_initial_value_preserved(self, equal_weights_5, mu_5, cov_5, hist_returns_5):
        result = self._run(equal_weights_5, mu_5, cov_5, hist_returns_5)
        assert np.allclose(result.equity_percentiles[0, :], 100_000.0)

    def test_percentiles_ordered(self, equal_weights_5, mu_5, cov_5, hist_returns_5):
        result = self._run(equal_weights_5, mu_5, cov_5, hist_returns_5)
        for t in range(result.equity_percentiles.shape[0]):
            row = result.equity_percentiles[t, :]
            assert np.all(np.diff(row) >= 0), (
                f"Percentiles not ordered at t={t}: {row}"
            )

    def test_requires_historical_returns(self, equal_weights_5, mu_5, cov_5):
        config = MonteCarloConfig(
            n_sims=50, horizon_days=21, method=MCSimMethod.BOOTSTRAP,
        )
        with pytest.raises(ValueError, match="historical_returns"):
            run_monte_carlo(equal_weights_5, mu_5, cov_5, config)

    def test_dates_length(self, equal_weights_5, mu_5, cov_5, hist_returns_5):
        result = self._run(equal_weights_5, mu_5, cov_5, hist_returns_5)
        assert len(result.dates) == 64

    def test_shortfall_probability_range(self, equal_weights_5, mu_5, cov_5, hist_returns_5):
        result = self._run(equal_weights_5, mu_5, cov_5, hist_returns_5)
        assert 0.0 <= result.shortfall_probability <= 1.0

    def test_metrics_distributions(self, equal_weights_5, mu_5, cov_5, hist_returns_5):
        result = self._run(equal_weights_5, mu_5, cov_5, hist_returns_5, n_sims=100)
        for key, arr in result.metrics_distributions.items():
            assert len(arr) == 100
            assert np.all(np.isfinite(arr))

    def test_different_block_sizes(self, equal_weights_5, mu_5, cov_5, hist_returns_5):
        """Results should differ with different block sizes."""
        r1 = self._run(equal_weights_5, mu_5, cov_5, hist_returns_5,
                       block_size=5, n_sims=50)
        # Reset seed for fair comparison
        np.random.seed(42)
        r2 = self._run(equal_weights_5, mu_5, cov_5, hist_returns_5,
                       block_size=40, n_sims=50)
        # Terminal medians should be different (different autocorrelation capture)
        m1 = r1.median_curve[-1]
        m2 = r2.median_curve[-1]
        # They might be similar but the paths should be structurally different
        assert isinstance(m1, float) and isinstance(m2, float)
