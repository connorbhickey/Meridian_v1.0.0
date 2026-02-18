"""Tests for HMM regime detection and rolling volatility regime classification."""

import numpy as np
import pandas as pd
import pytest

from portopt.engine.regime import (
    RegimeInfo,
    RegimeResult,
    detect_regimes,
    regime_conditional_parameters,
    rolling_regime,
)
from portopt.engine.returns import simple_returns


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def market_returns(prices_5):
    """Equal-weight market return series."""
    rets = simple_returns(prices_5)
    return rets.mean(axis=1)


@pytest.fixture
def asset_returns(prices_5):
    return simple_returns(prices_5)


# ── detect_regimes ────────────────────────────────────────────────────


class TestDetectRegimes:
    def test_result_type(self, market_returns):
        result = detect_regimes(market_returns, n_regimes=2)
        assert isinstance(result, RegimeResult)

    def test_two_regimes(self, market_returns):
        result = detect_regimes(market_returns, n_regimes=2)
        assert len(result.regimes) == 2
        names = [r.name for r in result.regimes]
        assert names == ["Bull", "Bear"]

    def test_three_regimes(self, market_returns):
        result = detect_regimes(market_returns, n_regimes=3)
        assert len(result.regimes) == 3

    def test_regime_sequence_length(self, market_returns):
        result = detect_regimes(market_returns, n_regimes=2)
        assert len(result.regime_sequence) == len(market_returns)

    def test_regime_probabilities_shape(self, market_returns):
        n_regimes = 3
        result = detect_regimes(market_returns, n_regimes=n_regimes)
        assert result.regime_probabilities.shape == (len(market_returns), n_regimes)

    def test_probabilities_sum_to_one(self, market_returns):
        result = detect_regimes(market_returns, n_regimes=3)
        row_sums = result.regime_probabilities.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_transition_matrix_shape(self, market_returns):
        n_regimes = 3
        result = detect_regimes(market_returns, n_regimes=n_regimes)
        assert result.transition_matrix.shape == (n_regimes, n_regimes)

    def test_transition_matrix_rows_sum_to_one(self, market_returns):
        result = detect_regimes(market_returns, n_regimes=3)
        row_sums = result.transition_matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_current_regime_valid(self, market_returns):
        result = detect_regimes(market_returns, n_regimes=3)
        assert 0 <= result.current_regime < 3
        assert result.current_regime_name in [r.name for r in result.regimes]

    def test_regimes_sorted_by_volatility(self, market_returns):
        result = detect_regimes(market_returns, n_regimes=3)
        vols = [r.volatility for r in result.regimes]
        assert vols == sorted(vols), f"Regimes not sorted by vol: {vols}"

    def test_stationary_probabilities_sum_to_one(self, market_returns):
        result = detect_regimes(market_returns, n_regimes=3)
        total = sum(r.stationary_prob for r in result.regimes)
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_bic_is_finite(self, market_returns):
        result = detect_regimes(market_returns, n_regimes=2)
        assert np.isfinite(result.bic)

    def test_dates_preserved(self, market_returns):
        result = detect_regimes(market_returns, n_regimes=2)
        assert len(result.dates) == len(market_returns)

    def test_numpy_array_input(self, market_returns):
        """Accept numpy array (not just pd.Series)."""
        arr = market_returns.values
        result = detect_regimes(arr, n_regimes=2)
        assert isinstance(result, RegimeResult)
        assert len(result.regime_sequence) == len(arr)

    def test_too_few_observations_raises(self):
        short = pd.Series(np.random.randn(10))
        with pytest.raises(ValueError, match="at least"):
            detect_regimes(short, n_regimes=2)

    def test_regime_colors_assigned(self, market_returns):
        result = detect_regimes(market_returns, n_regimes=3)
        for r in result.regimes:
            assert r.color.startswith("#")


# ── rolling_regime ────────────────────────────────────────────────────


class TestRollingRegime:
    def test_returns_series(self, market_returns):
        labels = rolling_regime(market_returns, window=63)
        assert isinstance(labels, pd.Series)
        assert len(labels) == len(market_returns)

    def test_valid_labels(self, market_returns):
        labels = rolling_regime(market_returns, window=63)
        valid = {"Low Vol", "Normal", "High Vol"}
        non_null = labels.dropna()
        assert set(non_null.unique()).issubset(valid)

    def test_initial_nans(self, market_returns):
        window = 63
        labels = rolling_regime(market_returns, window=window)
        assert labels.iloc[:window - 1].isna().all()

    def test_custom_window(self, market_returns):
        labels = rolling_regime(market_returns, window=21)
        non_null = labels.dropna()
        assert len(non_null) > 0


# ── regime_conditional_parameters ─────────────────────────────────────


class TestRegimeConditionalParameters:
    def test_returns_dict(self, asset_returns, prices_5, market_returns):
        regime_result = detect_regimes(market_returns, n_regimes=2)
        params = regime_conditional_parameters(asset_returns, prices_5, regime_result)
        assert isinstance(params, dict)

    def test_has_mu_and_cov(self, asset_returns, prices_5, market_returns):
        regime_result = detect_regimes(market_returns, n_regimes=2)
        params = regime_conditional_parameters(asset_returns, prices_5, regime_result)
        for regime_id, data in params.items():
            assert "mu" in data
            assert "cov" in data
            assert isinstance(data["mu"], pd.Series)
            assert isinstance(data["cov"], pd.DataFrame)

    def test_mu_is_annualized(self, asset_returns, prices_5, market_returns):
        """Mu values should be annualized (roughly in range [-1, 2])."""
        regime_result = detect_regimes(market_returns, n_regimes=2)
        params = regime_conditional_parameters(asset_returns, prices_5, regime_result)
        for regime_id, data in params.items():
            for val in data["mu"].values:
                assert -2.0 < val < 3.0, f"Regime {regime_id} mu = {val} looks non-annualized"

    def test_cov_positive_semidefinite(self, asset_returns, prices_5, market_returns):
        regime_result = detect_regimes(market_returns, n_regimes=2)
        params = regime_conditional_parameters(asset_returns, prices_5, regime_result)
        for regime_id, data in params.items():
            eigvals = np.linalg.eigvalsh(data["cov"].values)
            assert np.all(eigvals >= -1e-8), (
                f"Regime {regime_id} cov not PSD: min eigval = {eigvals.min()}"
            )
