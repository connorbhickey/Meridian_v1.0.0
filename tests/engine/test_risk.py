"""Tests for covariance estimators and related utilities."""

import numpy as np
import pandas as pd
import pytest

from portopt.constants import CovEstimator
from portopt.engine.risk import (
    cov_to_corr,
    estimate_covariance,
    is_positive_definite,
    nearest_positive_definite,
)
from tests.conftest import assert_positive_definite


class TestEstimateCovariance:
    @pytest.mark.parametrize("method", list(CovEstimator))
    def test_all_methods_produce_valid_cov(self, prices_5, method):
        cov = estimate_covariance(prices_5, method=method)
        assert isinstance(cov, pd.DataFrame)
        assert cov.shape == (5, 5)
        # Symmetric
        np.testing.assert_allclose(cov.values, cov.values.T, atol=1e-10)
        # PSD
        assert_positive_definite(cov)

    def test_sample_cov_annualized(self, prices_5):
        cov = estimate_covariance(prices_5, method=CovEstimator.SAMPLE)
        # Diagonal should be annualized variances (roughly 0.01 to 0.09)
        diag = np.diag(cov.values)
        assert (diag > 0).all()
        assert (diag < 1.0).all()  # Reasonable annualized variance

    def test_ledoit_wolf_shrinks(self, prices_5):
        sample = estimate_covariance(prices_5, method=CovEstimator.SAMPLE)
        lw = estimate_covariance(prices_5, method=CovEstimator.LEDOIT_WOLF)
        # Ledoit-Wolf should have smaller off-diagonal elements (shrinkage)
        off_diag_sample = np.abs(sample.values[np.triu_indices(5, 1)]).mean()
        off_diag_lw = np.abs(lw.values[np.triu_indices(5, 1)]).mean()
        assert off_diag_lw <= off_diag_sample * 1.1  # Allow small tolerance


class TestCovToCorr:
    def test_diagonal_is_one(self, sample_cov):
        corr = cov_to_corr(sample_cov)
        np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-10)

    def test_values_in_range(self, sample_cov):
        corr = cov_to_corr(sample_cov)
        assert (corr.values >= -1.0 - 1e-10).all()
        assert (corr.values <= 1.0 + 1e-10).all()


class TestPositiveDefinite:
    def test_identity_is_pd(self):
        assert is_positive_definite(np.eye(3))

    def test_non_pd_detected(self):
        bad = np.array([[1, 2], [2, 1]])
        assert not is_positive_definite(bad)

    def test_nearest_pd(self):
        bad = np.array([[1, 2], [2, 1]])
        fixed = nearest_positive_definite(bad)
        assert is_positive_definite(fixed)
