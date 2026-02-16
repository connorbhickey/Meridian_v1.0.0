"""Tests for return estimators."""

import numpy as np
import pandas as pd
import pytest

from portopt.constants import ReturnEstimator
from portopt.engine.returns import estimate_returns, log_returns, simple_returns


class TestSimpleReturns:
    def test_shape(self, prices_5):
        ret = simple_returns(prices_5)
        assert ret.shape == (len(prices_5) - 1, 5)

    def test_no_nans(self, prices_5):
        ret = simple_returns(prices_5)
        assert not ret.isna().any().any()

    def test_correct_first_return(self, prices_5):
        ret = simple_returns(prices_5)
        sym = prices_5.columns[0]
        expected = prices_5[sym].iloc[1] / prices_5[sym].iloc[0] - 1
        assert abs(ret[sym].iloc[0] - expected) < 1e-10


class TestLogReturns:
    def test_shape(self, prices_5):
        ret = log_returns(prices_5)
        assert ret.shape == (len(prices_5) - 1, 5)

    def test_close_to_simple_for_small_returns(self, prices_5):
        lr = log_returns(prices_5)
        sr = simple_returns(prices_5)
        # For small returns, log and simple should be very close
        diff = (lr - sr).abs().mean().mean()
        assert diff < 0.01


class TestEstimateReturns:
    def test_historical_mean(self, prices_5):
        mu = estimate_returns(prices_5, method=ReturnEstimator.HISTORICAL_MEAN)
        assert isinstance(mu, pd.Series)
        assert len(mu) == 5
        assert not mu.isna().any()

    def test_returns_are_annualized(self, prices_5):
        mu = estimate_returns(prices_5, method=ReturnEstimator.HISTORICAL_MEAN)
        # Annualized returns should be in reasonable range (-50% to +100%)
        assert (mu > -0.5).all()
        assert (mu < 1.0).all()

    def test_exponential_method(self, prices_5):
        mu = estimate_returns(prices_5, method=ReturnEstimator.EXPONENTIAL)
        assert isinstance(mu, pd.Series)
        assert len(mu) == 5

    def test_capm_method(self, prices_5):
        mu = estimate_returns(prices_5, method=ReturnEstimator.CAPM)
        assert isinstance(mu, pd.Series)
        assert len(mu) == 5
