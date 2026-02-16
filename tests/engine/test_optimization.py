"""Tests for optimization methods — MVO, HRP, HERC, BL, TIC."""

import numpy as np
import pandas as pd
import pytest

from portopt.constants import (
    CovEstimator, LinkageMethod, OptMethod, ReturnEstimator, RiskMeasure,
)
from portopt.data.models import OptimizationResult
from portopt.engine.constraints import PortfolioConstraints
from portopt.engine.optimization.black_litterman import (
    BLView,
    BlackLittermanModel,
)
from portopt.engine.optimization.herc import herc_optimize
from portopt.engine.optimization.hrp import hrp_optimize
from portopt.engine.optimization.mean_variance import MeanVarianceOptimizer
from portopt.engine.optimization.tic import theory_implied_correlation
from portopt.engine.returns import estimate_returns
from portopt.engine.risk import estimate_covariance
from tests.conftest import assert_valid_weights


# ── MVO ────────────────────────────────────────────────────────────────

class TestMeanVarianceOptimizer:
    @pytest.mark.parametrize("method", [
        OptMethod.INVERSE_VARIANCE,
        OptMethod.MIN_VOLATILITY,
        OptMethod.MAX_SHARPE,
        OptMethod.MAX_DIVERSIFICATION,
        OptMethod.MAX_DECORRELATION,
        OptMethod.MAX_QUADRATIC_UTILITY,
    ])
    def test_basic_methods(self, prices_5, method):
        mu = estimate_returns(prices_5)
        cov = estimate_covariance(prices_5)
        opt = MeanVarianceOptimizer(mu, cov, method=method)
        result = opt.optimize()

        assert isinstance(result, OptimizationResult)
        assert_valid_weights(result.weights, list(prices_5.columns))
        assert result.volatility > 0
        assert isinstance(result.sharpe_ratio, float)

    def test_efficient_return(self, prices_5):
        mu = estimate_returns(prices_5)
        cov = estimate_covariance(prices_5)
        # Efficient Return needs target_risk (find return for given risk)
        constraints = PortfolioConstraints(target_risk=0.15)
        opt = MeanVarianceOptimizer(mu, cov, constraints=constraints,
                                    method=OptMethod.EFFICIENT_RETURN)
        result = opt.optimize()
        assert isinstance(result, OptimizationResult)
        assert_valid_weights(result.weights, list(prices_5.columns))

    def test_efficient_risk(self, prices_5):
        mu = estimate_returns(prices_5)
        cov = estimate_covariance(prices_5)
        # Efficient Risk needs target_return (find risk for given return)
        constraints = PortfolioConstraints(target_return=0.10)
        opt = MeanVarianceOptimizer(mu, cov, constraints=constraints,
                                    method=OptMethod.EFFICIENT_RISK)
        result = opt.optimize()
        assert isinstance(result, OptimizationResult)

    def test_min_vol_lower_than_max_sharpe(self, prices_5):
        mu = estimate_returns(prices_5)
        cov = estimate_covariance(prices_5)
        min_vol = MeanVarianceOptimizer(mu, cov, method=OptMethod.MIN_VOLATILITY).optimize()
        max_sh = MeanVarianceOptimizer(mu, cov, method=OptMethod.MAX_SHARPE).optimize()
        assert min_vol.volatility <= max_sh.volatility + 1e-4

    def test_efficient_frontier(self, prices_5):
        mu = estimate_returns(prices_5)
        cov = estimate_covariance(prices_5)
        opt = MeanVarianceOptimizer(mu, cov, method=OptMethod.MAX_SHARPE)
        frontier = opt.efficient_frontier(n_points=20)
        assert len(frontier) == 20
        # Frontier should be sorted by risk
        risks = [p.volatility for p in frontier]
        assert risks == sorted(risks)

    def test_weight_bounds(self, prices_5):
        mu = estimate_returns(prices_5)
        cov = estimate_covariance(prices_5)
        constraints = PortfolioConstraints(min_weight=0.05, max_weight=0.40)
        opt = MeanVarianceOptimizer(mu, cov, constraints=constraints,
                                    method=OptMethod.MAX_SHARPE)
        result = opt.optimize()
        for w in result.weights.values():
            assert w >= 0.05 - 1e-4
            assert w <= 0.40 + 1e-4


# ── HRP ────────────────────────────────────────────────────────────────

class TestHRP:
    def test_basic(self, prices_5):
        cov = estimate_covariance(prices_5)
        result = hrp_optimize(cov)
        assert isinstance(result, OptimizationResult)
        assert_valid_weights(result.weights, list(prices_5.columns))

    @pytest.mark.parametrize("linkage", list(LinkageMethod))
    def test_linkage_methods(self, prices_5, linkage):
        cov = estimate_covariance(prices_5)
        result = hrp_optimize(cov, linkage_method=linkage)
        assert_valid_weights(result.weights, list(prices_5.columns))

    def test_metadata_has_linkage(self, prices_5):
        cov = estimate_covariance(prices_5)
        result = hrp_optimize(cov)
        assert "linkage_matrix" in result.metadata


# ── HERC ───────────────────────────────────────────────────────────────

class TestHERC:
    def test_basic(self, prices_5):
        cov = estimate_covariance(prices_5)
        result = herc_optimize(cov)
        assert isinstance(result, OptimizationResult)
        assert_valid_weights(result.weights, list(prices_5.columns))

    @pytest.mark.parametrize("risk_measure", [RiskMeasure.VARIANCE, RiskMeasure.STD_DEV])
    def test_risk_measures(self, prices_5, risk_measure):
        cov = estimate_covariance(prices_5)
        returns = prices_5.pct_change().dropna()
        result = herc_optimize(cov, risk_measure=risk_measure, returns=returns)
        assert_valid_weights(result.weights, list(prices_5.columns))


# ── Black-Litterman ────────────────────────────────────────────────────

class TestBlackLitterman:
    def test_equilibrium_returns(self, prices_5):
        cov = estimate_covariance(prices_5)
        bl = BlackLittermanModel(cov)
        pi = bl.equilibrium_returns()
        assert isinstance(pi, pd.Series)
        assert len(pi) == 5

    def test_absolute_view(self, prices_5):
        cov = estimate_covariance(prices_5)
        bl = BlackLittermanModel(cov)
        views = [BLView(
            assets=["AAPL"],
            weights=[1.0],
            view_return=0.15,
            confidence=0.8,
        )]
        posterior_mu, posterior_cov = bl.posterior(views)
        assert isinstance(posterior_mu, pd.Series)
        assert len(posterior_mu) == 5

    def test_relative_view(self, prices_5):
        cov = estimate_covariance(prices_5)
        bl = BlackLittermanModel(cov)
        views = [BLView(
            assets=["AAPL", "MSFT"],
            weights=[1.0, -1.0],
            view_return=0.05,
            confidence=0.5,
        )]
        posterior_mu, _ = bl.posterior(views)
        assert isinstance(posterior_mu, pd.Series)


# ── TIC ────────────────────────────────────────────────────────────────

class TestTIC:
    def test_basic(self, prices_5):
        cov = estimate_covariance(prices_5)
        tic_corr = theory_implied_correlation(cov)
        assert isinstance(tic_corr, pd.DataFrame)
        assert tic_corr.shape == (5, 5)
        # Diagonal should be 1
        np.testing.assert_allclose(np.diag(tic_corr.values), 1.0, atol=1e-6)

    @pytest.mark.parametrize("linkage", [LinkageMethod.SINGLE, LinkageMethod.WARD])
    def test_linkage_methods(self, prices_5, linkage):
        cov = estimate_covariance(prices_5)
        tic_corr = theory_implied_correlation(cov, linkage_method=linkage)
        assert tic_corr.shape == (5, 5)
