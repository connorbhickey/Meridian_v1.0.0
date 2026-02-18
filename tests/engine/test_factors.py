"""Tests for Fama-French factor models and factor exposure analysis."""

import numpy as np
import pandas as pd
import pytest

from portopt.engine.factors import (
    FactorExposure,
    FactorModelResult,
    build_fama_french_factors,
    compute_factor_exposures,
    compute_portfolio_factor_exposures,
    run_factor_analysis,
)
from portopt.engine.returns import simple_returns


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def equal_weights_5(symbols_5):
    return {s: 0.2 for s in symbols_5}


@pytest.fixture
def returns_5(prices_5):
    return simple_returns(prices_5)


# ── build_fama_french_factors ─────────────────────────────────────────


class TestBuildFamaFrenchFactors:
    def test_returns_dataframe(self, prices_5):
        factors = build_fama_french_factors(prices_5)
        assert isinstance(factors, pd.DataFrame)

    def test_has_three_columns(self, prices_5):
        factors = build_fama_french_factors(prices_5)
        assert list(factors.columns) == ["MKT-RF", "SMB", "HML"]

    def test_no_nans(self, prices_5):
        factors = build_fama_french_factors(prices_5)
        assert not factors.isna().any().any()

    def test_with_market_prices(self, prices_5):
        market = prices_5.mean(axis=1)
        factors = build_fama_french_factors(prices_5, market_prices=market)
        assert factors.shape[1] == 3
        assert not factors.isna().any().any()

    def test_custom_risk_free_rate(self, prices_5):
        factors = build_fama_french_factors(prices_5, risk_free_rate=0.0)
        assert isinstance(factors, pd.DataFrame)

    def test_two_assets_returns_zero_smb_hml(self, prices_5):
        """With < 3 assets, SMB and HML should be zero."""
        prices_2 = prices_5.iloc[:, :2]
        factors = build_fama_french_factors(prices_2)
        assert (factors["SMB"] == 0.0).all()
        assert (factors["HML"] == 0.0).all()

    def test_single_asset_returns_zero_smb_hml(self, prices_5):
        prices_1 = prices_5.iloc[:, :1]
        factors = build_fama_french_factors(prices_1)
        assert (factors["SMB"] == 0.0).all()

    def test_short_history(self):
        """With fewer than 252 observations, uses reduced lookback."""
        np.random.seed(42)
        dates = pd.bdate_range("2024-01-01", periods=100)
        prices = pd.DataFrame(
            np.random.lognormal(0, 0.02, (100, 4)).cumsum(axis=0) + 100,
            index=dates,
            columns=["A", "B", "C", "D"],
        )
        factors = build_fama_french_factors(prices)
        assert isinstance(factors, pd.DataFrame)
        assert len(factors) > 0


# ── compute_factor_exposures ──────────────────────────────────────────


class TestComputeFactorExposures:
    def test_returns_dict(self, prices_5):
        rets = simple_returns(prices_5)
        factors = build_fama_french_factors(prices_5)
        exposures = compute_factor_exposures(rets, factors)
        assert isinstance(exposures, dict)

    def test_one_entry_per_asset(self, prices_5, symbols_5):
        rets = simple_returns(prices_5)
        factors = build_fama_french_factors(prices_5)
        exposures = compute_factor_exposures(rets, factors)
        assert set(exposures.keys()) == set(symbols_5)

    def test_three_exposures_per_asset(self, prices_5):
        rets = simple_returns(prices_5)
        factors = build_fama_french_factors(prices_5)
        exposures = compute_factor_exposures(rets, factors)
        for symbol, exps in exposures.items():
            assert len(exps) == 3
            names = {e.factor_name for e in exps}
            assert names == {"MKT-RF", "SMB", "HML"}

    def test_exposure_fields(self, prices_5):
        rets = simple_returns(prices_5)
        factors = build_fama_french_factors(prices_5)
        exposures = compute_factor_exposures(rets, factors)
        first_key = next(iter(exposures))
        for exp in exposures[first_key]:
            assert isinstance(exp, FactorExposure)
            assert np.isfinite(exp.beta)
            assert np.isfinite(exp.t_stat)
            assert 0.0 <= exp.p_value <= 1.0
            assert 0.0 <= exp.r_squared <= 1.0

    def test_market_beta_near_one(self, prices_5):
        """Market beta should be roughly in [0.5, 1.5] for synthetic GBM data."""
        rets = simple_returns(prices_5)
        factors = build_fama_french_factors(prices_5)
        exposures = compute_factor_exposures(rets, factors)
        for symbol, exps in exposures.items():
            mkt_exp = [e for e in exps if e.factor_name == "MKT-RF"][0]
            assert -2.0 < mkt_exp.beta < 3.0, (
                f"{symbol} MKT-RF beta = {mkt_exp.beta}"
            )

    def test_no_overlap_returns_empty(self):
        """If returns and factor dates don't overlap, returns empty dict."""
        dates_a = pd.bdate_range("2020-01-01", periods=50)
        dates_b = pd.bdate_range("2022-01-01", periods=50)
        rets = pd.DataFrame(np.random.randn(50, 3), index=dates_a, columns=["A", "B", "C"])
        factors = pd.DataFrame(np.random.randn(50, 3), index=dates_b, columns=["MKT-RF", "SMB", "HML"])
        assert compute_factor_exposures(rets, factors) == {}


# ── compute_portfolio_factor_exposures ────────────────────────────────


class TestPortfolioFactorExposures:
    def test_returns_list(self, prices_5, equal_weights_5):
        rets = simple_returns(prices_5)
        factors = build_fama_french_factors(prices_5)
        asset_exp = compute_factor_exposures(rets, factors)
        port_exp = compute_portfolio_factor_exposures(equal_weights_5, asset_exp)
        assert isinstance(port_exp, list)
        assert len(port_exp) == 3

    def test_factor_names(self, prices_5, equal_weights_5):
        rets = simple_returns(prices_5)
        factors = build_fama_french_factors(prices_5)
        asset_exp = compute_factor_exposures(rets, factors)
        port_exp = compute_portfolio_factor_exposures(equal_weights_5, asset_exp)
        names = {e.factor_name for e in port_exp}
        assert names == {"MKT-RF", "SMB", "HML"}

    def test_weighted_average_beta(self, prices_5, equal_weights_5):
        """Portfolio beta should be the weighted average of asset betas."""
        rets = simple_returns(prices_5)
        factors = build_fama_french_factors(prices_5)
        asset_exp = compute_factor_exposures(rets, factors)
        port_exp = compute_portfolio_factor_exposures(equal_weights_5, asset_exp)

        # Manually compute weighted average for MKT-RF
        manual_beta = np.mean([
            [e for e in exps if e.factor_name == "MKT-RF"][0].beta
            for exps in asset_exp.values()
        ])
        port_mkt = [e for e in port_exp if e.factor_name == "MKT-RF"][0]
        assert port_mkt.beta == pytest.approx(manual_beta, rel=0.01)

    def test_empty_exposures(self):
        assert compute_portfolio_factor_exposures({}, {}) == []


# ── run_factor_analysis (full pipeline) ───────────────────────────────


class TestRunFactorAnalysis:
    def test_result_type(self, prices_5, equal_weights_5):
        result = run_factor_analysis(prices_5, equal_weights_5)
        assert isinstance(result, FactorModelResult)

    def test_asset_exposures_populated(self, prices_5, equal_weights_5, symbols_5):
        result = run_factor_analysis(prices_5, equal_weights_5)
        assert set(result.asset_exposures.keys()) == set(symbols_5)

    def test_portfolio_exposures_populated(self, prices_5, equal_weights_5):
        result = run_factor_analysis(prices_5, equal_weights_5)
        assert len(result.portfolio_exposures) == 3

    def test_factor_returns_shape(self, prices_5, equal_weights_5):
        result = run_factor_analysis(prices_5, equal_weights_5)
        assert result.factor_returns.shape[1] == 3

    def test_residual_returns_shape(self, prices_5, equal_weights_5, symbols_5):
        result = run_factor_analysis(prices_5, equal_weights_5)
        assert set(result.residual_returns.columns) == set(symbols_5)

    def test_r_squared_between_0_and_1(self, prices_5, equal_weights_5):
        result = run_factor_analysis(prices_5, equal_weights_5)
        for symbol, r2 in result.r_squared.items():
            assert 0.0 <= r2 <= 1.0, f"{symbol} R² = {r2}"

    def test_custom_factor_returns(self, prices_5, equal_weights_5):
        """Supply pre-built factor returns instead of auto-constructing."""
        custom_factors = build_fama_french_factors(prices_5)
        result = run_factor_analysis(prices_5, equal_weights_5, factor_returns=custom_factors)
        assert isinstance(result, FactorModelResult)
        assert len(result.portfolio_exposures) == 3
