"""Tests for risk budgeting optimization — Euler decomposition, custom budgets, ERC."""

import numpy as np
import pandas as pd
import pytest

from portopt.engine.risk_budgeting import (
    RiskBudgetConfig,
    RiskContribution,
    compute_risk_contributions,
    equal_risk_contribution,
    optimize_risk_budget,
)
from portopt.data.models import OptimizationResult
from tests.conftest import assert_valid_weights


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def equal_weights_5(symbols_5, sample_cov):
    """Equal weights for 5 assets, aligned with covariance matrix symbols."""
    symbols = list(sample_cov.columns)
    return {s: 0.2 for s in symbols}


@pytest.fixture
def equal_budgets_5(symbols_5, sample_cov):
    """Equal risk budgets for 5 assets."""
    symbols = list(sample_cov.columns)
    return {s: 0.2 for s in symbols}


# ── RiskBudgetConfig ──────────────────────────────────────────────────


class TestRiskBudgetConfig:
    def test_valid_config(self):
        config = RiskBudgetConfig(risk_budgets={"A": 0.5, "B": 0.5})
        assert config.risk_measure == "volatility"

    def test_budgets_not_summing_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            RiskBudgetConfig(risk_budgets={"A": 0.3, "B": 0.3})

    def test_negative_budgets(self):
        with pytest.raises(ValueError, match="non-negative"):
            RiskBudgetConfig(risk_budgets={"A": -0.2, "B": 1.2})

    def test_invalid_risk_measure(self):
        with pytest.raises(ValueError, match="Unsupported"):
            RiskBudgetConfig(risk_budgets={"A": 1.0}, risk_measure="invalid")


# ── compute_risk_contributions ────────────────────────────────────────


class TestComputeRiskContributions:
    def test_returns_list(self, equal_weights_5, sample_cov):
        rc = compute_risk_contributions(equal_weights_5, sample_cov)
        assert isinstance(rc, list)
        assert all(isinstance(r, RiskContribution) for r in rc)

    def test_one_entry_per_symbol(self, equal_weights_5, sample_cov):
        rc = compute_risk_contributions(equal_weights_5, sample_cov)
        symbols = {r.symbol for r in rc}
        assert symbols == set(sample_cov.columns)

    def test_risk_contributions_sum_to_one(self, equal_weights_5, sample_cov):
        """Risk contribution percentages should sum to ~1.0."""
        rc = compute_risk_contributions(equal_weights_5, sample_cov)
        total = sum(r.risk_contribution_pct for r in rc)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_all_fields_finite(self, equal_weights_5, sample_cov):
        rc = compute_risk_contributions(equal_weights_5, sample_cov)
        for r in rc:
            assert np.isfinite(r.weight)
            assert np.isfinite(r.marginal_risk)
            assert np.isfinite(r.risk_contribution)
            assert np.isfinite(r.risk_contribution_pct)

    def test_marginal_risk_positive(self, equal_weights_5, sample_cov):
        """For equal-weight long-only, marginal risk should be positive."""
        rc = compute_risk_contributions(equal_weights_5, sample_cov)
        for r in rc:
            assert r.marginal_risk > 0, f"{r.symbol} MRC = {r.marginal_risk}"

    def test_sorted_by_symbol(self, equal_weights_5, sample_cov):
        rc = compute_risk_contributions(equal_weights_5, sample_cov)
        symbols = [r.symbol for r in rc]
        assert symbols == sorted(symbols)

    def test_zero_weight_assets(self, sample_cov):
        symbols = list(sample_cov.columns)
        weights = {s: 0.0 for s in symbols}
        weights[symbols[0]] = 1.0
        rc = compute_risk_contributions(weights, sample_cov)
        # Only the single holding should have nonzero RC
        non_zero = [r for r in rc if r.risk_contribution != 0.0]
        assert len(non_zero) == 1
        assert non_zero[0].symbol == symbols[0]


# ── optimize_risk_budget ──────────────────────────────────────────────


class TestOptimizeRiskBudget:
    def test_result_type(self, expected_returns, sample_cov, equal_budgets_5):
        result = optimize_risk_budget(expected_returns, sample_cov, equal_budgets_5)
        assert isinstance(result, OptimizationResult)

    def test_method_label(self, expected_returns, sample_cov, equal_budgets_5):
        result = optimize_risk_budget(expected_returns, sample_cov, equal_budgets_5)
        assert result.method == "Risk Budget"

    def test_weights_valid(self, expected_returns, sample_cov, equal_budgets_5, symbols_5):
        result = optimize_risk_budget(expected_returns, sample_cov, equal_budgets_5)
        assert_valid_weights(result.weights, list(sample_cov.columns))

    def test_positive_volatility(self, expected_returns, sample_cov, equal_budgets_5):
        result = optimize_risk_budget(expected_returns, sample_cov, equal_budgets_5)
        assert result.volatility > 0

    def test_sharpe_ratio_finite(self, expected_returns, sample_cov, equal_budgets_5):
        result = optimize_risk_budget(expected_returns, sample_cov, equal_budgets_5)
        assert np.isfinite(result.sharpe_ratio)

    def test_metadata_has_budgets(self, expected_returns, sample_cov, equal_budgets_5):
        result = optimize_risk_budget(expected_returns, sample_cov, equal_budgets_5)
        assert "risk_budgets" in result.metadata
        assert "actual_risk_contributions" in result.metadata

    def test_unequal_budgets(self, expected_returns, sample_cov):
        """Test asymmetric risk budget allocation."""
        symbols = list(sample_cov.columns)
        n = len(symbols)
        # First asset gets 50%, rest split equally
        budgets = {symbols[0]: 0.5}
        rest = 0.5 / (n - 1)
        for s in symbols[1:]:
            budgets[s] = rest

        result = optimize_risk_budget(expected_returns, sample_cov, budgets)
        assert_valid_weights(result.weights, symbols)
        # The first asset should have a larger weight (more risk budget)
        assert result.weights[symbols[0]] > rest

    def test_budgets_not_summing_to_one_raises(self, expected_returns, sample_cov):
        symbols = list(sample_cov.columns)
        budgets = {s: 0.1 for s in symbols}  # sums to 0.5
        with pytest.raises(ValueError, match="sum to 1.0"):
            optimize_risk_budget(expected_returns, sample_cov, budgets)

    def test_single_asset(self):
        mu = pd.Series([0.10], index=["A"])
        cov = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
        result = optimize_risk_budget(mu, cov, {"A": 1.0})
        assert result.weights == {"A": 1.0}


# ── equal_risk_contribution ───────────────────────────────────────────


class TestEqualRiskContribution:
    def test_result_type(self, expected_returns, sample_cov):
        result = equal_risk_contribution(expected_returns, sample_cov)
        assert isinstance(result, OptimizationResult)

    def test_method_label(self, expected_returns, sample_cov):
        result = equal_risk_contribution(expected_returns, sample_cov)
        assert result.method == "Equal Risk Contribution"

    def test_weights_valid(self, expected_returns, sample_cov):
        result = equal_risk_contribution(expected_returns, sample_cov)
        assert_valid_weights(result.weights, list(sample_cov.columns))

    def test_risk_contributions_roughly_equal(self, expected_returns, sample_cov):
        """After ERC optimization, risk contributions should be approximately equal."""
        result = equal_risk_contribution(expected_returns, sample_cov)
        rc_pct = result.metadata.get("actual_risk_contributions", {})
        if rc_pct:
            values = list(rc_pct.values())
            target = 1.0 / len(values)
            for v in values:
                assert v == pytest.approx(target, abs=0.05), (
                    f"RC% = {v:.4f}, expected ~{target:.4f}"
                )

    def test_positive_volatility(self, expected_returns, sample_cov):
        result = equal_risk_contribution(expected_returns, sample_cov)
        assert result.volatility > 0

    def test_metadata_has_budgets(self, expected_returns, sample_cov):
        result = equal_risk_contribution(expected_returns, sample_cov)
        assert "risk_budgets" in result.metadata
        budgets = result.metadata["risk_budgets"]
        n = len(sample_cov.columns)
        for v in budgets.values():
            assert v == pytest.approx(1.0 / n)

    def test_single_asset(self):
        mu = pd.Series([0.10], index=["A"])
        cov = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
        result = equal_risk_contribution(mu, cov)
        assert result.weights == {"A": 1.0}

    def test_two_assets(self):
        """Two-asset ERC should give weights inversely proportional to volatility."""
        mu = pd.Series([0.10, 0.12], index=["A", "B"])
        # A has vol=10%, B has vol=20% with zero correlation
        cov = pd.DataFrame(
            [[0.01, 0.0], [0.0, 0.04]],
            index=["A", "B"], columns=["A", "B"],
        )
        result = equal_risk_contribution(mu, cov)
        # A (lower vol) should get higher weight
        assert result.weights["A"] > result.weights["B"]
