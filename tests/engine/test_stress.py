"""Tests for the stress testing engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portopt.engine.stress import (
    HISTORICAL_SCENARIOS,
    ReverseStressResult,
    StressResult,
    StressScenario,
    reverse_stress_test,
    run_all_stress_tests,
    run_stress_test,
)

from tests.conftest import _make_prices


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def equal_weights() -> dict[str, float]:
    return {"AAPL": 0.25, "MSFT": 0.25, "GOOG": 0.25, "AMZN": 0.25}


@pytest.fixture
def tech_heavy_weights() -> dict[str, float]:
    return {"QQQ": 0.60, "AGG": 0.30, "GLD": 0.10}


@pytest.fixture
def sector_map() -> dict[str, str]:
    return {
        "AAPL": "tech",
        "MSFT": "tech",
        "GOOG": "tech",
        "AMZN": "consumer",
        "JPM": "financials",
        "JNJ": "healthcare",
    }


@pytest.fixture
def sample_cov_4() -> pd.DataFrame:
    """Simple 4x4 covariance matrix."""
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN"]
    prices = _make_prices(symbols, n_days=252)
    returns = prices.pct_change().dropna()
    return returns.cov() * 252


@pytest.fixture
def gfc_scenario() -> StressScenario:
    return HISTORICAL_SCENARIOS["2008 GFC"]


# ── StressScenario Tests ────────────────────────────────────────────────

class TestStressScenarios:
    def test_stress_result_type(self, equal_weights, gfc_scenario):
        result = run_stress_test(equal_weights, gfc_scenario)
        assert isinstance(result, StressResult)

    def test_portfolio_impact_negative_for_crash(self, equal_weights, gfc_scenario):
        result = run_stress_test(equal_weights, gfc_scenario)
        assert result.portfolio_impact < 0, "GFC should produce negative portfolio impact"

    def test_asset_impacts_weighted_sum_equals_portfolio(self, equal_weights, gfc_scenario):
        result = run_stress_test(equal_weights, gfc_scenario)
        # Without cov, portfolio impact should be the exact weighted sum
        expected = sum(
            equal_weights[sym] * result.asset_impacts[sym]
            for sym in equal_weights
        )
        assert abs(result.portfolio_impact - expected) < 1e-10

    def test_stressed_value_consistent(self, equal_weights, gfc_scenario):
        initial = 100_000.0
        result = run_stress_test(equal_weights, gfc_scenario, initial_value=initial)
        expected_value = initial * (1.0 + result.portfolio_impact)
        assert abs(result.stressed_value - expected_value) < 0.01

    def test_all_named_scenarios_run(self, equal_weights):
        results = run_all_stress_tests(equal_weights)
        assert len(results) == len(HISTORICAL_SCENARIOS)
        for r in results:
            assert isinstance(r, StressResult)
            assert r.scenario.name in HISTORICAL_SCENARIOS

    def test_results_sorted_worst_first(self, equal_weights):
        results = run_all_stress_tests(equal_weights)
        impacts = [r.portfolio_impact for r in results]
        assert impacts == sorted(impacts), "Results should be sorted worst first"

    def test_custom_scenario(self, equal_weights):
        custom = StressScenario(
            name="Custom",
            description="Test custom scenario",
            shocks={"equity": -0.15, "tech": -0.20},
        )
        result = run_stress_test(equal_weights, custom)
        assert isinstance(result, StressResult)
        assert result.scenario.name == "Custom"

    def test_zero_shock_no_impact(self, equal_weights):
        zero = StressScenario(
            name="Zero", description="No shocks",
            shocks={"equity": 0.0, "tech": 0.0, "bond": 0.0, "gold": 0.0},
        )
        result = run_stress_test(equal_weights, zero)
        assert abs(result.portfolio_impact) < 1e-10
        assert abs(result.stressed_value - result.initial_value) < 0.01

    def test_sector_map_applied(self, sector_map, gfc_scenario):
        weights = {"AAPL": 0.50, "JPM": 0.50}
        result = run_stress_test(weights, gfc_scenario, sector_map=sector_map)
        # AAPL → tech (-0.45), JPM → financials (-0.55)
        assert abs(result.asset_impacts["AAPL"] - (-0.45)) < 1e-10
        assert abs(result.asset_impacts["JPM"] - (-0.55)) < 1e-10

    def test_equal_weight_portfolio_math(self, gfc_scenario):
        weights = {"SPY": 0.50, "AGG": 0.50}
        result = run_stress_test(weights, gfc_scenario)
        # SPY→equity(-0.40), AGG→bond(+0.05)
        expected = 0.50 * (-0.40) + 0.50 * 0.05
        assert abs(result.portfolio_impact - expected) < 1e-10

    def test_initial_value_respected(self, equal_weights, gfc_scenario):
        r1 = run_stress_test(equal_weights, gfc_scenario, initial_value=50_000)
        r2 = run_stress_test(equal_weights, gfc_scenario, initial_value=200_000)
        assert r1.initial_value == 50_000
        assert r2.initial_value == 200_000
        # Impact percentage should be the same
        assert abs(r1.portfolio_impact - r2.portfolio_impact) < 1e-10

    def test_cov_contagion_adjusts_impact(self, equal_weights, sample_cov_4, gfc_scenario):
        r_without = run_stress_test(equal_weights, gfc_scenario, cov=None)
        r_with = run_stress_test(equal_weights, gfc_scenario, cov=sample_cov_4)
        # With positive correlation, contagion should amplify the negative impact
        # (make it more negative or at minimum equal)
        assert r_with.portfolio_impact <= r_without.portfolio_impact + 1e-10

    def test_tech_heavy_portfolio(self, tech_heavy_weights):
        """Tech-heavy portfolio should suffer more in Dot-Com Bust than Flash Crash."""
        dotcom = HISTORICAL_SCENARIOS["Dot-Com Bust"]
        flash = HISTORICAL_SCENARIOS["Flash Crash"]
        r_dotcom = run_stress_test(tech_heavy_weights, dotcom)
        r_flash = run_stress_test(tech_heavy_weights, flash)
        assert r_dotcom.portfolio_impact < r_flash.portfolio_impact


# ── Reverse Stress Test ─────────────────────────────────────────────────

class TestReverseStress:
    def test_reverse_result_type(self, equal_weights, sample_cov_4):
        result = reverse_stress_test(
            equal_weights, target_drawdown=-0.20, cov=sample_cov_4, n_scenarios=50,
        )
        assert isinstance(result, ReverseStressResult)

    def test_finds_scenarios_near_target(self, equal_weights, sample_cov_4):
        target = -0.20
        result = reverse_stress_test(
            equal_weights, target_drawdown=target, cov=sample_cov_4, n_scenarios=200,
        )
        assert len(result.worst_scenarios) > 0
        # The worst scenarios should approximate the target drawdown
        for s in result.worst_scenarios:
            assert abs(s["impact"] - target) < 0.05, (
                f"Scenario impact {s['impact']:.4f} too far from target {target}"
            )

    def test_result_ordering(self, equal_weights, sample_cov_4):
        result = reverse_stress_test(
            equal_weights, target_drawdown=-0.15, cov=sample_cov_4, n_scenarios=100,
        )
        if len(result.worst_scenarios) > 1:
            impacts = [s["impact"] for s in result.worst_scenarios]
            assert impacts == sorted(impacts), "Worst scenarios should be sorted"

    def test_empty_when_no_overlap(self):
        """Weights have no overlap with cov columns."""
        weights = {"ZZZ": 0.5, "YYY": 0.5}
        cov = pd.DataFrame(
            [[0.04, 0.01], [0.01, 0.03]],
            index=["AAPL", "MSFT"], columns=["AAPL", "MSFT"],
        )
        result = reverse_stress_test(weights, -0.20, cov, n_scenarios=10)
        assert result.worst_scenarios == []
