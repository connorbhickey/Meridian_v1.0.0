"""Tests for strategy comparison engine — compare_methods, parameter_sweep, bootstrap."""

import numpy as np
import pandas as pd
import pytest

from portopt.constants import CovEstimator, ReturnEstimator
from portopt.data.models import OptimizationResult
from portopt.engine.constraints import PortfolioConstraints
from portopt.engine.strategy_compare import (
    ComparisonResult,
    PairwiseTest,
    StrategyResult,
    bootstrap_paired_test,
    compare_methods,
    parameter_sweep,
    _default_sweep_values,
    _modify_constraints,
)
from tests.conftest import assert_valid_weights


# ── compare_methods ────────────────────────────────────────────────────


class TestCompareMethods:
    """Tests for the compare_methods() top-level function."""

    def test_default_methods_produces_results_for_all(self, prices_5):
        """Default methods list should produce a result for each method."""
        result = compare_methods(prices_5)
        assert isinstance(result, ComparisonResult)
        expected_names = {
            "max_sharpe", "min_volatility", "max_diversification",
            "inverse_variance", "hrp",
        }
        actual_names = {s.name for s in result.strategies}
        assert actual_names == expected_names

    def test_default_methods_count(self, prices_5):
        """Default call should produce exactly 5 strategy results."""
        result = compare_methods(prices_5)
        assert len(result.strategies) == 5

    def test_custom_methods_list(self, prices_5):
        """Custom methods list should only run the specified methods."""
        methods = ["max_sharpe", "hrp"]
        result = compare_methods(prices_5, methods=methods)
        assert len(result.strategies) == 2
        actual_names = {s.name for s in result.strategies}
        assert actual_names == {"max_sharpe", "hrp"}

    def test_invalid_method_skipped(self, prices_5):
        """Unknown method names should be skipped without raising."""
        methods = ["max_sharpe", "bogus_method", "hrp"]
        result = compare_methods(prices_5, methods=methods)
        actual_names = {s.name for s in result.strategies}
        assert "bogus_method" not in actual_names
        assert len(result.strategies) == 2

    def test_all_invalid_methods_returns_empty(self, prices_5):
        """If every method is invalid, strategies list is empty."""
        result = compare_methods(prices_5, methods=["fake1", "fake2"])
        assert len(result.strategies) == 0
        assert result.best_sharpe == ""
        assert result.best_return == ""
        assert result.lowest_vol == ""
        assert result.lowest_drawdown == ""

    def test_single_method(self, prices_5):
        """A single-method list should work and produce one result."""
        result = compare_methods(prices_5, methods=["min_volatility"])
        assert len(result.strategies) == 1
        assert result.strategies[0].name == "min_volatility"

    def test_strategy_results_have_valid_weights(self, prices_5):
        """Each strategy result should contain valid portfolio weights."""
        result = compare_methods(prices_5)
        symbols = list(prices_5.columns)
        for strat in result.strategies:
            assert isinstance(strat.opt_result, OptimizationResult)
            assert_valid_weights(strat.opt_result.weights, symbols)

    def test_strategy_results_have_metrics(self, prices_5):
        """Each strategy result should contain computed metrics dict."""
        result = compare_methods(prices_5)
        for strat in result.strategies:
            assert isinstance(strat.metrics, dict)
            assert "sharpe_ratio" in strat.metrics
            assert "annualized_return" in strat.metrics
            assert "annualized_volatility" in strat.metrics
            assert "max_drawdown" in strat.metrics

    def test_strategy_results_have_backtest_returns(self, prices_5):
        """Each strategy result should have a backtest_returns array."""
        result = compare_methods(prices_5)
        n_days = len(prices_5.pct_change().dropna())
        for strat in result.strategies:
            assert strat.backtest_returns is not None
            assert isinstance(strat.backtest_returns, np.ndarray)
            assert len(strat.backtest_returns) == n_days

    def test_best_sharpe_is_correct(self, prices_5):
        """best_sharpe should be the name of the strategy with the highest Sharpe."""
        result = compare_methods(prices_5)
        best = max(
            result.strategies,
            key=lambda s: s.metrics.get("sharpe_ratio", -999),
        )
        assert result.best_sharpe == best.name

    def test_best_return_is_correct(self, prices_5):
        """best_return should be the name of the strategy with the highest annualized return."""
        result = compare_methods(prices_5)
        best = max(
            result.strategies,
            key=lambda s: s.metrics.get("annualized_return", -999),
        )
        assert result.best_return == best.name

    def test_lowest_vol_is_correct(self, prices_5):
        """lowest_vol should be the strategy with the smallest annualized volatility."""
        result = compare_methods(prices_5)
        best = min(
            result.strategies,
            key=lambda s: s.metrics.get("annualized_volatility", 999),
        )
        assert result.lowest_vol == best.name

    def test_lowest_drawdown_is_correct(self, prices_5):
        """lowest_drawdown should be the strategy with the highest (least negative) max_drawdown."""
        result = compare_methods(prices_5)
        best = max(
            result.strategies,
            key=lambda s: s.metrics.get("max_drawdown", -999),
        )
        assert result.lowest_drawdown == best.name

    def test_pairwise_tests_populated(self, prices_5):
        """With N strategies, should produce N*(N-1)/2 pairwise tests."""
        result = compare_methods(prices_5)
        n = len(result.strategies)
        expected_pairs = n * (n - 1) // 2
        assert len(result.pairwise_tests) == expected_pairs

    def test_pairwise_tests_have_correct_fields(self, prices_5):
        """Each pairwise test should have all required fields set."""
        result = compare_methods(prices_5, methods=["max_sharpe", "hrp"])
        assert len(result.pairwise_tests) == 1
        test = result.pairwise_tests[0]
        assert isinstance(test, PairwiseTest)
        assert test.strategy_a != ""
        assert test.strategy_b != ""
        assert isinstance(test.test_statistic, float)
        assert isinstance(test.p_value, float)
        assert 0.0 <= test.p_value <= 1.0
        assert isinstance(test.mean_diff, float)
        assert isinstance(test.significant, bool)

    def test_single_method_no_pairwise_tests(self, prices_5):
        """A single method should produce zero pairwise tests."""
        result = compare_methods(prices_5, methods=["hrp"])
        assert len(result.pairwise_tests) == 0

    def test_empty_prices_returns_empty_result(self):
        """Empty price DataFrame should return a ComparisonResult with no strategies."""
        empty = pd.DataFrame()
        result = compare_methods(empty)
        assert isinstance(result, ComparisonResult)
        assert len(result.strategies) == 0
        assert result.best_sharpe == ""
        assert result.best_return == ""
        assert result.lowest_vol == ""
        assert result.lowest_drawdown == ""

    def test_custom_constraints(self, prices_5):
        """Custom constraints should be respected by all methods."""
        constraints = PortfolioConstraints(min_weight=0.05, max_weight=0.50)
        result = compare_methods(prices_5, constraints=constraints,
                                 methods=["max_sharpe", "min_volatility"])
        for strat in result.strategies:
            # HRP/HERC don't take constraints, but MVO methods should respect them
            for w in strat.opt_result.weights.values():
                assert w >= 0.05 - 1e-4
                assert w <= 0.50 + 1e-4

    def test_custom_risk_free_rate(self, prices_5):
        """Different risk-free rates should produce different Sharpe ratios."""
        result_low = compare_methods(prices_5, methods=["max_sharpe"],
                                     risk_free_rate=0.0)
        result_high = compare_methods(prices_5, methods=["max_sharpe"],
                                      risk_free_rate=0.10)
        sharpe_low = result_low.strategies[0].metrics["sharpe_ratio"]
        sharpe_high = result_high.strategies[0].metrics["sharpe_ratio"]
        # Higher risk-free rate means lower Sharpe (excess return shrinks)
        assert sharpe_low > sharpe_high

    def test_case_insensitive_method_names(self, prices_5):
        """Method names should be case-insensitive (lowered internally)."""
        result = compare_methods(prices_5, methods=["MAX_SHARPE", "HRP"])
        actual_names = {s.name for s in result.strategies}
        assert "MAX_SHARPE" in actual_names or "max_sharpe" in actual_names
        assert len(result.strategies) == 2

    def test_three_asset_universe(self, prices_3):
        """Should work with a small 3-asset universe."""
        result = compare_methods(prices_3, methods=["max_sharpe", "hrp"])
        assert len(result.strategies) == 2
        for strat in result.strategies:
            assert_valid_weights(strat.opt_result.weights, list(prices_3.columns))


# ── parameter_sweep ────────────────────────────────────────────────────


class TestParameterSweep:
    """Tests for the parameter_sweep() function."""

    def test_risk_aversion_sweep_count(self, prices_5):
        """Risk aversion sweep with default values should return correct count."""
        defaults = _default_sweep_values("risk_aversion")
        results = parameter_sweep(prices_5, method="max_quadratic_utility",
                                  param_name="risk_aversion")
        assert len(results) == len(defaults)

    def test_risk_aversion_sweep_labels(self, prices_5):
        """Each result in a risk aversion sweep should have a descriptive label."""
        results = parameter_sweep(prices_5, method="max_quadratic_utility",
                                  param_name="risk_aversion")
        for r in results:
            assert "risk_aversion=" in r.name
            assert "max_quadratic_utility" in r.name

    def test_risk_aversion_sweep_has_valid_weights(self, prices_5):
        """Each sweep result should contain valid portfolio weights."""
        results = parameter_sweep(prices_5, method="max_quadratic_utility",
                                  param_name="risk_aversion")
        symbols = list(prices_5.columns)
        for r in results:
            assert_valid_weights(r.opt_result.weights, symbols)

    def test_max_weight_sweep(self, prices_5):
        """Max weight sweep should return results for each tested value."""
        values = [0.3, 0.5, 1.0]
        results = parameter_sweep(prices_5, method="max_sharpe",
                                  param_name="max_weight",
                                  param_values=values)
        assert len(results) == len(values)
        for r in results:
            assert "max_weight=" in r.name

    def test_max_weight_sweep_respects_constraint(self, prices_5):
        """Lower max_weight should cap individual weights."""
        values = [0.3, 1.0]
        results = parameter_sweep(prices_5, method="max_sharpe",
                                  param_name="max_weight",
                                  param_values=values)
        # The result with max_weight=0.3 should have no weight > 0.3 + tolerance
        tight_result = results[0]
        for w in tight_result.opt_result.weights.values():
            assert w <= 0.3 + 1e-3

    def test_custom_param_values(self, prices_5):
        """Explicit param_values list overrides defaults."""
        values = [1.0, 5.0, 50.0]
        results = parameter_sweep(prices_5, method="max_quadratic_utility",
                                  param_name="risk_aversion",
                                  param_values=values)
        assert len(results) == 3
        for val, r in zip(values, results):
            assert f"risk_aversion={val}" in r.name

    def test_default_param_values_used_when_none(self, prices_5):
        """When param_values is None, _default_sweep_values() values are used."""
        defaults = _default_sweep_values("risk_aversion")
        results = parameter_sweep(prices_5, method="max_quadratic_utility",
                                  param_name="risk_aversion",
                                  param_values=None)
        assert len(results) == len(defaults)

    def test_invalid_method_raises(self, prices_5):
        """An unknown method name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            parameter_sweep(prices_5, method="totally_fake")

    def test_sweep_results_have_metrics(self, prices_5):
        """Each sweep result should have computed metrics."""
        results = parameter_sweep(prices_5, method="max_quadratic_utility",
                                  param_name="risk_aversion",
                                  param_values=[1.0, 5.0])
        for r in results:
            assert isinstance(r.metrics, dict)
            assert "sharpe_ratio" in r.metrics
            assert "annualized_volatility" in r.metrics

    def test_sweep_results_have_backtest_returns(self, prices_5):
        """Each sweep result should have a backtest_returns array."""
        results = parameter_sweep(prices_5, method="max_quadratic_utility",
                                  param_name="risk_aversion",
                                  param_values=[1.0, 5.0])
        n_days = len(prices_5.pct_change().dropna())
        for r in results:
            assert r.backtest_returns is not None
            assert len(r.backtest_returns) == n_days

    def test_higher_risk_aversion_lower_volatility(self, prices_5):
        """Higher risk aversion should generally produce lower volatility."""
        results = parameter_sweep(prices_5, method="max_quadratic_utility",
                                  param_name="risk_aversion",
                                  param_values=[0.5, 20.0])
        vol_low_aversion = results[0].metrics["annualized_volatility"]
        vol_high_aversion = results[1].metrics["annualized_volatility"]
        # High risk aversion should have equal or lower vol
        assert vol_high_aversion <= vol_low_aversion + 1e-3

    def test_min_weight_sweep(self, prices_5):
        """Min weight sweep should produce results."""
        results = parameter_sweep(prices_5, method="max_sharpe",
                                  param_name="min_weight",
                                  param_values=[0.0, 0.05])
        assert len(results) == 2
        # With min_weight=0.05, no weight should be below 0.05
        tight_result = results[1]
        for w in tight_result.opt_result.weights.values():
            assert w >= 0.05 - 1e-4

    def test_lookback_sweep_runs(self, prices_5):
        """Lookback sweep should run (even though lookback is externally handled)."""
        results = parameter_sweep(prices_5, method="max_sharpe",
                                  param_name="lookback",
                                  param_values=[120, 252])
        # Lookback doesn't modify constraints, so results may be identical
        assert len(results) == 2

    def test_three_asset_sweep(self, prices_3):
        """Sweep should work with smaller asset universe."""
        results = parameter_sweep(prices_3, method="max_quadratic_utility",
                                  param_name="risk_aversion",
                                  param_values=[1.0, 10.0])
        assert len(results) == 2
        for r in results:
            assert_valid_weights(r.opt_result.weights, list(prices_3.columns))


# ── bootstrap_paired_test ──────────────────────────────────────────────


class TestBootstrapPairedTest:
    """Tests for the bootstrap_paired_test() function."""

    def test_identical_returns_not_significant(self):
        """Identical return series should yield a non-significant result."""
        np.random.seed(42)
        returns = np.random.randn(500) * 0.01
        result = bootstrap_paired_test(returns, returns, n_bootstrap=5000)
        assert isinstance(result, PairwiseTest)
        assert result.p_value > 0.05
        assert not result.significant
        assert result.mean_diff == pytest.approx(0.0, abs=1e-15)

    def test_very_different_returns_significant(self):
        """Very different return series should yield a significant result."""
        np.random.seed(42)
        n = 500
        # Strategy A: strong positive drift
        returns_a = np.random.randn(n) * 0.005 + 0.005
        # Strategy B: strong negative drift
        returns_b = np.random.randn(n) * 0.005 - 0.005
        result = bootstrap_paired_test(returns_a, returns_b, n_bootstrap=5000)
        assert result.p_value < 0.05
        assert result.significant
        assert result.mean_diff > 0  # A is better

    def test_returns_correct_fields(self):
        """Result should contain all expected fields."""
        np.random.seed(42)
        a = np.random.randn(200) * 0.01
        b = np.random.randn(200) * 0.01
        result = bootstrap_paired_test(a, b)
        assert hasattr(result, "strategy_a")
        assert hasattr(result, "strategy_b")
        assert hasattr(result, "test_statistic")
        assert hasattr(result, "p_value")
        assert hasattr(result, "mean_diff")
        assert hasattr(result, "significant")
        # strategy_a/b are empty strings (filled by caller)
        assert result.strategy_a == ""
        assert result.strategy_b == ""

    def test_p_value_between_0_and_1(self):
        """p-value should always be in [0, 1]."""
        np.random.seed(42)
        a = np.random.randn(300) * 0.01
        b = np.random.randn(300) * 0.01 + 0.001
        result = bootstrap_paired_test(a, b, n_bootstrap=2000)
        assert 0.0 <= result.p_value <= 1.0

    def test_mean_diff_sign(self):
        """mean_diff should be positive when A has higher mean, negative otherwise."""
        np.random.seed(42)
        n = 1000
        a = np.ones(n) * 0.01
        b = np.ones(n) * 0.005
        result = bootstrap_paired_test(a, b)
        assert result.mean_diff > 0

    def test_mean_diff_sign_reversed(self):
        """mean_diff should be negative when B has higher mean."""
        np.random.seed(42)
        n = 1000
        a = np.ones(n) * 0.005
        b = np.ones(n) * 0.01
        result = bootstrap_paired_test(a, b)
        assert result.mean_diff < 0

    def test_different_length_arrays(self):
        """Should handle different-length arrays by truncating to min length."""
        np.random.seed(42)
        a = np.random.randn(500) * 0.01
        b = np.random.randn(300) * 0.01
        result = bootstrap_paired_test(a, b, n_bootstrap=1000)
        assert isinstance(result, PairwiseTest)
        assert isinstance(result.p_value, float)

    def test_custom_alpha(self):
        """Custom alpha level should be respected for significance determination."""
        np.random.seed(42)
        n = 500
        a = np.random.randn(n) * 0.01 + 0.002
        b = np.random.randn(n) * 0.01
        # With a very strict alpha, result may not be significant
        result_strict = bootstrap_paired_test(a, b, n_bootstrap=5000, alpha=0.001)
        # With a loose alpha, result is more likely significant
        result_loose = bootstrap_paired_test(a, b, n_bootstrap=5000, alpha=0.50)
        # If strict is significant, loose must also be
        if result_strict.significant:
            assert result_loose.significant
        # Loose should be at least as likely to be significant
        assert result_loose.p_value == pytest.approx(result_strict.p_value, abs=0.1)

    def test_small_n_bootstrap(self):
        """Small n_bootstrap should still produce valid results."""
        np.random.seed(42)
        a = np.random.randn(100) * 0.01
        b = np.random.randn(100) * 0.01
        result = bootstrap_paired_test(a, b, n_bootstrap=50)
        assert isinstance(result, PairwiseTest)
        assert 0.0 <= result.p_value <= 1.0

    def test_test_statistic_is_finite(self):
        """Test statistic should be a finite number."""
        np.random.seed(42)
        a = np.random.randn(300) * 0.01
        b = np.random.randn(300) * 0.01 + 0.002
        result = bootstrap_paired_test(a, b, n_bootstrap=2000)
        assert np.isfinite(result.test_statistic)

    def test_constant_returns_zero_diff(self):
        """Constant identical returns should produce zero mean_diff."""
        a = np.ones(200) * 0.01
        b = np.ones(200) * 0.01
        result = bootstrap_paired_test(a, b, n_bootstrap=500)
        assert result.mean_diff == pytest.approx(0.0, abs=1e-15)


# ── Internal helpers ───────────────────────────────────────────────────


class TestDefaultSweepValues:
    """Tests for _default_sweep_values() helper."""

    def test_risk_aversion_defaults(self):
        """risk_aversion should return a list of positive floats."""
        vals = _default_sweep_values("risk_aversion")
        assert isinstance(vals, list)
        assert len(vals) >= 3
        assert all(v > 0 for v in vals)
        # Should be sorted ascending
        assert vals == sorted(vals)

    def test_lookback_defaults(self):
        """lookback should return a list of positive integers."""
        vals = _default_sweep_values("lookback")
        assert isinstance(vals, list)
        assert len(vals) >= 2
        assert all(v > 0 for v in vals)

    def test_max_weight_defaults(self):
        """max_weight should return values in (0, 1]."""
        vals = _default_sweep_values("max_weight")
        assert isinstance(vals, list)
        assert len(vals) >= 2
        assert all(0.0 < v <= 1.0 for v in vals)

    def test_min_weight_defaults(self):
        """min_weight should return non-negative values."""
        vals = _default_sweep_values("min_weight")
        assert isinstance(vals, list)
        assert len(vals) >= 2
        assert all(v >= 0.0 for v in vals)

    def test_unknown_param_returns_fallback(self):
        """Unknown parameter name should return a fallback default list."""
        vals = _default_sweep_values("nonexistent_param")
        assert isinstance(vals, list)
        assert len(vals) >= 2


class TestModifyConstraints:
    """Tests for _modify_constraints() helper."""

    def test_risk_aversion_modified(self):
        """Should set risk_aversion on the cloned constraints."""
        base = PortfolioConstraints(risk_aversion=1.0)
        modified = _modify_constraints(base, "risk_aversion", 10.0)
        assert modified.risk_aversion == pytest.approx(10.0)
        # Original should be unchanged
        assert base.risk_aversion == pytest.approx(1.0)

    def test_max_weight_modified(self):
        """Should set max_weight on the cloned constraints."""
        base = PortfolioConstraints(max_weight=1.0)
        modified = _modify_constraints(base, "max_weight", 0.3)
        assert modified.max_weight == pytest.approx(0.3)
        assert base.max_weight == pytest.approx(1.0)

    def test_min_weight_modified(self):
        """Should set min_weight on the cloned constraints."""
        base = PortfolioConstraints(min_weight=0.0)
        modified = _modify_constraints(base, "min_weight", 0.05)
        assert modified.min_weight == pytest.approx(0.05)
        assert base.min_weight == pytest.approx(0.0)

    def test_lookback_is_noop(self):
        """Lookback modification should not raise (it's a pass-through)."""
        base = PortfolioConstraints()
        modified = _modify_constraints(base, "lookback", 252)
        # Should return a valid constraints object unchanged
        assert isinstance(modified, PortfolioConstraints)

    def test_unknown_param_does_not_raise(self):
        """Unknown parameter name should not raise, just log a warning."""
        base = PortfolioConstraints()
        modified = _modify_constraints(base, "bogus_param", 42.0)
        assert isinstance(modified, PortfolioConstraints)

    def test_deep_copy_independence(self):
        """Modified constraints should be fully independent of original."""
        base = PortfolioConstraints(
            risk_aversion=1.0,
            max_weight=0.5,
            min_weight=0.01,
            weight_bounds={"AAPL": (0.0, 0.3)},
        )
        modified = _modify_constraints(base, "risk_aversion", 99.0)
        # Mutating the modified copy should not affect the original
        modified.max_weight = 0.9
        modified.weight_bounds["MSFT"] = (0.0, 0.2)
        assert base.max_weight == pytest.approx(0.5)
        assert "MSFT" not in base.weight_bounds


# ── Dataclass construction ─────────────────────────────────────────────


class TestDataclasses:
    """Verify the strategy comparison dataclasses are well-formed."""

    def test_strategy_result_defaults(self):
        """StrategyResult should have sensible defaults."""
        opt = OptimizationResult(method="test", weights={"A": 1.0})
        sr = StrategyResult(name="test", opt_result=opt)
        assert sr.backtest_returns is None
        assert sr.metrics == {}

    def test_pairwise_test_construction(self):
        """PairwiseTest should hold all fields."""
        pt = PairwiseTest(
            strategy_a="A", strategy_b="B",
            test_statistic=2.5, p_value=0.01,
            mean_diff=0.001, significant=True,
        )
        assert pt.strategy_a == "A"
        assert pt.strategy_b == "B"
        assert pt.test_statistic == pytest.approx(2.5)
        assert pt.p_value == pytest.approx(0.01)
        assert pt.mean_diff == pytest.approx(0.001)
        assert pt.significant is True

    def test_comparison_result_defaults(self):
        """ComparisonResult should have empty defaults."""
        cr = ComparisonResult(strategies=[])
        assert cr.pairwise_tests == []
        assert cr.best_sharpe == ""
        assert cr.best_return == ""
        assert cr.lowest_vol == ""
        assert cr.lowest_drawdown == ""
