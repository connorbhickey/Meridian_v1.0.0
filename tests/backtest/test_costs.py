"""Tests for transaction cost models."""

import pytest

from portopt.backtest.costs import (
    CompositeCost,
    FixedCost,
    ProportionalCost,
    SpreadCost,
    ZeroCost,
    create_cost_model,
)


class TestZeroCost:
    def test_always_zero(self):
        model = ZeroCost()
        assert model.compute_cost("AAPL", 0.10, 1_000_000) == 0.0
        assert model.compute_cost("MSFT", -0.50, 500_000) == 0.0


class TestFixedCost:
    def test_fixed_per_trade(self):
        model = FixedCost(cost_per_trade=10.0)
        cost = model.compute_cost("AAPL", 0.05, 1_000_000)
        assert cost == 10.0

    def test_zero_change_no_cost(self):
        model = FixedCost(cost_per_trade=10.0)
        cost = model.compute_cost("AAPL", 0.0, 1_000_000)
        assert cost == 0.0


class TestProportionalCost:
    def test_proportional(self):
        model = ProportionalCost(rate=0.001)  # 10 bps
        cost = model.compute_cost("AAPL", 0.10, 1_000_000)
        expected = 0.001 * abs(0.10) * 1_000_000  # $100
        assert cost == pytest.approx(expected)

    def test_negative_change(self):
        model = ProportionalCost(rate=0.001)
        cost = model.compute_cost("AAPL", -0.10, 1_000_000)
        assert cost > 0  # Cost is always positive


class TestSpreadCost:
    def test_spread_cost(self):
        model = SpreadCost(spread_bps=5.0)
        cost = model.compute_cost("AAPL", 0.10, 1_000_000)
        assert cost > 0


class TestCompositeCost:
    def test_sum_of_models(self):
        fixed = FixedCost(cost_per_trade=5.0)
        prop = ProportionalCost(rate=0.001)
        composite = CompositeCost(models=[fixed, prop])
        cost = composite.compute_cost("AAPL", 0.10, 1_000_000)
        expected = 5.0 + 0.001 * 0.10 * 1_000_000
        assert cost == pytest.approx(expected)


class TestComputeTotalCost:
    def test_multiple_trades(self):
        model = FixedCost(cost_per_trade=10.0)
        changes = {"AAPL": 0.10, "MSFT": -0.05, "GOOG": 0.0}
        total = model.compute_total_cost(changes, 1_000_000)
        assert total == 20.0  # Only 2 non-zero trades


class TestCreateCostModel:
    def test_create_zero(self):
        model = create_cost_model("zero")
        assert isinstance(model, ZeroCost)

    def test_create_fixed(self):
        model = create_cost_model("fixed", cost_per_trade=7.0)
        assert isinstance(model, FixedCost)

    def test_create_proportional(self):
        model = create_cost_model("proportional", rate=0.002)
        assert isinstance(model, ProportionalCost)
