"""Tests for portfolio constraints."""

import pytest

from portopt.engine.constraints import PortfolioConstraints


class TestPortfolioConstraints:
    def test_defaults(self):
        c = PortfolioConstraints()
        assert c.long_only is True
        assert c.min_weight == 0.0
        assert c.max_weight == 1.0
        assert c.leverage == 1.0

    def test_get_bounds_default(self):
        c = PortfolioConstraints(min_weight=0.0, max_weight=0.5)
        lo, hi = c.get_bounds("AAPL")
        assert lo == 0.0
        assert hi == 0.5

    def test_get_bounds_override(self):
        c = PortfolioConstraints(
            min_weight=0.0,
            max_weight=0.5,
            weight_bounds={"AAPL": (0.1, 0.3)},
        )
        lo, hi = c.get_bounds("AAPL")
        assert lo == 0.1
        assert hi == 0.3
        # Non-overridden symbol uses defaults
        lo2, hi2 = c.get_bounds("MSFT")
        assert lo2 == 0.0
        assert hi2 == 0.5

    def test_get_all_bounds(self):
        c = PortfolioConstraints(min_weight=0.05, max_weight=0.40)
        bounds = c.get_all_bounds(["AAPL", "MSFT", "GOOG"])
        assert len(bounds) == 3
        for lo, hi in bounds:
            assert lo == 0.05
            assert hi == 0.40

    def test_validate_good_weights(self):
        c = PortfolioConstraints(min_weight=0.0, max_weight=0.5)
        issues = c.validate({"AAPL": 0.3, "MSFT": 0.3, "GOOG": 0.4})
        assert len(issues) == 0

    def test_validate_over_max(self):
        c = PortfolioConstraints(max_weight=0.4)
        issues = c.validate({"AAPL": 0.6, "MSFT": 0.4})
        assert len(issues) > 0
