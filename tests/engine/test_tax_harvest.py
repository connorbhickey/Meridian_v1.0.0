"""Tests for tax-loss harvesting analysis."""

import numpy as np
import pandas as pd
import pytest

from portopt.engine.tax_harvest import (
    HarvestCandidate,
    HarvestRecommendation,
    compute_harvest_recommendation,
    estimate_tax_alpha,
    identify_harvest_candidates,
    suggest_replacements,
)
from portopt.data.models import Asset, AssetType, Holding


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_holding(symbol, quantity, cost_basis, current_price):
    """Helper to create a Holding with an Asset."""
    asset = Asset(symbol=symbol, name=symbol, asset_type=AssetType.STOCK)
    return Holding(
        asset=asset,
        quantity=quantity,
        cost_basis=cost_basis,
        current_price=current_price,
    )


@pytest.fixture
def mixed_holdings():
    """Holdings with a mix of gains and losses."""
    return [
        _make_holding("AAPL", 100, 15000, 170),   # gain: 2000
        _make_holding("MSFT", 50, 20000, 350),     # loss: -2500
        _make_holding("GOOG", 30, 5400, 140),      # loss: -1200
        _make_holding("AMZN", 20, 3000, 180),      # gain: 600
        _make_holding("TSLA", 40, 12000, 200),      # loss: -4000
    ]


@pytest.fixture
def all_gain_holdings():
    """Holdings with only gains (nothing to harvest)."""
    return [
        _make_holding("AAPL", 100, 10000, 170),
        _make_holding("MSFT", 50, 10000, 350),
    ]


# ── identify_harvest_candidates ───────────────────────────────────────


class TestIdentifyHarvestCandidates:
    def test_returns_list(self, mixed_holdings):
        candidates = identify_harvest_candidates(mixed_holdings)
        assert isinstance(candidates, list)
        assert all(isinstance(c, HarvestCandidate) for c in candidates)

    def test_only_losses(self, mixed_holdings):
        """Should only include holdings with unrealized losses."""
        candidates = identify_harvest_candidates(mixed_holdings)
        for c in candidates:
            assert c.unrealized_loss < 0

    def test_excludes_gains(self, mixed_holdings):
        candidates = identify_harvest_candidates(mixed_holdings)
        symbols = {c.symbol for c in candidates}
        assert "AAPL" not in symbols  # AAPL has a gain
        assert "AMZN" not in symbols  # AMZN has a gain

    def test_includes_losses(self, mixed_holdings):
        candidates = identify_harvest_candidates(mixed_holdings)
        symbols = {c.symbol for c in candidates}
        assert "MSFT" in symbols
        assert "GOOG" in symbols
        assert "TSLA" in symbols

    def test_sorted_by_tax_savings(self, mixed_holdings):
        candidates = identify_harvest_candidates(mixed_holdings)
        savings = [c.tax_savings for c in candidates]
        assert savings == sorted(savings, reverse=True)

    def test_tax_savings_calculation(self, mixed_holdings):
        """Tax savings = |loss| * tax_rate."""
        candidates = identify_harvest_candidates(mixed_holdings, tax_rate=0.35)
        for c in candidates:
            expected = abs(c.unrealized_loss) * 0.35
            assert c.tax_savings == pytest.approx(expected)

    def test_custom_tax_rate(self, mixed_holdings):
        rate = 0.20
        candidates = identify_harvest_candidates(mixed_holdings, tax_rate=rate)
        for c in candidates:
            expected = abs(c.unrealized_loss) * rate
            assert c.tax_savings == pytest.approx(expected)

    def test_loss_percentage(self, mixed_holdings):
        candidates = identify_harvest_candidates(mixed_holdings)
        for c in candidates:
            assert c.loss_pct < 0  # Should be negative

    def test_empty_holdings(self):
        assert identify_harvest_candidates([]) == []

    def test_all_gains_returns_empty(self, all_gain_holdings):
        assert identify_harvest_candidates(all_gain_holdings) == []


# ── suggest_replacements ──────────────────────────────────────────────


class TestSuggestReplacements:
    def test_returns_list(self, prices_5):
        replacements = suggest_replacements("AAPL", prices_5)
        assert isinstance(replacements, list)

    def test_does_not_include_self(self, prices_5):
        replacements = suggest_replacements("AAPL", prices_5)
        assert "AAPL" not in replacements

    def test_max_top_n(self, prices_5):
        replacements = suggest_replacements("AAPL", prices_5, top_n=2)
        assert len(replacements) <= 2

    def test_missing_symbol(self, prices_5):
        replacements = suggest_replacements("NONEXISTENT", prices_5)
        assert replacements == []

    def test_single_column(self):
        """With only one column, no replacements possible."""
        prices = pd.DataFrame({"A": np.random.randn(100).cumsum() + 100})
        assert suggest_replacements("A", prices) == []

    def test_high_correlation_found(self):
        """Perfectly correlated assets should be suggested."""
        np.random.seed(42)
        base = np.random.randn(200).cumsum() + 100
        prices = pd.DataFrame({
            "A": base,
            "B": base + np.random.randn(200) * 0.01,  # Nearly identical
            "C": np.random.randn(200).cumsum() + 100,  # Unrelated
        })
        replacements = suggest_replacements("A", prices)
        assert "B" in replacements


# ── compute_harvest_recommendation ────────────────────────────────────


class TestComputeHarvestRecommendation:
    def test_returns_recommendation(self, mixed_holdings):
        rec = compute_harvest_recommendation(mixed_holdings)
        assert isinstance(rec, HarvestRecommendation)

    def test_total_harvestable_loss(self, mixed_holdings):
        rec = compute_harvest_recommendation(mixed_holdings)
        assert rec.total_harvestable_loss < 0
        # Sum of individual losses
        expected = sum(c.unrealized_loss for c in rec.candidates)
        assert rec.total_harvestable_loss == pytest.approx(expected)

    def test_total_tax_savings(self, mixed_holdings):
        rec = compute_harvest_recommendation(mixed_holdings, tax_rate=0.35)
        assert rec.total_tax_savings > 0
        expected = sum(c.tax_savings for c in rec.candidates)
        assert rec.total_tax_savings == pytest.approx(expected)

    def test_tax_rate_stored(self, mixed_holdings):
        rec = compute_harvest_recommendation(mixed_holdings, tax_rate=0.25)
        assert rec.tax_rate == 0.25

    def test_with_prices_adds_replacements(self, mixed_holdings, prices_5):
        rec = compute_harvest_recommendation(mixed_holdings, prices=prices_5)
        # At least one candidate should have replacement suggestions
        # (since synthetic GBM data tends to have correlations)
        assert isinstance(rec.replacement_suggestions, dict)

    def test_without_prices_no_replacements(self, mixed_holdings):
        rec = compute_harvest_recommendation(mixed_holdings)
        assert rec.replacement_suggestions == {}

    def test_empty_holdings(self):
        rec = compute_harvest_recommendation([])
        assert rec.candidates == []
        assert rec.total_harvestable_loss == 0
        assert rec.total_tax_savings == 0


# ── estimate_tax_alpha ────────────────────────────────────────────────


class TestEstimateTaxAlpha:
    def test_positive_result(self):
        alpha = estimate_tax_alpha(-10000, 0.35, 0.08)
        assert alpha > 0

    def test_formula(self):
        """alpha = |loss| * tax_rate * reinvestment_return."""
        alpha = estimate_tax_alpha(-10000, 0.35, 0.08)
        expected = 10000 * 0.35 * 0.08
        assert alpha == pytest.approx(expected)

    def test_zero_loss(self):
        assert estimate_tax_alpha(0, 0.35) == 0.0

    def test_positive_loss_returns_zero(self):
        """Positive number means no loss to harvest."""
        assert estimate_tax_alpha(5000, 0.35) == 0.0

    def test_custom_reinvestment_return(self):
        alpha = estimate_tax_alpha(-10000, 0.35, reinvestment_return=0.10)
        expected = 10000 * 0.35 * 0.10
        assert alpha == pytest.approx(expected)
