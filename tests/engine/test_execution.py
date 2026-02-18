"""Tests for execution simulation — market impact, slippage, and capacity analysis."""

import numpy as np
import pandas as pd
import pytest

from portopt.engine.execution import (
    CapacityResult,
    ExecutionSummary,
    TradeExecution,
    analyze_capacity,
    estimate_volatilities_from_prices,
    estimate_volumes_from_prices,
    linear_impact,
    simulate_execution,
    sqrt_impact,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def simple_prices():
    """Simple price dict for three symbols."""
    return {"AAPL": 150.0, "MSFT": 300.0, "GOOG": 100.0}


@pytest.fixture
def simple_volumes():
    """ADV in shares for three symbols."""
    return {"AAPL": 50_000_000.0, "MSFT": 25_000_000.0, "GOOG": 30_000_000.0}


@pytest.fixture
def simple_volatilities():
    """Daily volatilities for three symbols."""
    return {"AAPL": 0.02, "MSFT": 0.015, "GOOG": 0.025}


@pytest.fixture
def simple_weights():
    """Target weights for three symbols."""
    return {"AAPL": 0.40, "MSFT": 0.35, "GOOG": 0.25}


@pytest.fixture
def price_dataframe():
    """Synthetic price DataFrame for estimation helpers."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    data = {}
    for sym, base in [("AAPL", 150.0), ("MSFT", 300.0), ("GOOG", 100.0)]:
        # Random walk with small drift
        returns = np.random.randn(60) * 0.01 + 0.0003
        cumret = np.cumprod(1 + returns)
        data[sym] = base * cumret
    return pd.DataFrame(data, index=dates)


# ── sqrt_impact ─────────────────────────────────────────────────────


class TestSqrtImpact:
    def test_zero_trade_value(self):
        assert sqrt_impact(0.0, 1_000_000.0, 0.02) == 0.0

    def test_negative_trade_value(self):
        assert sqrt_impact(-100.0, 1_000_000.0, 0.02) == 0.0

    def test_zero_adv(self):
        assert sqrt_impact(100_000.0, 0.0, 0.02) == 0.0

    def test_negative_adv(self):
        assert sqrt_impact(100_000.0, -500.0, 0.02) == 0.0

    def test_positive_values_return_positive_impact(self):
        impact = sqrt_impact(500_000.0, 10_000_000.0, 0.02)
        assert impact > 0

    def test_higher_trade_value_higher_impact(self):
        low = sqrt_impact(100_000.0, 10_000_000.0, 0.02)
        high = sqrt_impact(500_000.0, 10_000_000.0, 0.02)
        assert high > low

    def test_monotone_increasing_in_trade_value(self):
        """Impact is monotonically increasing as trade value grows."""
        adv = 10_000_000.0
        vol = 0.02
        trade_values = [50_000, 100_000, 500_000, 1_000_000, 5_000_000]
        impacts = [sqrt_impact(tv, adv, vol) for tv in trade_values]
        for i in range(1, len(impacts)):
            assert impacts[i] > impacts[i - 1]

    def test_higher_volatility_higher_impact(self):
        low_vol = sqrt_impact(500_000.0, 10_000_000.0, 0.01)
        high_vol = sqrt_impact(500_000.0, 10_000_000.0, 0.04)
        assert high_vol > low_vol

    def test_volatility_scales_linearly(self):
        """Impact should double when volatility doubles."""
        impact_1 = sqrt_impact(500_000.0, 10_000_000.0, 0.01, eta=0.1)
        impact_2 = sqrt_impact(500_000.0, 10_000_000.0, 0.02, eta=0.1)
        assert impact_2 == pytest.approx(2.0 * impact_1, rel=1e-10)

    def test_known_value(self):
        """Check formula: eta * sigma * 10000 * sqrt(Q / ADV)."""
        # Q = 1_000_000, ADV = 10_000_000, sigma = 0.02, eta = 0.1
        # participation = 0.1
        # impact = 0.1 * 0.02 * 10000 * sqrt(0.1) = 20 * 0.31623 = 6.3246
        expected = 0.1 * 0.02 * 10000 * np.sqrt(1_000_000 / 10_000_000)
        actual = sqrt_impact(1_000_000.0, 10_000_000.0, 0.02, eta=0.1)
        assert actual == pytest.approx(expected, rel=1e-10)

    def test_eta_parameter(self):
        """Higher eta gives proportionally higher impact."""
        impact_low = sqrt_impact(500_000.0, 10_000_000.0, 0.02, eta=0.05)
        impact_high = sqrt_impact(500_000.0, 10_000_000.0, 0.02, eta=0.20)
        assert impact_high == pytest.approx(4.0 * impact_low, rel=1e-10)


# ── linear_impact ───────────────────────────────────────────────────


class TestLinearImpact:
    def test_zero_trade_value(self):
        assert linear_impact(0.0, 1_000_000.0) == 0.0

    def test_negative_trade_value(self):
        assert linear_impact(-100.0, 1_000_000.0) == 0.0

    def test_zero_adv(self):
        assert linear_impact(100_000.0, 0.0) == 0.0

    def test_negative_adv(self):
        assert linear_impact(100_000.0, -500.0) == 0.0

    def test_positive_values_return_positive(self):
        impact = linear_impact(500_000.0, 10_000_000.0)
        assert impact > 0

    def test_proportional_to_participation(self):
        """Doubling trade value (i.e., doubling participation) doubles impact."""
        impact_1 = linear_impact(500_000.0, 10_000_000.0, lambda_coeff=0.1)
        impact_2 = linear_impact(1_000_000.0, 10_000_000.0, lambda_coeff=0.1)
        assert impact_2 == pytest.approx(2.0 * impact_1, rel=1e-10)

    def test_known_value(self):
        """Check formula: lambda * (Q / ADV) * 10000."""
        # Q = 1M, ADV = 10M, lambda = 0.1 => 0.1 * 0.1 * 10000 = 100 bps
        expected = 0.1 * (1_000_000 / 10_000_000) * 10000
        actual = linear_impact(1_000_000.0, 10_000_000.0, lambda_coeff=0.1)
        assert actual == pytest.approx(expected, rel=1e-10)

    def test_lambda_coefficient(self):
        """Higher lambda gives proportionally higher impact."""
        low = linear_impact(500_000.0, 10_000_000.0, lambda_coeff=0.05)
        high = linear_impact(500_000.0, 10_000_000.0, lambda_coeff=0.20)
        assert high == pytest.approx(4.0 * low, rel=1e-10)

    def test_monotone_increasing(self):
        adv = 10_000_000.0
        trade_values = [50_000, 100_000, 500_000, 1_000_000, 5_000_000]
        impacts = [linear_impact(tv, adv) for tv in trade_values]
        for i in range(1, len(impacts)):
            assert impacts[i] > impacts[i - 1]


# ── simulate_execution ──────────────────────────────────────────────


class TestSimulateExecution:
    def test_basic_rebalance_returns_summary(
        self, simple_prices, simple_volumes, simple_volatilities
    ):
        weight_changes = {"AAPL": 0.10, "MSFT": -0.05, "GOOG": -0.05}
        summary = simulate_execution(
            weight_changes=weight_changes,
            portfolio_value=1_000_000.0,
            prices=simple_prices,
            volumes=simple_volumes,
            volatilities=simple_volatilities,
        )
        assert isinstance(summary, ExecutionSummary)
        assert len(summary.trades) == 3
        assert summary.total_cost > 0
        assert summary.total_notional > 0
        assert summary.cost_as_pct > 0

    def test_all_trade_fields_populated(
        self, simple_prices, simple_volumes, simple_volatilities
    ):
        weight_changes = {"AAPL": 0.10}
        summary = simulate_execution(
            weight_changes=weight_changes,
            portfolio_value=1_000_000.0,
            prices=simple_prices,
            volumes=simple_volumes,
            volatilities=simple_volatilities,
        )
        trade = summary.trades[0]
        assert isinstance(trade, TradeExecution)
        assert trade.symbol == "AAPL"
        assert trade.side == "BUY"
        assert trade.target_shares > 0
        assert trade.filled_shares > 0
        assert 0 < trade.fill_rate <= 1.0
        assert trade.avg_price > 0
        assert trade.market_price == 150.0
        assert trade.slippage_bps > 0
        assert trade.impact_cost > 0
        assert trade.spread_cost > 0
        assert trade.total_cost == pytest.approx(
            trade.impact_cost + trade.spread_cost, rel=1e-10
        )

    def test_zero_weight_change_no_trades(
        self, simple_prices, simple_volumes, simple_volatilities
    ):
        weight_changes = {"AAPL": 0.0, "MSFT": 0.0}
        summary = simulate_execution(
            weight_changes=weight_changes,
            portfolio_value=1_000_000.0,
            prices=simple_prices,
            volumes=simple_volumes,
            volatilities=simple_volatilities,
        )
        assert len(summary.trades) == 0
        assert summary.total_cost == 0.0
        assert summary.avg_fill_rate == 1.0

    def test_tiny_weight_change_filtered(
        self, simple_prices, simple_volumes, simple_volatilities
    ):
        """Changes below 1e-10 threshold are skipped."""
        weight_changes = {"AAPL": 1e-12, "MSFT": -5e-11}
        summary = simulate_execution(
            weight_changes=weight_changes,
            portfolio_value=1_000_000.0,
            prices=simple_prices,
            volumes=simple_volumes,
            volatilities=simple_volatilities,
        )
        assert len(summary.trades) == 0

    def test_buy_side(self, simple_prices, simple_volumes, simple_volatilities):
        weight_changes = {"AAPL": 0.05}
        summary = simulate_execution(
            weight_changes=weight_changes,
            portfolio_value=1_000_000.0,
            prices=simple_prices,
            volumes=simple_volumes,
            volatilities=simple_volatilities,
        )
        trade = summary.trades[0]
        assert trade.side == "BUY"
        # BUY avg_price should be above market price (paying slippage)
        assert trade.avg_price > trade.market_price

    def test_sell_side(self, simple_prices, simple_volumes, simple_volatilities):
        weight_changes = {"MSFT": -0.10}
        summary = simulate_execution(
            weight_changes=weight_changes,
            portfolio_value=1_000_000.0,
            prices=simple_prices,
            volumes=simple_volumes,
            volatilities=simple_volatilities,
        )
        trade = summary.trades[0]
        assert trade.side == "SELL"
        # SELL avg_price should be below market price (giving up slippage)
        assert trade.avg_price < trade.market_price

    def test_partial_fill_when_exceeds_adv(self):
        """Trade exceeding max_participation * ADV is partially filled."""
        # Tiny ADV so trade will exceed the cap
        prices = {"SMALL": 10.0}
        volumes = {"SMALL": 1_000.0}  # ADV = 1000 shares => $10,000
        weight_changes = {"SMALL": 0.50}
        portfolio_value = 10_000_000.0  # 50% * 10M = $5M trade, ADV = $10K

        summary = simulate_execution(
            weight_changes=weight_changes,
            portfolio_value=portfolio_value,
            prices=prices,
            volumes=volumes,
            max_participation=0.10,
        )
        trade = summary.trades[0]
        # Target: 5,000,000 / 10 = 500,000 shares
        assert trade.target_shares == pytest.approx(500_000.0, rel=1e-10)
        # Filled: 0.10 * 1,000 = 100 shares
        assert trade.filled_shares == pytest.approx(100.0, rel=1e-10)
        assert trade.fill_rate == pytest.approx(100.0 / 500_000.0, rel=1e-6)
        assert trade.fill_rate < 1.0

    def test_full_fill_when_within_adv(
        self, simple_prices, simple_volumes, simple_volatilities
    ):
        """Small trade relative to ADV should fill completely."""
        weight_changes = {"AAPL": 0.01}  # 1% of $1M = $10K vs ADV = $7.5B
        summary = simulate_execution(
            weight_changes=weight_changes,
            portfolio_value=1_000_000.0,
            prices=simple_prices,
            volumes=simple_volumes,
            volatilities=simple_volatilities,
        )
        trade = summary.trades[0]
        assert trade.fill_rate == pytest.approx(1.0, rel=1e-10)
        assert trade.filled_shares == pytest.approx(trade.target_shares, rel=1e-10)

    def test_spread_cost_calculation(self):
        """Spread cost = filled_dollars * (spread_bps / 2) / 10000."""
        prices = {"X": 100.0}
        volumes = {"X": 1_000_000.0}
        weight_changes = {"X": 0.10}
        spread_bps = 10.0

        summary = simulate_execution(
            weight_changes=weight_changes,
            portfolio_value=1_000_000.0,
            prices=prices,
            volumes=volumes,
            spread_bps=spread_bps,
        )
        trade = summary.trades[0]
        filled_dollars = trade.filled_shares * trade.market_price
        expected_spread = filled_dollars * (spread_bps / 2) / 10000
        assert trade.spread_cost == pytest.approx(expected_spread, rel=1e-10)

    def test_sqrt_impact_model(
        self, simple_prices, simple_volumes, simple_volatilities
    ):
        weight_changes = {"AAPL": 0.10}
        summary = simulate_execution(
            weight_changes=weight_changes,
            portfolio_value=1_000_000.0,
            prices=simple_prices,
            volumes=simple_volumes,
            volatilities=simple_volatilities,
            impact_model="sqrt",
        )
        assert len(summary.trades) == 1
        assert summary.trades[0].impact_cost > 0

    def test_linear_impact_model(
        self, simple_prices, simple_volumes, simple_volatilities
    ):
        weight_changes = {"AAPL": 0.10}
        summary = simulate_execution(
            weight_changes=weight_changes,
            portfolio_value=1_000_000.0,
            prices=simple_prices,
            volumes=simple_volumes,
            volatilities=simple_volatilities,
            impact_model="linear",
        )
        assert len(summary.trades) == 1
        assert summary.trades[0].impact_cost > 0

    def test_different_impact_models_give_different_costs(
        self, simple_prices, simple_volumes, simple_volatilities
    ):
        weight_changes = {"AAPL": 0.10, "MSFT": -0.05}
        sqrt_summary = simulate_execution(
            weight_changes=weight_changes,
            portfolio_value=1_000_000.0,
            prices=simple_prices,
            volumes=simple_volumes,
            volatilities=simple_volatilities,
            impact_model="sqrt",
        )
        lin_summary = simulate_execution(
            weight_changes=weight_changes,
            portfolio_value=1_000_000.0,
            prices=simple_prices,
            volumes=simple_volumes,
            volatilities=simple_volatilities,
            impact_model="linear",
        )
        # Both positive but generally different amounts
        assert sqrt_summary.total_impact_cost > 0
        assert lin_summary.total_impact_cost > 0
        assert sqrt_summary.total_impact_cost != pytest.approx(
            lin_summary.total_impact_cost, rel=0.01
        )

    def test_summary_aggregation(
        self, simple_prices, simple_volumes, simple_volatilities
    ):
        """Summary totals should equal the sum of per-trade values."""
        weight_changes = {"AAPL": 0.15, "MSFT": -0.10, "GOOG": 0.05}
        summary = simulate_execution(
            weight_changes=weight_changes,
            portfolio_value=1_000_000.0,
            prices=simple_prices,
            volumes=simple_volumes,
            volatilities=simple_volatilities,
        )
        expected_impact = sum(t.impact_cost for t in summary.trades)
        expected_spread = sum(t.spread_cost for t in summary.trades)
        expected_notional = sum(
            t.filled_shares * t.market_price for t in summary.trades
        )
        assert summary.total_impact_cost == pytest.approx(expected_impact, rel=1e-10)
        assert summary.total_spread_cost == pytest.approx(expected_spread, rel=1e-10)
        assert summary.total_cost == pytest.approx(
            expected_impact + expected_spread, rel=1e-10
        )
        assert summary.total_notional == pytest.approx(expected_notional, rel=1e-10)
        assert summary.cost_as_pct == pytest.approx(
            summary.total_cost / summary.total_notional * 100, rel=1e-10
        )

    def test_avg_slippage_and_fill_rate(
        self, simple_prices, simple_volumes, simple_volatilities
    ):
        weight_changes = {"AAPL": 0.10, "MSFT": 0.05}
        summary = simulate_execution(
            weight_changes=weight_changes,
            portfolio_value=1_000_000.0,
            prices=simple_prices,
            volumes=simple_volumes,
            volatilities=simple_volatilities,
        )
        slippages = [t.slippage_bps for t in summary.trades]
        fill_rates = [t.fill_rate for t in summary.trades]
        assert summary.avg_slippage_bps == pytest.approx(np.mean(slippages), rel=1e-10)
        assert summary.avg_fill_rate == pytest.approx(np.mean(fill_rates), rel=1e-10)

    def test_default_volatilities(self, simple_prices, simple_volumes):
        """When volatilities=None, defaults to 0.02 for each symbol."""
        weight_changes = {"AAPL": 0.10}
        summary = simulate_execution(
            weight_changes=weight_changes,
            portfolio_value=1_000_000.0,
            prices=simple_prices,
            volumes=simple_volumes,
            volatilities=None,
        )
        assert len(summary.trades) == 1
        assert summary.trades[0].impact_cost > 0

    def test_missing_price_symbol_skipped(self, simple_volumes, simple_volatilities):
        """Symbol not in prices dict is skipped."""
        weight_changes = {"UNKNOWN": 0.10}
        summary = simulate_execution(
            weight_changes=weight_changes,
            portfolio_value=1_000_000.0,
            prices={"AAPL": 150.0},
            volumes=simple_volumes,
            volatilities=simple_volatilities,
        )
        assert len(summary.trades) == 0

    def test_zero_price_symbol_skipped(self, simple_volumes, simple_volatilities):
        """Symbol with price=0 is skipped."""
        weight_changes = {"AAPL": 0.10}
        summary = simulate_execution(
            weight_changes=weight_changes,
            portfolio_value=1_000_000.0,
            prices={"AAPL": 0.0},
            volumes=simple_volumes,
            volatilities=simple_volatilities,
        )
        assert len(summary.trades) == 0

    def test_larger_portfolio_higher_costs(
        self, simple_prices, simple_volumes, simple_volatilities
    ):
        """Bigger portfolio => bigger trade => higher total cost."""
        weight_changes = {"AAPL": 0.10}
        small = simulate_execution(
            weight_changes=weight_changes,
            portfolio_value=100_000.0,
            prices=simple_prices,
            volumes=simple_volumes,
            volatilities=simple_volatilities,
        )
        large = simulate_execution(
            weight_changes=weight_changes,
            portfolio_value=10_000_000.0,
            prices=simple_prices,
            volumes=simple_volumes,
            volatilities=simple_volatilities,
        )
        assert large.total_cost > small.total_cost


# ── analyze_capacity ────────────────────────────────────────────────


class TestAnalyzeCapacity:
    def test_returns_capacity_result(
        self, simple_weights, simple_prices, simple_volumes, simple_volatilities
    ):
        result = analyze_capacity(
            weights=simple_weights,
            prices=simple_prices,
            volumes=simple_volumes,
            volatilities=simple_volatilities,
        )
        assert isinstance(result, CapacityResult)
        assert result.max_capacity_usd > 0
        assert result.impact_threshold_bps == 50.0
        assert result.bottleneck_symbol in ("AAPL", "MSFT", "GOOG")
        assert result.bottleneck_adv_pct > 0

    def test_capacity_curve_non_empty(
        self, simple_weights, simple_prices, simple_volumes
    ):
        result = analyze_capacity(
            weights=simple_weights,
            prices=simple_prices,
            volumes=simple_volumes,
        )
        assert len(result.capacity_curve) > 0

    def test_capacity_curve_monotone_increasing_impact(
        self, simple_weights, simple_prices, simple_volumes
    ):
        """Impact should increase monotonically with AUM."""
        result = analyze_capacity(
            weights=simple_weights,
            prices=simple_prices,
            volumes=simple_volumes,
        )
        impacts = [bps for _, bps in result.capacity_curve]
        for i in range(1, len(impacts)):
            assert impacts[i] >= impacts[i - 1]

    def test_capacity_curve_aum_increasing(
        self, simple_weights, simple_prices, simple_volumes
    ):
        """AUM values in the curve should be monotonically increasing."""
        result = analyze_capacity(
            weights=simple_weights,
            prices=simple_prices,
            volumes=simple_volumes,
        )
        aums = [aum for aum, _ in result.capacity_curve]
        for i in range(1, len(aums)):
            assert aums[i] > aums[i - 1]

    def test_higher_volumes_higher_capacity(self, simple_weights, simple_prices):
        """More liquid market => higher capacity."""
        low_vol = {"AAPL": 100_000.0, "MSFT": 50_000.0, "GOOG": 80_000.0}
        high_vol = {"AAPL": 50_000_000.0, "MSFT": 25_000_000.0, "GOOG": 30_000_000.0}

        low_result = analyze_capacity(
            weights=simple_weights,
            prices=simple_prices,
            volumes=low_vol,
        )
        high_result = analyze_capacity(
            weights=simple_weights,
            prices=simple_prices,
            volumes=high_vol,
        )
        assert high_result.max_capacity_usd >= low_result.max_capacity_usd

    def test_bottleneck_is_least_liquid(self, simple_prices):
        """Bottleneck should be the symbol with lowest ADV relative to weight."""
        weights = {"AAPL": 0.30, "MSFT": 0.30, "GOOG": 0.40}
        # GOOG has the lowest ADV in dollar terms relative to weight
        volumes = {
            "AAPL": 10_000_000.0,  # ADV$ = 10M * 150 = 1.5B
            "MSFT": 5_000_000.0,   # ADV$ = 5M * 300 = 1.5B
            "GOOG": 1_000.0,       # ADV$ = 1K * 100 = 100K (very illiquid)
        }
        result = analyze_capacity(
            weights=weights,
            prices=simple_prices,
            volumes=volumes,
        )
        assert result.bottleneck_symbol == "GOOG"

    def test_custom_aum_range(self, simple_weights, simple_prices, simple_volumes):
        """Custom aum_range is respected."""
        custom_range = [1_000_000.0, 5_000_000.0, 10_000_000.0]
        result = analyze_capacity(
            weights=simple_weights,
            prices=simple_prices,
            volumes=simple_volumes,
            aum_range=custom_range,
        )
        assert len(result.capacity_curve) == 3
        aums = [aum for aum, _ in result.capacity_curve]
        assert aums == pytest.approx(custom_range)

    def test_custom_threshold(self, simple_weights, simple_prices, simple_volumes):
        """Lower threshold should give lower or equal capacity."""
        tight = analyze_capacity(
            weights=simple_weights,
            prices=simple_prices,
            volumes=simple_volumes,
            impact_threshold_bps=10.0,
        )
        loose = analyze_capacity(
            weights=simple_weights,
            prices=simple_prices,
            volumes=simple_volumes,
            impact_threshold_bps=100.0,
        )
        assert tight.max_capacity_usd <= loose.max_capacity_usd

    def test_default_volatilities(
        self, simple_weights, simple_prices, simple_volumes
    ):
        """When volatilities=None, defaults are used without error."""
        result = analyze_capacity(
            weights=simple_weights,
            prices=simple_prices,
            volumes=simple_volumes,
            volatilities=None,
        )
        assert isinstance(result, CapacityResult)
        assert result.max_capacity_usd > 0

    def test_linear_impact_model(
        self, simple_weights, simple_prices, simple_volumes
    ):
        result = analyze_capacity(
            weights=simple_weights,
            prices=simple_prices,
            volumes=simple_volumes,
            impact_model="linear",
        )
        assert isinstance(result, CapacityResult)
        assert result.max_capacity_usd > 0

    def test_zero_weight_symbol_ignored(self, simple_prices, simple_volumes):
        """Symbols with zero weight do not become the bottleneck."""
        weights = {"AAPL": 0.50, "MSFT": 0.50, "GOOG": 0.0}
        result = analyze_capacity(
            weights=weights,
            prices=simple_prices,
            volumes=simple_volumes,
        )
        assert result.bottleneck_symbol != "GOOG"


# ── estimate_volumes_from_prices ────────────────────────────────────


class TestEstimateVolumesFromPrices:
    def test_returns_dict_with_correct_keys(self, price_dataframe):
        volumes = estimate_volumes_from_prices(price_dataframe)
        assert isinstance(volumes, dict)
        assert set(volumes.keys()) == {"AAPL", "MSFT", "GOOG"}

    def test_default_values_positive(self, price_dataframe):
        volumes = estimate_volumes_from_prices(price_dataframe)
        for sym, v in volumes.items():
            assert v > 0, f"Volume for {sym} should be positive"

    def test_default_is_one_million(self, price_dataframe):
        volumes = estimate_volumes_from_prices(price_dataframe)
        for v in volumes.values():
            assert v == pytest.approx(1_000_000.0)


# ── estimate_volatilities_from_prices ───────────────────────────────


class TestEstimateVolatilitiesFromPrices:
    def test_returns_dict_with_correct_keys(self, price_dataframe):
        vols = estimate_volatilities_from_prices(price_dataframe)
        assert isinstance(vols, dict)
        assert set(vols.keys()) == {"AAPL", "MSFT", "GOOG"}

    def test_volatilities_positive(self, price_dataframe):
        vols = estimate_volatilities_from_prices(price_dataframe)
        for sym, v in vols.items():
            assert v > 0, f"Volatility for {sym} should be positive"

    def test_reasonable_daily_volatility(self, price_dataframe):
        """Daily vol from ~1% daily returns should be in a reasonable range."""
        vols = estimate_volatilities_from_prices(price_dataframe)
        for sym, v in vols.items():
            assert 0.001 < v < 0.10, f"Vol for {sym} = {v} seems unreasonable"

    def test_single_column(self):
        """Works correctly with a single-column DataFrame."""
        np.random.seed(7)
        prices = pd.DataFrame(
            {"SPY": 100.0 * np.cumprod(1 + np.random.randn(30) * 0.01)},
            index=pd.date_range("2024-01-01", periods=30, freq="B"),
        )
        vols = estimate_volatilities_from_prices(prices)
        assert "SPY" in vols
        assert vols["SPY"] > 0
