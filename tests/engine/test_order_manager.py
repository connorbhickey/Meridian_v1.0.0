"""Tests for the order manager engine module."""

from __future__ import annotations

from datetime import datetime

import pytest

from portopt.data.models import Asset, AssetType, Holding, OptimizationResult, Portfolio
from portopt.engine.order_manager import (
    OrderBatch,
    OrderStatus,
    OrderType,
    TradeOrder,
    format_orders_text,
    generate_orders,
    generate_rebalance_orders,
    orders_to_dicts,
    prioritize_orders,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def aapl_holding() -> Holding:
    return Holding(
        asset=Asset("AAPL", "Apple Inc."),
        quantity=100.0,
        cost_basis=15_000.0,
        current_price=175.0,
        account="Individual",
    )


@pytest.fixture
def msft_holding() -> Holding:
    return Holding(
        asset=Asset("MSFT", "Microsoft Corp"),
        quantity=50.0,
        cost_basis=18_000.0,
        current_price=420.0,
        account="Individual",
    )


@pytest.fixture
def goog_holding() -> Holding:
    return Holding(
        asset=Asset("GOOG", "Alphabet Inc."),
        quantity=30.0,
        cost_basis=4_000.0,
        current_price=150.0,
        account="Individual",
    )


@pytest.fixture
def portfolio(aapl_holding, msft_holding, goog_holding) -> Portfolio:
    """Portfolio: AAPL=$17500, MSFT=$21000, GOOG=$4500 -> total=$43000."""
    return Portfolio(
        name="Test Portfolio",
        holdings=[aapl_holding, msft_holding, goog_holding],
    )


@pytest.fixture
def empty_portfolio() -> Portfolio:
    return Portfolio(name="Empty", holdings=[])


@pytest.fixture
def target_weights() -> dict[str, float]:
    return {"AAPL": 0.40, "MSFT": 0.40, "GOOG": 0.20}


@pytest.fixture
def prices() -> dict[str, float]:
    return {"AAPL": 175.0, "MSFT": 420.0, "GOOG": 150.0}


@pytest.fixture
def opt_result(target_weights) -> OptimizationResult:
    return OptimizationResult(
        method="max_sharpe",
        weights=target_weights,
        expected_return=0.12,
        volatility=0.18,
        sharpe_ratio=0.67,
    )


@pytest.fixture
def basic_batch(portfolio, target_weights) -> OrderBatch:
    """A pre-generated batch for tests that need an existing batch."""
    return generate_orders(portfolio, target_weights)


# ── Helper ────────────────────────────────────────────────────────────


def _find_order(batch: OrderBatch, symbol: str) -> TradeOrder | None:
    for order in batch.orders:
        if order.symbol == symbol:
            return order
    return None


def _orders_by_side(batch: OrderBatch, side: str) -> list[TradeOrder]:
    return [o for o in batch.orders if o.side == side]


# ── TestGenerateOrders ────────────────────────────────────────────────


class TestGenerateOrders:
    """Tests for generate_orders()."""

    def test_basic_rebalance_produces_orders(self, portfolio, target_weights):
        batch = generate_orders(portfolio, target_weights)
        assert len(batch.orders) > 0

    def test_correct_buy_sell_sides(self, portfolio, target_weights):
        """Portfolio weights: AAPL~40.7%, MSFT~48.8%, GOOG~10.5%.
        Target: AAPL=40%, MSFT=40%, GOOG=20%.
        Expect MSFT sell, GOOG buy. AAPL is near-target and may be filtered."""
        batch = generate_orders(portfolio, target_weights)
        msft_order = _find_order(batch, "MSFT")
        goog_order = _find_order(batch, "GOOG")
        assert msft_order is not None
        assert msft_order.side == "SELL"
        assert goog_order is not None
        assert goog_order.side == "BUY"

    def test_sells_appear_before_buys(self, portfolio, target_weights):
        batch = generate_orders(portfolio, target_weights)
        sell_seen = False
        buy_seen = False
        for order in batch.orders:
            if order.side == "SELL":
                assert not buy_seen, "Found SELL after a BUY order"
                sell_seen = True
            elif order.side == "BUY":
                buy_seen = True

    def test_market_orders_have_no_limit_price(self, portfolio, target_weights):
        batch = generate_orders(portfolio, target_weights, order_type=OrderType.MARKET)
        for order in batch.orders:
            assert order.limit_price is None

    def test_limit_orders_have_limit_prices(self, portfolio, target_weights):
        batch = generate_orders(
            portfolio, target_weights, order_type=OrderType.LIMIT, limit_offset_pct=1.0
        )
        for order in batch.orders:
            assert order.limit_price is not None
            assert order.limit_price > 0

    def test_limit_buy_price_above_market(self, portfolio, target_weights, prices):
        batch = generate_orders(
            portfolio, target_weights,
            prices=prices,
            order_type=OrderType.LIMIT,
            limit_offset_pct=1.0,
        )
        for order in batch.orders:
            if order.side == "BUY":
                market = prices[order.symbol]
                assert order.limit_price > market

    def test_limit_sell_price_below_market(self, portfolio, target_weights, prices):
        batch = generate_orders(
            portfolio, target_weights,
            prices=prices,
            order_type=OrderType.LIMIT,
            limit_offset_pct=1.0,
        )
        for order in batch.orders:
            if order.side == "SELL":
                market = prices[order.symbol]
                assert order.limit_price < market

    def test_limit_offset_magnitude(self, portfolio, target_weights, prices):
        offset = 2.0
        batch = generate_orders(
            portfolio, target_weights,
            prices=prices,
            order_type=OrderType.LIMIT,
            limit_offset_pct=offset,
        )
        for order in batch.orders:
            market = prices[order.symbol]
            expected_offset = market * (offset / 100.0)
            if order.side == "BUY":
                assert order.limit_price == pytest.approx(market + expected_offset)
            else:
                assert order.limit_price == pytest.approx(market - expected_offset)

    def test_small_trades_filtered(self, portfolio):
        """Weights that produce a tiny delta should be filtered by min_trade_value."""
        # Set target very close to current weights so delta is tiny
        current_w = portfolio.weights
        # Nudge AAPL by a tiny amount
        target = {s: w for s, w in current_w.items()}
        target["AAPL"] = current_w["AAPL"] + 0.0001
        target["MSFT"] = current_w["MSFT"] - 0.0001

        batch = generate_orders(portfolio, target, min_trade_value=100.0)
        # The tiny deltas ($4.30 each) should be filtered out
        assert len(batch.orders) == 0

    def test_min_trade_value_respected(self, portfolio, target_weights):
        batch_strict = generate_orders(portfolio, target_weights, min_trade_value=5000.0)
        batch_loose = generate_orders(portfolio, target_weights, min_trade_value=1.0)
        assert len(batch_strict.orders) <= len(batch_loose.orders)

    def test_round_lots(self, portfolio, target_weights):
        batch = generate_orders(portfolio, target_weights, round_lots=True)
        for order in batch.orders:
            assert order.quantity % 100 == 0

    def test_round_lots_may_filter_small_orders(self, portfolio, target_weights):
        """Round lots that round to zero should be removed."""
        batch = generate_orders(portfolio, target_weights, round_lots=True)
        for order in batch.orders:
            assert order.quantity > 0

    def test_empty_portfolio_returns_empty_batch(self, empty_portfolio, target_weights):
        batch = generate_orders(empty_portfolio, target_weights)
        assert len(batch.orders) == 0
        assert batch.total_buy_value == 0.0
        assert batch.total_sell_value == 0.0

    def test_zero_target_weights_generates_sells(self, portfolio):
        target = {"AAPL": 0.0, "MSFT": 0.0, "GOOG": 0.0}
        batch = generate_orders(portfolio, target)
        for order in batch.orders:
            assert order.side == "SELL"
        # All three symbols should have sells
        sold_symbols = {o.symbol for o in batch.orders}
        assert sold_symbols == {"AAPL", "MSFT", "GOOG"}

    def test_zero_target_empty_dict_generates_sells(self, portfolio):
        """An empty target dict means 0% target for all current holdings."""
        batch = generate_orders(portfolio, {})
        for order in batch.orders:
            assert order.side == "SELL"

    def test_new_symbol_in_target_creates_buy(self, portfolio, prices):
        target = {"AAPL": 0.30, "MSFT": 0.30, "GOOG": 0.20, "TSLA": 0.20}
        prices_with_tsla = {**prices, "TSLA": 250.0}
        batch = generate_orders(portfolio, target, prices=prices_with_tsla)
        tsla_order = _find_order(batch, "TSLA")
        assert tsla_order is not None
        assert tsla_order.side == "BUY"

    def test_new_symbol_without_price_skipped(self, portfolio, prices):
        """New symbol with no price available should be skipped."""
        target = {"AAPL": 0.30, "MSFT": 0.30, "GOOG": 0.20, "TSLA": 0.20}
        batch = generate_orders(portfolio, target, prices=prices)
        tsla_order = _find_order(batch, "TSLA")
        assert tsla_order is None

    def test_price_override_used(self, portfolio, target_weights):
        custom_prices = {"AAPL": 200.0, "MSFT": 400.0, "GOOG": 160.0}
        batch = generate_orders(portfolio, target_weights, prices=custom_prices)
        # The batch should use custom prices for cost estimation
        assert batch.total_estimated_cost >= 0

    def test_estimated_costs_positive(self, portfolio, target_weights):
        batch = generate_orders(portfolio, target_weights)
        for order in batch.orders:
            assert order.estimated_cost >= 0
        assert batch.total_estimated_cost >= 0

    def test_order_timestamps(self, portfolio, target_weights):
        before = datetime.now()
        batch = generate_orders(portfolio, target_weights)
        after = datetime.now()
        for order in batch.orders:
            assert order.created_at is not None
            assert before <= order.created_at <= after

    def test_order_status_defaults_to_pending(self, portfolio, target_weights):
        batch = generate_orders(portfolio, target_weights)
        for order in batch.orders:
            assert order.status == OrderStatus.PENDING

    def test_order_type_propagated(self, portfolio, target_weights):
        for otype in [OrderType.MARKET, OrderType.LIMIT, OrderType.STOP_LIMIT]:
            batch = generate_orders(portfolio, target_weights, order_type=otype)
            for order in batch.orders:
                assert order.order_type == otype

    def test_account_from_holding(self, portfolio, target_weights):
        batch = generate_orders(portfolio, target_weights)
        for order in batch.orders:
            if order.symbol in {"AAPL", "MSFT", "GOOG"}:
                assert order.account == "Individual"

    def test_spread_bps_affects_cost(self, portfolio, target_weights):
        batch_narrow = generate_orders(portfolio, target_weights, spread_bps=1.0)
        batch_wide = generate_orders(portfolio, target_weights, spread_bps=20.0)
        assert batch_wide.total_estimated_cost > batch_narrow.total_estimated_cost


# ── TestGenerateRebalanceOrders ───────────────────────────────────────


class TestGenerateRebalanceOrders:
    """Tests for generate_rebalance_orders() wrapper."""

    def test_extracts_weights_from_result(self, portfolio, opt_result, target_weights):
        batch = generate_rebalance_orders(portfolio, opt_result)
        direct_batch = generate_orders(portfolio, target_weights)
        assert len(batch.orders) == len(direct_batch.orders)

    def test_kwargs_forwarded(self, portfolio, opt_result):
        batch = generate_rebalance_orders(
            portfolio, opt_result, order_type=OrderType.LIMIT, limit_offset_pct=2.0
        )
        for order in batch.orders:
            assert order.order_type == OrderType.LIMIT
            assert order.limit_price is not None

    def test_prices_forwarded(self, portfolio, opt_result, prices):
        batch = generate_rebalance_orders(portfolio, opt_result, prices=prices)
        assert batch.portfolio_value > 0

    def test_min_trade_value_forwarded(self, portfolio, opt_result):
        batch_strict = generate_rebalance_orders(
            portfolio, opt_result, min_trade_value=50_000.0
        )
        batch_loose = generate_rebalance_orders(
            portfolio, opt_result, min_trade_value=1.0
        )
        assert len(batch_strict.orders) <= len(batch_loose.orders)


# ── TestPrioritizeOrders ──────────────────────────────────────────────


class TestPrioritizeOrders:
    """Tests for prioritize_orders()."""

    def test_all_sells_kept(self, portfolio):
        target = {"AAPL": 0.0, "MSFT": 0.0, "GOOG": 0.60}
        batch = generate_orders(portfolio, target)
        prioritized = prioritize_orders(batch, available_cash=0.0)
        sells = _orders_by_side(prioritized, "SELL")
        original_sells = _orders_by_side(batch, "SELL")
        assert len(sells) == len(original_sells)

    def test_buys_trimmed_when_cash_constrained(self, portfolio):
        """Set a very low cash budget so buys must be trimmed."""
        target = {"AAPL": 0.0, "MSFT": 0.0, "GOOG": 1.0}
        batch = generate_orders(portfolio, target)
        # With zero cash and sells proceeding, buys should be constrained
        prioritized = prioritize_orders(batch, available_cash=0.0)
        buys_original = _orders_by_side(batch, "BUY")
        buys_trimmed = _orders_by_side(prioritized, "BUY")
        if buys_original:
            original_buy_qty = sum(o.quantity for o in buys_original)
            trimmed_buy_qty = sum(o.quantity for o in buys_trimmed)
            assert trimmed_buy_qty <= original_buy_qty + 1e-8

    def test_no_cash_no_sells_means_no_buys(self):
        """An order batch with only buys and no cash should result in no buys."""
        buy_order = TradeOrder(
            symbol="AAPL",
            side="BUY",
            quantity=100.0,
            order_type=OrderType.MARKET,
            estimated_cost=0.5,
        )
        batch = OrderBatch(
            orders=[buy_order],
            portfolio_value=50_000.0,
            total_buy_value=17_500.0,
            total_sell_value=0.0,
            total_estimated_cost=0.5,
            net_cash_flow=-17_500.0,
            turnover=0.35,
        )
        prioritized = prioritize_orders(batch, available_cash=0.0)
        assert len(prioritized.orders) == 0

    def test_partial_fill_when_budget_runs_out(self):
        """When budget can only partially cover a buy, the quantity is reduced."""
        buy_order = TradeOrder(
            symbol="AAPL",
            side="BUY",
            quantity=100.0,
            order_type=OrderType.LIMIT,
            limit_price=175.0,
            estimated_cost=1.0,
        )
        batch = OrderBatch(
            orders=[buy_order],
            portfolio_value=50_000.0,
            total_buy_value=17_500.0,
            total_sell_value=0.0,
            total_estimated_cost=1.0,
            net_cash_flow=-17_500.0,
            turnover=0.35,
        )
        # Budget = $8750, enough for 50 shares at $175
        prioritized = prioritize_orders(batch, available_cash=8_750.0)
        assert len(prioritized.orders) == 1
        assert prioritized.orders[0].quantity == pytest.approx(50.0)
        assert "Trimmed" in prioritized.orders[0].notes

    def test_full_budget_keeps_all_buys(self):
        """Ample cash keeps all buy orders intact."""
        buy_order = TradeOrder(
            symbol="AAPL",
            side="BUY",
            quantity=100.0,
            order_type=OrderType.LIMIT,
            limit_price=175.0,
            estimated_cost=1.0,
        )
        batch = OrderBatch(
            orders=[buy_order],
            portfolio_value=50_000.0,
            total_buy_value=17_500.0,
            total_sell_value=0.0,
            total_estimated_cost=1.0,
            net_cash_flow=-17_500.0,
            turnover=0.35,
        )
        prioritized = prioritize_orders(batch, available_cash=50_000.0)
        assert len(prioritized.orders) == 1
        assert prioritized.orders[0].quantity == pytest.approx(100.0)

    def test_sell_proceeds_add_to_budget(self):
        """Sell proceeds should supplement the available cash for buys."""
        sell_order = TradeOrder(
            symbol="MSFT",
            side="SELL",
            quantity=10.0,
            order_type=OrderType.LIMIT,
            limit_price=420.0,
            estimated_cost=0.5,
        )
        buy_order = TradeOrder(
            symbol="GOOG",
            side="BUY",
            quantity=25.0,
            order_type=OrderType.LIMIT,
            limit_price=150.0,
            estimated_cost=0.5,
        )
        batch = OrderBatch(
            orders=[sell_order, buy_order],
            portfolio_value=50_000.0,
            total_buy_value=3_750.0,
            total_sell_value=4_200.0,
            total_estimated_cost=1.0,
            net_cash_flow=450.0,
            turnover=0.159,
        )
        # Zero cash, but sell proceeds ($4200) should cover the buy ($3750)
        prioritized = prioritize_orders(batch, available_cash=0.0)
        buys = _orders_by_side(prioritized, "BUY")
        assert len(buys) == 1
        assert buys[0].quantity == pytest.approx(25.0)

    def test_none_cash_defaults_to_zero(self, basic_batch):
        prioritized = prioritize_orders(basic_batch, available_cash=None)
        assert isinstance(prioritized, OrderBatch)


# ── TestOrdersToDicts ─────────────────────────────────────────────────


class TestOrdersToDicts:
    """Tests for orders_to_dicts()."""

    def test_correct_keys(self, basic_batch):
        dicts = orders_to_dicts(basic_batch)
        expected_keys = {"date", "symbol", "side", "quantity", "price", "cost", "weight_after"}
        for d in dicts:
            assert set(d.keys()) == expected_keys

    def test_output_length_matches_orders(self, basic_batch):
        dicts = orders_to_dicts(basic_batch)
        assert len(dicts) == len(basic_batch.orders)

    def test_target_weights_populate_weight_after(self, basic_batch, target_weights):
        dicts = orders_to_dicts(basic_batch, target_weights=target_weights)
        for d in dicts:
            symbol = d["symbol"]
            if symbol in target_weights:
                assert d["weight_after"] == pytest.approx(target_weights[symbol])

    def test_no_target_weights_gives_zero(self, basic_batch):
        dicts = orders_to_dicts(basic_batch, target_weights=None)
        for d in dicts:
            assert d["weight_after"] == 0.0

    def test_prices_used_for_market_orders(self, portfolio, target_weights, prices):
        batch = generate_orders(portfolio, target_weights, order_type=OrderType.MARKET)
        dicts = orders_to_dicts(batch, prices=prices)
        for d in dicts:
            if d["symbol"] in prices:
                # For market orders without limit_price, should use provided prices
                assert d["price"] == pytest.approx(prices[d["symbol"]])

    def test_limit_price_used_when_present(self, portfolio, target_weights, prices):
        batch = generate_orders(
            portfolio, target_weights,
            prices=prices,
            order_type=OrderType.LIMIT,
            limit_offset_pct=1.0,
        )
        dicts = orders_to_dicts(batch)
        for d in dicts:
            assert d["price"] > 0

    def test_date_format(self, basic_batch):
        dicts = orders_to_dicts(basic_batch)
        for d in dicts:
            if d["date"]:
                datetime.strptime(d["date"], "%Y-%m-%d")

    def test_empty_batch(self, empty_portfolio, target_weights):
        batch = generate_orders(empty_portfolio, target_weights)
        dicts = orders_to_dicts(batch)
        assert dicts == []


# ── TestFormatOrdersText ──────────────────────────────────────────────


class TestFormatOrdersText:
    """Tests for format_orders_text()."""

    def test_returns_non_empty_string(self, basic_batch):
        text = format_orders_text(basic_batch)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_contains_side(self, basic_batch):
        text = format_orders_text(basic_batch)
        assert "Side" in text

    def test_contains_symbol_header(self, basic_batch):
        text = format_orders_text(basic_batch)
        assert "Symbol" in text

    def test_contains_qty_header(self, basic_batch):
        text = format_orders_text(basic_batch)
        assert "Qty" in text

    def test_contains_order_symbols(self, basic_batch):
        text = format_orders_text(basic_batch)
        for order in basic_batch.orders:
            assert order.symbol in text

    def test_contains_total_buy_summary(self, basic_batch):
        text = format_orders_text(basic_batch)
        assert "Total Buy" in text

    def test_contains_total_sell_summary(self, basic_batch):
        text = format_orders_text(basic_batch)
        assert "Total Sell" in text

    def test_contains_turnover(self, basic_batch):
        text = format_orders_text(basic_batch)
        assert "Turnover" in text

    def test_contains_portfolio_value(self, basic_batch):
        text = format_orders_text(basic_batch)
        assert "Portfolio" in text

    def test_empty_batch_still_formats(self, empty_portfolio, target_weights):
        batch = generate_orders(empty_portfolio, target_weights)
        text = format_orders_text(batch)
        assert "Total Buy" in text


# ── TestOrderBatch ────────────────────────────────────────────────────


class TestOrderBatch:
    """Tests for OrderBatch computed fields."""

    def test_turnover_calculation(self, portfolio, target_weights):
        batch = generate_orders(portfolio, target_weights)
        total_traded = batch.total_buy_value + batch.total_sell_value
        expected_turnover = total_traded / batch.portfolio_value
        assert batch.turnover == pytest.approx(expected_turnover)

    def test_net_cash_flow_equals_sells_minus_buys(self, portfolio, target_weights):
        batch = generate_orders(portfolio, target_weights)
        expected_net = batch.total_sell_value - batch.total_buy_value
        assert batch.net_cash_flow == pytest.approx(expected_net)

    def test_all_sell_batch_positive_net_cash(self, portfolio):
        target = {"AAPL": 0.0, "MSFT": 0.0, "GOOG": 0.0}
        batch = generate_orders(portfolio, target)
        assert batch.net_cash_flow > 0
        assert batch.total_buy_value == 0.0
        assert batch.total_sell_value > 0.0

    def test_portfolio_value_propagated(self, portfolio, target_weights):
        batch = generate_orders(portfolio, target_weights)
        assert batch.portfolio_value == pytest.approx(portfolio.total_value)

    def test_created_at_set(self, portfolio, target_weights):
        batch = generate_orders(portfolio, target_weights)
        assert batch.created_at is not None
        assert isinstance(batch.created_at, datetime)

    def test_total_estimated_cost_is_sum_of_order_costs(self, portfolio, target_weights):
        batch = generate_orders(portfolio, target_weights)
        expected = sum(o.estimated_cost for o in batch.orders)
        assert batch.total_estimated_cost == pytest.approx(expected)


# ── TestEnums ─────────────────────────────────────────────────────────


class TestEnums:
    """Tests for OrderStatus and OrderType enums."""

    def test_order_status_values(self):
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.SUBMITTED.value == "submitted"
        assert OrderStatus.PARTIALLY_FILLED.value == "partially_filled"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"

    def test_order_type_values(self):
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"
        assert OrderType.STOP_LIMIT.value == "stop_limit"

    def test_order_status_count(self):
        assert len(OrderStatus) == 6

    def test_order_type_count(self):
        assert len(OrderType) == 4
