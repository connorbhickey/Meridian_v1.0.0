"""Trade order generation — converts optimization weight changes into executable orders.

Given a current portfolio and target weights (from an optimization result),
this module computes the share-level trades required to rebalance, estimates
transaction costs, and produces order batches ready for review or execution.

This is an ENGINE module — it has zero GUI knowledge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from portopt.data.models import OptimizationResult, Portfolio

logger = logging.getLogger(__name__)


# ── Enums ────────────────────────────────────────────────────────────


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


# ── Dataclasses ──────────────────────────────────────────────────────


@dataclass
class TradeOrder:
    """A single trade order for one symbol."""

    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    account: str = ""
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_avg_price: float = 0.0
    estimated_cost: float = 0.0
    notes: str = ""
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class OrderBatch:
    """A collection of trade orders produced from a single rebalance."""

    orders: list[TradeOrder]
    portfolio_value: float
    total_buy_value: float
    total_sell_value: float
    total_estimated_cost: float
    net_cash_flow: float  # positive = sells exceed buys
    turnover: float  # total traded value / portfolio value
    created_at: datetime = field(default_factory=datetime.now)


# ── Constants ────────────────────────────────────────────────────────

DEFAULT_SPREAD_BPS = 5.0
DEFAULT_MIN_TRADE_VALUE = 50.0
LOT_SIZE = 100


# ── Core functions ───────────────────────────────────────────────────


def generate_orders(
    portfolio: Portfolio,
    target_weights: dict[str, float],
    prices: dict[str, float] | None = None,
    order_type: OrderType = OrderType.MARKET,
    limit_offset_pct: float = 0.5,
    min_trade_value: float = DEFAULT_MIN_TRADE_VALUE,
    round_lots: bool = False,
    spread_bps: float = DEFAULT_SPREAD_BPS,
) -> OrderBatch:
    """Generate trade orders to rebalance a portfolio toward target weights.

    Args:
        portfolio: Current portfolio with holdings and prices.
        target_weights: {symbol: target_weight} mapping (weights should sum to ~1).
        prices: {symbol: price} override. Falls back to holding current_price.
        order_type: Order type for all generated orders.
        limit_offset_pct: For LIMIT orders, offset from market price (percent).
            Buys use price * (1 + offset), sells use price * (1 - offset).
        min_trade_value: Minimum dollar value to generate an order (filters dust).
        round_lots: If True, round share quantities to nearest 100-share lot.
        spread_bps: Assumed bid-ask spread in basis points for cost estimation.

    Returns:
        OrderBatch containing all generated orders and summary statistics.
    """
    now = datetime.now()
    port_value = portfolio.total_value
    if port_value <= 0:
        logger.warning("Portfolio value is zero — no orders generated.")
        return _empty_batch(port_value, now)

    # Build price lookup: explicit prices override holding prices
    price_map = _build_price_map(portfolio, prices)
    current_weights = portfolio.weights

    # Collect all symbols that appear in either current or target
    all_symbols = set(current_weights.keys()) | set(target_weights.keys())

    orders: list[TradeOrder] = []

    for symbol in all_symbols:
        cur_w = current_weights.get(symbol, 0.0)
        tgt_w = target_weights.get(symbol, 0.0)
        delta_w = tgt_w - cur_w

        if abs(delta_w) < 1e-8:
            continue

        price = price_map.get(symbol, 0.0)
        if price <= 0:
            logger.warning("No price for %s — skipping order.", symbol)
            continue

        # Convert weight delta to dollar value, then to shares
        trade_value = abs(delta_w) * port_value
        if trade_value < min_trade_value:
            continue

        shares = trade_value / price
        if round_lots:
            shares = _round_to_lot(shares)
            if shares == 0:
                continue

        side = "BUY" if delta_w > 0 else "SELL"

        # Compute limit price if applicable
        limit_price = _compute_limit_price(order_type, price, side, limit_offset_pct)

        # Estimate transaction cost (half-spread model)
        estimated_cost = shares * price * (spread_bps / 2.0) / 10_000.0

        holding = portfolio.get_holding(symbol)
        account = holding.account if holding else ""

        orders.append(TradeOrder(
            symbol=symbol,
            side=side,
            quantity=shares,
            order_type=order_type,
            limit_price=limit_price,
            account=account,
            estimated_cost=estimated_cost,
            created_at=now,
        ))

    # Sort: sells first (to free cash), then buys; within each group, by value descending
    orders.sort(key=lambda o: (o.side != "SELL", -(o.quantity * price_map.get(o.symbol, 0))))

    return _build_batch(orders, port_value, price_map, now)


def generate_rebalance_orders(
    portfolio: Portfolio,
    optimization_result: OptimizationResult,
    prices: dict[str, float] | None = None,
    **kwargs,
) -> OrderBatch:
    """Convenience wrapper: extract target weights from an OptimizationResult.

    Args:
        portfolio: Current portfolio.
        optimization_result: Result containing target weights.
        prices: Optional price overrides.
        **kwargs: Forwarded to generate_orders (order_type, min_trade_value, etc.).

    Returns:
        OrderBatch with the computed trade orders.
    """
    return generate_orders(
        portfolio,
        optimization_result.weights,
        prices=prices,
        **kwargs,
    )


def prioritize_orders(
    batch: OrderBatch,
    available_cash: float | None = None,
) -> OrderBatch:
    """Prioritize and trim orders given a cash constraint.

    Sells are always kept in full. Buys are prioritized by absolute trade
    value (largest first) and trimmed if total buy value exceeds available
    cash plus sell proceeds.

    Args:
        batch: Original order batch.
        available_cash: Starting cash available. If None, assumes zero.

    Returns:
        A new OrderBatch with potentially reduced buy quantities.
    """
    if available_cash is None:
        available_cash = 0.0

    sells = [o for o in batch.orders if o.side == "SELL"]
    buys = [o for o in batch.orders if o.side == "BUY"]

    sell_proceeds = sum(
        o.quantity * (o.limit_price or 0.0) if o.limit_price else o.quantity * 0.0
        for o in sells
    )
    # Re-estimate sell proceeds from batch-level data when limit prices are unavailable
    sell_proceeds = max(sell_proceeds, batch.total_sell_value)
    budget = available_cash + sell_proceeds

    # Sort buys by dollar value descending (biggest deviations first)
    buys.sort(key=lambda o: o.quantity * (o.limit_price or 1.0), reverse=True)

    trimmed_buys: list[TradeOrder] = []
    spent = 0.0

    for order in buys:
        price_est = order.limit_price if order.limit_price else 1.0
        order_value = order.quantity * price_est
        remaining = budget - spent

        if remaining <= 0:
            break

        if order_value <= remaining:
            trimmed_buys.append(order)
            spent += order_value
        else:
            # Partial fill: reduce quantity to fit budget
            affordable_qty = remaining / price_est if price_est > 0 else 0.0
            if affordable_qty > 0:
                trimmed_order = TradeOrder(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=affordable_qty,
                    order_type=order.order_type,
                    limit_price=order.limit_price,
                    stop_price=order.stop_price,
                    account=order.account,
                    estimated_cost=order.estimated_cost * (affordable_qty / order.quantity),
                    notes=f"Trimmed from {order.quantity:.2f} shares (cash-constrained)",
                    created_at=order.created_at,
                )
                trimmed_buys.append(trimmed_order)
            break

    final_orders = sells + trimmed_buys
    return _build_batch(final_orders, batch.portfolio_value, {}, batch.created_at)


def orders_to_dicts(
    batch: OrderBatch,
    target_weights: dict[str, float] | None = None,
    prices: dict[str, float] | None = None,
) -> list[dict]:
    """Convert an OrderBatch to dicts for the TradeBlotterPanel.

    Each dict contains: date, symbol, side, quantity, price, cost, weight_after.

    Args:
        batch: The order batch to convert.
        target_weights: Optional target weights to populate weight_after.
        prices: Optional {symbol: price} for market-order price display.

    Returns:
        List of trade dicts compatible with TradeBlotterPanel.set_trades().
    """
    if target_weights is None:
        target_weights = {}
    if prices is None:
        prices = {}

    result = []
    for order in batch.orders:
        price = order.limit_price or order.filled_avg_price or prices.get(order.symbol, 0.0)
        result.append({
            "date": order.created_at.strftime("%Y-%m-%d") if order.created_at else "",
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "price": price,
            "cost": order.estimated_cost,
            "weight_after": target_weights.get(order.symbol, 0.0),
        })
    return result


def format_orders_text(batch: OrderBatch) -> str:
    """Format an OrderBatch as a human-readable text table.

    Args:
        batch: The order batch to format.

    Returns:
        Multi-line string suitable for console display or text export.
    """
    lines: list[str] = []
    lines.append(f"Order Batch  |  Portfolio: ${batch.portfolio_value:,.2f}")
    lines.append(f"Generated:     {batch.created_at:%Y-%m-%d %H:%M:%S}")
    lines.append("-" * 78)
    lines.append(
        f"{'Side':<6} {'Symbol':<8} {'Qty':>10} {'Type':<12} "
        f"{'Limit':>10} {'Est Cost':>10} {'Status':<10}"
    )
    lines.append("-" * 78)

    for order in batch.orders:
        limit_str = f"${order.limit_price:,.2f}" if order.limit_price else "MKT"
        lines.append(
            f"{order.side:<6} {order.symbol:<8} {order.quantity:>10,.2f} "
            f"{order.order_type.value:<12} {limit_str:>10} "
            f"${order.estimated_cost:>9,.4f} {order.status.value:<10}"
        )

    lines.append("-" * 78)
    lines.append(
        f"Total Buy:  ${batch.total_buy_value:>12,.2f}   "
        f"Total Sell: ${batch.total_sell_value:>12,.2f}"
    )
    lines.append(
        f"Est. Costs: ${batch.total_estimated_cost:>12,.4f}   "
        f"Net Flow:   ${batch.net_cash_flow:>12,.2f}"
    )
    lines.append(f"Turnover:   {batch.turnover:>12.2%}")
    return "\n".join(lines)


# ── Private helpers ──────────────────────────────────────────────────


def _build_price_map(
    portfolio: Portfolio,
    prices: dict[str, float] | None,
) -> dict[str, float]:
    """Merge explicit prices with holding current prices."""
    price_map: dict[str, float] = {}
    for holding in portfolio.holdings:
        if holding.current_price > 0:
            price_map[holding.asset.symbol] = holding.current_price
    if prices:
        price_map.update(prices)
    return price_map


def _round_to_lot(shares: float) -> float:
    """Round share count to nearest lot (100 shares)."""
    return float(round(shares / LOT_SIZE) * LOT_SIZE)


def _compute_limit_price(
    order_type: OrderType,
    market_price: float,
    side: str,
    offset_pct: float,
) -> float | None:
    """Compute limit price for LIMIT / STOP_LIMIT orders."""
    if order_type not in (OrderType.LIMIT, OrderType.STOP_LIMIT):
        return None
    if side == "BUY":
        return market_price * (1.0 + offset_pct / 100.0)
    return market_price * (1.0 - offset_pct / 100.0)


def _empty_batch(portfolio_value: float, created_at: datetime) -> OrderBatch:
    """Return an empty OrderBatch."""
    return OrderBatch(
        orders=[],
        portfolio_value=portfolio_value,
        total_buy_value=0.0,
        total_sell_value=0.0,
        total_estimated_cost=0.0,
        net_cash_flow=0.0,
        turnover=0.0,
        created_at=created_at,
    )


def _build_batch(
    orders: list[TradeOrder],
    portfolio_value: float,
    price_map: dict[str, float],
    created_at: datetime,
) -> OrderBatch:
    """Compute summary statistics and wrap orders into an OrderBatch."""
    total_buy = 0.0
    total_sell = 0.0
    total_cost = 0.0

    for order in orders:
        price = price_map.get(order.symbol, 0.0)
        # Fall back to limit price if price_map is empty (e.g. from prioritize_orders)
        if price <= 0 and order.limit_price:
            price = order.limit_price
        value = order.quantity * price
        if order.side == "BUY":
            total_buy += value
        else:
            total_sell += value
        total_cost += order.estimated_cost

    total_traded = total_buy + total_sell
    turnover = total_traded / portfolio_value if portfolio_value > 0 else 0.0
    net_cash = total_sell - total_buy

    return OrderBatch(
        orders=orders,
        portfolio_value=portfolio_value,
        total_buy_value=total_buy,
        total_sell_value=total_sell,
        total_estimated_cost=total_cost,
        net_cash_flow=net_cash,
        turnover=turnover,
        created_at=created_at,
    )
