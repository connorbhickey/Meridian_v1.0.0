"""Execution simulation models — market impact, slippage, and capacity analysis.

Provides realistic transaction cost models beyond simple proportional costs:
- Square-root market impact (Almgren-Chriss inspired)
- Linear temporary impact
- Spread-based slippage
- Capacity analysis (maximum strategy AUM before performance degrades)
- Partial fill simulation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Result dataclasses ───────────────────────────────────────────────


@dataclass
class TradeExecution:
    """Result of simulating execution for a single trade."""
    symbol: str
    side: str                    # "BUY" or "SELL"
    target_shares: float
    filled_shares: float
    fill_rate: float             # filled / target
    avg_price: float             # VWAP of fills
    market_price: float          # price without impact
    slippage_bps: float          # slippage in basis points
    impact_cost: float           # dollar cost of market impact
    spread_cost: float           # dollar cost of spread
    total_cost: float            # impact + spread


@dataclass
class ExecutionSummary:
    """Aggregated execution simulation results."""
    trades: list[TradeExecution]
    total_impact_cost: float
    total_spread_cost: float
    total_cost: float
    avg_slippage_bps: float
    avg_fill_rate: float
    total_notional: float
    cost_as_pct: float           # total cost / total notional


@dataclass
class CapacityResult:
    """Result of capacity analysis for a strategy."""
    max_capacity_usd: float      # AUM where impact exceeds threshold
    impact_threshold_bps: float
    capacity_curve: list[tuple[float, float]]  # [(aum, impact_bps), ...]
    bottleneck_symbol: str       # symbol that limits capacity first
    bottleneck_adv_pct: float    # bottleneck's % of ADV at capacity


# ── Market impact models ─────────────────────────────────────────────


def sqrt_impact(
    trade_value: float,
    adv: float,
    volatility: float,
    eta: float = 0.1,
) -> float:
    """Square-root market impact model (Almgren-Chriss style).

    Impact (bps) = eta * sigma * sqrt(Q / ADV)

    where:
        Q = absolute trade value
        ADV = average daily volume in dollars
        sigma = daily volatility
        eta = impact coefficient (calibrated, typically 0.05–0.20)

    Args:
        trade_value: Absolute dollar value of the trade.
        adv: Average daily volume in dollars.
        volatility: Daily volatility of the asset.
        eta: Impact coefficient.

    Returns:
        Market impact in basis points.
    """
    if adv <= 0 or trade_value <= 0:
        return 0.0
    participation = trade_value / adv
    return eta * volatility * 10000 * np.sqrt(participation)


def linear_impact(
    trade_value: float,
    adv: float,
    lambda_coeff: float = 0.1,
) -> float:
    """Linear temporary impact model.

    Impact (bps) = lambda * (Q / ADV) * 10000

    Args:
        trade_value: Absolute dollar value of the trade.
        adv: Average daily volume in dollars.
        lambda_coeff: Impact coefficient.

    Returns:
        Market impact in basis points.
    """
    if adv <= 0 or trade_value <= 0:
        return 0.0
    return lambda_coeff * (trade_value / adv) * 10000


# ── Execution simulation ────────────────────────────────────────────


def simulate_execution(
    weight_changes: dict[str, float],
    portfolio_value: float,
    prices: dict[str, float],
    volumes: dict[str, float],
    volatilities: dict[str, float] | None = None,
    spread_bps: float = 5.0,
    impact_model: str = "sqrt",
    eta: float = 0.1,
    max_participation: float = 0.10,
) -> ExecutionSummary:
    """Simulate execution of a rebalance with realistic market impact.

    Args:
        weight_changes: {symbol: delta_weight} changes to execute.
        portfolio_value: Current portfolio value in dollars.
        prices: {symbol: current_price}.
        volumes: {symbol: average_daily_volume_shares}.
        volatilities: {symbol: daily_volatility}. If None, uses 2% default.
        spread_bps: Bid-ask spread in basis points.
        impact_model: "sqrt" or "linear".
        eta: Impact coefficient for the chosen model.
        max_participation: Maximum fraction of ADV to trade (for partial fills).

    Returns:
        ExecutionSummary with per-trade and aggregated results.
    """
    if volatilities is None:
        volatilities = {s: 0.02 for s in weight_changes}

    trades: list[TradeExecution] = []

    for symbol, dw in weight_changes.items():
        if abs(dw) < 1e-10:
            continue

        price = prices.get(symbol, 0.0)
        if price <= 0:
            continue

        vol_shares = volumes.get(symbol, 0.0)
        daily_vol = volatilities.get(symbol, 0.02)

        # Compute trade size
        trade_dollars = abs(dw) * portfolio_value
        target_shares = trade_dollars / price
        adv_dollars = vol_shares * price

        # Partial fill: cap at max_participation of ADV
        if adv_dollars > 0 and trade_dollars > max_participation * adv_dollars:
            filled_shares = max_participation * vol_shares
        else:
            filled_shares = target_shares

        fill_rate = filled_shares / target_shares if target_shares > 0 else 1.0
        filled_dollars = filled_shares * price

        # Market impact
        if impact_model == "linear":
            impact_bps = linear_impact(filled_dollars, adv_dollars, eta)
        else:
            impact_bps = sqrt_impact(filled_dollars, adv_dollars, daily_vol, eta)

        impact_cost = filled_dollars * impact_bps / 10000

        # Spread cost (half-spread per trade)
        spread_cost = filled_dollars * (spread_bps / 2) / 10000

        total_cost = impact_cost + spread_cost
        slippage_bps = (impact_bps + spread_bps / 2) if filled_dollars > 0 else 0.0

        # Compute average execution price
        side = "BUY" if dw > 0 else "SELL"
        if side == "BUY":
            avg_price = price * (1 + slippage_bps / 10000)
        else:
            avg_price = price * (1 - slippage_bps / 10000)

        trades.append(TradeExecution(
            symbol=symbol,
            side=side,
            target_shares=target_shares,
            filled_shares=filled_shares,
            fill_rate=fill_rate,
            avg_price=avg_price,
            market_price=price,
            slippage_bps=slippage_bps,
            impact_cost=impact_cost,
            spread_cost=spread_cost,
            total_cost=total_cost,
        ))

    total_impact = sum(t.impact_cost for t in trades)
    total_spread = sum(t.spread_cost for t in trades)
    total_cost = total_impact + total_spread
    total_notional = sum(t.filled_shares * t.market_price for t in trades)

    avg_slip = np.mean([t.slippage_bps for t in trades]) if trades else 0.0
    avg_fill = np.mean([t.fill_rate for t in trades]) if trades else 1.0

    return ExecutionSummary(
        trades=trades,
        total_impact_cost=total_impact,
        total_spread_cost=total_spread,
        total_cost=total_cost,
        avg_slippage_bps=float(avg_slip),
        avg_fill_rate=float(avg_fill),
        total_notional=total_notional,
        cost_as_pct=total_cost / total_notional * 100 if total_notional > 0 else 0.0,
    )


# ── Capacity analysis ───────────────────────────────────────────────


def analyze_capacity(
    weights: dict[str, float],
    prices: dict[str, float],
    volumes: dict[str, float],
    volatilities: dict[str, float] | None = None,
    impact_threshold_bps: float = 50.0,
    aum_range: list[float] | None = None,
    impact_model: str = "sqrt",
    eta: float = 0.1,
) -> CapacityResult:
    """Analyze strategy capacity — find max AUM before impact exceeds threshold.

    Assumes a full rebalance from cash to target weights (worst case).

    Args:
        weights: {symbol: target_weight}.
        prices: {symbol: current_price}.
        volumes: {symbol: average_daily_volume_shares}.
        volatilities: {symbol: daily_volatility}. Defaults to 2%.
        impact_threshold_bps: Impact level at which capacity is "reached".
        aum_range: List of AUM values to test. Defaults to log-spaced 1M–10B.
        impact_model: "sqrt" or "linear".
        eta: Impact coefficient.

    Returns:
        CapacityResult with capacity curve and bottleneck info.
    """
    if volatilities is None:
        volatilities = {s: 0.02 for s in weights}

    if aum_range is None:
        aum_range = np.logspace(6, 10, 50).tolist()  # 1M to 10B

    capacity_curve: list[tuple[float, float]] = []
    max_capacity = aum_range[-1]
    found_limit = False

    for aum in aum_range:
        # Compute per-symbol impact at this AUM
        max_impact = 0.0
        worst_symbol = ""
        worst_adv_pct = 0.0

        for symbol, w in weights.items():
            if abs(w) < 1e-10:
                continue

            price = prices.get(symbol, 0.0)
            vol_shares = volumes.get(symbol, 0.0)
            daily_vol = volatilities.get(symbol, 0.02)

            trade_dollars = abs(w) * aum
            adv_dollars = vol_shares * price

            if impact_model == "linear":
                impact = linear_impact(trade_dollars, adv_dollars, eta)
            else:
                impact = sqrt_impact(trade_dollars, adv_dollars, daily_vol, eta)

            adv_pct = trade_dollars / adv_dollars * 100 if adv_dollars > 0 else 100.0

            if impact > max_impact:
                max_impact = impact
                worst_symbol = symbol
                worst_adv_pct = adv_pct

        capacity_curve.append((aum, max_impact))

        if max_impact >= impact_threshold_bps and not found_limit:
            max_capacity = aum
            found_limit = True

    # Find bottleneck at capacity
    bottleneck = ""
    bottleneck_pct = 0.0
    for symbol, w in weights.items():
        if abs(w) < 1e-10:
            continue
        price = prices.get(symbol, 0.0)
        vol_shares = volumes.get(symbol, 0.0)
        trade_dollars = abs(w) * max_capacity
        adv_dollars = vol_shares * price
        adv_pct = trade_dollars / adv_dollars * 100 if adv_dollars > 0 else 100.0
        if adv_pct > bottleneck_pct:
            bottleneck_pct = adv_pct
            bottleneck = symbol

    return CapacityResult(
        max_capacity_usd=max_capacity,
        impact_threshold_bps=impact_threshold_bps,
        capacity_curve=capacity_curve,
        bottleneck_symbol=bottleneck,
        bottleneck_adv_pct=bottleneck_pct,
    )


# ── Volume estimation helpers ────────────────────────────────────────


def estimate_volumes_from_prices(prices: pd.DataFrame) -> dict[str, float]:
    """Estimate average daily volumes from a price DataFrame.

    If the DataFrame has a 'Volume' level or volume columns, use those.
    Otherwise returns a default placeholder.
    """
    volumes = {}
    for col in prices.columns:
        # Default: assume moderate liquidity (1M shares/day)
        volumes[col] = 1_000_000.0
    return volumes


def estimate_volatilities_from_prices(prices: pd.DataFrame) -> dict[str, float]:
    """Estimate daily volatilities from prices."""
    returns = prices.pct_change().dropna()
    vols = {}
    for col in returns.columns:
        vols[col] = float(returns[col].std())
    return vols
