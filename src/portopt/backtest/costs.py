"""Transaction cost models for backtesting.

Models:
- ZeroCost: No transaction costs
- FixedCost: Fixed cost per trade
- ProportionalCost: Percentage of trade value
- TieredCost: Percentage varies by trade size
- SpreadCost: Bid-ask spread model
- CompositeCost: Combine multiple cost models
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class BaseCostModel(ABC):
    """Abstract base for transaction cost models."""

    @abstractmethod
    def compute_cost(
        self,
        symbol: str,
        weight_change: float,
        portfolio_value: float,
        price: float = 0.0,
    ) -> float:
        """Compute the transaction cost for a single trade.

        Args:
            symbol: Asset ticker.
            weight_change: Change in portfolio weight (signed).
            portfolio_value: Total portfolio value.
            price: Current asset price (for spread/share-based models).

        Returns:
            Cost in currency units (always non-negative).
        """
        ...

    def compute_total_cost(
        self,
        weight_changes: dict[str, float],
        portfolio_value: float,
        prices: dict[str, float] | None = None,
    ) -> float:
        """Compute total cost across all trades in a rebalance.

        Args:
            weight_changes: {symbol: weight_change} for each asset traded.
            portfolio_value: Total portfolio value.
            prices: {symbol: price} for spread-based models.

        Returns:
            Total cost in currency units.
        """
        prices = prices or {}
        total = 0.0
        for symbol, dw in weight_changes.items():
            if abs(dw) < 1e-10:
                continue
            price = prices.get(symbol, 0.0)
            total += self.compute_cost(symbol, dw, portfolio_value, price)
        return total


class ZeroCost(BaseCostModel):
    """No transaction costs."""

    def compute_cost(self, symbol, weight_change, portfolio_value, price=0.0):
        return 0.0


@dataclass
class FixedCost(BaseCostModel):
    """Fixed cost per trade (e.g. $5 per trade).

    Attributes:
        cost_per_trade: Fixed dollar amount per trade.
    """
    cost_per_trade: float = 5.0

    def compute_cost(self, symbol, weight_change, portfolio_value, price=0.0):
        if abs(weight_change) < 1e-10:
            return 0.0
        return self.cost_per_trade


@dataclass
class ProportionalCost(BaseCostModel):
    """Proportional cost: percentage of trade notional value.

    Attributes:
        rate: Cost as fraction of trade value (e.g. 0.001 = 10 bps).
    """
    rate: float = 0.001  # 10 basis points

    def compute_cost(self, symbol, weight_change, portfolio_value, price=0.0):
        trade_value = abs(weight_change) * portfolio_value
        return trade_value * self.rate


@dataclass
class TieredCost(BaseCostModel):
    """Tiered cost: rate depends on trade size.

    Tiers are defined as (max_value, rate) pairs sorted ascending.
    Trade value below first tier max uses first rate, etc.

    Attributes:
        tiers: List of (max_trade_value, rate) tuples, sorted ascending.
               Last tier should have max_value = inf.
    """
    tiers: list[tuple[float, float]] = field(default_factory=lambda: [
        (10_000, 0.002),     # 20 bps for trades < $10k
        (100_000, 0.001),    # 10 bps for trades $10k-$100k
        (float("inf"), 0.0005),  # 5 bps for trades > $100k
    ])

    def compute_cost(self, symbol, weight_change, portfolio_value, price=0.0):
        trade_value = abs(weight_change) * portfolio_value
        for max_val, rate in sorted(self.tiers, key=lambda t: t[0]):
            if trade_value <= max_val:
                return trade_value * rate
        # Fallback: use last tier
        return trade_value * self.tiers[-1][1]


@dataclass
class SpreadCost(BaseCostModel):
    """Bid-ask spread model: cost = half-spread * trade value.

    Attributes:
        spread_bps: Half bid-ask spread in basis points (default 5 bps = 0.05%).
        spread_overrides: Per-symbol spread overrides in bps.
    """
    spread_bps: float = 5.0
    spread_overrides: dict[str, float] = field(default_factory=dict)

    def compute_cost(self, symbol, weight_change, portfolio_value, price=0.0):
        bps = self.spread_overrides.get(symbol, self.spread_bps)
        trade_value = abs(weight_change) * portfolio_value
        return trade_value * bps / 10_000


@dataclass
class CompositeCost(BaseCostModel):
    """Combine multiple cost models (costs are additive).

    Attributes:
        models: List of cost models to combine.
    """
    models: list[BaseCostModel] = field(default_factory=list)

    def compute_cost(self, symbol, weight_change, portfolio_value, price=0.0):
        return sum(
            m.compute_cost(symbol, weight_change, portfolio_value, price)
            for m in self.models
        )


def create_cost_model(
    model_type: str,
    **kwargs,
) -> BaseCostModel:
    """Factory function to create a cost model from string type.

    Args:
        model_type: One of "zero", "fixed", "proportional", "tiered", "spread", "composite".
        **kwargs: Arguments passed to the cost model constructor.
    """
    models = {
        "zero": ZeroCost,
        "fixed": FixedCost,
        "proportional": ProportionalCost,
        "tiered": TieredCost,
        "spread": SpreadCost,
        "composite": CompositeCost,
    }
    cls = models.get(model_type.lower())
    if cls is None:
        raise ValueError(f"Unknown cost model: {model_type}. Options: {list(models.keys())}")
    return cls(**kwargs)
