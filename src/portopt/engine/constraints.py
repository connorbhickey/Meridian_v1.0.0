"""Portfolio weight constraints for optimization."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PortfolioConstraints:
    """Constraints applied during portfolio optimization.

    Attributes:
        min_weight: Minimum weight per asset (0 = long-only, negative = allow short).
        max_weight: Maximum weight per asset (1 = no concentration limit).
        weight_bounds: Per-asset (min, max) bounds. Overrides min/max_weight for specific assets.
        sector_constraints: Sector name -> (min_weight, max_weight) for sector allocation.
        max_turnover: Maximum total turnover from current weights (None = unconstrained).
        long_only: If True, enforces min_weight >= 0 (overrides min_weight).
        max_short: Maximum total short exposure (absolute).
        target_return: Target return for efficient return optimization.
        target_risk: Target risk for efficient risk optimization.
        risk_aversion: Risk aversion parameter for quadratic utility.
        market_neutral: If True, weights must sum to 0.
        leverage: Target leverage (weights sum to this value, default=1).
    """
    min_weight: float = 0.0
    max_weight: float = 1.0
    weight_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    sector_constraints: dict[str, tuple[float, float]] = field(default_factory=dict)
    sector_map: dict[str, str] = field(default_factory=dict)  # symbol -> sector
    max_turnover: float | None = None
    current_weights: dict[str, float] = field(default_factory=dict)
    long_only: bool = True
    max_short: float = 0.0
    target_return: float | None = None
    target_risk: float | None = None
    risk_aversion: float = 1.0
    market_neutral: bool = False
    leverage: float = 1.0

    def get_bounds(self, symbol: str) -> tuple[float, float]:
        """Get weight bounds for a specific asset."""
        if symbol in self.weight_bounds:
            return self.weight_bounds[symbol]
        lo = max(self.min_weight, 0.0) if self.long_only else self.min_weight
        return (lo, self.max_weight)

    def get_all_bounds(self, symbols: list[str]) -> list[tuple[float, float]]:
        """Get bounds for all symbols as a list of (min, max) tuples."""
        return [self.get_bounds(s) for s in symbols]

    def validate(self, weights: dict[str, float]) -> list[str]:
        """Validate weights against constraints. Returns list of violation messages."""
        violations = []
        w = np.array(list(weights.values()))

        # Sum constraint
        total = np.sum(w)
        if self.market_neutral:
            if abs(total) > 1e-6:
                violations.append(f"Market neutral violated: sum={total:.6f}")
        else:
            if abs(total - self.leverage) > 1e-4:
                violations.append(f"Leverage violated: sum={total:.4f}, target={self.leverage}")

        # Bounds
        for sym, wt in weights.items():
            lo, hi = self.get_bounds(sym)
            if wt < lo - 1e-6:
                violations.append(f"{sym}: weight {wt:.4f} < min {lo:.4f}")
            if wt > hi + 1e-6:
                violations.append(f"{sym}: weight {wt:.4f} > max {hi:.4f}")

        # Long-only
        if self.long_only and np.any(w < -1e-6):
            violations.append("Long-only violated: negative weights found")

        # Turnover
        if self.max_turnover is not None and self.current_weights:
            turnover = sum(
                abs(weights.get(s, 0) - self.current_weights.get(s, 0))
                for s in set(list(weights.keys()) + list(self.current_weights.keys()))
            )
            if turnover > self.max_turnover + 1e-6:
                violations.append(f"Turnover {turnover:.4f} > max {self.max_turnover:.4f}")

        # Sector constraints
        if self.sector_constraints and self.sector_map:
            sector_weights: dict[str, float] = {}
            for sym, wt in weights.items():
                sector = self.sector_map.get(sym, "Unknown")
                sector_weights[sector] = sector_weights.get(sector, 0) + wt
            for sector, (s_min, s_max) in self.sector_constraints.items():
                sw = sector_weights.get(sector, 0)
                if sw < s_min - 1e-6:
                    violations.append(f"Sector {sector}: weight {sw:.4f} < min {s_min:.4f}")
                if sw > s_max + 1e-6:
                    violations.append(f"Sector {sector}: weight {sw:.4f} > max {s_max:.4f}")

        return violations
