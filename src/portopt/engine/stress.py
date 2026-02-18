"""Stress testing engine — historical scenarios, custom shocks, and reverse stress tests.

Zero GUI knowledge. All inputs/outputs are plain Python/numpy objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ── Data Structures ─────────────────────────────────────────────────────

@dataclass
class StressScenario:
    """A single stress scenario with sector-level shocks."""
    name: str
    description: str
    shocks: dict[str, float]   # {sector_or_factor: return_shock}  e.g. {"equity": -0.40}
    duration_days: int = 1


@dataclass
class StressResult:
    """Output from a single stress test."""
    scenario: StressScenario
    portfolio_impact: float        # weighted-average return shock
    asset_impacts: dict[str, float]  # per-asset shock applied
    stressed_value: float
    initial_value: float


@dataclass
class ReverseStressResult:
    """Output from reverse stress testing."""
    target_drawdown: float
    worst_scenarios: list[dict]    # [{direction, impact, scaled_shocks}, ...]


# ── Default Sector Mapping ──────────────────────────────────────────────

DEFAULT_SECTOR_MAP: dict[str, str] = {
    # Broad market ETFs
    "SPY": "equity", "VOO": "equity", "IVV": "equity", "VTI": "equity",
    "QQQ": "tech", "TQQQ": "tech",
    "IWM": "equity", "IWF": "equity", "IWD": "equity",
    # Bond ETFs
    "AGG": "bond", "BND": "bond", "TLT": "bond", "IEF": "bond",
    "LQD": "bond", "HYG": "bond", "SHY": "bond",
    # Commodities
    "GLD": "gold", "IAU": "gold", "SLV": "gold",
    "USO": "energy", "XLE": "energy",
    # Sector ETFs
    "XLK": "tech", "XLF": "financials", "XLV": "healthcare",
    "XLY": "consumer", "XLP": "consumer", "XLI": "industrials",
    "XLB": "materials", "XLU": "utilities", "XLRE": "realestate",
}


# ── Historical Scenarios ────────────────────────────────────────────────

HISTORICAL_SCENARIOS: dict[str, StressScenario] = {
    "2008 GFC": StressScenario(
        name="2008 GFC",
        description="Global Financial Crisis — severe equity drawdown, flight to bonds/gold",
        shocks={"equity": -0.40, "tech": -0.45, "financials": -0.55,
                "bond": 0.05, "gold": 0.10, "energy": -0.35,
                "healthcare": -0.25, "consumer": -0.30,
                "industrials": -0.40, "materials": -0.35,
                "utilities": -0.20, "realestate": -0.40},
        duration_days=252,
    ),
    "COVID Crash": StressScenario(
        name="COVID Crash",
        description="COVID-19 pandemic sell-off — rapid equity decline, mixed bond/gold",
        shocks={"equity": -0.34, "tech": -0.25, "financials": -0.35,
                "bond": 0.03, "gold": -0.03, "energy": -0.50,
                "healthcare": -0.15, "consumer": -0.30,
                "industrials": -0.35, "materials": -0.25,
                "utilities": -0.20, "realestate": -0.25},
        duration_days=23,
    ),
    "2022 Rate Hikes": StressScenario(
        name="2022 Rate Hikes",
        description="Aggressive Fed tightening — bonds and equities both fall",
        shocks={"equity": -0.25, "tech": -0.33, "financials": -0.15,
                "bond": -0.15, "gold": -0.05, "energy": 0.30,
                "healthcare": -0.10, "consumer": -0.20,
                "industrials": -0.15, "materials": -0.10,
                "utilities": -0.05, "realestate": -0.25},
        duration_days=252,
    ),
    "Dot-Com Bust": StressScenario(
        name="Dot-Com Bust",
        description="Tech bubble burst — severe tech losses, defensive sectors resilient",
        shocks={"equity": -0.45, "tech": -0.70, "financials": -0.20,
                "bond": 0.08, "gold": -0.05, "energy": -0.10,
                "healthcare": -0.15, "consumer": -0.10,
                "industrials": -0.30, "materials": -0.20,
                "utilities": 0.05, "realestate": -0.10},
        duration_days=504,
    ),
    "Flash Crash": StressScenario(
        name="Flash Crash",
        description="2010-style intraday crash — brief but sharp equity sell-off",
        shocks={"equity": -0.09, "tech": -0.10, "financials": -0.08,
                "bond": 0.01, "gold": 0.02, "energy": -0.07,
                "healthcare": -0.06, "consumer": -0.07,
                "industrials": -0.08, "materials": -0.07,
                "utilities": -0.04, "realestate": -0.06},
        duration_days=1,
    ),
}


# ── Core Functions ──────────────────────────────────────────────────────

def _resolve_sector(symbol: str, sector_map: dict[str, str] | None) -> str:
    """Look up sector for a symbol. Falls back to 'equity'."""
    if sector_map and symbol in sector_map:
        return sector_map[symbol]
    if symbol in DEFAULT_SECTOR_MAP:
        return DEFAULT_SECTOR_MAP[symbol]
    return "equity"


def run_stress_test(
    weights: dict[str, float],
    scenario: StressScenario,
    initial_value: float = 100_000.0,
    sector_map: dict[str, str] | None = None,
    cov: pd.DataFrame | None = None,
) -> StressResult:
    """Apply a stress scenario to a portfolio.

    For each asset, looks up its sector and applies the corresponding shock.
    If cov is provided and the scenario has a contagion effect, correlation
    adjustments are applied.

    Parameters
    ----------
    weights : dict mapping symbol -> portfolio weight
    scenario : StressScenario with sector-level shocks
    initial_value : starting portfolio value
    sector_map : optional symbol -> sector mapping (supplements DEFAULT_SECTOR_MAP)
    cov : optional covariance matrix for correlation-based adjustments
    """
    asset_impacts: dict[str, float] = {}

    for symbol, weight in weights.items():
        sector = _resolve_sector(symbol, sector_map)
        # Apply the shock for this sector; default to equity shock if sector not in shocks
        shock = scenario.shocks.get(sector, scenario.shocks.get("equity", 0.0))
        asset_impacts[symbol] = shock

    # Portfolio-level impact is the weighted sum
    portfolio_impact = sum(
        weights[sym] * asset_impacts[sym] for sym in weights
    )

    # Optional: correlation-based contagion adjustment
    if cov is not None and len(weights) > 1:
        try:
            syms = [s for s in weights if s in cov.columns]
            if len(syms) > 1:
                sub_cov = cov.loc[syms, syms]
                corr = sub_cov.values / np.outer(
                    np.sqrt(np.diag(sub_cov.values)),
                    np.sqrt(np.diag(sub_cov.values)),
                )
                avg_corr = (corr.sum() - len(syms)) / (len(syms) * (len(syms) - 1))
                # Higher correlation → amplify the stress slightly
                contagion_factor = 1.0 + 0.1 * max(0, avg_corr - 0.3)
                portfolio_impact *= contagion_factor
        except Exception:
            pass  # Fail silently — correlation adjustment is optional

    stressed_value = initial_value * (1.0 + portfolio_impact)

    return StressResult(
        scenario=scenario,
        portfolio_impact=portfolio_impact,
        asset_impacts=asset_impacts,
        stressed_value=stressed_value,
        initial_value=initial_value,
    )


def run_all_stress_tests(
    weights: dict[str, float],
    scenarios: list[StressScenario] | None = None,
    initial_value: float = 100_000.0,
    sector_map: dict[str, str] | None = None,
    cov: pd.DataFrame | None = None,
) -> list[StressResult]:
    """Run multiple stress scenarios and return sorted results (worst first).

    If scenarios is None, runs all HISTORICAL_SCENARIOS.
    """
    if scenarios is None:
        scenarios = list(HISTORICAL_SCENARIOS.values())

    results = [
        run_stress_test(weights, s, initial_value, sector_map, cov)
        for s in scenarios
    ]
    # Sort by impact (most negative first)
    results.sort(key=lambda r: r.portfolio_impact)
    return results


def reverse_stress_test(
    weights: dict[str, float],
    target_drawdown: float,
    cov: pd.DataFrame,
    n_scenarios: int = 100,
    initial_value: float = 100_000.0,
) -> ReverseStressResult:
    """Find shock directions that produce a target drawdown level.

    Generates random return vectors from the covariance structure and scales
    them to approximate the target drawdown. Returns the worst scenarios
    sorted by severity.

    Parameters
    ----------
    weights : portfolio weights
    target_drawdown : target loss (negative, e.g. -0.20 for 20% loss)
    cov : covariance matrix
    n_scenarios : number of random directions to sample
    initial_value : starting portfolio value
    """
    symbols = [s for s in weights if s in cov.columns]
    if not symbols:
        return ReverseStressResult(
            target_drawdown=target_drawdown, worst_scenarios=[]
        )

    w = np.array([weights[s] for s in symbols])
    sub_cov = cov.loc[symbols, symbols].values

    # Generate random return vectors from N(0, cov)
    rng = np.random.default_rng(42)
    random_returns = rng.multivariate_normal(
        mean=np.zeros(len(symbols)), cov=sub_cov, size=n_scenarios
    )

    worst = []
    for i in range(n_scenarios):
        r = random_returns[i]
        port_return = w @ r

        if abs(port_return) < 1e-10:
            continue

        # Scale the return vector so portfolio impact matches target_drawdown
        scale = target_drawdown / port_return
        if scale < 0:
            # Flip direction — we want losses
            r = -r
            port_return = -port_return
            scale = target_drawdown / port_return

        scaled_r = r * scale
        actual_impact = w @ scaled_r

        worst.append({
            "direction": i,
            "impact": actual_impact,
            "scaled_shocks": {sym: float(scaled_r[j]) for j, sym in enumerate(symbols)},
        })

    # Sort by impact (most negative = worst)
    worst.sort(key=lambda x: x["impact"])
    # Keep top-N worst
    worst = worst[:min(10, len(worst))]

    return ReverseStressResult(
        target_drawdown=target_drawdown,
        worst_scenarios=worst,
    )
