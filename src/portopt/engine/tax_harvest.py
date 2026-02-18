"""Tax-loss harvesting analysis.

Identifies portfolio holdings with unrealized losses that can be harvested
to offset capital gains, suggests correlated replacement securities to
maintain market exposure, and estimates the tax alpha from reinvesting
the resulting tax savings.

This is a pure computation module -- zero GUI imports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Data Models ──────────────────────────────────────────────────────────


@dataclass
class HarvestCandidate:
    """A single holding eligible for tax-loss harvesting."""

    symbol: str
    quantity: float
    cost_basis: float
    market_value: float
    unrealized_loss: float  # Negative = loss (harvestable)
    loss_pct: float  # As percentage (negative)
    tax_savings: float  # At given tax rate (positive number)
    wash_sale_risk: bool  # True if correlated replacement exists


@dataclass
class HarvestRecommendation:
    """Complete tax-loss harvesting recommendation for a portfolio."""

    candidates: list[HarvestCandidate]  # Sorted by tax savings (largest first)
    total_harvestable_loss: float  # Sum of all unrealized losses (negative number)
    total_tax_savings: float  # Sum of all tax savings (positive number)
    replacement_suggestions: dict[str, list[str]]  # {symbol: [correlated alternatives]}
    tax_rate: float


# ── Functions ────────────────────────────────────────────────────────────


def identify_harvest_candidates(
    holdings: list,
    tax_rate: float = 0.35,
) -> list[HarvestCandidate]:
    """Identify holdings with unrealized losses suitable for tax-loss harvesting.

    Parameters
    ----------
    holdings : list[Holding]
        Portfolio holdings to scan for harvestable losses.
    tax_rate : float
        Combined federal + state tax rate applied to capital gains.
        Defaults to 0.35 (35%).

    Returns
    -------
    list[HarvestCandidate]
        Candidates sorted by tax_savings descending (largest savings first).
    """
    if not holdings:
        return []

    candidates: list[HarvestCandidate] = []

    for holding in holdings:
        pnl = holding.unrealized_pnl

        # Only consider positions with unrealized losses
        if pnl >= 0:
            continue

        cost_basis = holding.cost_basis
        unrealized_loss = pnl  # Negative number
        loss_pct = (unrealized_loss / cost_basis) * 100 if cost_basis > 0 else 0.0
        tax_savings = abs(unrealized_loss) * tax_rate

        candidates.append(
            HarvestCandidate(
                symbol=holding.asset.symbol,
                quantity=holding.quantity,
                cost_basis=cost_basis,
                market_value=holding.market_value,
                unrealized_loss=unrealized_loss,
                loss_pct=loss_pct,
                tax_savings=tax_savings,
                wash_sale_risk=False,  # Set later by suggest_replacements
            )
        )

    # Sort by tax_savings descending (largest savings first)
    candidates.sort(key=lambda c: c.tax_savings, reverse=True)

    logger.debug(
        "Identified %d harvest candidates from %d holdings",
        len(candidates),
        len(holdings),
    )
    return candidates


def suggest_replacements(
    symbol: str,
    prices: pd.DataFrame,
    top_n: int = 3,
) -> list[str]:
    """Suggest correlated replacement securities for a harvested position.

    Finds securities whose returns are highly correlated (> 0.8) with
    the given symbol, which can serve as replacement positions to maintain
    similar market exposure while avoiding wash-sale rule violations
    (since they are different securities).

    Parameters
    ----------
    symbol : str
        Ticker symbol of the position being harvested.
    prices : pd.DataFrame
        Price history DataFrame with symbols as columns and dates as index.
    top_n : int
        Maximum number of replacement suggestions to return.

    Returns
    -------
    list[str]
        Up to ``top_n`` most correlated symbols (correlation > 0.8),
        sorted by correlation descending.
    """
    if symbol not in prices.columns:
        logger.debug("Symbol %s not found in prices DataFrame", symbol)
        return []

    if prices.shape[1] < 2:
        return []

    # Compute returns
    returns = prices.pct_change().dropna()

    if returns.empty or symbol not in returns.columns:
        return []

    # Correlation of this symbol with all others
    target_returns = returns[symbol]
    correlations = returns.corr()[symbol].drop(symbol, errors="ignore")

    # Filter for correlation > 0.8
    high_corr = correlations[correlations > 0.8]

    if high_corr.empty:
        return []

    # Sort by correlation descending and take top_n
    high_corr = high_corr.sort_values(ascending=False)
    replacements = high_corr.head(top_n).index.tolist()

    logger.debug(
        "Found %d replacement suggestions for %s (top corr: %.3f)",
        len(replacements),
        symbol,
        high_corr.iloc[0] if len(high_corr) > 0 else 0.0,
    )
    return replacements


def compute_harvest_recommendation(
    holdings: list,
    prices: pd.DataFrame | None = None,
    tax_rate: float = 0.35,
) -> HarvestRecommendation:
    """Compute a complete tax-loss harvesting recommendation.

    Full pipeline: identifies harvest candidates, suggests correlated
    replacements, flags wash-sale risks, and computes aggregate totals.

    Parameters
    ----------
    holdings : list[Holding]
        Portfolio holdings to analyze.
    prices : pd.DataFrame | None
        Price history for replacement suggestions. If None, no
        replacements are suggested and wash_sale_risk remains False.
    tax_rate : float
        Combined federal + state tax rate. Defaults to 0.35 (35%).

    Returns
    -------
    HarvestRecommendation
        Complete recommendation with candidates, totals, and replacements.
    """
    candidates = identify_harvest_candidates(holdings, tax_rate=tax_rate)

    replacement_suggestions: dict[str, list[str]] = {}

    if prices is not None and not prices.empty:
        for candidate in candidates:
            replacements = suggest_replacements(candidate.symbol, prices)
            if replacements:
                replacement_suggestions[candidate.symbol] = replacements
                candidate.wash_sale_risk = True

    total_harvestable_loss = sum(c.unrealized_loss for c in candidates)
    total_tax_savings = sum(c.tax_savings for c in candidates)

    recommendation = HarvestRecommendation(
        candidates=candidates,
        total_harvestable_loss=total_harvestable_loss,
        total_tax_savings=total_tax_savings,
        replacement_suggestions=replacement_suggestions,
        tax_rate=tax_rate,
    )

    logger.info(
        "Harvest recommendation: %d candidates, total loss $%.2f, "
        "tax savings $%.2f",
        len(candidates),
        total_harvestable_loss,
        total_tax_savings,
    )
    return recommendation


def estimate_tax_alpha(
    harvested_losses: float,
    tax_rate: float,
    reinvestment_return: float = 0.08,
) -> float:
    """Estimate the annual return benefit from reinvesting tax savings.

    Tax alpha represents the additional portfolio return generated by
    harvesting losses, receiving a tax refund (or offset), and reinvesting
    that capital at the expected market return.

    Parameters
    ----------
    harvested_losses : float
        Total harvested losses (should be negative). If positive, returns 0.0.
    tax_rate : float
        Combined federal + state tax rate.
    reinvestment_return : float
        Expected annual return on reinvested tax savings. Defaults to 0.08 (8%).

    Returns
    -------
    float
        Estimated annual tax alpha (positive number).
    """
    if harvested_losses >= 0:
        return 0.0

    tax_alpha = abs(harvested_losses) * tax_rate * reinvestment_return

    logger.debug(
        "Tax alpha estimate: $%.2f (losses=$%.2f, rate=%.1f%%, return=%.1f%%)",
        tax_alpha,
        harvested_losses,
        tax_rate * 100,
        reinvestment_return * 100,
    )
    return tax_alpha
