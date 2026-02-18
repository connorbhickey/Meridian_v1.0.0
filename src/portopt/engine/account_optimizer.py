"""Multi-account household-level portfolio optimization with tax-aware asset location.

Supports per-account constraints, tax-efficient asset placement across account
types (taxable, tax-deferred, tax-exempt), and household-level weight aggregation.

This is an ENGINE module — zero GUI imports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from portopt.data.models import OptimizationResult
from portopt.engine.constraints import PortfolioConstraints

logger = logging.getLogger(__name__)


# ── Enums ────────────────────────────────────────────────────────────


class AccountType(Enum):
    TAXABLE = "taxable"
    TAX_DEFERRED = "tax_deferred"
    TAX_EXEMPT = "tax_exempt"


# ── Data Models ──────────────────────────────────────────────────────


@dataclass
class AccountConstraints:
    """Constraints and metadata for a single account within a household.

    Attributes:
        account_id: Unique identifier for this account.
        account_type: Tax treatment of the account.
        total_value: Current market value of the account.
        constraints: Per-account weight constraints for optimization.
        allowed_symbols: If set, only these symbols may be held. None means all allowed.
        excluded_symbols: Symbols that must not be held in this account.
        max_positions: Maximum number of distinct positions. None means unlimited.
    """

    account_id: str
    account_type: AccountType
    total_value: float
    constraints: PortfolioConstraints
    allowed_symbols: set[str] | None = None
    excluded_symbols: set[str] = field(default_factory=set)
    max_positions: int | None = None


@dataclass
class AssetTaxProfile:
    """Tax characteristics of an asset, used for optimal location decisions.

    Attributes:
        symbol: Ticker symbol.
        dividend_yield: Annual dividend yield as a decimal (e.g. 0.03 = 3%).
        qualified_dividend_pct: Fraction of dividends that qualify for lower tax rates.
        turnover_rate: Annual portfolio turnover for funds (0 for individual stocks).
        tax_cost_ratio: Estimated annual tax drag as a fraction of return.
    """

    symbol: str
    dividend_yield: float = 0.0
    qualified_dividend_pct: float = 1.0
    turnover_rate: float = 0.0
    tax_cost_ratio: float = 0.0


@dataclass
class HouseholdOptResult:
    """Result of a household-level multi-account optimization.

    Attributes:
        account_weights: Per-account weight allocations {account_id: {symbol: weight}}.
        household_weights: Aggregate household-level weights across all accounts.
        tax_savings_estimate: Estimated annual tax savings from optimal asset location.
        asset_locations: Recommended primary account for each symbol {symbol: account_id}.
        per_account_results: Individual OptimizationResult for each account.
    """

    account_weights: dict[str, dict[str, float]]
    household_weights: dict[str, float]
    tax_savings_estimate: float
    asset_locations: dict[str, str]
    per_account_results: dict[str, OptimizationResult]


# ── Default Tax Rates ────────────────────────────────────────────────

DEFAULT_TAX_RATES: dict[str, float] = {
    "ordinary_income": 0.35,
    "qualified_dividend": 0.15,
    "long_term_capital_gain": 0.15,
    "short_term_capital_gain": 0.35,
}


# ── Tax Efficiency Scoring ───────────────────────────────────────────


def compute_tax_efficiency_score(
    profile: AssetTaxProfile,
    account_type: AccountType,
) -> float:
    """Compute how tax-efficient placing an asset in a given account type is.

    Returns a score between 0.0 (poor fit) and 1.0 (ideal fit).

    Scoring logic:
    - Tax-deferred and tax-exempt accounts shelter all income equally,
      so tax-inefficient assets (high dividends, non-qualified income,
      high turnover) benefit most from placement there.
    - In taxable accounts, assets with low dividends, high qualified
      dividend ratios, and low turnover are most efficient.

    The inefficiency score is built from three normalized components:
    - Dividend burden: yield weighted by the non-qualified fraction
    - Turnover burden: how much short-term gain the fund generates
    - Explicit tax cost ratio (if provided)

    Args:
        profile: Tax characteristics of the asset.
        account_type: Type of account being considered.

    Returns:
        Float in [0, 1] where 1.0 means ideal placement.
    """
    # Build a 0-1 "tax inefficiency" measure from raw characteristics.
    # Dividend burden: high yield + low qualified pct = very inefficient
    # Scale: 5% non-qualified dividends maps to ~1.0
    dividend_burden = profile.dividend_yield * (1.0 - profile.qualified_dividend_pct) / 0.05
    # Qualified dividends are still taxed, just at lower rates
    dividend_burden += profile.dividend_yield * profile.qualified_dividend_pct / 0.10

    # Turnover burden: 50% annual turnover maps to ~1.0
    turnover_burden = profile.turnover_rate / 0.50

    # Explicit tax cost ratio: 2% maps to ~1.0
    explicit_burden = profile.tax_cost_ratio / 0.02 if profile.tax_cost_ratio > 0 else 0.0

    raw_inefficiency = max(dividend_burden, explicit_burden) + turnover_burden * 0.3
    # Clamp to [0, 1]
    inefficiency = float(np.clip(raw_inefficiency, 0.0, 1.0))

    if account_type in (AccountType.TAX_DEFERRED, AccountType.TAX_EXEMPT):
        # Tax-advantaged accounts shelter everything. The more tax-inefficient
        # the asset, the higher the benefit of placing it here.
        return 0.3 + 0.7 * inefficiency

    # Taxable account: penalize high tax drag
    return 1.0 - 0.8 * inefficiency


def _estimate_tax_drag(profile: AssetTaxProfile) -> float:
    """Estimate the annual tax drag of an asset as a decimal.

    Combines dividend tax cost, turnover-driven capital gains tax, and the
    explicit tax_cost_ratio into a single drag figure.

    Args:
        profile: Tax characteristics of the asset.

    Returns:
        Estimated annual tax drag as a decimal (e.g. 0.02 = 2%).
    """
    if profile.tax_cost_ratio > 0:
        return profile.tax_cost_ratio

    rates = DEFAULT_TAX_RATES
    qualified_tax = profile.dividend_yield * profile.qualified_dividend_pct * rates["qualified_dividend"]
    ordinary_tax = profile.dividend_yield * (1.0 - profile.qualified_dividend_pct) * rates["ordinary_income"]
    turnover_tax = profile.turnover_rate * 0.05 * rates["short_term_capital_gain"]

    return qualified_tax + ordinary_tax + turnover_tax


# ── Optimal Asset Location ───────────────────────────────────────────


def optimal_asset_location(
    symbols: list[str],
    tax_profiles: dict[str, AssetTaxProfile],
    accounts: list[AccountConstraints],
) -> dict[str, str]:
    """Determine optimal account placement for each asset.

    Algorithm:
        1. Score each (symbol, account_type) pair for tax efficiency.
        2. Sort assets by tax-inefficiency (most inefficient first).
        3. Greedily assign tax-inefficient assets to tax-advantaged accounts.
        4. Remaining assets go to the largest taxable account.

    Args:
        symbols: List of asset symbols to place.
        tax_profiles: Tax profile for each symbol. Missing symbols get a
            default (tax-neutral) profile.
        accounts: Available accounts with types and sizes.

    Returns:
        Mapping of {symbol: account_id} for recommended placement.
    """
    if not accounts:
        return {}

    # Build capacity tracker: how much value each account can absorb
    capacity = {acct.account_id: acct.total_value for acct in accounts}

    # Score each asset's tax-inefficiency in a taxable account (lower = more inefficient)
    inefficiency_scores: list[tuple[str, float]] = []
    for sym in symbols:
        profile = tax_profiles.get(sym, AssetTaxProfile(symbol=sym))
        taxable_score = compute_tax_efficiency_score(profile, AccountType.TAXABLE)
        inefficiency_scores.append((sym, taxable_score))

    # Sort: most tax-inefficient first (lowest taxable score)
    inefficiency_scores.sort(key=lambda pair: pair[1])

    # Separate accounts by type for priority assignment
    tax_advantaged = [
        a for a in accounts
        if a.account_type in (AccountType.TAX_DEFERRED, AccountType.TAX_EXEMPT)
    ]
    taxable = [a for a in accounts if a.account_type == AccountType.TAXABLE]

    # Sort advantaged accounts by value descending (fill largest first)
    tax_advantaged.sort(key=lambda a: a.total_value, reverse=True)

    locations: dict[str, str] = {}
    total_household = sum(a.total_value for a in accounts)
    if total_household <= 0:
        return {}

    # Threshold: assets with taxable efficiency above this are "tax-efficient"
    # and should prefer taxable accounts, not consume scarce tax-advantaged space.
    efficiency_threshold = 0.7

    for sym, taxable_score in inefficiency_scores:
        placed = False
        is_tax_inefficient = taxable_score < efficiency_threshold

        if is_tax_inefficient:
            # Tax-inefficient: prefer tax-advantaged accounts
            for acct in tax_advantaged:
                if _is_symbol_allowed(sym, acct) and capacity[acct.account_id] > 0:
                    locations[sym] = acct.account_id
                    placed = True
                    break
        else:
            # Tax-efficient: prefer taxable accounts
            for acct in taxable:
                if _is_symbol_allowed(sym, acct):
                    locations[sym] = acct.account_id
                    placed = True
                    break

        # Fall back: try any remaining account
        if not placed:
            for acct in accounts:
                if _is_symbol_allowed(sym, acct):
                    locations[sym] = acct.account_id
                    placed = True
                    break

        if not placed:
            logger.warning("No eligible account found for %s", sym)

    return locations


def _is_symbol_allowed(symbol: str, account: AccountConstraints) -> bool:
    """Check whether a symbol may be held in an account.

    Args:
        symbol: Ticker symbol.
        account: Account constraints to check against.

    Returns:
        True if the symbol is allowed in this account.
    """
    if symbol in account.excluded_symbols:
        return False
    if account.allowed_symbols is not None and symbol not in account.allowed_symbols:
        return False
    return True


# ── Household Optimization ───────────────────────────────────────────


def optimize_household(
    target_weights: dict[str, float],
    accounts: list[AccountConstraints],
    tax_profiles: dict[str, AssetTaxProfile] | None = None,
    mu: np.ndarray | None = None,
    cov: np.ndarray | None = None,
) -> HouseholdOptResult:
    """Optimize asset allocation across multiple accounts at the household level.

    Uses proportional allocation adjusted by tax-aware asset location preferences.
    Each account receives a share of each asset proportional to its value fraction,
    with adjustments to place tax-inefficient assets in tax-advantaged accounts.

    Args:
        target_weights: Household-level target weights {symbol: weight}.
        accounts: List of account constraints and metadata.
        tax_profiles: Optional tax profiles for asset location optimization.
            If None, uses simple proportional allocation without tax adjustment.
        mu: Expected returns vector (reserved for future use).
        cov: Covariance matrix (reserved for future use).

    Returns:
        HouseholdOptResult with per-account and aggregate weights.
    """
    symbols = list(target_weights.keys())
    total_household = sum(a.total_value for a in accounts)

    if total_household <= 0:
        logger.warning("Household total value is zero; returning empty result")
        return _empty_result(symbols, accounts)

    # Determine asset locations
    asset_locations: dict[str, str] = {}
    if tax_profiles:
        asset_locations = optimal_asset_location(symbols, tax_profiles, accounts)

    # Compute account value fractions
    value_fractions = {
        acct.account_id: acct.total_value / total_household for acct in accounts
    }

    # Allocate weights per account
    account_weights: dict[str, dict[str, float]] = {}
    for acct in accounts:
        acct_weights = _allocate_account_weights(
            acct, symbols, target_weights, value_fractions, asset_locations,
        )
        account_weights[acct.account_id] = acct_weights

    # Normalize: ensure household-level weights match targets
    account_weights = _normalize_household_weights(
        account_weights, target_weights, accounts, total_household,
    )

    # Compute aggregate household weights
    household_weights = _compute_household_weights(account_weights, accounts, total_household)

    # Estimate tax savings
    tax_savings = 0.0
    if tax_profiles:
        tax_savings = estimate_tax_savings(
            account_weights, accounts, tax_profiles, DEFAULT_TAX_RATES,
        )

    # Build per-account OptimizationResult
    per_account_results: dict[str, OptimizationResult] = {}
    for acct in accounts:
        weights = account_weights[acct.account_id]
        per_account_results[acct.account_id] = OptimizationResult(
            method="household_proportional",
            weights=weights,
            metadata={"account_type": acct.account_type.value},
        )

    return HouseholdOptResult(
        account_weights=account_weights,
        household_weights=household_weights,
        tax_savings_estimate=tax_savings,
        asset_locations=asset_locations,
        per_account_results=per_account_results,
    )


def _allocate_account_weights(
    account: AccountConstraints,
    symbols: list[str],
    target_weights: dict[str, float],
    value_fractions: dict[str, float],
    asset_locations: dict[str, str],
) -> dict[str, float]:
    """Compute raw weight allocation for a single account.

    Starts from proportional allocation (account's value fraction of each target
    weight), then boosts weights for assets that are location-optimized into this
    account and reduces weights for assets assigned elsewhere.

    Args:
        account: The account to allocate for.
        symbols: All symbols in the household.
        target_weights: Household-level target weights.
        value_fractions: Each account's fraction of total household value.
        asset_locations: Preferred account for each symbol.

    Returns:
        Raw (un-normalized) weight dict for this account.
    """
    frac = value_fractions.get(account.account_id, 0.0)
    weights: dict[str, float] = {}

    for sym in symbols:
        if not _is_symbol_allowed(sym, account):
            weights[sym] = 0.0
            continue

        base = target_weights.get(sym, 0.0) * frac

        # Apply location preference: boost if this is the preferred account
        if asset_locations.get(sym) == account.account_id:
            base *= 1.5
        elif sym in asset_locations and asset_locations[sym] != account.account_id:
            base *= 0.5

        # Enforce per-asset bounds from account constraints
        lo, hi = account.constraints.get_bounds(sym)
        weights[sym] = float(np.clip(base, lo * frac, hi * frac) if frac > 0 else 0.0)

    return weights


def _normalize_household_weights(
    account_weights: dict[str, dict[str, float]],
    target_weights: dict[str, float],
    accounts: list[AccountConstraints],
    total_household: float,
) -> dict[str, dict[str, float]]:
    """Normalize per-account weights so the household aggregate matches targets.

    Scales each symbol's allocation across accounts so that the value-weighted
    sum equals the target household weight.

    Args:
        account_weights: Current per-account weight dicts.
        target_weights: Desired household-level weights.
        accounts: Account metadata (for value lookups).
        total_household: Sum of all account values.

    Returns:
        Adjusted per-account weight dicts.
    """
    for sym, target_w in target_weights.items():
        # Current household weight for this symbol
        current_w = sum(
            account_weights[a.account_id].get(sym, 0.0) * a.total_value / total_household
            for a in accounts
        )

        if current_w <= 0:
            continue

        scale = target_w / current_w
        for acct_id, weights in account_weights.items():
            if sym in weights and weights[sym] > 0:
                weights[sym] *= scale

    # Ensure each account's weights sum to 1.0
    for acct_id, weights in account_weights.items():
        total = sum(weights.values())
        if total > 0:
            for sym in weights:
                weights[sym] /= total

    return account_weights


def _compute_household_weights(
    account_weights: dict[str, dict[str, float]],
    accounts: list[AccountConstraints],
    total_household: float,
) -> dict[str, float]:
    """Compute aggregate household-level weights from per-account allocations.

    Each account's weights are scaled by its fraction of total household value.

    Args:
        account_weights: Per-account weight dicts.
        accounts: Account metadata.
        total_household: Sum of all account values.

    Returns:
        Household-level weights {symbol: weight}.
    """
    household: dict[str, float] = {}
    for acct in accounts:
        frac = acct.total_value / total_household if total_household > 0 else 0.0
        for sym, w in account_weights[acct.account_id].items():
            household[sym] = household.get(sym, 0.0) + w * frac

    return household


def _empty_result(
    symbols: list[str],
    accounts: list[AccountConstraints],
) -> HouseholdOptResult:
    """Build an empty HouseholdOptResult when optimization cannot proceed.

    Args:
        symbols: List of symbols.
        accounts: List of accounts.

    Returns:
        HouseholdOptResult with zero weights everywhere.
    """
    zero_weights = {sym: 0.0 for sym in symbols}
    return HouseholdOptResult(
        account_weights={a.account_id: dict(zero_weights) for a in accounts},
        household_weights=dict(zero_weights),
        tax_savings_estimate=0.0,
        asset_locations={},
        per_account_results={
            a.account_id: OptimizationResult(
                method="household_proportional",
                weights=dict(zero_weights),
            )
            for a in accounts
        },
    )


# ── Tax Savings Estimation ───────────────────────────────────────────


def estimate_tax_savings(
    account_weights: dict[str, dict[str, float]],
    accounts: list[AccountConstraints],
    tax_profiles: dict[str, AssetTaxProfile],
    tax_rates: dict[str, float] | None = None,
) -> float:
    """Estimate annual tax savings from optimal vs. naive asset location.

    Compares the tax cost of the optimized placement against a naive baseline
    where every account holds the same proportional allocation.

    Args:
        account_weights: Per-account weight allocations from optimization.
        accounts: Account metadata.
        tax_profiles: Tax profile for each symbol.
        tax_rates: Tax rate assumptions. Uses DEFAULT_TAX_RATES if None.

    Returns:
        Estimated annual tax savings as a dollar amount (based on account values).
    """
    if tax_rates is None:
        tax_rates = DEFAULT_TAX_RATES

    total_household = sum(a.total_value for a in accounts)
    if total_household <= 0:
        return 0.0

    # Tax cost of the optimized allocation
    optimized_cost = 0.0
    for acct in accounts:
        weights = account_weights.get(acct.account_id, {})
        for sym, w in weights.items():
            if w <= 0:
                continue
            profile = tax_profiles.get(sym, AssetTaxProfile(symbol=sym))
            dollar_alloc = w * acct.total_value
            drag = _tax_cost_in_account(profile, acct.account_type, tax_rates)
            optimized_cost += dollar_alloc * drag

    # Tax cost of a naive allocation (uniform across all accounts)
    naive_cost = 0.0
    all_symbols = set()
    for weights in account_weights.values():
        all_symbols.update(weights.keys())

    for acct in accounts:
        for sym in all_symbols:
            # Naive: each account holds same proportional weight
            household_w = sum(
                account_weights.get(a.account_id, {}).get(sym, 0.0) * a.total_value / total_household
                for a in accounts
            )
            dollar_alloc = household_w * acct.total_value
            profile = tax_profiles.get(sym, AssetTaxProfile(symbol=sym))
            drag = _tax_cost_in_account(profile, acct.account_type, tax_rates)
            naive_cost += dollar_alloc * drag

    return max(0.0, naive_cost - optimized_cost)


def _tax_cost_in_account(
    profile: AssetTaxProfile,
    account_type: AccountType,
    tax_rates: dict[str, float],
) -> float:
    """Compute the annual tax cost rate for an asset in a specific account type.

    Tax-deferred and tax-exempt accounts have zero current tax cost (taxes
    are deferred or eliminated). Taxable accounts incur taxes on dividends
    and realized capital gains.

    Args:
        profile: Tax characteristics of the asset.
        account_type: Account type for tax treatment.
        tax_rates: Tax rate assumptions.

    Returns:
        Annual tax cost as a decimal fraction of the position value.
    """
    if account_type in (AccountType.TAX_DEFERRED, AccountType.TAX_EXEMPT):
        return 0.0

    # Taxable account: full tax drag applies
    return _estimate_tax_drag(profile)
