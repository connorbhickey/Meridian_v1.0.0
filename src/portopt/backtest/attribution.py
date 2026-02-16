"""Return attribution: Brinson and factor-based decomposition.

Implements:
- Brinson-Hood-Beebower (BHB) attribution: allocation, selection, interaction
- Factor attribution: decompose returns into factor exposures
- Time-series contribution: daily contribution of each asset to portfolio return
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


@dataclass
class BrinsonAttribution:
    """Brinson-Hood-Beebower attribution results.

    Decomposes active return into:
    - Allocation effect: over/underweight winning/losing sectors
    - Selection effect: stock picking within sectors
    - Interaction effect: combined allocation + selection
    """
    allocation: dict[str, float] = field(default_factory=dict)
    selection: dict[str, float] = field(default_factory=dict)
    interaction: dict[str, float] = field(default_factory=dict)
    total_allocation: float = 0.0
    total_selection: float = 0.0
    total_interaction: float = 0.0
    total_active: float = 0.0
    sectors: list[str] = field(default_factory=list)


def brinson_attribution(
    portfolio_weights: dict[str, float],
    benchmark_weights: dict[str, float],
    portfolio_returns: dict[str, float],
    benchmark_returns: dict[str, float],
    sector_map: dict[str, str],
) -> BrinsonAttribution:
    """Single-period Brinson-Hood-Beebower attribution.

    Args:
        portfolio_weights: {symbol: weight} for portfolio.
        benchmark_weights: {symbol: weight} for benchmark.
        portfolio_returns: {symbol: return} for portfolio assets.
        benchmark_returns: {symbol: return} for benchmark assets.
        sector_map: {symbol: sector} mapping.

    Returns:
        BrinsonAttribution with per-sector and total effects.
    """
    # Aggregate to sector level
    all_symbols = set(
        list(portfolio_weights.keys()) +
        list(benchmark_weights.keys())
    )
    sectors_set = set()
    for sym in all_symbols:
        sectors_set.add(sector_map.get(sym, "Unknown"))
    sectors = sorted(sectors_set)

    # Sector-level weights and returns
    port_sector_w: dict[str, float] = {}
    bench_sector_w: dict[str, float] = {}
    port_sector_r: dict[str, float] = {}
    bench_sector_r: dict[str, float] = {}

    for sector in sectors:
        # Portfolio sector weight and return
        pw = 0.0
        pr_num = 0.0
        for sym in all_symbols:
            if sector_map.get(sym, "Unknown") != sector:
                continue
            w = portfolio_weights.get(sym, 0.0)
            pw += w
            pr_num += w * portfolio_returns.get(sym, 0.0)
        port_sector_w[sector] = pw
        port_sector_r[sector] = pr_num / pw if pw > 0 else 0.0

        # Benchmark sector weight and return
        bw = 0.0
        br_num = 0.0
        for sym in all_symbols:
            if sector_map.get(sym, "Unknown") != sector:
                continue
            w = benchmark_weights.get(sym, 0.0)
            bw += w
            br_num += w * benchmark_returns.get(sym, 0.0)
        bench_sector_w[sector] = bw
        bench_sector_r[sector] = br_num / bw if bw > 0 else 0.0

    # Total benchmark return
    total_bench_r = sum(
        bench_sector_w.get(s, 0.0) * bench_sector_r.get(s, 0.0)
        for s in sectors
    )

    # Compute effects
    allocation = {}
    selection = {}
    interaction = {}

    for sector in sectors:
        wp = port_sector_w.get(sector, 0.0)
        wb = bench_sector_w.get(sector, 0.0)
        rp = port_sector_r.get(sector, 0.0)
        rb = bench_sector_r.get(sector, 0.0)

        # Allocation: (wp - wb) * (rb - total_bench_r)
        allocation[sector] = (wp - wb) * (rb - total_bench_r)

        # Selection: wb * (rp - rb)
        selection[sector] = wb * (rp - rb)

        # Interaction: (wp - wb) * (rp - rb)
        interaction[sector] = (wp - wb) * (rp - rb)

    return BrinsonAttribution(
        allocation=allocation,
        selection=selection,
        interaction=interaction,
        total_allocation=sum(allocation.values()),
        total_selection=sum(selection.values()),
        total_interaction=sum(interaction.values()),
        total_active=sum(allocation.values()) + sum(selection.values()) + sum(interaction.values()),
        sectors=sectors,
    )


def multi_period_brinson(
    portfolio_weights_series: list[dict[str, float]],
    benchmark_weights_series: list[dict[str, float]],
    portfolio_returns_series: list[dict[str, float]],
    benchmark_returns_series: list[dict[str, float]],
    sector_map: dict[str, str],
) -> BrinsonAttribution:
    """Multi-period Brinson attribution (arithmetic linking).

    Averages single-period attributions across all periods.
    """
    n = len(portfolio_weights_series)
    if n == 0:
        return BrinsonAttribution()

    all_sectors: set[str] = set()
    results = []

    for i in range(n):
        result = brinson_attribution(
            portfolio_weights_series[i],
            benchmark_weights_series[i],
            portfolio_returns_series[i],
            benchmark_returns_series[i],
            sector_map,
        )
        results.append(result)
        all_sectors.update(result.sectors)

    sectors = sorted(all_sectors)
    avg_alloc = {s: np.mean([r.allocation.get(s, 0.0) for r in results]) for s in sectors}
    avg_sel = {s: np.mean([r.selection.get(s, 0.0) for r in results]) for s in sectors}
    avg_inter = {s: np.mean([r.interaction.get(s, 0.0) for r in results]) for s in sectors}

    return BrinsonAttribution(
        allocation=avg_alloc,
        selection=avg_sel,
        interaction=avg_inter,
        total_allocation=sum(avg_alloc.values()),
        total_selection=sum(avg_sel.values()),
        total_interaction=sum(avg_inter.values()),
        total_active=sum(avg_alloc.values()) + sum(avg_sel.values()) + sum(avg_inter.values()),
        sectors=sectors,
    )


@dataclass
class FactorAttribution:
    """Factor-based attribution results."""
    factor_exposures: dict[str, float] = field(default_factory=dict)   # beta to each factor
    factor_contributions: dict[str, float] = field(default_factory=dict)  # return from each factor
    specific_return: float = 0.0        # Residual / idiosyncratic return
    r_squared: float = 0.0             # Fraction of variance explained
    total_return: float = 0.0


def factor_attribution(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
) -> FactorAttribution:
    """Decompose portfolio returns using factor model (OLS regression).

    Args:
        portfolio_returns: Daily portfolio return series.
        factor_returns: DataFrame of daily factor returns (columns = factor names).

    Returns:
        FactorAttribution with exposures and contributions.
    """
    # Align dates
    common = portfolio_returns.index.intersection(factor_returns.index)
    if len(common) < 10:
        return FactorAttribution()

    y = portfolio_returns.loc[common].values
    X = factor_returns.loc[common].values
    factor_names = list(factor_returns.columns)

    # Add intercept
    X_with_const = np.column_stack([np.ones(len(X)), X])

    # OLS regression
    try:
        beta, residuals, _, _ = np.linalg.lstsq(X_with_const, y, rcond=None)
    except np.linalg.LinAlgError:
        return FactorAttribution()

    alpha = beta[0]
    factor_betas = beta[1:]

    # Factor contributions = beta * mean factor return * frequency
    factor_mean = np.mean(X, axis=0)
    contributions = factor_betas * factor_mean * 252

    # R-squared
    y_pred = X_with_const @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return FactorAttribution(
        factor_exposures=dict(zip(factor_names, factor_betas.tolist())),
        factor_contributions=dict(zip(factor_names, contributions.tolist())),
        specific_return=float(alpha * 252),
        r_squared=float(r_squared),
        total_return=float(np.mean(y) * 252),
    )


def contribution_analysis(
    weights_history: list[dict[str, float]],
    asset_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Compute daily contribution of each asset to portfolio return.

    Args:
        weights_history: List of weight dicts (one per day).
        asset_returns: Daily asset return DataFrame.

    Returns:
        DataFrame with same shape as asset_returns, values = w_i * r_i.
    """
    n = min(len(weights_history), len(asset_returns))
    symbols = list(asset_returns.columns)
    dates = asset_returns.index[:n]

    contributions = pd.DataFrame(0.0, index=dates, columns=symbols)

    for i in range(n):
        w = weights_history[i]
        r = asset_returns.iloc[i]
        for s in symbols:
            contributions.iloc[i][s] = w.get(s, 0.0) * float(r.get(s, 0.0))

    return contributions


def cumulative_contribution(
    weights_history: list[dict[str, float]],
    asset_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Compute cumulative contribution of each asset over time.

    Returns DataFrame where each column shows the cumulative contribution
    of that asset to the portfolio return.
    """
    daily = contribution_analysis(weights_history, asset_returns)
    return daily.cumsum()
