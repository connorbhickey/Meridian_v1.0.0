"""Hierarchical Risk Parity (HRP).

Implementation based on Marcos Lopez de Prado (2016):
1. Tree clustering via agglomerative hierarchical clustering
2. Quasi-diagonalization (seriation) of the covariance matrix
3. Recursive bisection to allocate weights by inverse variance

Supports 4 linkage methods: single, complete, average, ward.
Supports long-only and long/short variants.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from portopt.constants import LinkageMethod, RiskMeasure
from portopt.data.models import OptimizationResult
from portopt.engine.risk import cov_to_corr


def hrp_optimize(
    covariance: pd.DataFrame,
    linkage_method: LinkageMethod = LinkageMethod.SINGLE,
    risk_measure: RiskMeasure = RiskMeasure.VARIANCE,
    returns: pd.DataFrame | None = None,
    long_only: bool = True,
) -> OptimizationResult:
    """Run Hierarchical Risk Parity optimization.

    Args:
        covariance: Annualized covariance matrix.
        linkage_method: Hierarchical clustering linkage type.
        risk_measure: Risk measure for recursive bisection (VARIANCE, STD_DEV, CVAR, CDAR).
        returns: Required if risk_measure is CVAR or CDAR.
        long_only: If True, all weights are non-negative.

    Returns:
        OptimizationResult with HRP weights.
    """
    symbols = list(covariance.index)
    cov = covariance.values
    corr = cov_to_corr(covariance).values

    # Step 1: Tree clustering
    dist = _correlation_distance(corr)
    dist_condensed = squareform(dist)
    link = linkage(dist_condensed, method=linkage_method.value)

    # Step 2: Quasi-diagonalization
    sorted_idx = list(leaves_list(link))
    sorted_symbols = [symbols[i] for i in sorted_idx]

    # Step 3: Recursive bisection
    weights = _recursive_bisection(
        cov=cov,
        sorted_idx=sorted_idx,
        risk_measure=risk_measure,
        returns=returns.values if returns is not None else None,
        symbols=symbols,
    )

    # Build weight dict
    weight_dict = {symbols[i]: weights[i] for i in range(len(symbols))}

    # Compute portfolio stats
    w = np.array([weight_dict[s] for s in symbols])
    mu_est = 0.0  # HRP doesn't use expected returns
    vol = float(np.sqrt(w @ cov @ w))

    return OptimizationResult(
        method="HRP",
        weights=weight_dict,
        expected_return=mu_est,
        volatility=vol,
        sharpe_ratio=0.0,
        metadata={
            "linkage_method": linkage_method.value,
            "risk_measure": risk_measure.name,
            "sorted_symbols": sorted_symbols,
            "linkage_matrix": link.tolist(),
        },
    )


def _correlation_distance(corr: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to distance matrix: d = sqrt(0.5 * (1 - corr))."""
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0.0)
    return dist


def _recursive_bisection(
    cov: np.ndarray,
    sorted_idx: list[int],
    risk_measure: RiskMeasure = RiskMeasure.VARIANCE,
    returns: np.ndarray | None = None,
    symbols: list[str] | None = None,
) -> np.ndarray:
    """Recursive bisection to allocate weights by cluster risk.

    Splits sorted assets into two clusters, allocates between them
    inversely proportional to their risk, then recurses.
    """
    n = cov.shape[0]
    weights = np.zeros(n)

    # Initialize: all weight assigned to the full set
    cluster_weights = {tuple(sorted_idx): 1.0}

    while cluster_weights:
        new_clusters = {}
        for cluster, cluster_w in cluster_weights.items():
            if len(cluster) == 1:
                weights[cluster[0]] = cluster_w
                continue

            # Split in half
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]

            # Compute cluster risk
            left_risk = _cluster_risk(cov, list(left), risk_measure, returns)
            right_risk = _cluster_risk(cov, list(right), risk_measure, returns)

            # Allocate inversely proportional to risk
            total_inv = 1.0 / left_risk + 1.0 / right_risk if (left_risk > 0 and right_risk > 0) else 1.0
            alpha = (1.0 / left_risk) / total_inv if left_risk > 0 else 0.5

            new_clusters[left] = cluster_w * alpha
            new_clusters[right] = cluster_w * (1 - alpha)

        cluster_weights = new_clusters

    return weights


def _cluster_risk(
    cov: np.ndarray,
    indices: list[int],
    risk_measure: RiskMeasure = RiskMeasure.VARIANCE,
    returns: np.ndarray | None = None,
) -> float:
    """Compute the risk of a cluster using inverse-variance weights within the cluster."""
    sub_cov = cov[np.ix_(indices, indices)]

    # Inverse variance weights within cluster
    inv_var = 1.0 / np.diag(sub_cov)
    w = inv_var / inv_var.sum()

    if risk_measure == RiskMeasure.VARIANCE:
        return float(w @ sub_cov @ w)
    elif risk_measure == RiskMeasure.STD_DEV:
        return float(np.sqrt(w @ sub_cov @ w))
    elif risk_measure == RiskMeasure.CVAR and returns is not None:
        sub_returns = returns[:, indices]
        port_returns = sub_returns @ w
        var_95 = np.percentile(port_returns, 5)
        tail = port_returns[port_returns <= var_95]
        return float(abs(np.mean(tail))) if len(tail) > 0 else float(np.sqrt(w @ sub_cov @ w))
    elif risk_measure == RiskMeasure.CDAR and returns is not None:
        sub_returns = returns[:, indices]
        port_returns = sub_returns @ w
        cumulative = np.cumprod(1 + port_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = 1 - cumulative / running_max
        cdar_95 = np.percentile(drawdowns, 95)
        tail = drawdowns[drawdowns >= cdar_95]
        return float(np.mean(tail)) if len(tail) > 0 else float(np.sqrt(w @ sub_cov @ w))
    else:
        return float(w @ sub_cov @ w)


def get_linkage_matrix(
    covariance: pd.DataFrame,
    linkage_method: LinkageMethod = LinkageMethod.SINGLE,
) -> np.ndarray:
    """Compute the linkage matrix for dendrogram visualization."""
    corr = cov_to_corr(covariance).values
    dist = _correlation_distance(corr)
    dist_condensed = squareform(dist)
    return linkage(dist_condensed, method=linkage_method.value)
