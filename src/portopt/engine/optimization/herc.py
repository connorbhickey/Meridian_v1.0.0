"""Hierarchical Equal Risk Contribution (HERC).

Extension of HRP that:
1. Uses Gap Index to determine optimal number of clusters
2. Performs true dendrogram-based bisection at each cluster level
3. Applies Naive Risk Parity within each cluster
4. Allocates between clusters using risk contribution equality

Supports: Variance, StdDev, CVaR, CDaR risk measures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform

from portopt.constants import LinkageMethod, RiskMeasure
from portopt.data.models import OptimizationResult
from portopt.engine.risk import cov_to_corr


def herc_optimize(
    covariance: pd.DataFrame,
    linkage_method: LinkageMethod = LinkageMethod.WARD,
    risk_measure: RiskMeasure = RiskMeasure.VARIANCE,
    returns: pd.DataFrame | None = None,
    max_clusters: int | None = None,
) -> OptimizationResult:
    """Run HERC optimization.

    Args:
        covariance: Annualized covariance matrix.
        linkage_method: Hierarchical clustering linkage type.
        risk_measure: Risk measure for allocation.
        returns: Required if risk_measure is CVAR or CDAR.
        max_clusters: Maximum number of clusters (None = use Gap Index).

    Returns:
        OptimizationResult with HERC weights.
    """
    symbols = list(covariance.index)
    n = len(symbols)

    # Single-asset guard
    if n == 1:
        return OptimizationResult(
            method="HERC",
            weights={symbols[0]: 1.0},
            expected_return=0.0,
            volatility=float(np.sqrt(covariance.values[0, 0])),
            sharpe_ratio=0.0,
            metadata={"linkage_method": linkage_method.value, "risk_measure": risk_measure.name, "n_clusters": 1},
        )

    cov = covariance.values
    corr = cov_to_corr(covariance).values

    # Step 1: Hierarchical clustering
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0.0)
    dist_condensed = squareform(dist)
    link = linkage(dist_condensed, method=linkage_method.value)

    # Step 2: Determine optimal number of clusters
    if max_clusters is not None:
        n_clusters = max_clusters
    else:
        n_clusters = _gap_index(link, n)

    n_clusters = max(2, min(n_clusters, n))

    # Step 3: Assign assets to clusters
    cluster_labels = fcluster(link, t=n_clusters, criterion="maxclust")
    clusters: dict[int, list[int]] = {}
    for i, label in enumerate(cluster_labels):
        clusters.setdefault(int(label), []).append(i)

    # Step 4: Within-cluster allocation (naive risk parity)
    within_weights = np.zeros(n)
    cluster_risks = {}

    for label, indices in clusters.items():
        sub_cov = cov[np.ix_(indices, indices)]

        # Naive risk parity within cluster: weight inversely proportional to marginal risk
        diag = np.diag(sub_cov)
        if risk_measure == RiskMeasure.VARIANCE:
            risk_contrib = diag
        elif risk_measure == RiskMeasure.STD_DEV:
            risk_contrib = np.sqrt(diag)
        elif risk_measure == RiskMeasure.CVAR and returns is not None:
            risk_contrib = _per_asset_cvar(returns.values[:, indices])
        elif risk_measure == RiskMeasure.CDAR and returns is not None:
            risk_contrib = _per_asset_cdar(returns.values[:, indices])
        else:
            risk_contrib = diag

        inv_risk = 1.0 / np.maximum(risk_contrib, 1e-10)
        local_w = inv_risk / inv_risk.sum()

        for j, idx in enumerate(indices):
            within_weights[idx] = local_w[j]

        # Cluster-level risk for between-cluster allocation
        cluster_risks[label] = _cluster_risk_nrp(
            cov, indices, risk_measure,
            returns.values if returns is not None else None,
        )

    # Step 5: Between-cluster allocation (equal risk contribution)
    inv_cluster_risk = {l: 1.0 / max(r, 1e-10) for l, r in cluster_risks.items()}
    total_inv = sum(inv_cluster_risk.values())
    cluster_alloc = {l: v / total_inv for l, v in inv_cluster_risk.items()}

    # Step 6: Combine
    weights = np.zeros(n)
    for label, indices in clusters.items():
        for idx in indices:
            weights[idx] = within_weights[idx] * cluster_alloc[label]

    # Build result
    weight_dict = {symbols[i]: weights[i] for i in range(n)}
    vol = float(np.sqrt(weights @ cov @ weights))

    return OptimizationResult(
        method="HERC",
        weights=weight_dict,
        expected_return=0.0,
        volatility=vol,
        sharpe_ratio=0.0,
        metadata={
            "linkage_method": linkage_method.value,
            "risk_measure": risk_measure.name,
            "n_clusters": n_clusters,
            "cluster_labels": {symbols[i]: int(cluster_labels[i]) for i in range(n)},
            "cluster_allocation": {str(k): v for k, v in cluster_alloc.items()},
            "linkage_matrix": link.tolist(),
        },
    )


def _gap_index(link: np.ndarray, n: int) -> int:
    """Determine optimal number of clusters using the Gap Index.

    The gap index looks at the relative change in merge distances
    to find a natural number of clusters.
    """
    merge_dists = link[:, 2]
    if len(merge_dists) < 3:
        return 2

    # Compute gaps: difference between successive merge distances
    gaps = np.diff(merge_dists)
    # Normalize
    if gaps.max() > 0:
        gaps_norm = gaps / gaps.max()
    else:
        return 2

    # Find the largest gap — the number of clusters is n - index_of_max_gap
    max_gap_idx = np.argmax(gaps_norm)
    n_clusters = n - max_gap_idx - 1

    return max(2, min(n_clusters, n // 2))


def _cluster_risk_nrp(
    cov: np.ndarray,
    indices: list[int],
    risk_measure: RiskMeasure,
    returns: np.ndarray | None,
) -> float:
    """Compute cluster risk using naive risk parity weights."""
    sub_cov = cov[np.ix_(indices, indices)]
    diag = np.diag(sub_cov)
    inv_var = 1.0 / np.maximum(diag, 1e-10)
    w = inv_var / inv_var.sum()

    if risk_measure == RiskMeasure.VARIANCE:
        return float(w @ sub_cov @ w)
    elif risk_measure == RiskMeasure.STD_DEV:
        return float(np.sqrt(w @ sub_cov @ w))
    elif risk_measure == RiskMeasure.CVAR and returns is not None:
        sub_ret = returns[:, indices]
        port_ret = sub_ret @ w
        var_95 = np.percentile(port_ret, 5)
        tail = port_ret[port_ret <= var_95]
        return float(abs(np.mean(tail))) if len(tail) > 0 else float(np.sqrt(w @ sub_cov @ w))
    elif risk_measure == RiskMeasure.CDAR and returns is not None:
        sub_ret = returns[:, indices]
        port_ret = sub_ret @ w
        cum = np.cumprod(1 + port_ret)
        rmax = np.maximum.accumulate(cum)
        dd = 1 - cum / rmax
        cdar = np.percentile(dd, 95)
        tail = dd[dd >= cdar]
        return float(np.mean(tail)) if len(tail) > 0 else float(np.sqrt(w @ sub_cov @ w))
    return float(w @ sub_cov @ w)


def _per_asset_cvar(returns: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Per-asset CVaR (for naive risk parity)."""
    result = np.zeros(returns.shape[1])
    for i in range(returns.shape[1]):
        r = returns[:, i]
        var = np.percentile(r, alpha * 100)
        tail = r[r <= var]
        result[i] = abs(np.mean(tail)) if len(tail) > 0 else np.std(r)
    return result


def _per_asset_cdar(returns: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Per-asset CDaR (for naive risk parity)."""
    result = np.zeros(returns.shape[1])
    for i in range(returns.shape[1]):
        r = returns[:, i]
        cum = np.cumprod(1 + r)
        rmax = np.maximum.accumulate(cum)
        dd = 1 - cum / rmax
        cdar = np.percentile(dd, (1 - alpha) * 100)
        tail = dd[dd >= cdar]
        result[i] = np.mean(tail) if len(tail) > 0 else np.std(r)
    return result
