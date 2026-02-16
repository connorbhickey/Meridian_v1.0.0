"""Minimum Spanning Tree (MST) for asset correlation networks.

Implements:
- Correlation-to-distance transformation
- MST construction via Prim's and Kruskal's algorithms (via networkx)
- Rolling-window MST evolution
- MST-based clustering
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx

from portopt.engine.risk import cov_to_corr


def build_correlation_network(
    covariance: pd.DataFrame,
) -> nx.Graph:
    """Build a complete weighted graph from a covariance matrix.

    Edge weights are correlation-derived distances: d = sqrt(2 * (1 - rho)).
    """
    corr = cov_to_corr(covariance)
    symbols = list(corr.index)
    G = nx.Graph()

    for i, s1 in enumerate(symbols):
        G.add_node(s1)
        for j, s2 in enumerate(symbols):
            if i < j:
                rho = corr.iloc[i, j]
                dist = np.sqrt(2 * (1 - rho))
                G.add_edge(s1, s2, weight=dist, correlation=rho)

    return G


def compute_mst(
    covariance: pd.DataFrame,
    algorithm: str = "kruskal",
) -> nx.Graph:
    """Compute the Minimum Spanning Tree of the asset correlation network.

    Args:
        covariance: Covariance matrix DataFrame.
        algorithm: "kruskal" or "prim".

    Returns:
        networkx Graph representing the MST.
    """
    G = build_correlation_network(covariance)
    mst = nx.minimum_spanning_tree(G, algorithm=algorithm)
    return mst


def mst_from_prices(
    prices: pd.DataFrame,
    algorithm: str = "kruskal",
) -> nx.Graph:
    """Build MST directly from price data."""
    from portopt.engine.risk import estimate_covariance
    from portopt.constants import CovEstimator

    cov = estimate_covariance(prices, CovEstimator.SAMPLE)
    return compute_mst(cov, algorithm)


def rolling_mst(
    prices: pd.DataFrame,
    window: int = 252,
    step: int = 21,
    algorithm: str = "kruskal",
) -> list[tuple[pd.Timestamp, nx.Graph]]:
    """Compute MST at regular intervals using a rolling window.

    Args:
        prices: Price DataFrame (index=dates, columns=symbols).
        window: Rolling window size in trading days.
        step: Step size between MST computations.
        algorithm: MST algorithm.

    Returns:
        List of (date, MST graph) tuples.
    """
    from portopt.engine.returns import log_returns

    rets = log_returns(prices)
    results = []

    for end in range(window, len(rets), step):
        start = end - window
        sub_rets = rets.iloc[start:end]
        cov = sub_rets.cov() * 252
        try:
            mst = compute_mst(cov, algorithm)
            date = rets.index[end - 1]
            results.append((date, mst))
        except Exception:
            continue

    return results


def mst_to_adjacency(mst: nx.Graph, symbols: list[str] | None = None) -> pd.DataFrame:
    """Convert MST to adjacency matrix DataFrame."""
    if symbols is None:
        symbols = sorted(mst.nodes())
    adj = nx.to_pandas_adjacency(mst, nodelist=symbols)
    return adj


def get_mst_layout(mst: nx.Graph) -> dict[str, tuple[float, float]]:
    """Compute node positions for MST visualization using spring layout."""
    return nx.spring_layout(mst, seed=42, k=2.0)


def get_mst_central_node(mst: nx.Graph) -> str:
    """Find the most central node in the MST (highest degree centrality)."""
    centrality = nx.degree_centrality(mst)
    return max(centrality, key=centrality.get)
