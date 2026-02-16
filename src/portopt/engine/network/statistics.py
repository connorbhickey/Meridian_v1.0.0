"""Network statistics for MST and correlation graphs.

Computes centrality measures, clustering, community structure,
and other graph-theoretic metrics useful for portfolio analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx


def compute_network_statistics(graph: nx.Graph) -> dict[str, float]:
    """Compute summary statistics for a network graph.

    Returns a dictionary of network-level metrics.
    """
    stats = {}
    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    stats["n_nodes"] = n
    stats["n_edges"] = m
    stats["density"] = nx.density(graph)

    # Average path length (only for connected graphs)
    if nx.is_connected(graph):
        stats["avg_path_length"] = nx.average_shortest_path_length(graph)
        stats["diameter"] = nx.diameter(graph)
    else:
        stats["avg_path_length"] = 0.0
        stats["diameter"] = 0

    # Degree statistics
    degrees = [d for _, d in graph.degree()]
    stats["avg_degree"] = float(np.mean(degrees)) if degrees else 0.0
    stats["max_degree"] = max(degrees) if degrees else 0
    stats["degree_std"] = float(np.std(degrees)) if degrees else 0.0

    # Average edge weight
    if m > 0:
        weights = [d.get("weight", 1.0) for _, _, d in graph.edges(data=True)]
        stats["avg_edge_weight"] = float(np.mean(weights))
        stats["total_tree_length"] = float(np.sum(weights))
    else:
        stats["avg_edge_weight"] = 0.0
        stats["total_tree_length"] = 0.0

    return stats


def compute_node_centralities(graph: nx.Graph) -> pd.DataFrame:
    """Compute centrality measures for each node.

    Returns DataFrame with columns: degree, betweenness, closeness, eigenvector.
    """
    nodes = sorted(graph.nodes())

    degree = dict(nx.degree_centrality(graph))
    betweenness = dict(nx.betweenness_centrality(graph))
    closeness = dict(nx.closeness_centrality(graph))

    try:
        eigenvector = dict(nx.eigenvector_centrality(graph, max_iter=500))
    except nx.PowerIterationFailedConvergence:
        eigenvector = {n: 0.0 for n in nodes}

    df = pd.DataFrame({
        "degree": pd.Series(degree),
        "betweenness": pd.Series(betweenness),
        "closeness": pd.Series(closeness),
        "eigenvector": pd.Series(eigenvector),
    })
    return df.loc[nodes]


def compute_degree_distribution(graph: nx.Graph) -> dict[int, int]:
    """Compute the degree distribution of the graph.

    Returns dict: degree -> count of nodes with that degree.
    """
    degrees = [d for _, d in graph.degree()]
    dist = {}
    for d in degrees:
        dist[d] = dist.get(d, 0) + 1
    return dict(sorted(dist.items()))


def find_hub_nodes(graph: nx.Graph, top_n: int = 5) -> list[tuple[str, int]]:
    """Find the top hub nodes (highest degree) in the network.

    Returns list of (node, degree) tuples sorted by degree descending.
    """
    degree_list = sorted(graph.degree(), key=lambda x: x[1], reverse=True)
    return [(str(node), deg) for node, deg in degree_list[:top_n]]


def find_leaf_nodes(graph: nx.Graph) -> list[str]:
    """Find all leaf nodes (degree 1) in the network."""
    return [str(n) for n, d in graph.degree() if d == 1]


def compute_mst_sector_composition(
    mst: nx.Graph,
    sector_map: dict[str, str],
) -> dict[str, dict]:
    """Analyze sector composition of MST edges and clusters.

    Returns per-sector statistics including number of nodes,
    intra-sector edges, and inter-sector edges.
    """
    sectors = {}
    for node in mst.nodes():
        sector = sector_map.get(str(node), "Unknown")
        sectors.setdefault(sector, {"nodes": [], "intra_edges": 0, "inter_edges": 0})
        sectors[sector]["nodes"].append(str(node))

    for u, v in mst.edges():
        s_u = sector_map.get(str(u), "Unknown")
        s_v = sector_map.get(str(v), "Unknown")
        if s_u == s_v:
            sectors[s_u]["intra_edges"] += 1
        else:
            sectors[s_u]["inter_edges"] += 1
            sectors[s_v]["inter_edges"] += 1

    # Compute summary
    result = {}
    for sector, data in sectors.items():
        result[sector] = {
            "n_nodes": len(data["nodes"]),
            "intra_edges": data["intra_edges"],
            "inter_edges": data["inter_edges"],
            "symbols": data["nodes"],
        }
    return result


def edge_weight_series(
    mst_series: list[tuple[pd.Timestamp, nx.Graph]],
) -> pd.DataFrame:
    """Convert a rolling MST series into total tree length over time.

    Args:
        mst_series: List of (date, MST graph) from rolling_mst().

    Returns:
        DataFrame with columns: date, total_tree_length, avg_edge_weight.
    """
    records = []
    for date, mst in mst_series:
        weights = [d.get("weight", 0) for _, _, d in mst.edges(data=True)]
        records.append({
            "date": date,
            "total_tree_length": sum(weights),
            "avg_edge_weight": float(np.mean(weights)) if weights else 0.0,
        })
    return pd.DataFrame(records)
