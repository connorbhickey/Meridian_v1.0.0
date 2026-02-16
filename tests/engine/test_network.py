"""Tests for MST network analysis."""

import networkx as nx
import numpy as np
import pytest

from portopt.engine.network.mst import (
    build_correlation_network,
    compute_mst,
    get_mst_central_node,
    get_mst_layout,
    mst_to_adjacency,
)
from portopt.engine.risk import estimate_covariance


class TestBuildCorrelationNetwork:
    def test_complete_graph(self, prices_5):
        cov = estimate_covariance(prices_5)
        G = build_correlation_network(cov)
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == 5
        # Complete graph: n*(n-1)/2 edges
        assert G.number_of_edges() == 10

    def test_edge_attributes(self, prices_5):
        cov = estimate_covariance(prices_5)
        G = build_correlation_network(cov)
        for u, v, data in G.edges(data=True):
            assert "weight" in data
            assert "correlation" in data
            assert data["weight"] >= 0
            assert -1 <= data["correlation"] <= 1


class TestComputeMST:
    def test_mst_has_n_minus_1_edges(self, prices_5):
        cov = estimate_covariance(prices_5)
        mst = compute_mst(cov)
        assert isinstance(mst, nx.Graph)
        assert mst.number_of_nodes() == 5
        assert mst.number_of_edges() == 4  # n-1

    def test_mst_is_tree(self, prices_5):
        cov = estimate_covariance(prices_5)
        mst = compute_mst(cov)
        assert nx.is_tree(mst)

    @pytest.mark.parametrize("algorithm", ["kruskal", "prim"])
    def test_algorithms(self, prices_5, algorithm):
        cov = estimate_covariance(prices_5)
        mst = compute_mst(cov, algorithm=algorithm)
        assert mst.number_of_edges() == 4


class TestMSTUtilities:
    def test_adjacency_matrix(self, prices_5):
        cov = estimate_covariance(prices_5)
        mst = compute_mst(cov)
        adj = mst_to_adjacency(mst)
        assert adj.shape == (5, 5)
        # Symmetric
        np.testing.assert_allclose(adj.values, adj.values.T, atol=1e-10)

    def test_layout(self, prices_5):
        cov = estimate_covariance(prices_5)
        mst = compute_mst(cov)
        pos = get_mst_layout(mst)
        assert len(pos) == 5
        for node, (x, y) in pos.items():
            assert isinstance(x, float)
            assert isinstance(y, float)

    def test_central_node(self, prices_5):
        cov = estimate_covariance(prices_5)
        mst = compute_mst(cov)
        central = get_mst_central_node(mst)
        assert central in list(prices_5.columns)
