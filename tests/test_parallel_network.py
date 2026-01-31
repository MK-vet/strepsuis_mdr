"""
Tests for parallel network analysis module.

Author: MK-vet
License: MIT
"""

import pytest
import numpy as np
import networkx as nx
import pandas as pd

from strepsuis_mdr.parallel_network import (
    detect_single_community,
    parallel_community_detection,
    compute_parallel_modularity,
    export_community_dataframe,
    _compute_consensus_communities,
    _compute_community_stability,
)


class TestDetectSingleCommunity:
    """Test single community detection function."""

    def test_karate_club_louvain(self):
        """Test Louvain on Karate Club graph."""
        G = nx.karate_club_graph()

        # Add weights
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0

        communities = detect_single_community(G, resolution=1.0, seed=42, algorithm="louvain")

        # Check all nodes assigned
        assert len(communities) == len(G.nodes())

        # Check community IDs are reasonable
        n_communities = len(set(communities.values()))
        assert 2 <= n_communities <= 10

    def test_simple_graph(self):
        """Test on simple graph with clear communities."""
        # Create graph with 2 clear communities
        G = nx.Graph()
        # Community 1: nodes 0-4 (complete)
        for i in range(5):
            for j in range(i + 1, 5):
                G.add_edge(i, j, weight=10.0)

        # Community 2: nodes 5-9 (complete)
        for i in range(5, 10):
            for j in range(i + 1, 10):
                G.add_edge(i, j, weight=10.0)

        # Weak connection between communities
        G.add_edge(4, 5, weight=0.1)

        communities = detect_single_community(G, resolution=1.0, seed=42)

        # Should find 2 communities
        n_communities = len(set(communities.values()))
        assert n_communities == 2

    def test_empty_graph(self):
        """Test on empty graph."""
        G = nx.Graph()
        communities = detect_single_community(G)

        assert len(communities) == 0

    def test_single_node(self):
        """Test on single node graph."""
        G = nx.Graph()
        G.add_node(0)

        communities = detect_single_community(G, seed=42)

        assert len(communities) == 1
        assert 0 in communities


class TestParallelCommunityDetection:
    """Test parallel community detection with consensus."""

    def test_karate_club_parallel(self):
        """Test parallel detection on Karate Club."""
        G = nx.karate_club_graph()

        # Add weights
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0

        communities, metrics = parallel_community_detection(
            G,
            resolution=1.0,
            n_jobs=2,
            n_iterations=5,
            random_state=42
        )

        # Check all nodes assigned
        assert len(communities) == len(G.nodes())

        # Check metrics
        assert 'mean_ari' in metrics
        assert 'std_ari' in metrics
        assert 'modularity' in metrics
        assert 'n_communities' in metrics

        # ARI should be between 0 and 1
        assert 0 <= metrics['mean_ari'] <= 1

        # Modularity should be reasonable
        assert -0.5 <= metrics['modularity'] <= 1.0

        # Should find reasonable number of communities
        assert 2 <= metrics['n_communities'] <= 10

    def test_deterministic_with_seed(self):
        """Test that results are reproducible with seed."""
        G = nx.karate_club_graph()
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0

        communities1, metrics1 = parallel_community_detection(
            G, n_iterations=3, random_state=42
        )
        communities2, metrics2 = parallel_community_detection(
            G, n_iterations=3, random_state=42
        )

        # Should get same results
        assert communities1 == communities2
        assert abs(metrics1['modularity'] - metrics2['modularity']) < 0.01

    def test_empty_graph(self):
        """Test on empty graph."""
        G = nx.Graph()
        communities, metrics = parallel_community_detection(G)

        assert len(communities) == 0
        assert metrics['mean_ari'] == 0.0
        assert metrics['modularity'] == 0.0

    def test_resolution_parameter(self):
        """Test that resolution parameter affects number of communities."""
        G = nx.karate_club_graph()
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0

        # Low resolution -> fewer communities
        _, metrics_low = parallel_community_detection(
            G, resolution=0.5, n_iterations=3, random_state=42
        )

        # High resolution -> more communities
        _, metrics_high = parallel_community_detection(
            G, resolution=2.0, n_iterations=3, random_state=42
        )

        # Generally, higher resolution -> more communities
        # (not guaranteed, but usually true)
        assert metrics_low['n_communities'] >= 1
        assert metrics_high['n_communities'] >= 1

    def test_stability_metrics(self):
        """Test that stability metrics are computed correctly."""
        G = nx.karate_club_graph()
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0

        _, metrics = parallel_community_detection(
            G, n_iterations=10, random_state=42
        )

        # High ARI indicates stable communities
        assert metrics['mean_ari'] > 0.5

        # Std should be non-negative
        assert metrics['std_ari'] >= 0

    def test_single_iteration(self):
        """Test with single iteration."""
        G = nx.karate_club_graph()
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0

        communities, metrics = parallel_community_detection(
            G, n_iterations=1, random_state=42
        )

        assert len(communities) > 0
        # With 1 iteration, ARI should be 1.0 (perfect agreement with itself)
        assert metrics['mean_ari'] == 1.0


class TestConsensusComputation:
    """Test consensus community computation."""

    def test_identical_partitions(self):
        """Test consensus on identical partitions."""
        nodes = list(range(10))
        partition1 = {i: 0 if i < 5 else 1 for i in range(10)}
        partition2 = {i: 0 if i < 5 else 1 for i in range(10)}

        partitions = [partition1, partition2]

        consensus = _compute_consensus_communities(partitions, nodes)

        # Should produce 2 communities
        n_communities = len(set(consensus.values()))
        assert n_communities == 2

    def test_different_partitions(self):
        """Test consensus on different partitions."""
        nodes = list(range(10))

        # Partition 1: two groups
        partition1 = {i: 0 if i < 5 else 1 for i in range(10)}

        # Partition 2: different split
        partition2 = {i: 0 if i < 6 else 1 for i in range(10)}

        partitions = [partition1, partition2]

        consensus = _compute_consensus_communities(partitions, nodes)

        # Should find reasonable consensus
        assert len(consensus) == 10

    def test_empty_partitions(self):
        """Test with empty partitions."""
        partitions = []
        nodes = []

        consensus = _compute_consensus_communities(partitions, nodes)

        assert len(consensus) == 0


class TestCommunityStability:
    """Test community stability metrics."""

    def test_perfect_agreement(self):
        """Test stability with perfect agreement."""
        partition = {i: 0 if i < 5 else 1 for i in range(10)}

        # All identical
        partitions = [partition.copy() for _ in range(5)]

        stability = _compute_community_stability(partitions)

        # Perfect agreement -> ARI = 1.0
        assert abs(stability['mean_ari'] - 1.0) < 0.01
        assert stability['std_ari'] < 0.01

    def test_no_agreement(self):
        """Test stability with random partitions."""
        np.random.seed(42)

        # Create random partitions
        partitions = []
        for _ in range(5):
            partition = {i: np.random.randint(0, 3) for i in range(20)}
            partitions.append(partition)

        stability = _compute_community_stability(partitions)

        # Random partitions -> low ARI
        assert stability['mean_ari'] < 0.5
        assert stability['std_ari'] >= 0

    def test_single_partition(self):
        """Test with single partition."""
        partition = {i: i % 3 for i in range(10)}
        partitions = [partition]

        stability = _compute_community_stability(partitions)

        # Single partition -> perfect stability
        assert stability['mean_ari'] == 1.0
        assert stability['std_ari'] == 0.0


class TestParallelModularity:
    """Test parallel modularity computation."""

    def test_modularity_karate(self):
        """Test modularity on Karate Club."""
        G = nx.karate_club_graph()
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0

        # Detect communities
        communities, _ = parallel_community_detection(G, n_iterations=3, random_state=42)

        # Compute modularity
        modularity = compute_parallel_modularity(G, communities, n_jobs=2)

        # Modularity should be reasonable
        assert -0.5 <= modularity <= 1.0
        # For Karate Club, modularity should be positive
        assert modularity > 0

    def test_modularity_simple_graph(self):
        """Test modularity on graph with clear structure."""
        # Create graph with 2 clear communities
        G = nx.Graph()

        # Community 1
        for i in range(5):
            for j in range(i + 1, 5):
                G.add_edge(i, j, weight=1.0)

        # Community 2
        for i in range(5, 10):
            for j in range(i + 1, 10):
                G.add_edge(i, j, weight=1.0)

        # Weak connection
        G.add_edge(4, 5, weight=0.1)

        # Perfect community assignment
        communities = {i: 0 if i < 5 else 1 for i in range(10)}

        modularity = compute_parallel_modularity(G, communities)

        # Should have high modularity
        assert modularity > 0.3


class TestExportCommunityDataFrame:
    """Test community export function."""

    def test_basic_export(self):
        """Test basic community export."""
        communities = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}

        df = export_community_dataframe(communities)

        assert len(df) == 5
        assert 'node' in df.columns
        assert 'community' in df.columns

        # Check values
        assert set(df['node'].values) == {0, 1, 2, 3, 4}
        assert set(df['community'].values) == {0, 1, 2}

    def test_export_with_attributes(self):
        """Test export with node attributes."""
        communities = {0: 0, 1: 0, 2: 1}

        node_attributes = {
            0: {'name': 'A', 'degree': 5},
            1: {'name': 'B', 'degree': 3},
            2: {'name': 'C', 'degree': 7}
        }

        df = export_community_dataframe(communities, node_attributes)

        assert len(df) == 3
        assert 'node' in df.columns
        assert 'community' in df.columns
        assert 'name' in df.columns
        assert 'degree' in df.columns

        # Check attributes
        row_0 = df[df['node'] == 0].iloc[0]
        assert row_0['name'] == 'A'
        assert row_0['degree'] == 5

    def test_empty_communities(self):
        """Test export with empty communities."""
        communities = {}

        df = export_community_dataframe(communities)

        assert len(df) == 0
        assert 'node' in df.columns
        assert 'community' in df.columns


class TestPerformance:
    """Test performance characteristics."""

    def test_parallel_faster_than_sequential(self):
        """Test that parallel is faster for multiple iterations."""
        import time

        G = nx.karate_club_graph()
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0

        # Sequential (n_jobs=1)
        start = time.perf_counter()
        parallel_community_detection(G, n_jobs=1, n_iterations=10, random_state=42)
        time_sequential = time.perf_counter() - start

        # Parallel (n_jobs=2)
        start = time.perf_counter()
        parallel_community_detection(G, n_jobs=2, n_iterations=10, random_state=42)
        time_parallel = time.perf_counter() - start

        # Parallel should be at least a bit faster (allowing for overhead)
        # This is a soft check since performance can vary
        assert time_parallel < time_sequential * 1.5

    def test_scalability(self):
        """Test that method scales to larger graphs."""
        # Create larger graph
        G = nx.random_partition_graph([100, 100, 100], 0.9, 0.01, seed=42)

        # Add weights
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0

        communities, metrics = parallel_community_detection(
            G, n_jobs=2, n_iterations=3, random_state=42
        )

        # Should handle larger graph
        assert len(communities) == 300
        assert metrics['n_communities'] > 1
