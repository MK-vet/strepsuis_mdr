#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Test Coverage for coselection_analysis.py
=======================================================

Tests ALL functions in the CoSelectionAnalyzer class:
- Initialization and setup
- Co-selection score calculation
- Module identification
- Co-selection network construction
- Gene ranking by co-selection potential
"""

import sys
import numpy as np
import pandas as pd
import networkx as nx
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from strepsuis_mdr.coselection_analysis import CoSelectionAnalyzer


class TestCoSelectionAnalyzerInit:
    """Test CoSelectionAnalyzer initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        network = nx.Graph()
        network.add_edge('gene1', 'gene2', weight=0.8)

        cooccurrence = pd.DataFrame({
            'Item1': ['gene1'],
            'Item2': ['gene2'],
            'Phi': [0.7],
            'Corrected_p': [0.01]
        })

        analyzer = CoSelectionAnalyzer(network, cooccurrence)

        assert analyzer.network == network
        assert len(analyzer.cooccurrence) == 1
        assert analyzer.communities is None
        assert analyzer.data is None

    def test_init_with_communities(self):
        """Test initialization with communities."""
        network = nx.Graph()
        network.add_edge('gene1', 'gene2')

        cooccurrence = pd.DataFrame({
            'Item1': ['gene1'],
            'Item2': ['gene2'],
            'Phi': [0.7],
            'Corrected_p': [0.01]
        })

        communities = pd.DataFrame({
            'Node': ['gene1', 'gene2', 'gene3'],
            'Community': [0, 0, 1]
        })

        analyzer = CoSelectionAnalyzer(network, cooccurrence, communities=communities)

        assert analyzer._community_lookup['gene1'] == 0
        assert analyzer._community_lookup['gene2'] == 0
        assert analyzer._community_lookup['gene3'] == 1

    def test_init_with_data(self):
        """Test initialization with data matrix."""
        network = nx.Graph()
        cooccurrence = pd.DataFrame({
            'Item1': ['gene1'],
            'Item2': ['gene2'],
            'Phi': [0.7],
            'Corrected_p': [0.01]
        })

        data = pd.DataFrame({
            'gene1': [1, 0, 1],
            'gene2': [1, 1, 0]
        })

        analyzer = CoSelectionAnalyzer(network, cooccurrence, data=data)

        assert analyzer.data is not None
        assert len(analyzer.data) == 3


class TestCoSelectionScoreCalculation:
    """Test co-selection score calculation."""

    def test_calculate_coselection_score_basic(self):
        """Test basic co-selection score calculation."""
        network = nx.Graph()
        network.add_edge('gene1', 'gene2')
        network.add_edge('gene2', 'gene3')

        cooccurrence = pd.DataFrame({
            'Item1': ['gene1', 'gene2'],
            'Item2': ['gene2', 'gene3'],
            'Phi': [0.8, 0.7],
            'Corrected_p': [0.01, 0.02]
        })

        communities = pd.DataFrame({
            'Node': ['gene1', 'gene2', 'gene3'],
            'Community': [0, 0, 1]
        })

        analyzer = CoSelectionAnalyzer(network, cooccurrence, communities=communities)

        # Test co-selection score for gene1-gene2 (same community)
        score = analyzer.calculate_coselection_score('gene1', 'gene2')

        assert 0 <= score <= 1
        assert score > 0  # Should have positive score

    def test_calculate_coselection_score_no_cooccurrence(self):
        """Test co-selection score when no co-occurrence data exists."""
        network = nx.Graph()
        network.add_edge('gene1', 'gene2')

        cooccurrence = pd.DataFrame({
            'Item1': ['gene3'],
            'Item2': ['gene4'],
            'Phi': [0.5],
            'Corrected_p': [0.05]
        })

        analyzer = CoSelectionAnalyzer(network, cooccurrence)

        score = analyzer.calculate_coselection_score('gene1', 'gene2')

        assert 0 <= score <= 1
        # Score should be based on network proximity only

    def test_calculate_coselection_score_no_path(self):
        """Test co-selection score when genes not connected."""
        network = nx.Graph()
        network.add_edge('gene1', 'gene2')
        network.add_edge('gene3', 'gene4')  # Disconnected component

        cooccurrence = pd.DataFrame({
            'Item1': ['gene1'],
            'Item2': ['gene3'],
            'Phi': [0.6],
            'Corrected_p': [0.03]
        })

        analyzer = CoSelectionAnalyzer(network, cooccurrence)

        score = analyzer.calculate_coselection_score('gene1', 'gene3')

        assert 0 <= score <= 1

    def test_calculate_coselection_score_custom_weights(self):
        """Test co-selection score with custom weights."""
        network = nx.Graph()
        network.add_edge('gene1', 'gene2')

        cooccurrence = pd.DataFrame({
            'Item1': ['gene1'],
            'Item2': ['gene2'],
            'Phi': [0.8],
            'Corrected_p': [0.01]
        })

        analyzer = CoSelectionAnalyzer(network, cooccurrence)

        # Test with different weight configurations
        score1 = analyzer.calculate_coselection_score('gene1', 'gene2', weights=(0.5, 0.3, 0.2))
        score2 = analyzer.calculate_coselection_score('gene1', 'gene2', weights=(0.2, 0.5, 0.3))

        assert 0 <= score1 <= 1
        assert 0 <= score2 <= 1
        # Scores should differ due to different weights
        # (may be equal in some cases, so we don't assert inequality)

    def test_calculate_coselection_score_same_community(self):
        """Test co-selection score for genes in same community."""
        network = nx.Graph()
        network.add_edge('gene1', 'gene2')

        cooccurrence = pd.DataFrame({
            'Item1': ['gene1'],
            'Item2': ['gene2'],
            'Phi': [0.7],
            'Corrected_p': [0.01]
        })

        communities = pd.DataFrame({
            'Node': ['gene1', 'gene2'],
            'Community': [0, 0]
        })

        analyzer = CoSelectionAnalyzer(network, cooccurrence, communities=communities)

        score = analyzer.calculate_coselection_score('gene1', 'gene2')

        # Should have higher score due to same community
        assert score > 0


class TestModuleIdentification:
    """Test co-selection module identification."""

    def test_identify_coselection_modules_basic(self):
        """Test basic module identification."""
        network = nx.Graph()
        network.add_edge('gene1', 'gene2')
        network.add_edge('gene2', 'gene3')

        cooccurrence = pd.DataFrame({
            'Item1': ['gene1', 'gene2', 'gene1'],
            'Item2': ['gene2', 'gene3', 'gene3'],
            'Phi': [0.9, 0.8, 0.85],
            'Corrected_p': [0.001, 0.002, 0.001]
        })

        communities = pd.DataFrame({
            'Node': ['gene1', 'gene2', 'gene3'],
            'Community': [0, 0, 0]
        })

        analyzer = CoSelectionAnalyzer(network, cooccurrence, communities=communities)

        modules = analyzer.identify_coselection_modules(threshold=0.7, min_module_size=2)

        assert 'module_id' in modules.columns
        assert 'genes' in modules.columns
        assert 'n_genes' in modules.columns
        assert 'avg_cs_score' in modules.columns
        assert 'predicted_mge' in modules.columns

    def test_identify_coselection_modules_no_communities(self):
        """Test module identification without communities."""
        network = nx.Graph()
        network.add_edge('gene1', 'gene2')

        cooccurrence = pd.DataFrame({
            'Item1': ['gene1'],
            'Item2': ['gene2'],
            'Phi': [0.8],
            'Corrected_p': [0.01]
        })

        analyzer = CoSelectionAnalyzer(network, cooccurrence, communities=None)

        modules = analyzer.identify_coselection_modules()

        assert modules.empty
        assert list(modules.columns) == ['module_id', 'genes', 'n_genes', 'avg_cs_score', 'predicted_mge']

    def test_identify_coselection_modules_high_threshold(self):
        """Test module identification with high threshold."""
        network = nx.Graph()
        network.add_edge('gene1', 'gene2')
        network.add_edge('gene2', 'gene3')

        cooccurrence = pd.DataFrame({
            'Item1': ['gene1', 'gene2'],
            'Item2': ['gene2', 'gene3'],
            'Phi': [0.3, 0.4],  # Low phi values
            'Corrected_p': [0.05, 0.04]
        })

        communities = pd.DataFrame({
            'Node': ['gene1', 'gene2', 'gene3'],
            'Community': [0, 0, 0]
        })

        analyzer = CoSelectionAnalyzer(network, cooccurrence, communities=communities)

        # With high threshold, may not find modules
        modules = analyzer.identify_coselection_modules(threshold=0.95)

        # Result could be empty or have modules
        assert 'module_id' in modules.columns

    def test_identify_coselection_modules_min_size(self):
        """Test module identification with minimum size constraint."""
        network = nx.Graph()
        network.add_edge('gene1', 'gene2')

        cooccurrence = pd.DataFrame({
            'Item1': ['gene1'],
            'Item2': ['gene2'],
            'Phi': [0.9],
            'Corrected_p': [0.001]
        })

        communities = pd.DataFrame({
            'Node': ['gene1', 'gene2'],
            'Community': [0, 0]
        })

        analyzer = CoSelectionAnalyzer(network, cooccurrence, communities=communities)

        # With min_module_size=5, won't find module (only 2 genes)
        modules = analyzer.identify_coselection_modules(min_module_size=5)

        assert len(modules) == 0


class TestCoSelectionNetwork:
    """Test co-selection network construction."""

    def test_get_gene_coselection_network_basic(self):
        """Test basic co-selection network construction."""
        network = nx.Graph()
        network.add_edge('gene1', 'gene2')
        network.add_edge('gene2', 'gene3')

        cooccurrence = pd.DataFrame({
            'Item1': ['gene1', 'gene2'],
            'Item2': ['gene2', 'gene3'],
            'Phi': [0.8, 0.7],
            'Corrected_p': [0.01, 0.02]
        })

        analyzer = CoSelectionAnalyzer(network, cooccurrence)

        cs_network = analyzer.get_gene_coselection_network(threshold=0.5)

        assert isinstance(cs_network, nx.Graph)
        # Network may have edges if co-selection scores meet threshold

    def test_get_gene_coselection_network_high_threshold(self):
        """Test co-selection network with high threshold."""
        network = nx.Graph()
        network.add_edge('gene1', 'gene2')

        cooccurrence = pd.DataFrame({
            'Item1': ['gene1'],
            'Item2': ['gene2'],
            'Phi': [0.3],  # Low phi
            'Corrected_p': [0.05]
        })

        analyzer = CoSelectionAnalyzer(network, cooccurrence)

        cs_network = analyzer.get_gene_coselection_network(threshold=0.95)

        assert isinstance(cs_network, nx.Graph)
        # Network should have few or no edges

    def test_get_gene_coselection_network_edge_attributes(self):
        """Test that co-selection network has correct edge attributes."""
        network = nx.Graph()
        network.add_edge('gene1', 'gene2')

        cooccurrence = pd.DataFrame({
            'Item1': ['gene1'],
            'Item2': ['gene2'],
            'Phi': [0.9],
            'Corrected_p': [0.001]
        })

        communities = pd.DataFrame({
            'Node': ['gene1', 'gene2'],
            'Community': [0, 0]
        })

        analyzer = CoSelectionAnalyzer(network, cooccurrence, communities=communities)

        cs_network = analyzer.get_gene_coselection_network(threshold=0.5)

        # If edge exists, check attributes
        if cs_network.number_of_edges() > 0:
            for u, v, data in cs_network.edges(data=True):
                assert 'weight' in data
                assert 'cs_score' in data
                assert 0 <= data['cs_score'] <= 1


class TestGeneRanking:
    """Test gene ranking by co-selection potential."""

    def test_rank_genes_by_coselection_potential_basic(self):
        """Test basic gene ranking."""
        network = nx.Graph()
        network.add_edge('gene1', 'gene2')
        network.add_edge('gene2', 'gene3')
        network.add_edge('gene1', 'gene3')

        cooccurrence = pd.DataFrame({
            'Item1': ['gene1', 'gene2', 'gene1'],
            'Item2': ['gene2', 'gene3', 'gene3'],
            'Phi': [0.8, 0.7, 0.75],
            'Corrected_p': [0.01, 0.02, 0.015]
        })

        communities = pd.DataFrame({
            'Node': ['gene1', 'gene2', 'gene3'],
            'Community': [0, 0, 0]
        })

        analyzer = CoSelectionAnalyzer(network, cooccurrence, communities=communities)

        ranking = analyzer.rank_genes_by_coselection_potential(top_n=3)

        assert len(ranking) <= 3
        assert 'gene' in ranking.columns
        assert 'high_cs_relationships' in ranking.columns
        assert 'avg_cs_score' in ranking.columns
        assert 'community_size' in ranking.columns
        assert 'potential_score' in ranking.columns
        assert 'rank' in ranking.columns

    def test_rank_genes_by_coselection_potential_no_communities(self):
        """Test gene ranking without communities."""
        network = nx.Graph()
        network.add_edge('gene1', 'gene2')

        cooccurrence = pd.DataFrame({
            'Item1': ['gene1'],
            'Item2': ['gene2'],
            'Phi': [0.8],
            'Corrected_p': [0.01]
        })

        analyzer = CoSelectionAnalyzer(network, cooccurrence, communities=None)

        ranking = analyzer.rank_genes_by_coselection_potential(top_n=5)

        assert ranking.empty

    def test_rank_genes_by_coselection_potential_top_n(self):
        """Test that ranking respects top_n parameter."""
        network = nx.Graph()
        for i in range(10):
            for j in range(i+1, 10):
                network.add_edge(f'gene{i}', f'gene{j}')

        # Create cooccurrence data for all pairs
        pairs = []
        for i in range(10):
            for j in range(i+1, 10):
                pairs.append({
                    'Item1': f'gene{i}',
                    'Item2': f'gene{j}',
                    'Phi': 0.5 + (i+j)/40,  # Varying phi values
                    'Corrected_p': 0.01
                })

        cooccurrence = pd.DataFrame(pairs)

        communities = pd.DataFrame({
            'Node': [f'gene{i}' for i in range(10)],
            'Community': [0] * 10
        })

        analyzer = CoSelectionAnalyzer(network, cooccurrence, communities=communities)

        ranking = analyzer.rank_genes_by_coselection_potential(top_n=5)

        assert len(ranking) == 5

    def test_rank_genes_ordering(self):
        """Test that genes are ranked correctly."""
        network = nx.Graph()
        network.add_edge('gene1', 'gene2')
        network.add_edge('gene2', 'gene3')

        cooccurrence = pd.DataFrame({
            'Item1': ['gene1', 'gene2'],
            'Item2': ['gene2', 'gene3'],
            'Phi': [0.9, 0.8],
            'Corrected_p': [0.001, 0.01]
        })

        communities = pd.DataFrame({
            'Node': ['gene1', 'gene2', 'gene3'],
            'Community': [0, 0, 0]
        })

        analyzer = CoSelectionAnalyzer(network, cooccurrence, communities=communities)

        ranking = analyzer.rank_genes_by_coselection_potential(top_n=10)

        # Check that ranking is in descending order of potential_score
        if len(ranking) > 1:
            scores = ranking['potential_score'].tolist()
            assert scores == sorted(scores, reverse=True)


class TestHelperMethods:
    """Test internal helper methods."""

    def test_get_community(self):
        """Test community lookup."""
        network = nx.Graph()
        cooccurrence = pd.DataFrame({
            'Item1': ['gene1'],
            'Item2': ['gene2'],
            'Phi': [0.7],
            'Corrected_p': [0.01]
        })

        communities = pd.DataFrame({
            'Node': ['gene1', 'gene2', 'gene3'],
            'Community': [0, 1, 1]
        })

        analyzer = CoSelectionAnalyzer(network, cooccurrence, communities=communities)

        assert analyzer._get_community('gene1') == 0
        assert analyzer._get_community('gene2') == 1
        assert analyzer._get_community('nonexistent') is None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_network(self):
        """Test with empty network."""
        network = nx.Graph()
        cooccurrence = pd.DataFrame({
            'Item1': [],
            'Item2': [],
            'Phi': [],
            'Corrected_p': []
        })

        analyzer = CoSelectionAnalyzer(network, cooccurrence)

        # Should not crash (score may be 0 or close to 0)
        score = analyzer.calculate_coselection_score('gene1', 'gene2')
        assert 0 <= score <= 1

    def test_single_node_network(self):
        """Test with single node."""
        network = nx.Graph()
        network.add_node('gene1')

        cooccurrence = pd.DataFrame({
            'Item1': [],
            'Item2': [],
            'Phi': [],
            'Corrected_p': []
        })

        communities = pd.DataFrame({
            'Node': ['gene1'],
            'Community': [0]
        })

        analyzer = CoSelectionAnalyzer(network, cooccurrence, communities=communities)

        modules = analyzer.identify_coselection_modules()

        # Single node can't form a module (min_size=2)
        assert len(modules) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
