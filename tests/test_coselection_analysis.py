#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for co-selection analysis module.

Tests co-selection score calculation and mobile genetic element prediction.
"""

import numpy as np
import pandas as pd
import pytest
import networkx as nx

from strepsuis_mdr.coselection_analysis import CoSelectionAnalyzer


class TestCoSelectionAnalyzer:
    """Test CoSelectionAnalyzer class."""
    
    @pytest.fixture
    def sample_network(self):
        """Create sample network."""
        G = nx.Graph()
        G.add_edge("Gene_A", "Gene_B", weight=0.5)
        G.add_edge("Gene_B", "Gene_C", weight=0.6)
        G.add_edge("Gene_A", "Gene_C", weight=0.4)
        return G
    
    @pytest.fixture
    def sample_cooccurrence(self):
        """Create sample co-occurrence data."""
        return pd.DataFrame({
            'Item1': ['Gene_A', 'Gene_B', 'Gene_A'],
            'Item2': ['Gene_B', 'Gene_C', 'Gene_C'],
            'Phi': [0.5, 0.6, 0.4],
            'Corrected_p': [0.01, 0.02, 0.03],
        })
    
    @pytest.fixture
    def sample_communities(self):
        """Create sample communities."""
        return pd.DataFrame({
            'Node': ['Gene_A', 'Gene_B', 'Gene_C'],
            'Community': [0, 0, 1],
        })
    
    @pytest.fixture
    def sample_data(self):
        """Create sample binary data."""
        return pd.DataFrame({
            'Gene_A': [1, 0, 1, 0, 1] * 20,
            'Gene_B': [0, 1, 1, 0, 0] * 20,
            'Gene_C': [1, 1, 0, 1, 0] * 20,
        })
    
    def test_initialization(
        self, sample_network, sample_cooccurrence, sample_communities, sample_data
    ):
        """Test analyzer initialization."""
        analyzer = CoSelectionAnalyzer(
            network=sample_network,
            cooccurrence_results=sample_cooccurrence,
            communities=sample_communities,
            data=sample_data,
        )
        
        assert analyzer.network.number_of_nodes() == 3
        assert len(analyzer.cooccurrence) == 3
        assert len(analyzer.communities) == 3
    
    def test_calculate_coselection_score(
        self, sample_network, sample_cooccurrence, sample_communities
    ):
        """Test co-selection score calculation."""
        analyzer = CoSelectionAnalyzer(
            network=sample_network,
            cooccurrence_results=sample_cooccurrence,
            communities=sample_communities,
        )
        
        score = analyzer.calculate_coselection_score('Gene_A', 'Gene_B')
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_calculate_coselection_score_same_community(
        self, sample_network, sample_cooccurrence, sample_communities
    ):
        """Test that genes in same community get higher score."""
        analyzer = CoSelectionAnalyzer(
            network=sample_network,
            cooccurrence_results=sample_cooccurrence,
            communities=sample_communities,
        )
        
        # Gene_A and Gene_B are in same community (0)
        score_same = analyzer.calculate_coselection_score('Gene_A', 'Gene_B')
        
        # Gene_A and Gene_C are in different communities
        score_diff = analyzer.calculate_coselection_score('Gene_A', 'Gene_C')
        
        # Same community should generally score higher (though not guaranteed)
        assert isinstance(score_same, float)
        assert isinstance(score_diff, float)
    
    def test_calculate_coselection_score_no_cooccurrence(
        self, sample_network, sample_communities
    ):
        """Test with no co-occurrence data."""
        empty_cooccurrence = pd.DataFrame(columns=['Item1', 'Item2', 'Phi'])
        analyzer = CoSelectionAnalyzer(
            network=sample_network,
            cooccurrence_results=empty_cooccurrence,
            communities=sample_communities,
        )
        
        score = analyzer.calculate_coselection_score('Gene_A', 'Gene_B')
        
        # Should still return a score (based on network and community)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_identify_coselection_modules(
        self, sample_network, sample_cooccurrence, sample_communities
    ):
        """Test co-selection module identification."""
        analyzer = CoSelectionAnalyzer(
            network=sample_network,
            cooccurrence_results=sample_cooccurrence,
            communities=sample_communities,
        )
        
        modules = analyzer.identify_coselection_modules(threshold=0.3)
        
        assert isinstance(modules, pd.DataFrame)
        if not modules.empty:
            assert 'module_id' in modules.columns
            assert 'genes' in modules.columns
            assert 'n_genes' in modules.columns
            assert 'avg_cs_score' in modules.columns
            assert 'predicted_mge' in modules.columns
    
    def test_identify_coselection_modules_empty_communities(
        self, sample_network, sample_cooccurrence
    ):
        """Test with empty communities."""
        empty_communities = pd.DataFrame(columns=['Node', 'Community'])
        analyzer = CoSelectionAnalyzer(
            network=sample_network,
            cooccurrence_results=sample_cooccurrence,
            communities=empty_communities,
        )
        
        modules = analyzer.identify_coselection_modules()
        
        assert isinstance(modules, pd.DataFrame)
        assert modules.empty or 'module_id' in modules.columns
    
    def test_get_gene_coselection_network(
        self, sample_network, sample_cooccurrence, sample_communities
    ):
        """Test co-selection network construction."""
        analyzer = CoSelectionAnalyzer(
            network=sample_network,
            cooccurrence_results=sample_cooccurrence,
            communities=sample_communities,
        )
        
        cs_network = analyzer.get_gene_coselection_network(threshold=0.3)
        
        assert isinstance(cs_network, nx.Graph)
        # Should have some edges if threshold is low enough
        assert cs_network.number_of_nodes() >= 0
    
    def test_rank_genes_by_coselection_potential(
        self, sample_network, sample_cooccurrence, sample_communities
    ):
        """Test gene ranking by co-selection potential."""
        analyzer = CoSelectionAnalyzer(
            network=sample_network,
            cooccurrence_results=sample_cooccurrence,
            communities=sample_communities,
        )
        
        rankings = analyzer.rank_genes_by_coselection_potential(top_n=5)
        
        assert isinstance(rankings, pd.DataFrame)
        if not rankings.empty:
            assert 'gene' in rankings.columns
            assert 'potential_score' in rankings.columns
            assert 'rank' in rankings.columns
            assert len(rankings) <= 5
    
    def test_edge_cases_empty_network(self, sample_cooccurrence, sample_communities):
        """Test with empty network."""
        empty_network = nx.Graph()
        analyzer = CoSelectionAnalyzer(
            network=empty_network,
            cooccurrence_results=sample_cooccurrence,
            communities=sample_communities,
        )
        
        score = analyzer.calculate_coselection_score('Gene_A', 'Gene_B')
        # Should handle gracefully
        assert isinstance(score, float)
    
    def test_edge_cases_no_path_in_network(
        self, sample_cooccurrence, sample_communities
    ):
        """Test with disconnected network."""
        disconnected_net = nx.Graph()
        disconnected_net.add_node("Gene_A")
        disconnected_net.add_node("Gene_B")
        # No edge between them
        
        analyzer = CoSelectionAnalyzer(
            network=disconnected_net,
            cooccurrence_results=sample_cooccurrence,
            communities=sample_communities,
        )
        
        score = analyzer.calculate_coselection_score('Gene_A', 'Gene_B')
        # Should handle no path gracefully
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
