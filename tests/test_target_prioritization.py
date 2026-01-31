#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for intervention target prioritization module.

Tests multi-criteria ranking for drug/vaccine targets.
"""

import numpy as np
import pandas as pd
import pytest
import networkx as nx

from strepsuis_mdr.target_prioritization import InterventionTargetRanker


class TestInterventionTargetRanker:
    """Test InterventionTargetRanker class."""
    
    @pytest.fixture
    def sample_network(self):
        """Create sample network."""
        G = nx.Graph()
        G.add_edge("Gene_A", "Gene_B", weight=0.5)
        G.add_edge("Gene_B", "Gene_C", weight=0.6)
        G.add_node("Gene_D")
        return G
    
    @pytest.fixture
    def sample_gene_associations(self):
        """Create sample gene associations."""
        return pd.DataFrame({
            'Gene': ['Gene_A', 'Gene_B', 'Gene_C'],
            'Phenotype': ['MDR', 'MDR', 'Resistance'],
            'Phi': [0.7, 0.6, 0.5],
            'Corrected_p': [0.01, 0.02, 0.03],
        })
    
    @pytest.fixture
    def sample_communities(self):
        """Create sample communities."""
        return pd.DataFrame({
            'Node': ['Gene_A', 'Gene_B', 'Gene_C', 'Gene_D'],
            'Community': [0, 0, 1, 1],
        })
    
    @pytest.fixture
    def sample_data(self):
        """Create sample binary data."""
        return pd.DataFrame({
            'Gene_A': [1, 0, 1, 0, 1] * 20,
            'Gene_B': [0, 1, 1, 0, 0] * 20,
            'Gene_C': [1, 1, 0, 1, 0] * 20,
            'Gene_D': [0, 0, 1, 1, 0] * 20,
        })
    
    @pytest.fixture
    def sample_coselection_modules(self):
        """Create sample co-selection modules."""
        return pd.DataFrame({
            'module_id': [0, 1],
            'genes': [['Gene_A', 'Gene_B'], ['Gene_C']],
            'n_genes': [2, 1],
            'avg_cs_score': [0.8, 0.6],
            'predicted_mge': [True, False],
        })
    
    def test_initialization(
        self, sample_network, sample_gene_associations, sample_communities, sample_data
    ):
        """Test ranker initialization."""
        ranker = InterventionTargetRanker(
            network=sample_network,
            gene_associations=sample_gene_associations,
            communities=sample_communities,
            data=sample_data,
        )
        
        assert ranker.network.number_of_nodes() == 4
        assert len(ranker.gene_associations) == 3
    
    def test_rank_intervention_targets(
        self, sample_network, sample_gene_associations, sample_communities, sample_data
    ):
        """Test target ranking."""
        ranker = InterventionTargetRanker(
            network=sample_network,
            gene_associations=sample_gene_associations,
            communities=sample_communities,
            data=sample_data,
        )
        
        rankings = ranker.rank_intervention_targets()
        
        assert isinstance(rankings, pd.DataFrame)
        assert len(rankings) > 0
        assert 'gene' in rankings.columns
        assert 'priority_score' in rankings.columns
        assert 'priority_rank' in rankings.columns
        assert 'mdr_association' in rankings.columns
    
    def test_rank_intervention_targets_with_coselection(
        self, sample_network, sample_gene_associations, sample_communities,
        sample_data, sample_coselection_modules
    ):
        """Test ranking with co-selection modules."""
        ranker = InterventionTargetRanker(
            network=sample_network,
            gene_associations=sample_gene_associations,
            communities=sample_communities,
            data=sample_data,
            coselection_modules=sample_coselection_modules,
        )
        
        rankings = ranker.rank_intervention_targets()
        
        assert isinstance(rankings, pd.DataFrame)
        # Genes in MGE modules should have mobility_score > 0
        gene_a_ranking = rankings[rankings['gene'] == 'Gene_A']
        if not gene_a_ranking.empty:
            assert 'predicted_mobile' in rankings.columns
    
    def test_get_top_targets(
        self, sample_network, sample_gene_associations, sample_communities, sample_data
    ):
        """Test getting top targets with filtering."""
        ranker = InterventionTargetRanker(
            network=sample_network,
            gene_associations=sample_gene_associations,
            communities=sample_communities,
            data=sample_data,
        )
        
        top_targets = ranker.get_top_targets(top_n=5, min_mdr_association=0.3, min_prevalence=0.1)
        
        assert isinstance(top_targets, pd.DataFrame)
        assert len(top_targets) <= 5
        if not top_targets.empty:
            assert all(top_targets['mdr_association'] >= 0.3)
            assert all(top_targets['prevalence'] >= 0.1)
    
    def test_generate_target_report(
        self, sample_network, sample_gene_associations, sample_communities, sample_data, tmp_path
    ):
        """Test target report generation."""
        ranker = InterventionTargetRanker(
            network=sample_network,
            gene_associations=sample_gene_associations,
            communities=sample_communities,
            data=sample_data,
        )
        
        output_path = tmp_path / "target_report.txt"
        report = ranker.generate_target_report(top_n=5, output_path=str(output_path))
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "INTERVENTION TARGET" in report
        assert output_path.exists()
    
    def test_edge_cases_empty_associations(
        self, sample_network, sample_communities, sample_data
    ):
        """Test with empty gene associations."""
        empty_assoc = pd.DataFrame(columns=['Gene', 'Phenotype', 'Phi', 'Corrected_p'])
        ranker = InterventionTargetRanker(
            network=sample_network,
            gene_associations=empty_assoc,
            communities=sample_communities,
            data=sample_data,
        )
        
        rankings = ranker.rank_intervention_targets()
        
        assert isinstance(rankings, pd.DataFrame)
        # Should still rank based on network centrality and prevalence
        assert len(rankings) > 0
    
    def test_edge_cases_empty_network(
        self, sample_gene_associations, sample_communities, sample_data
    ):
        """Test with empty network."""
        empty_network = nx.Graph()
        ranker = InterventionTargetRanker(
            network=empty_network,
            gene_associations=sample_gene_associations,
            communities=sample_communities,
            data=sample_data,
        )
        
        rankings = ranker.rank_intervention_targets()
        
        assert isinstance(rankings, pd.DataFrame)
        # Should handle gracefully
        assert len(rankings) == 0
    
    def test_priority_scores_are_sorted(
        self, sample_network, sample_gene_associations, sample_communities, sample_data
    ):
        """Test that priority scores are properly sorted."""
        ranker = InterventionTargetRanker(
            network=sample_network,
            gene_associations=sample_gene_associations,
            communities=sample_communities,
            data=sample_data,
        )
        
        rankings = ranker.rank_intervention_targets()
        
        if len(rankings) > 1:
            # Should be sorted descending by priority_score
            scores = rankings['priority_score'].values
            assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
            
            # Ranks should be sequential
            ranks = rankings['priority_rank'].values
            assert all(ranks[i] == i+1 for i in range(len(ranks)))
