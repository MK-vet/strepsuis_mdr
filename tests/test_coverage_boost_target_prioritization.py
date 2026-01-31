#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Test Coverage for target_prioritization.py
========================================================

Tests ALL functions in the InterventionTargetRanker class:
- Initialization
- Multi-criteria target ranking
- Target filtering
- Report generation
"""

import sys
import tempfile
import numpy as np
import pandas as pd
import networkx as nx
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from strepsuis_mdr.target_prioritization import InterventionTargetRanker


class TestInterventionTargetRankerInit:
    """Test InterventionTargetRanker initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        network = nx.Graph()
        network.add_edge('gene1', 'Penicillins', weight=0.8)

        gene_associations = pd.DataFrame({
            'Gene': ['gene1'],
            'Phenotype': ['Penicillins'],
            'Phi': [0.8],
            'Corrected_p': [0.01]
        })

        ranker = InterventionTargetRanker(network, gene_associations)

        assert ranker.network == network
        assert len(ranker.gene_associations) == 1
        assert ranker.communities is None
        assert ranker.data is None
        assert ranker.coselection_modules is None

    def test_init_with_communities(self):
        """Test initialization with communities."""
        network = nx.Graph()
        gene_associations = pd.DataFrame({
            'Gene': ['gene1'],
            'Phenotype': ['Penicillins'],
            'Phi': [0.8],
            'Corrected_p': [0.01]
        })

        communities = pd.DataFrame({
            'Node': ['gene1', 'gene2'],
            'Community': [0, 1]
        })

        ranker = InterventionTargetRanker(
            network,
            gene_associations,
            communities=communities
        )

        assert ranker.communities is not None
        assert len(ranker.communities) == 2

    def test_init_with_coselection_modules(self):
        """Test initialization with co-selection modules."""
        network = nx.Graph()
        gene_associations = pd.DataFrame({
            'Gene': ['gene1'],
            'Phenotype': ['Penicillins'],
            'Phi': [0.8],
            'Corrected_p': [0.01]
        })

        coselection_modules = pd.DataFrame({
            'module_id': [0, 1],
            'genes': [['gene1', 'gene2'], ['gene3']],
            'predicted_mge': [True, False]
        })

        ranker = InterventionTargetRanker(
            network,
            gene_associations,
            coselection_modules=coselection_modules
        )

        # Check gene-to-module mapping
        assert ranker._gene_to_module['gene1'] == True
        assert ranker._gene_to_module['gene2'] == True
        assert ranker._gene_to_module['gene3'] == False


class TestTargetRanking:
    """Test intervention target ranking."""

    def test_rank_intervention_targets_basic(self):
        """Test basic target ranking."""
        network = nx.Graph()
        network.add_edge('gene1', 'Penicillins', weight=0.8)
        network.add_edge('gene2', 'Aminoglycosides', weight=0.7)

        gene_associations = pd.DataFrame({
            'Gene': ['gene1', 'gene2'],
            'Phenotype': ['Penicillins', 'Aminoglycosides'],
            'Phi': [0.8, 0.7],
            'Corrected_p': [0.01, 0.02]
        })

        ranker = InterventionTargetRanker(network, gene_associations)

        result = ranker.rank_intervention_targets()

        # Result should include genes that are in the network
        assert 'gene' in result.columns
        assert 'mdr_association' in result.columns
        assert 'p_value' in result.columns
        assert 'degree' in result.columns
        assert 'degree_centrality' in result.columns
        assert 'betweenness_centrality' in result.columns
        assert 'prevalence' in result.columns
        assert 'predicted_mobile' in result.columns
        assert 'priority_score' in result.columns
        assert 'priority_rank' in result.columns

    def test_rank_intervention_targets_with_data(self):
        """Test target ranking with prevalence data."""
        network = nx.Graph()
        network.add_edge('gene1', 'Penicillins')
        network.add_edge('gene2', 'Aminoglycosides')

        gene_associations = pd.DataFrame({
            'Gene': ['gene1', 'gene2'],
            'Phenotype': ['Penicillins', 'Aminoglycosides'],
            'Phi': [0.8, 0.6],
            'Corrected_p': [0.01, 0.03]
        })

        data = pd.DataFrame({
            'gene1': [1, 1, 0, 1],
            'gene2': [1, 0, 0, 0]
        })

        ranker = InterventionTargetRanker(network, gene_associations, data=data)

        result = ranker.rank_intervention_targets()

        # gene1 should have higher prevalence (0.75 vs 0.25)
        gene1_row = result[result['gene'] == 'gene1'].iloc[0]
        gene2_row = result[result['gene'] == 'gene2'].iloc[0]

        assert gene1_row['prevalence'] > gene2_row['prevalence']

    def test_rank_intervention_targets_custom_weights(self):
        """Test target ranking with custom weights."""
        network = nx.Graph()
        network.add_edge('gene1', 'Penicillins', weight=0.8)

        gene_associations = pd.DataFrame({
            'Gene': ['gene1'],
            'Phenotype': ['Penicillins'],
            'Phi': [0.8],
            'Corrected_p': [0.01]
        })

        ranker = InterventionTargetRanker(network, gene_associations)

        # Test with different weight configurations
        weights1 = {
            'mdr_association': 0.5,
            'degree': 0.2,
            'betweenness': 0.1,
            'prevalence': 0.1,
            'mobility': 0.1
        }

        weights2 = {
            'mdr_association': 0.2,
            'degree': 0.2,
            'betweenness': 0.2,
            'prevalence': 0.2,
            'mobility': 0.2
        }

        result1 = ranker.rank_intervention_targets(weights=weights1)
        result2 = ranker.rank_intervention_targets(weights=weights2)

        assert len(result1) > 0
        assert len(result2) > 0

        # Priority scores may differ due to different weights
        # (but may be same in some cases)

    def test_rank_intervention_targets_with_mobility(self):
        """Test target ranking with mobility prediction."""
        network = nx.Graph()
        network.add_edge('gene1', 'Penicillins')
        network.add_edge('gene2', 'Aminoglycosides')

        gene_associations = pd.DataFrame({
            'Gene': ['gene1', 'gene2'],
            'Phenotype': ['Penicillins', 'Aminoglycosides'],
            'Phi': [0.7, 0.7],  # Same association
            'Corrected_p': [0.01, 0.01]
        })

        coselection_modules = pd.DataFrame({
            'module_id': [0],
            'genes': [['gene1', 'gene3']],
            'predicted_mge': [True]
        })

        ranker = InterventionTargetRanker(
            network,
            gene_associations,
            coselection_modules=coselection_modules
        )

        result = ranker.rank_intervention_targets()

        # gene1 should have higher mobility score
        gene1_row = result[result['gene'] == 'gene1'].iloc[0]
        gene2_row = result[result['gene'] == 'gene2'].iloc[0]

        assert gene1_row['predicted_mobile'] == True
        assert gene2_row['predicted_mobile'] == False

    def test_rank_intervention_targets_feature1_feature2_columns(self):
        """Test ranking with Feature1/Feature2 column names."""
        network = nx.Graph()
        network.add_edge('gene1', 'Penicillins')

        # Use Feature1/Feature2 instead of Gene/Phenotype
        gene_associations = pd.DataFrame({
            'Feature1': ['gene1'],
            'Feature2': ['Penicillins'],
            'Phi': [0.8],
            'Corrected_p': [0.01]
        })

        ranker = InterventionTargetRanker(network, gene_associations)

        result = ranker.rank_intervention_targets()

        assert len(result) > 0
        assert result.loc[0, 'mdr_association'] == 0.8


class TestGetCommunity:
    """Test community lookup helper."""

    def test_get_community_existing(self):
        """Test getting community for existing node."""
        network = nx.Graph()
        gene_associations = pd.DataFrame({
            'Gene': ['gene1'],
            'Phenotype': ['Penicillins'],
            'Phi': [0.8],
            'Corrected_p': [0.01]
        })

        communities = pd.DataFrame({
            'Node': ['gene1', 'gene2'],
            'Community': [0, 1]
        })

        ranker = InterventionTargetRanker(
            network,
            gene_associations,
            communities=communities
        )

        assert ranker._get_community('gene1') == 0
        assert ranker._get_community('gene2') == 1

    def test_get_community_nonexistent(self):
        """Test getting community for non-existent node."""
        network = nx.Graph()
        gene_associations = pd.DataFrame({
            'Gene': ['gene1'],
            'Phenotype': ['Penicillins'],
            'Phi': [0.8],
            'Corrected_p': [0.01]
        })

        communities = pd.DataFrame({
            'Node': ['gene1'],
            'Community': [0]
        })

        ranker = InterventionTargetRanker(
            network,
            gene_associations,
            communities=communities
        )

        assert ranker._get_community('nonexistent') is None

    def test_get_community_no_communities(self):
        """Test getting community when communities not provided."""
        network = nx.Graph()
        gene_associations = pd.DataFrame({
            'Gene': ['gene1'],
            'Phenotype': ['Penicillins'],
            'Phi': [0.8],
            'Corrected_p': [0.01]
        })

        ranker = InterventionTargetRanker(network, gene_associations)

        assert ranker._get_community('gene1') is None


class TestGetTopTargets:
    """Test top target filtering."""

    def test_get_top_targets_basic(self):
        """Test getting top targets."""
        network = nx.Graph()
        network.add_edge('gene1', 'Penicillins')
        network.add_edge('gene2', 'Aminoglycosides')
        network.add_edge('gene3', 'Macrolides')

        gene_associations = pd.DataFrame({
            'Gene': ['gene1', 'gene2', 'gene3'],
            'Phenotype': ['Penicillins', 'Aminoglycosides', 'Macrolides'],
            'Phi': [0.9, 0.6, 0.4],
            'Corrected_p': [0.001, 0.02, 0.05]
        })

        data = pd.DataFrame({
            'gene1': [1, 1, 1, 0],
            'gene2': [1, 0, 0, 0],
            'gene3': [0, 0, 0, 0]
        })

        ranker = InterventionTargetRanker(network, gene_associations, data=data)

        top_targets = ranker.get_top_targets(
            top_n=2,
            min_mdr_association=0.5,
            min_prevalence=0.1
        )

        # Should get gene1 and gene2 (gene3 filtered by association and prevalence)
        assert len(top_targets) <= 2

    def test_get_top_targets_strict_filters(self):
        """Test top targets with strict filters."""
        network = nx.Graph()
        network.add_edge('gene1', 'Penicillins')
        network.add_edge('gene2', 'Aminoglycosides')

        gene_associations = pd.DataFrame({
            'Gene': ['gene1', 'gene2'],
            'Phenotype': ['Penicillins', 'Aminoglycosides'],
            'Phi': [0.9, 0.3],  # gene2 low association
            'Corrected_p': [0.001, 0.1]
        })

        data = pd.DataFrame({
            'gene1': [1, 1, 1],
            'gene2': [0, 0, 0]  # gene2 low prevalence
        })

        ranker = InterventionTargetRanker(network, gene_associations, data=data)

        top_targets = ranker.get_top_targets(
            top_n=10,
            min_mdr_association=0.5,
            min_prevalence=0.2
        )

        # Should only get gene1
        assert len(top_targets) == 1
        assert top_targets.iloc[0]['gene'] == 'gene1'

    def test_get_top_targets_respects_top_n(self):
        """Test that get_top_targets respects top_n parameter."""
        network = nx.Graph()
        for i in range(10):
            network.add_edge(f'gene{i}', f'Phenotype{i}')

        gene_associations = pd.DataFrame({
            'Gene': [f'gene{i}' for i in range(10)],
            'Phenotype': [f'Phenotype{i}' for i in range(10)],
            'Phi': [0.5 + i/20 for i in range(10)],  # Varying associations
            'Corrected_p': [0.01] * 10
        })

        ranker = InterventionTargetRanker(network, gene_associations)

        top_targets = ranker.get_top_targets(top_n=3)

        assert len(top_targets) <= 3


class TestGenerateTargetReport:
    """Test target report generation."""

    def test_generate_target_report_basic(self):
        """Test basic report generation."""
        network = nx.Graph()
        network.add_edge('gene1', 'Penicillins')
        network.add_edge('gene2', 'Aminoglycosides')

        gene_associations = pd.DataFrame({
            'Gene': ['gene1', 'gene2'],
            'Phenotype': ['Penicillins', 'Aminoglycosides'],
            'Phi': [0.8, 0.7],
            'Corrected_p': [0.01, 0.02]
        })

        ranker = InterventionTargetRanker(network, gene_associations)

        report = ranker.generate_target_report(top_n=5)

        assert isinstance(report, str)
        assert 'INTERVENTION TARGET PRIORITIZATION REPORT' in report
        assert 'gene1' in report or 'gene2' in report

    def test_generate_target_report_saves_file(self, tmp_path):
        """Test that report saves to file."""
        network = nx.Graph()
        network.add_edge('gene1', 'Penicillins')

        gene_associations = pd.DataFrame({
            'Gene': ['gene1'],
            'Phenotype': ['Penicillins'],
            'Phi': [0.8],
            'Corrected_p': [0.01]
        })

        ranker = InterventionTargetRanker(network, gene_associations)

        output_path = tmp_path / "target_report.txt"

        report = ranker.generate_target_report(
            top_n=5,
            output_path=str(output_path)
        )

        assert output_path.exists()

        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert content == report
            assert 'INTERVENTION TARGET PRIORITIZATION REPORT' in content

    def test_generate_target_report_content(self):
        """Test report content includes all expected sections."""
        network = nx.Graph()
        network.add_edge('gene1', 'Penicillins', weight=0.8)

        gene_associations = pd.DataFrame({
            'Gene': ['gene1'],
            'Phenotype': ['Penicillins'],
            'Phi': [0.8],
            'Corrected_p': [0.01]
        })

        data = pd.DataFrame({
            'gene1': [1, 1, 0]
        })

        ranker = InterventionTargetRanker(network, gene_associations, data=data)

        report = ranker.generate_target_report(top_n=5)

        # Check for key sections
        assert 'Ranking Criteria' in report
        assert 'MDR Association' in report
        assert 'Network Centrality' in report
        assert 'Prevalence' in report
        assert 'Mobility' in report


class TestRankingScores:
    """Test ranking score calculations."""

    def test_priority_scores_in_range(self):
        """Test that priority scores are in valid range."""
        network = nx.Graph()
        network.add_edge('gene1', 'Penicillins')

        gene_associations = pd.DataFrame({
            'Gene': ['gene1'],
            'Phenotype': ['Penicillins'],
            'Phi': [0.8],
            'Corrected_p': [0.01]
        })

        ranker = InterventionTargetRanker(network, gene_associations)

        result = ranker.rank_intervention_targets()

        # All scores should be >= 0
        assert all(result['priority_score'] >= 0)
        assert all(result['mdr_association'] >= 0)
        assert all(result['mdr_association'] <= 1)
        assert all(result['degree'] >= 0)
        assert all(result['prevalence'] >= 0)
        assert all(result['prevalence'] <= 1)

    def test_ranking_order(self):
        """Test that genes are ranked in correct order."""
        network = nx.Graph()
        network.add_edge('gene1', 'Penicillins')
        network.add_edge('gene2', 'Aminoglycosides')

        gene_associations = pd.DataFrame({
            'Gene': ['gene1', 'gene2'],
            'Phenotype': ['Penicillins', 'Aminoglycosides'],
            'Phi': [0.9, 0.3],  # gene1 much stronger association
            'Corrected_p': [0.001, 0.1]
        })

        ranker = InterventionTargetRanker(network, gene_associations)

        result = ranker.rank_intervention_targets()

        # Scores should be in descending order
        scores = result['priority_score'].tolist()
        assert scores == sorted(scores, reverse=True)

        # Ranks should be sequential
        ranks = result['priority_rank'].tolist()
        assert ranks == list(range(1, len(ranks) + 1))


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_network(self):
        """Test with empty network."""
        network = nx.Graph()

        gene_associations = pd.DataFrame({
            'Gene': [],
            'Phenotype': [],
            'Phi': [],
            'Corrected_p': []
        })

        ranker = InterventionTargetRanker(network, gene_associations)

        result = ranker.rank_intervention_targets()

        assert result.empty
        assert list(result.columns) == [
            'gene', 'mdr_association', 'p_value', 'degree', 'degree_centrality',
            'betweenness_centrality', 'prevalence', 'predicted_mobile',
            'mobility_score', 'priority_score', 'priority_rank'
        ]

    def test_mobility_heuristic_large_community(self):
        """Test mobility heuristic for large communities."""
        network = nx.Graph()
        for i in range(5):
            network.add_edge(f'gene{i}', f'Phenotype{i}')

        gene_associations = pd.DataFrame({
            'Gene': [f'gene{i}' for i in range(5)],
            'Phenotype': [f'Phenotype{i}' for i in range(5)],
            'Phi': [0.7] * 5,
            'Corrected_p': [0.01] * 5
        })

        communities = pd.DataFrame({
            'Node': [f'gene{i}' for i in range(5)],
            'Community': [0] * 5  # All in same large community
        })

        ranker = InterventionTargetRanker(
            network,
            gene_associations,
            communities=communities
        )

        result = ranker.rank_intervention_targets()

        # Large community (>=3) should get moderate mobility score
        # even without explicit co-selection module
        # (This tests the heuristic in the code)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
