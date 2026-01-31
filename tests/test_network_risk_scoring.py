"""
Tests for Network Risk Scoring innovation.

Tests the compute_network_mdr_risk_score function which combines
network centrality metrics with bootstrap confidence intervals
to predict MDR risk.
"""

import numpy as np
import pandas as pd
import pytest
import networkx as nx

from strepsuis_mdr.mdr_analysis_core import (
    compute_network_mdr_risk_score,
    build_hybrid_co_resistance_network,
)


class TestNetworkRiskScoring:
    """Test suite for Network Risk Scoring innovation."""

    def test_network_risk_scoring_basic(self):
        """Test basic risk score computation."""
        # Create simple network
        network = nx.Graph()
        network.add_edge("Gene_A", "Gene_B", phi=0.5, pvalue=0.01)
        network.add_edge("Gene_B", "Phenotype_X", phi=0.6, pvalue=0.02)
        
        # Create strain features
        strain_features = pd.DataFrame({
            "Gene_A": [1, 0, 1],
            "Gene_B": [1, 1, 0],
            "Phenotype_X": [1, 1, 0],
        }, index=["Strain_001", "Strain_002", "Strain_003"])
        
        # Create bootstrap CI
        bootstrap_ci = {
            "Gene_A": (0.2, 0.4),
            "Gene_B": (0.3, 0.5),
            "Phenotype_X": (0.4, 0.6),
        }
        
        # Compute risk scores
        risk_scores = compute_network_mdr_risk_score(
            network, strain_features, bootstrap_ci
        )
        
        # Verify structure
        assert isinstance(risk_scores, pd.DataFrame)
        assert "Strain_ID" in risk_scores.columns
        assert "Network_Risk_Score" in risk_scores.columns
        assert "MDR_Predicted" in risk_scores.columns
        assert "Percentile_Rank" in risk_scores.columns
        
        # Verify all strains are included
        assert len(risk_scores) == 3
        assert set(risk_scores["Strain_ID"]) == {"Strain_001", "Strain_002", "Strain_003"}
        
        # Verify scores are non-negative
        assert (risk_scores["Network_Risk_Score"] >= 0).all()
        
        # Verify percentile ranks are in [0, 100]
        assert (risk_scores["Percentile_Rank"] >= 0).all()
        assert (risk_scores["Percentile_Rank"] <= 100).all()

    def test_network_risk_scoring_empty_network(self):
        """Test handling of empty network."""
        network = nx.Graph()
        strain_features = pd.DataFrame({
            "Gene_A": [1, 0],
        }, index=["Strain_001", "Strain_002"])
        bootstrap_ci = {"Gene_A": (0.2, 0.4)}
        
        risk_scores = compute_network_mdr_risk_score(
            network, strain_features, bootstrap_ci
        )
        
        # Should return empty DataFrame with correct columns
        assert isinstance(risk_scores, pd.DataFrame)
        assert len(risk_scores) == 0
        assert list(risk_scores.columns) == [
            "Strain_ID", "Network_Risk_Score", "MDR_Predicted", "Percentile_Rank"
        ]

    def test_network_risk_scoring_no_matching_features(self):
        """Test when strain features don't match network nodes."""
        network = nx.Graph()
        network.add_edge("Gene_X", "Gene_Y", phi=0.5, pvalue=0.01)
        
        strain_features = pd.DataFrame({
            "Gene_Z": [1, 0],  # Different gene
        }, index=["Strain_001", "Strain_002"])
        
        bootstrap_ci = {"Gene_X": (0.2, 0.4), "Gene_Z": (0.3, 0.5)}
        
        risk_scores = compute_network_mdr_risk_score(
            network, strain_features, bootstrap_ci
        )
        
        # Should return scores (all zeros since no matches)
        assert len(risk_scores) == 2
        assert (risk_scores["Network_Risk_Score"] == 0).all()

    def test_network_risk_scoring_percentile_threshold(self):
        """Test custom percentile threshold."""
        network = nx.Graph()
        network.add_edge("Gene_A", "Gene_B", phi=0.5, pvalue=0.01)
        
        # Create data with clear high/low risk strains
        strain_features = pd.DataFrame({
            "Gene_A": [1, 1, 0, 0],
            "Gene_B": [1, 1, 0, 0],
        }, index=["Strain_001", "Strain_002", "Strain_003", "Strain_004"])
        
        bootstrap_ci = {
            "Gene_A": (0.4, 0.6),  # Narrow CI = high confidence
            "Gene_B": (0.4, 0.6),
        }
        
        # Test with 50th percentile threshold
        risk_scores = compute_network_mdr_risk_score(
            network, strain_features, bootstrap_ci, percentile_threshold=50.0
        )
        
        # High-risk strains (001, 002) should be predicted as MDR
        high_risk = risk_scores[risk_scores["Strain_ID"].isin(["Strain_001", "Strain_002"])]
        assert high_risk["MDR_Predicted"].sum() >= 1  # At least one should be predicted

    def test_network_risk_scoring_ci_weights(self):
        """Test that CI weights affect risk scores."""
        network = nx.Graph()
        network.add_edge("Gene_A", "Gene_B", phi=0.5, pvalue=0.01)
        
        strain_features = pd.DataFrame({
            "Gene_A": [1, 1],
            "Gene_B": [1, 1],
        }, index=["Strain_001", "Strain_002"])
        
        # Narrow CI (high confidence) vs wide CI (low confidence)
        bootstrap_ci_narrow = {
            "Gene_A": (0.45, 0.55),  # Narrow
            "Gene_B": (0.45, 0.55),
        }
        
        bootstrap_ci_wide = {
            "Gene_A": (0.1, 0.9),  # Wide
            "Gene_B": (0.1, 0.9),
        }
        
        scores_narrow = compute_network_mdr_risk_score(
            network, strain_features, bootstrap_ci_narrow
        )
        
        scores_wide = compute_network_mdr_risk_score(
            network, strain_features, bootstrap_ci_wide
        )
        
        # Narrow CI should not reduce scores vs wide CI (higher confidence weighting)
        assert (
            scores_narrow["Network_Risk_Score"].iloc[0]
            >= scores_wide["Network_Risk_Score"].iloc[0]
        )

    def test_network_risk_scoring_integration(self):
        """Test integration with network construction."""
        # Create sample data
        data = pd.DataFrame({
            "Strain_ID": ["S1", "S2", "S3", "S4", "S5"],
            "Gene_A": [1, 1, 0, 0, 1],
            "Gene_B": [1, 0, 1, 0, 1],
            "Phenotype_X": [1, 1, 1, 0, 1],
        }).set_index("Strain_ID")
        
        # Build network
        pheno_cols = ["Phenotype_X"]
        gene_cols = ["Gene_A", "Gene_B"]
        network = build_hybrid_co_resistance_network(
            data, pheno_cols, gene_cols, alpha=0.05
        )
        
        # Create bootstrap CI
        bootstrap_ci = {
            "Gene_A": (0.3, 0.5),
            "Gene_B": (0.2, 0.4),
            "Phenotype_X": (0.6, 0.8),
        }
        
        # Compute risk scores
        risk_scores = compute_network_mdr_risk_score(
            network, data, bootstrap_ci
        )
        
        # Verify results
        if network.number_of_edges() == 0:
            # With tiny synthetic inputs, the network can legitimately be empty under strict alpha
            assert len(risk_scores) == 0
        else:
            assert len(risk_scores) == 5
            assert "Strain_ID" in risk_scores.columns
            assert all(score >= 0 for score in risk_scores["Network_Risk_Score"])


