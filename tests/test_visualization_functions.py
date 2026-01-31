"""
Tests for visualization functions in mdr_analysis_core.

Target: Increase coverage for create_network_risk_scoring_visualizations
and create_sequential_patterns_visualizations from 0% to 80%+
"""

import pandas as pd
import pytest

from strepsuis_mdr.mdr_analysis_core import (
    create_network_risk_scoring_visualizations,
    create_sequential_patterns_visualizations,
)


class TestNetworkRiskScoringVisualizations:
    """Test create_network_risk_scoring_visualizations function."""

    def test_empty_risk_scores(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame(columns=["Strain_ID", "Network_Risk_Score", "MDR_Predicted", "Percentile_Rank"])
        
        html, fig_hist, fig_ranking, fig_dist = create_network_risk_scoring_visualizations(empty_df)
        
        assert html == "<p>No risk scores to visualize.</p>"
        assert fig_hist is None
        assert fig_ranking is None
        assert fig_dist is None

    def test_basic_risk_scores(self):
        """Test with basic risk scores data."""
        risk_scores = pd.DataFrame({
            "Strain_ID": [f"Strain_{i}" for i in range(20)],
            "Network_Risk_Score": [0.1 + i * 0.05 for i in range(20)],
            "MDR_Predicted": [True if i >= 10 else False for i in range(20)],
            "Percentile_Rank": [i * 5 for i in range(20)]
        })
        
        html, fig_hist, fig_ranking, fig_dist = create_network_risk_scoring_visualizations(risk_scores)
        
        assert html is not None
        assert len(html) > 0
        assert "risk_histogram" in html
        assert "risk_ranking" in html
        assert "risk_distribution" in html
        assert fig_hist is not None
        assert fig_ranking is not None
        assert fig_dist is not None

    def test_small_dataset(self):
        """Test with small dataset (< 10 strains)."""
        risk_scores = pd.DataFrame({
            "Strain_ID": [f"Strain_{i}" for i in range(5)],
            "Network_Risk_Score": [0.1, 0.2, 0.3, 0.4, 0.5],
            "MDR_Predicted": [False, False, True, True, True],
            "Percentile_Rank": [20, 40, 60, 80, 100]
        })
        
        html, fig_hist, fig_ranking, fig_dist = create_network_risk_scoring_visualizations(risk_scores)
        
        assert html is not None
        assert fig_hist is not None
        # Should handle small datasets gracefully
        assert fig_ranking is not None

    def test_all_mdr_predicted_true(self):
        """Test with all strains predicted as MDR."""
        risk_scores = pd.DataFrame({
            "Strain_ID": [f"Strain_{i}" for i in range(10)],
            "Network_Risk_Score": [0.5 + i * 0.1 for i in range(10)],
            "MDR_Predicted": [True] * 10,
            "Percentile_Rank": [i * 10 for i in range(10)]
        })
        
        html, fig_hist, fig_ranking, fig_dist = create_network_risk_scoring_visualizations(risk_scores)
        
        assert html is not None
        assert fig_dist is not None

    def test_all_mdr_predicted_false(self):
        """Test with all strains predicted as non-MDR."""
        risk_scores = pd.DataFrame({
            "Strain_ID": [f"Strain_{i}" for i in range(10)],
            "Network_Risk_Score": [0.1 + i * 0.05 for i in range(10)],
            "MDR_Predicted": [False] * 10,
            "Percentile_Rank": [i * 10 for i in range(10)]
        })
        
        html, fig_hist, fig_ranking, fig_dist = create_network_risk_scoring_visualizations(risk_scores)
        
        assert html is not None
        assert fig_dist is not None

    def test_identical_risk_scores(self):
        """Test with identical risk scores."""
        risk_scores = pd.DataFrame({
            "Strain_ID": [f"Strain_{i}" for i in range(10)],
            "Network_Risk_Score": [0.5] * 10,
            "MDR_Predicted": [True, False] * 5,
            "Percentile_Rank": [50] * 10
        })
        
        html, fig_hist, fig_ranking, fig_dist = create_network_risk_scoring_visualizations(risk_scores)
        
        assert html is not None
        assert fig_hist is not None


class TestSequentialPatternsVisualizations:
    """Test create_sequential_patterns_visualizations function."""

    def test_empty_patterns(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame(columns=["Pattern", "Support", "Confidence", "Lift", "P_Value"])
        
        html, fig_bar, fig_scatter = create_sequential_patterns_visualizations(empty_df, "AMR Genes")
        
        assert "No sequential patterns detected" in html
        assert "AMR Genes" in html
        assert fig_bar is None
        assert fig_scatter is None

    def test_basic_patterns(self):
        """Test with basic patterns data."""
        patterns = pd.DataFrame({
            "Pattern": ["A→B", "B→C", "C→D", "A→C", "B→D"],
            "Support": [0.3, 0.25, 0.2, 0.15, 0.1],
            "Confidence": [0.8, 0.75, 0.7, 0.65, 0.6],
            "Lift": [1.5, 1.4, 1.3, 1.2, 1.1],
            "P_Value": [0.001, 0.01, 0.02, 0.03, 0.05]
        })
        
        html, fig_bar, fig_scatter = create_sequential_patterns_visualizations(patterns, "AMR Genes")
        
        assert html is not None
        assert len(html) > 0
        assert "pattern_bar" in html or "patterns" in html.lower()
        assert fig_bar is not None
        assert fig_scatter is not None

    def test_single_pattern(self):
        """Test with single pattern."""
        patterns = pd.DataFrame({
            "Pattern": ["A→B"],
            "Support": [0.5],
            "Confidence": [0.9],
            "Lift": [1.8],
            "P_Value": [0.001]
        })
        
        html, fig_bar, fig_scatter = create_sequential_patterns_visualizations(patterns, "MIC Phenotypes")
        
        assert html is not None
        assert fig_bar is not None
        assert fig_scatter is not None

    def test_many_patterns(self):
        """Test with many patterns (> 20)."""
        patterns = pd.DataFrame({
            "Pattern": [f"Gene_{i}→Gene_{i+1}" for i in range(30)],
            "Support": [0.1 + i * 0.01 for i in range(30)],
            "Confidence": [0.5 + i * 0.01 for i in range(30)],
            "Lift": [1.0 + i * 0.05 for i in range(30)],
            "P_Value": [0.01 + i * 0.001 for i in range(30)]
        })
        
        html, fig_bar, fig_scatter = create_sequential_patterns_visualizations(patterns, "AMR Genes")
        
        assert html is not None
        assert fig_bar is not None
        assert fig_scatter is not None

    def test_patterns_with_high_confidence(self):
        """Test with patterns having high confidence."""
        patterns = pd.DataFrame({
            "Pattern": ["A→B", "B→C", "C→D"],
            "Support": [0.4, 0.35, 0.3],
            "Confidence": [0.95, 0.92, 0.90],
            "Lift": [2.0, 1.9, 1.8],
            "P_Value": [0.001, 0.002, 0.003]
        })
        
        html, fig_bar, fig_scatter = create_sequential_patterns_visualizations(patterns, "AMR Genes")
        
        assert html is not None
        assert fig_bar is not None

    def test_patterns_with_low_confidence(self):
        """Test with patterns having low confidence."""
        patterns = pd.DataFrame({
            "Pattern": ["A→B", "B→C"],
            "Support": [0.1, 0.08],
            "Confidence": [0.3, 0.25],
            "Lift": [0.8, 0.7],
            "P_Value": [0.1, 0.15]
        })
        
        html, fig_bar, fig_scatter = create_sequential_patterns_visualizations(patterns, "MIC Phenotypes")
        
        assert html is not None
        assert fig_bar is not None
        assert fig_scatter is not None

    def test_different_data_types(self):
        """Test with different data_type parameters."""
        patterns = pd.DataFrame({
            "Pattern": ["A→B", "B→C"],
            "Support": [0.3, 0.25],
            "Confidence": [0.7, 0.65],
            "Lift": [1.4, 1.3],
            "P_Value": [0.01, 0.02]
        })
        
        for data_type in ["AMR Genes", "MIC Phenotypes", "Resistance Patterns"]:
            html, fig_bar, fig_scatter = create_sequential_patterns_visualizations(patterns, data_type)
            assert html is not None
            assert data_type in html or "patterns" in html.lower()
