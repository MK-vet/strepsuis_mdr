"""
Tests for Sequential Pattern Detection innovation.

Tests the detect_sequential_resistance_patterns function which identifies
order-dependent resistance acquisition patterns.
"""

import numpy as np
import pandas as pd
import pytest

from strepsuis_mdr.mdr_analysis_core import detect_sequential_resistance_patterns


class TestSequentialPatterns:
    """Test suite for Sequential Pattern Detection innovation."""

    def test_sequential_patterns_basic(self):
        """Test basic sequential pattern detection."""
        # Create data with clear sequential pattern: A→B
        data = pd.DataFrame({
            "Gene_A": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            "Gene_B": [1, 1, 0, 1, 1, 0, 0, 0, 0, 0],  # B often follows A
            "Gene_C": [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        })
        
        patterns = detect_sequential_resistance_patterns(
            data, min_support=0.1, min_confidence=0.3
        )
        
        # Verify structure
        assert isinstance(patterns, pd.DataFrame)
        assert "Pattern" in patterns.columns
        assert "Support" in patterns.columns
        assert "Confidence" in patterns.columns
        assert "Lift" in patterns.columns
        assert "P_Value" in patterns.columns

    def test_sequential_patterns_empty_data(self):
        """Test handling of empty data."""
        data = pd.DataFrame()
        
        patterns = detect_sequential_resistance_patterns(data)
        
        assert isinstance(patterns, pd.DataFrame)
        assert len(patterns) == 0
        assert list(patterns.columns) == ["Pattern", "Support", "Confidence", "Lift", "P_Value"]

    def test_sequential_patterns_insufficient_features(self):
        """Test with insufficient features."""
        data = pd.DataFrame({
            "Gene_A": [1, 0, 1],
        })
        
        patterns = detect_sequential_resistance_patterns(data)
        
        assert isinstance(patterns, pd.DataFrame)
        assert len(patterns) == 0

    def test_sequential_patterns_min_support_filtering(self):
        """Test that min_support filters patterns correctly."""
        data = pd.DataFrame({
            "Gene_A": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Low frequency
            "Gene_B": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            "Gene_C": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # High frequency
            "Gene_D": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        })
        
        # High min_support should filter out low-frequency patterns
        patterns_high = detect_sequential_resistance_patterns(
            data, min_support=0.5, min_confidence=0.3
        )
        
        # Low min_support should include more patterns
        patterns_low = detect_sequential_resistance_patterns(
            data, min_support=0.1, min_confidence=0.3
        )
        
        # High threshold should have fewer or equal patterns
        assert len(patterns_high) <= len(patterns_low)

    def test_sequential_patterns_min_confidence_filtering(self):
        """Test that min_confidence filters patterns correctly."""
        # Create data where A→B has high confidence
        data = pd.DataFrame({
            "Gene_A": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            "Gene_B": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # B follows A 75% of time
            "Gene_C": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        })
        
        # Low confidence threshold
        patterns_low = detect_sequential_resistance_patterns(
            data, min_support=0.1, min_confidence=0.3
        )
        
        # High confidence threshold
        patterns_high = detect_sequential_resistance_patterns(
            data, min_support=0.1, min_confidence=0.8
        )
        
        # High threshold should have fewer or equal patterns
        assert len(patterns_high) <= len(patterns_low)

    def test_sequential_patterns_correlation_threshold(self):
        """Test correlation threshold parameter."""
        # Create data with varying correlations
        data = pd.DataFrame({
            "Gene_A": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            "Gene_B": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # High correlation with A
            "Gene_C": [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Low correlation with A
        })
        
        # Low correlation threshold
        patterns_low = detect_sequential_resistance_patterns(
            data, min_support=0.1, correlation_threshold=0.1
        )
        
        # High correlation threshold
        patterns_high = detect_sequential_resistance_patterns(
            data, min_support=0.1, correlation_threshold=0.5
        )
        
        # High threshold should have fewer patterns (only strong correlations)
        assert len(patterns_high) <= len(patterns_low)

    def test_sequential_patterns_pattern_format(self):
        """Test that patterns are formatted correctly."""
        data = pd.DataFrame({
            "Gene_A": [1, 1, 0, 0],
            "Gene_B": [1, 0, 1, 0],
        })
        
        patterns = detect_sequential_resistance_patterns(
            data, min_support=0.1, min_confidence=0.3
        )
        
        if len(patterns) > 0:
            # Check pattern format (should contain →)
            assert "→" in patterns["Pattern"].iloc[0]
            
            # Check numeric columns are in valid ranges
            assert (patterns["Support"] >= 0).all()
            assert (patterns["Support"] <= 1).all()
            assert (patterns["Confidence"] >= 0).all()
            assert (patterns["Confidence"] <= 1).all()
            assert (patterns["Lift"] >= 0).all()
            assert (patterns["P_Value"] >= 0).all()
            assert (patterns["P_Value"] <= 1).all()

    def test_sequential_patterns_sorted_by_confidence(self):
        """Test that patterns are sorted by confidence."""
        data = pd.DataFrame({
            "Gene_A": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            "Gene_B": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            "Gene_C": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        })
        
        patterns = detect_sequential_resistance_patterns(
            data, min_support=0.1, min_confidence=0.3
        )
        
        if len(patterns) > 1:
            # Should be sorted descending by confidence
            confidences = patterns["Confidence"].values
            assert all(confidences[i] >= confidences[i + 1] for i in range(len(confidences) - 1))

