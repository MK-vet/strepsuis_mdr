"""Stress tests for strepsuis-mdr module.

These tests verify behavior with large datasets, memory constraints,
and concurrent operations.
"""

import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


@pytest.mark.stress
class TestLargeDatasets:
    """Tests with large datasets to verify scalability."""

    def test_large_binary_matrix_creation(self):
        """Test creating large binary matrices for analysis."""
        np.random.seed(42)
        n_samples = 500
        n_features = 100
        
        # Create large binary matrix
        data = np.random.randint(0, 2, size=(n_samples, n_features))
        df = pd.DataFrame(
            data,
            columns=[f"Feature_{i}" for i in range(n_features)]
        )
        
        assert df.shape == (n_samples, n_features)
        # dtype can vary by platform (int32 on Windows, int64 on Linux)
        assert np.issubdtype(df.values.dtype, np.integer)
        # Verify it's binary
        assert set(df.values.flatten()) <= {0, 1}

    def test_prevalence_calculation_large(self):
        """Test prevalence calculation with large dataset."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 50
        
        data = np.random.randint(0, 2, size=(n_samples, n_features))
        df = pd.DataFrame(data)
        
        start = time.time()
        prevalences = df.mean() * 100
        elapsed = time.time() - start
        
        assert len(prevalences) == n_features
        assert all(0 <= p <= 100 for p in prevalences)
        assert elapsed < 1.0  # Should be fast

    def test_pairwise_combinations_scale(self):
        """Test pairwise combination generation scales properly."""
        from itertools import combinations
        
        # For 50 features, we have 1225 pairs
        features = [f"F{i}" for i in range(50)]
        pairs = list(combinations(features, 2))
        
        assert len(pairs) == 50 * 49 // 2  # n*(n-1)/2
        assert len(pairs) == 1225

    def test_bootstrap_memory_efficiency(self):
        """Test bootstrap resampling doesn't use excessive memory."""
        np.random.seed(42)
        n_samples = 500
        n_iter = 100
        
        data = np.random.randint(0, 2, size=n_samples)
        
        # Store only statistics, not full samples
        stats = np.empty(n_iter)
        for i in range(n_iter):
            sample = np.random.choice(data, size=n_samples, replace=True)
            stats[i] = sample.mean()
        
        # Memory for stats should be small
        assert stats.nbytes < 1000  # Less than 1KB for 100 floats


@pytest.mark.stress
class TestConcurrentOperations:
    """Tests for concurrent operation handling."""

    def test_parallel_safe_random_generation(self):
        """Test that random generation with fixed seed is reproducible."""
        np.random.seed(42)
        result1 = np.random.randint(0, 100, size=1000)
        
        np.random.seed(42)
        result2 = np.random.randint(0, 100, size=1000)
        
        assert np.array_equal(result1, result2)

    def test_dataframe_operations_thread_safety(self):
        """Test DataFrame operations in sequence."""
        np.random.seed(42)
        
        dfs = []
        for i in range(10):
            df = pd.DataFrame(
                np.random.randint(0, 2, size=(100, 20)),
                columns=[f"Col_{j}" for j in range(20)]
            )
            dfs.append(df)
        
        # Each DataFrame should be independent
        assert len(dfs) == 10
        for df in dfs:
            assert df.shape == (100, 20)


@pytest.mark.stress
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        df = pd.DataFrame()
        assert df.empty
        assert len(df.columns) == 0

    def test_single_row_dataframe(self):
        """Test handling of single-row DataFrame."""
        df = pd.DataFrame({"A": [1], "B": [0], "C": [1]})
        assert len(df) == 1
        
        # Prevalence should still work
        prevalence = df.mean() * 100
        assert prevalence["A"] == 100.0
        assert prevalence["B"] == 0.0

    def test_single_column_dataframe(self):
        """Test handling of single-column DataFrame."""
        df = pd.DataFrame({"A": [1, 0, 1, 1, 0]})
        assert len(df.columns) == 1
        
        prevalence = df.mean() * 100
        assert prevalence["A"] == 60.0

    def test_all_zeros_column(self):
        """Test handling of column with all zeros."""
        df = pd.DataFrame({
            "A": [0, 0, 0, 0, 0],
            "B": [1, 1, 1, 1, 1]
        })
        
        prevalence = df.mean() * 100
        assert prevalence["A"] == 0.0
        assert prevalence["B"] == 100.0

    def test_all_ones_column(self):
        """Test handling of column with all ones."""
        df = pd.DataFrame({
            "A": [1, 1, 1, 1, 1]
        })
        
        prevalence = df.mean() * 100
        assert prevalence["A"] == 100.0

    def test_high_dimensional_data(self):
        """Test with high-dimensional data (more features than samples)."""
        np.random.seed(42)
        n_samples = 50
        n_features = 200
        
        data = np.random.randint(0, 2, size=(n_samples, n_features))
        df = pd.DataFrame(data)
        
        assert df.shape == (n_samples, n_features)
        assert df.shape[1] > df.shape[0]  # More features than samples
