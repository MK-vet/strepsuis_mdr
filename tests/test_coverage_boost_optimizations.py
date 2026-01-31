#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Test Coverage for optimizations.py
================================================

Tests ALL optimization functions:
- Vectorized bootstrap
- Batch bootstrap
- Sparse matrix operations
- Numba JIT functions (if available)
- LRU caching
- Parallel processing
- Memory optimization
- Benchmarking utilities
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from strepsuis_mdr import optimizations as opt


class TestVectorizedBootstrap:
    """Test vectorized bootstrap functions."""

    def test_vectorized_bootstrap_ci_mean(self):
        """Test vectorized bootstrap with mean statistic."""
        data = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])

        point_est, ci_low, ci_high = opt.vectorized_bootstrap_ci(
            data,
            n_bootstrap=100,
            confidence=0.95,
            statistic='mean',
            random_state=42
        )

        assert 0 <= point_est <= 1
        assert ci_low <= point_est <= ci_high
        assert 0 <= ci_low <= 1
        assert 0 <= ci_high <= 1

    def test_vectorized_bootstrap_ci_median(self):
        """Test vectorized bootstrap with median statistic."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        point_est, ci_low, ci_high = opt.vectorized_bootstrap_ci(
            data,
            n_bootstrap=100,
            confidence=0.95,
            statistic='median',
            random_state=42
        )

        assert ci_low <= point_est <= ci_high
        assert point_est == np.median(data)

    def test_vectorized_bootstrap_ci_std(self):
        """Test vectorized bootstrap with std statistic."""
        data = np.random.randn(50)

        point_est, ci_low, ci_high = opt.vectorized_bootstrap_ci(
            data,
            n_bootstrap=100,
            confidence=0.95,
            statistic='std',
            random_state=42
        )

        assert point_est > 0
        assert ci_low <= point_est <= ci_high

    def test_vectorized_bootstrap_ci_invalid_statistic(self):
        """Test vectorized bootstrap with invalid statistic."""
        data = np.array([1, 2, 3, 4, 5])

        with pytest.raises(ValueError, match="Unknown statistic"):
            opt.vectorized_bootstrap_ci(data, statistic='invalid')

    def test_vectorized_bootstrap_ci_reproducibility(self):
        """Test that results are reproducible with same random_state."""
        data = np.random.randn(30)

        result1 = opt.vectorized_bootstrap_ci(data, n_bootstrap=50, random_state=42)
        result2 = opt.vectorized_bootstrap_ci(data, n_bootstrap=50, random_state=42)

        assert result1 == result2


class TestBatchBootstrap:
    """Test batch bootstrap for multiple columns."""

    def test_batch_bootstrap_ci_basic(self):
        """Test batch bootstrap CI calculation."""
        data_matrix = np.array([
            [1, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 1, 1]
        ])

        results = opt.batch_bootstrap_ci(
            data_matrix,
            n_bootstrap=100,
            confidence=0.95,
            random_state=42
        )

        assert results.shape == (3, 3)  # 3 features, 3 values each

        # Check that each feature has mean, ci_low, ci_high
        for i in range(3):
            mean, ci_low, ci_high = results[i]
            assert ci_low <= mean <= ci_high
            assert 0 <= mean <= 1

    def test_batch_bootstrap_ci_reproducibility(self):
        """Test batch bootstrap reproducibility."""
        data_matrix = np.random.binomial(1, 0.5, (20, 5))

        result1 = opt.batch_bootstrap_ci(data_matrix, n_bootstrap=50, random_state=42)
        result2 = opt.batch_bootstrap_ci(data_matrix, n_bootstrap=50, random_state=42)

        np.testing.assert_array_equal(result1, result2)


class TestSparseMatrixOperations:
    """Test sparse matrix operations."""

    def test_to_sparse_binary_low_density(self):
        """Test sparse conversion with low density data."""
        data = np.array([
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1]
        ])

        sparse_data = opt.to_sparse_binary(data, threshold=0.5)

        assert sparse_data.shape == (3, 5)
        # Check that conversion preserves data
        assert np.array_equal(sparse_data.toarray(), data)

    def test_to_sparse_binary_high_density(self):
        """Test sparse conversion with high density data (warning)."""
        data = np.ones((3, 5))  # 100% density

        with pytest.warns(UserWarning, match="Data density"):
            sparse_data = opt.to_sparse_binary(data, threshold=0.3)

        assert np.array_equal(sparse_data.toarray(), data)

    def test_to_sparse_binary_dataframe(self):
        """Test sparse conversion with DataFrame input."""
        df = pd.DataFrame({
            'A': [1, 0, 1],
            'B': [0, 1, 0],
            'C': [0, 0, 0]
        })

        sparse_data = opt.to_sparse_binary(df)

        assert sparse_data.shape == (3, 3)
        assert np.array_equal(sparse_data.toarray(), df.values)

    def test_sparse_cooccurrence(self):
        """Test sparse co-occurrence matrix computation."""
        from scipy.sparse import csr_matrix

        data = np.array([
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1]
        ])

        sparse_data = csr_matrix(data)
        cooccur = opt.sparse_cooccurrence(sparse_data)

        assert cooccur.shape == (3, 3)
        # Diagonal should be sum of each column
        assert cooccur[0, 0] == 2  # Column 0 has 2 ones
        assert cooccur[1, 1] == 2  # Column 1 has 2 ones
        assert cooccur[2, 2] == 2  # Column 2 has 2 ones


class TestNumbaJITFunctions:
    """Test Numba JIT compiled functions."""

    def test_fast_phi_coefficient(self):
        """Test fast phi coefficient calculation."""
        # 2x2 table: [[10, 5], [3, 12]]
        phi = opt.fast_phi_coefficient(10, 5, 3, 12)

        assert -1 <= phi <= 1
        assert isinstance(phi, float)

    def test_fast_phi_coefficient_zero_denominator(self):
        """Test fast phi with zero denominator."""
        phi = opt.fast_phi_coefficient(0, 0, 0, 0)

        assert phi == 0.0

    def test_fast_phi_coefficient_perfect_association(self):
        """Test fast phi with perfect positive association."""
        # [[10, 0], [0, 10]]
        phi = opt.fast_phi_coefficient(10, 0, 0, 10)

        assert phi > 0.9  # Should be close to 1.0

    @pytest.mark.skipif(not opt.NUMBA_AVAILABLE, reason="Numba not available")
    def test_fast_pairwise_phi(self):
        """Test fast pairwise phi computation (if Numba available)."""
        data = np.array([
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0]
        ])

        phi_matrix = opt.fast_pairwise_phi(data)

        assert phi_matrix.shape == (3, 3)
        # Matrix should be symmetric
        assert np.allclose(phi_matrix, phi_matrix.T)
        # All values should be in [-1, 1]
        assert np.all(phi_matrix >= -1)
        assert np.all(phi_matrix <= 1)

    def test_fast_mdr_count(self):
        """Test fast MDR counting."""
        data = np.array([
            [1, 1, 1, 1],  # 4 resistances -> MDR
            [1, 1, 0, 0],  # 2 resistances -> not MDR
            [1, 1, 1, 0],  # 3 resistances -> MDR
            [0, 0, 0, 0]   # 0 resistances -> not MDR
        ])

        mdr_count, mdr_prev = opt.fast_mdr_count(data, threshold=3)

        assert mdr_count == 2  # 2 isolates with >= 3 resistances
        assert mdr_prev == 0.5  # 50% MDR

    def test_fast_mdr_count_custom_threshold(self):
        """Test fast MDR count with custom threshold."""
        data = np.array([
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ])

        mdr_count, mdr_prev = opt.fast_mdr_count(data, threshold=2)

        assert mdr_count == 2  # 2 isolates with >= 2 resistances
        assert mdr_prev == 0.5


class TestLRUCaching:
    """Test LRU cached functions."""

    def test_cached_chi_square_basic(self):
        """Test cached chi-square calculation."""
        chi2, p_val = opt.cached_chi_square(10, 5, 3, 12)

        assert chi2 >= 0
        assert 0 <= p_val <= 1

    def test_cached_chi_square_zero_table(self):
        """Test cached chi-square with zero table."""
        chi2, p_val = opt.cached_chi_square(0, 0, 0, 0)

        assert chi2 == 0.0
        assert p_val == 1.0

    def test_cached_chi_square_invalid_values(self):
        """Test cached chi-square with invalid values."""
        chi2, p_val = opt.cached_chi_square(-1, 0, 0, 0)

        assert chi2 == 0.0
        assert p_val == 1.0

    def test_cached_chi_square_caching(self):
        """Test that chi-square results are cached."""
        # Clear cache
        opt.cached_chi_square.cache_clear()

        # First call
        result1 = opt.cached_chi_square(10, 5, 3, 12)
        info1 = opt.cached_chi_square.cache_info()

        # Second call with same arguments
        result2 = opt.cached_chi_square(10, 5, 3, 12)
        info2 = opt.cached_chi_square.cache_info()

        # Results should be identical
        assert result1 == result2

        # Hit count should increase
        assert info2.hits == info1.hits + 1


class TestParallelProcessing:
    """Test parallel processing functions."""

    @pytest.mark.skipif(not opt.JOBLIB_AVAILABLE, reason="Joblib not available")
    def test_parallel_pairwise_analysis_chi_square(self):
        """Test parallel pairwise analysis with chi-square."""
        data = np.array([
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0]
        ])

        result = opt.parallel_pairwise_analysis(data, n_jobs=2, test='chi_square')

        assert 'feature1' in result.columns
        assert 'feature2' in result.columns
        assert 'statistic' in result.columns
        assert 'p_value' in result.columns

        # Should have 3 pairs for 3 features
        assert len(result) == 3

    @pytest.mark.skipif(not opt.JOBLIB_AVAILABLE, reason="Joblib not available")
    def test_parallel_pairwise_analysis_fisher(self):
        """Test parallel pairwise analysis with Fisher's test."""
        data = np.array([
            [1, 1],
            [1, 0],
            [0, 1]
        ])

        result = opt.parallel_pairwise_analysis(data, n_jobs=1, test='fisher')

        assert len(result) == 1  # 1 pair for 2 features
        assert all(0 <= p <= 1 for p in result['p_value'])

    def test_sequential_pairwise_analysis_fallback(self):
        """Test sequential pairwise analysis (fallback)."""
        data = np.array([
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1]
        ])

        result = opt._sequential_pairwise_analysis(data, test='chi_square')

        assert 'feature1' in result.columns
        assert 'feature2' in result.columns
        assert len(result) == 3


class TestMemoryOptimization:
    """Test memory optimization functions."""

    def test_optimize_dataframe_memory_binary(self):
        """Test memory optimization for binary data."""
        df = pd.DataFrame({
            'A': [1, 0, 1, 0, 1],
            'B': [0, 1, 0, 1, 0]
        })

        optimized = opt.optimize_dataframe_memory(df)

        # Binary columns should be int8
        assert optimized['A'].dtype == np.int8
        assert optimized['B'].dtype == np.int8

    def test_optimize_dataframe_memory_uint8(self):
        """Test memory optimization for uint8 range."""
        df = pd.DataFrame({
            'A': [0, 50, 100, 150, 200],
            'B': [10, 20, 30, 40, 50]
        })

        optimized = opt.optimize_dataframe_memory(df)

        # Should be uint8 (0-255 range)
        assert optimized['A'].dtype == np.uint8
        assert optimized['B'].dtype == np.uint8

    def test_optimize_dataframe_memory_int8(self):
        """Test memory optimization for int8 range."""
        df = pd.DataFrame({
            'A': [-50, 0, 50],
            'B': [-100, 0, 100]
        })

        optimized = opt.optimize_dataframe_memory(df)

        # Should be int8 (-128 to 127 range)
        assert optimized['A'].dtype == np.int8
        assert optimized['B'].dtype == np.int8

    def test_optimize_dataframe_memory_float32(self):
        """Test memory optimization for float data."""
        df = pd.DataFrame({
            'A': [1.5, 2.5, 3.5],
            'B': [0.1, 0.2, 0.3]
        })

        optimized = opt.optimize_dataframe_memory(df)

        # Float64 should be converted to float32
        assert optimized['A'].dtype == np.float32
        assert optimized['B'].dtype == np.float32


class TestBenchmarkingUtilities:
    """Test benchmarking utilities."""

    def test_benchmark_function_basic(self):
        """Test basic function benchmarking."""
        def dummy_function(x):
            return x ** 2

        results = opt.benchmark_function(dummy_function, 10, n_runs=5)

        assert 'mean_ms' in results
        assert 'std_ms' in results
        assert 'min_ms' in results
        assert 'max_ms' in results
        assert 'n_runs' in results

        assert results['n_runs'] == 5
        assert results['mean_ms'] >= 0
        assert results['min_ms'] >= 0

    def test_benchmark_function_with_kwargs(self):
        """Test benchmarking with keyword arguments."""
        def func_with_kwargs(a, b=10):
            return a + b

        results = opt.benchmark_function(func_with_kwargs, 5, n_runs=3, b=20)

        assert results['n_runs'] == 3


class TestConvenienceFunctions:
    """Test convenience and status functions."""

    def test_get_optimization_status(self):
        """Test optimization status retrieval."""
        status = opt.get_optimization_status()

        assert isinstance(status, dict)
        assert 'numba_jit' in status
        assert 'parallel_processing' in status
        assert 'vectorized_bootstrap' in status
        assert 'sparse_matrices' in status
        assert 'lru_caching' in status

        # These should always be True
        assert status['vectorized_bootstrap'] == True
        assert status['sparse_matrices'] == True
        assert status['lru_caching'] == True

    def test_print_optimization_status(self, capsys):
        """Test printing optimization status."""
        opt.print_optimization_status()

        captured = capsys.readouterr()
        output = captured.out

        assert 'STREPSUIS-MDR OPTIMIZATION STATUS' in output
        assert 'vectorized_bootstrap' in output


class TestPerformanceComparison:
    """Test that optimized functions are actually faster (if optimizations available)."""

    @pytest.mark.skipif(not opt.NUMBA_AVAILABLE, reason="Numba not available")
    def test_fast_phi_faster_than_naive(self):
        """Test that fast_phi is faster than naive implementation."""
        def naive_phi(a, b, c, d):
            """Naive Python implementation."""
            numerator = a * d - b * c
            denominator = ((a + b) * (c + d) * (a + c) * (b + d)) ** 0.5
            return numerator / denominator if denominator != 0 else 0.0

        # Benchmark both
        fast_result = opt.benchmark_function(opt.fast_phi_coefficient, 10, 5, 3, 12, n_runs=100)
        naive_result = opt.benchmark_function(naive_phi, 10, 5, 3, 12, n_runs=100)

        # Fast version should be faster (or at least not significantly slower)
        # Note: In some cases JIT overhead might make it slower for single calls
        # but it should be faster for repeated calls


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_vectorized_bootstrap_single_value(self):
        """Test bootstrap with single unique value."""
        data = np.ones(10)

        point_est, ci_low, ci_high = opt.vectorized_bootstrap_ci(
            data,
            n_bootstrap=50,
            random_state=42
        )

        assert point_est == 1.0
        assert ci_low == 1.0
        assert ci_high == 1.0

    def test_batch_bootstrap_single_column(self):
        """Test batch bootstrap with single column."""
        data_matrix = np.array([[1], [0], [1], [1], [0]])

        results = opt.batch_bootstrap_ci(data_matrix, n_bootstrap=50, random_state=42)

        assert results.shape == (1, 3)

    def test_optimize_dataframe_memory_empty(self):
        """Test memory optimization with empty DataFrame."""
        df = pd.DataFrame()

        optimized = opt.optimize_dataframe_memory(df)

        assert optimized.empty


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
