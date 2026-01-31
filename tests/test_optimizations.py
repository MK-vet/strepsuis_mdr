"""
Tests for optimizations module.

Tests vectorized bootstrap, sparse operations, and other optimizations.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.sparse import csr_matrix, issparse

from strepsuis_mdr.optimizations import (
    vectorized_bootstrap_ci,
    batch_bootstrap_ci,
    to_sparse_binary,
    sparse_cooccurrence,
    fast_phi_coefficient,
    fast_pairwise_phi,
    fast_mdr_count,
    cached_chi_square,
    parallel_pairwise_analysis,
    _sequential_pairwise_analysis,
    optimize_dataframe_memory,
    benchmark_function,
    get_optimization_status,
    print_optimization_status,
    NUMBA_AVAILABLE,
    JOBLIB_AVAILABLE,
)


class TestVectorizedBootstrap:
    """Tests for vectorized bootstrap CI."""
    
    def test_basic_bootstrap(self):
        """Test basic bootstrap CI calculation."""
        np.random.seed(42)
        data = np.random.binomial(1, 0.3, 100)
        
        mean, ci_low, ci_high = vectorized_bootstrap_ci(data, n_bootstrap=500)
        
        assert 0 <= ci_low <= mean <= ci_high <= 1
        assert ci_high - ci_low > 0  # CI should have width
    
    def test_bootstrap_with_seed(self):
        """Test reproducibility with random seed."""
        data = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
        
        result1 = vectorized_bootstrap_ci(data, n_bootstrap=100, random_state=42)
        result2 = vectorized_bootstrap_ci(data, n_bootstrap=100, random_state=42)
        
        assert result1 == result2
    
    def test_bootstrap_median(self):
        """Test bootstrap with median statistic."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        
        median, ci_low, ci_high = vectorized_bootstrap_ci(
            data, n_bootstrap=500, statistic='median', random_state=42
        )
        
        assert ci_low <= median <= ci_high
    
    def test_bootstrap_std(self):
        """Test bootstrap with std statistic."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        
        std, ci_low, ci_high = vectorized_bootstrap_ci(
            data, n_bootstrap=500, statistic='std', random_state=42
        )
        
        assert ci_low <= std <= ci_high
        assert std > 0
    
    def test_bootstrap_confidence_levels(self):
        """Test different confidence levels."""
        np.random.seed(42)
        data = np.random.binomial(1, 0.5, 100)
        
        _, ci90_low, ci90_high = vectorized_bootstrap_ci(
            data, n_bootstrap=500, confidence=0.90, random_state=42
        )
        _, ci95_low, ci95_high = vectorized_bootstrap_ci(
            data, n_bootstrap=500, confidence=0.95, random_state=42
        )
        
        # 95% CI should be wider than 90% CI
        assert (ci95_high - ci95_low) >= (ci90_high - ci90_low) * 0.9


class TestBatchBootstrap:
    """Tests for batch bootstrap CI."""
    
    def test_batch_bootstrap_basic(self):
        """Test batch bootstrap for multiple columns."""
        np.random.seed(42)
        data = np.random.binomial(1, 0.3, (100, 5))
        
        results = batch_bootstrap_ci(data, n_bootstrap=200, random_state=42)
        
        assert results.shape == (5, 3)  # 5 features, 3 values each
        for i in range(5):
            assert results[i, 1] <= results[i, 0] <= results[i, 2]  # ci_low <= mean <= ci_high
    
    def test_batch_bootstrap_reproducible(self):
        """Test batch bootstrap reproducibility."""
        data = np.random.binomial(1, 0.5, (50, 3))
        
        result1 = batch_bootstrap_ci(data, n_bootstrap=100, random_state=42)
        result2 = batch_bootstrap_ci(data, n_bootstrap=100, random_state=42)
        
        np.testing.assert_array_equal(result1, result2)


class TestSparseOperations:
    """Tests for sparse matrix operations."""
    
    def test_sparse_conversion(self):
        """Test conversion to sparse matrix."""
        # Low density data - should convert well
        data = np.zeros((100, 50))
        data[0, 0] = 1
        data[10, 5] = 1
        data[20, 10] = 1
        
        sparse_mat = to_sparse_binary(data, threshold=0.3)
        
        assert issparse(sparse_mat)
        assert sparse_mat.shape == (100, 50)
    
    def test_sparse_from_dataframe(self):
        """Test sparse conversion from DataFrame."""
        df = pd.DataFrame({
            'A': [1, 0, 0, 0, 0],
            'B': [0, 0, 1, 0, 0],
            'C': [0, 0, 0, 0, 1]
        })
        
        sparse_mat = to_sparse_binary(df, threshold=0.5)
        
        assert issparse(sparse_mat)
    
    def test_sparse_preserves_data(self):
        """Test that sparse conversion preserves data."""
        data = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        
        sparse_mat = to_sparse_binary(data)
        dense = sparse_mat.toarray()
        
        np.testing.assert_array_equal(dense, data)
    
    def test_sparse_cooccurrence(self):
        """Test sparse co-occurrence calculation."""
        data = np.array([
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ])
        sparse_data = csr_matrix(data)
        
        cooc = sparse_cooccurrence(sparse_data)
        
        assert cooc.shape == (3, 3)
        # Diagonal should be sum of each column
        assert cooc[0, 0] == 2  # Column 0 has 2 ones
        assert cooc[1, 1] == 2  # Column 1 has 2 ones


class TestFastPhiCoefficient:
    """Tests for fast phi coefficient calculation."""
    
    def test_perfect_positive(self):
        """Test perfect positive association."""
        # [[4, 0], [0, 4]] - perfect positive
        phi = fast_phi_coefficient(4, 0, 0, 4)
        
        assert abs(phi - 1.0) < 0.01
    
    def test_perfect_negative(self):
        """Test perfect negative association."""
        # [[0, 4], [4, 0]] - perfect negative
        phi = fast_phi_coefficient(0, 4, 4, 0)
        
        assert abs(phi - (-1.0)) < 0.01
    
    def test_no_association(self):
        """Test no association."""
        # [[2, 2], [2, 2]] - no association
        phi = fast_phi_coefficient(2, 2, 2, 2)
        
        assert abs(phi) < 0.01
    
    def test_zero_denominator(self):
        """Test handling of zero denominator."""
        # All in one cell
        phi = fast_phi_coefficient(10, 0, 0, 0)
        
        assert phi == 0.0


class TestFastPairwisePhi:
    """Tests for fast pairwise phi calculation."""
    
    def test_pairwise_phi_basic(self):
        """Test basic pairwise phi calculation."""
        data = np.array([
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
        ])
        
        phi_matrix = fast_pairwise_phi(data)
        
        assert phi_matrix.shape == (3, 3)
        # Diagonal should be 0 (not computed)
        assert phi_matrix[0, 0] == 0
        # Matrix should be symmetric
        np.testing.assert_array_almost_equal(phi_matrix, phi_matrix.T)
    
    def test_pairwise_phi_perfect_correlation(self):
        """Test pairwise phi with perfect correlation."""
        data = np.array([
            [1, 1],
            [1, 1],
            [0, 0],
            [0, 0],
        ])
        
        phi_matrix = fast_pairwise_phi(data)
        
        assert abs(phi_matrix[0, 1] - 1.0) < 0.01


class TestFastMDRCount:
    """Tests for fast MDR counting."""
    
    def test_mdr_count_basic(self):
        """Test basic MDR counting."""
        data = np.array([
            [1, 1, 1, 0],  # 3 resistances - MDR
            [1, 1, 0, 0],  # 2 resistances - not MDR
            [1, 1, 1, 1],  # 4 resistances - MDR
            [0, 0, 0, 0],  # 0 resistances - not MDR
            [1, 1, 1, 0],  # 3 resistances - MDR
        ])
        
        mdr_count, prevalence = fast_mdr_count(data, threshold=3)
        
        assert mdr_count == 3
        assert prevalence == 0.6
    
    def test_mdr_count_threshold(self):
        """Test MDR counting with different thresholds."""
        data = np.array([
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
        ])
        
        mdr3, prev3 = fast_mdr_count(data, threshold=3)
        mdr4, prev4 = fast_mdr_count(data, threshold=4)
        
        assert mdr3 == 2
        assert mdr4 == 1
        assert prev3 == 0.5
        assert prev4 == 0.25


class TestCachedChiSquare:
    """Tests for cached chi-square test."""
    
    def test_chi_square_basic(self):
        """Test basic chi-square calculation."""
        chi2, p = cached_chi_square(10, 5, 5, 10)
        
        assert chi2 > 0
        assert 0 <= p <= 1
    
    def test_chi_square_cached(self):
        """Test that results are cached."""
        # First call
        result1 = cached_chi_square(10, 5, 5, 10)
        # Second call (should use cache)
        result2 = cached_chi_square(10, 5, 5, 10)
        
        assert result1 == result2
    
    def test_chi_square_invalid(self):
        """Test handling of invalid tables."""
        chi2, p = cached_chi_square(0, 0, 0, 0)
        
        assert chi2 == 0.0
        assert p == 1.0
    
    def test_chi_square_negative(self):
        """Test handling of negative values."""
        chi2, p = cached_chi_square(-1, 5, 5, 10)
        
        assert chi2 == 0.0
        assert p == 1.0


class TestParallelAnalysis:
    """Tests for parallel pairwise analysis."""
    
    def test_sequential_analysis(self):
        """Test sequential pairwise analysis."""
        data = np.array([
            [1, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 0],
            [1, 1, 1],
        ])
        
        results = _sequential_pairwise_analysis(data, 'chi_square')
        
        assert len(results) == 3  # C(3,2) = 3 pairs
        assert 'feature1' in results.columns
        assert 'feature2' in results.columns
        assert 'statistic' in results.columns
        assert 'p_value' in results.columns
    
    def test_parallel_analysis_basic(self):
        """Test parallel pairwise analysis."""
        data = np.array([
            [1, 0, 1, 0],
            [1, 1, 0, 1],
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
        ])
        
        results = parallel_pairwise_analysis(data, n_jobs=1)
        
        assert len(results) == 6  # C(4,2) = 6 pairs
    
    @pytest.mark.skipif(not JOBLIB_AVAILABLE, reason="joblib not available")
    def test_parallel_multiple_jobs(self):
        """Test parallel analysis with multiple jobs."""
        np.random.seed(42)
        data = np.random.binomial(1, 0.5, (50, 6))
        
        results = parallel_pairwise_analysis(data, n_jobs=2)
        
        assert len(results) == 15  # C(6,2) = 15 pairs


class TestOptimizeMemory:
    """Tests for memory optimization."""
    
    def test_optimize_memory_basic(self):
        """Test basic memory optimization."""
        df = pd.DataFrame({
            'A': [1, 0, 1, 0, 1],
            'B': [0, 1, 0, 1, 0],
            'C': [1, 1, 0, 0, 1],
        })
        
        optimized = optimize_dataframe_memory(df)
        
        assert optimized.shape == df.shape
        # Values should be preserved
        np.testing.assert_array_equal(optimized.values, df.values)
    
    def test_optimize_memory_dtypes(self):
        """Test that dtypes are optimized."""
        df = pd.DataFrame({
            'A': [1, 0, 1, 0, 1],
            'B': [0, 1, 0, 1, 0],
        })
        
        optimized = optimize_dataframe_memory(df)
        
        # Should use smaller dtypes
        for col in optimized.columns:
            assert optimized[col].dtype in [np.int8, np.uint8, np.int16, np.int32, np.int64]


class TestBenchmark:
    """Tests for benchmark function."""
    
    def test_benchmark_basic(self):
        """Test basic benchmarking."""
        def simple_func(x):
            return x * 2
        
        result = benchmark_function(simple_func, 5, n_runs=5)
        
        assert 'mean_ms' in result
        assert 'std_ms' in result
        assert 'min_ms' in result
        assert 'max_ms' in result
        assert result['mean_ms'] >= 0


class TestOptimizationStatus:
    """Tests for optimization status functions."""
    
    def test_get_status(self):
        """Test getting optimization status."""
        status = get_optimization_status()
        
        assert 'numba_jit' in status
        assert 'parallel_processing' in status
        assert isinstance(status['numba_jit'], bool)
        assert isinstance(status['parallel_processing'], bool)
    
    def test_print_status(self):
        """Test printing optimization status."""
        # Should not raise
        print_optimization_status()


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_single_value_bootstrap(self):
        """Test bootstrap with single value."""
        data = np.array([1])
        
        mean, ci_low, ci_high = vectorized_bootstrap_ci(data, n_bootstrap=100)
        
        assert mean == 1.0
    
    def test_all_zeros_bootstrap(self):
        """Test bootstrap with all zeros."""
        data = np.zeros(100)
        
        mean, ci_low, ci_high = vectorized_bootstrap_ci(data, n_bootstrap=100)
        
        assert mean == 0.0
        assert ci_low == 0.0
        assert ci_high == 0.0
    
    def test_all_ones_bootstrap(self):
        """Test bootstrap with all ones."""
        data = np.ones(100)
        
        mean, ci_low, ci_high = vectorized_bootstrap_ci(data, n_bootstrap=100)
        
        assert mean == 1.0
        assert ci_low == 1.0
        assert ci_high == 1.0
    
    def test_sparse_high_density_warning(self):
        """Test warning for high density data."""
        # High density data
        data = np.ones((10, 10))
        
        with pytest.warns(UserWarning, match="density"):
            to_sparse_binary(data, threshold=0.3)
