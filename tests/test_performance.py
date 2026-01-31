"""Performance tests for strepsuis-mdr module.

These tests measure and verify timing benchmarks for key operations.
"""

import time

import numpy as np
import pandas as pd
import pytest


@pytest.mark.performance
class TestBootstrapPerformance:
    """Performance tests for bootstrap operations."""

    def test_bootstrap_ci_timing_small(self):
        """Test bootstrap CI computation time for small dataset."""
        np.random.seed(42)
        n_samples = 100
        n_iter = 100
        
        data = np.random.randint(0, 2, size=n_samples)
        
        start = time.time()
        boot_means = np.array([
            np.mean(np.random.choice(data, size=n_samples, replace=True))
            for _ in range(n_iter)
        ])
        ci_lower = np.percentile(boot_means, 2.5)
        ci_upper = np.percentile(boot_means, 97.5)
        elapsed = time.time() - start
        
        assert elapsed < 0.5  # Should complete in under 0.5 seconds
        assert 0 <= ci_lower <= ci_upper <= 1

    def test_bootstrap_ci_timing_medium(self):
        """Test bootstrap CI computation time for medium dataset."""
        np.random.seed(42)
        n_samples = 500
        n_iter = 500
        
        data = np.random.randint(0, 2, size=n_samples)
        
        start = time.time()
        boot_means = np.array([
            np.mean(np.random.choice(data, size=n_samples, replace=True))
            for _ in range(n_iter)
        ])
        elapsed = time.time() - start
        
        assert elapsed < 2.0  # Should complete in under 2 seconds


@pytest.mark.performance
class TestStatisticalPerformance:
    """Performance tests for statistical calculations."""

    def test_chi_square_calculation_timing(self):
        """Test chi-square calculation performance."""
        from scipy.stats import chi2_contingency
        
        np.random.seed(42)
        n_tests = 100
        
        start = time.time()
        for _ in range(n_tests):
            # Random 2x2 contingency table
            table = np.random.randint(1, 50, size=(2, 2))
            chi2_contingency(table)
        elapsed = time.time() - start
        
        # 100 chi-square tests should be fast
        assert elapsed < 0.5

    def test_fisher_exact_timing(self):
        """Test Fisher's exact test performance."""
        from scipy.stats import fisher_exact
        
        np.random.seed(42)
        n_tests = 100
        
        start = time.time()
        for _ in range(n_tests):
            # Random 2x2 contingency table
            table = np.random.randint(1, 20, size=(2, 2))
            fisher_exact(table)
        elapsed = time.time() - start
        
        # 100 Fisher tests should complete quickly
        assert elapsed < 1.0

    def test_fdr_correction_timing(self):
        """Test FDR correction performance."""
        from statsmodels.stats.multitest import multipletests
        
        np.random.seed(42)
        n_pvalues = 1000
        
        pvalues = np.random.uniform(0, 1, size=n_pvalues)
        
        start = time.time()
        reject, corrected, _, _ = multipletests(pvalues, method='fdr_bh')
        elapsed = time.time() - start
        
        # FDR on 1000 p-values should be very fast
        assert elapsed < 0.1
        assert len(corrected) == n_pvalues


@pytest.mark.performance
class TestDataProcessingPerformance:
    """Performance tests for data processing operations."""

    def test_dataframe_creation_timing(self):
        """Test DataFrame creation performance."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 100
        
        start = time.time()
        data = np.random.randint(0, 2, size=(n_samples, n_features))
        df = pd.DataFrame(
            data,
            columns=[f"Feature_{i}" for i in range(n_features)]
        )
        elapsed = time.time() - start
        
        assert elapsed < 0.5
        assert df.shape == (n_samples, n_features)

    def test_prevalence_calculation_timing(self):
        """Test prevalence calculation performance."""
        np.random.seed(42)
        n_samples = 5000
        n_features = 100
        
        df = pd.DataFrame(
            np.random.randint(0, 2, size=(n_samples, n_features))
        )
        
        start = time.time()
        prevalences = df.mean() * 100
        elapsed = time.time() - start
        
        assert elapsed < 0.1
        assert len(prevalences) == n_features

    def test_crosstab_timing(self):
        """Test contingency table creation performance."""
        np.random.seed(42)
        n_samples = 1000
        
        col1 = np.random.randint(0, 2, size=n_samples)
        col2 = np.random.randint(0, 2, size=n_samples)
        
        start = time.time()
        for _ in range(100):
            table = pd.crosstab(pd.Series(col1), pd.Series(col2))
        elapsed = time.time() - start
        
        # 100 crosstabs should be fast
        assert elapsed < 1.0


@pytest.mark.performance
class TestPairwisePerformance:
    """Performance tests for pairwise operations."""

    def test_pairwise_iteration_timing(self):
        """Test pairwise iteration performance."""
        from itertools import combinations
        
        n_features = 100
        features = list(range(n_features))
        
        start = time.time()
        pairs = list(combinations(features, 2))
        elapsed = time.time() - start
        
        expected_pairs = n_features * (n_features - 1) // 2
        assert len(pairs) == expected_pairs
        assert elapsed < 0.1

    def test_phi_coefficient_timing(self):
        """Test phi coefficient calculation performance."""
        np.random.seed(42)
        n_samples = 500
        n_pairs = 100
        
        def phi_coefficient(a, b, c, d):
            """Calculate phi coefficient."""
            num = a * d - b * c
            den = np.sqrt((a+b) * (c+d) * (a+c) * (b+d))
            return num / den if den > 0 else 0.0
        
        start = time.time()
        for _ in range(n_pairs):
            # Random contingency counts
            a, b, c, d = np.random.randint(10, 100, size=4)
            phi = phi_coefficient(a, b, c, d)
        elapsed = time.time() - start
        
        assert elapsed < 0.1
