"""
Tests for uncertainty quantification module.

Tests bootstrap CI, permutation tests, FDR correction, and stability metrics.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from strepsuis_mdr.uncertainty import (
    UncertaintyQuantifier,
    apply_default_uncertainty
)


class TestUncertaintyQuantifier:
    """Tests for UncertaintyQuantifier class."""

    def test_initialization(self):
        """Test basic initialization."""
        uq = UncertaintyQuantifier(n_bootstrap=500, random_state=42)
        assert uq.n_bootstrap == 500
        assert uq.random_state == 42
        assert uq.confidence_level == 0.95

    def test_bootstrap_ci_mean(self):
        """Test bootstrap CI for mean."""
        np.random.seed(42)
        data = np.random.normal(100, 15, 1000)

        uq = UncertaintyQuantifier(n_bootstrap=500, random_state=42)
        value, ci_low, ci_high = uq.bootstrap_ci(data, np.mean)

        # Check that point estimate is close to true mean
        assert 95 < value < 105

        # Check that CI contains true mean
        assert ci_low < 100 < ci_high

        # Check CI width is reasonable
        ci_width = ci_high - ci_low
        assert 1.5 < ci_width < 8  # Approximately 2 * 1.96 * SE

    def test_bootstrap_ci_median(self):
        """Test bootstrap CI for median."""
        np.random.seed(42)
        data = np.random.exponential(10, 500)

        uq = UncertaintyQuantifier(n_bootstrap=500, random_state=42)
        value, ci_low, ci_high = uq.bootstrap_ci(data, np.median)

        assert ci_low < value < ci_high

    def test_permutation_test_two_sample(self):
        """Test two-sample permutation test."""
        np.random.seed(42)
        # Two groups with different means
        group1 = np.random.normal(10, 1, 100)
        group2 = np.random.normal(12, 1, 100)

        uq = UncertaintyQuantifier(n_permutations=500, random_state=42)
        result = uq.permutation_test(
            group1, group2,
            statistic=lambda x, y: np.mean(x) - np.mean(y)
        )

        # Should detect significant difference
        assert result['p_value'] < 0.05
        assert 'observed_stat' in result
        assert 'null_mean' in result

    def test_permutation_test_no_difference(self):
        """Test permutation test with no true difference."""
        np.random.seed(42)
        # Two groups from same distribution
        group1 = np.random.normal(10, 1, 100)
        group2 = np.random.normal(10, 1, 100)

        uq = UncertaintyQuantifier(n_permutations=500, random_state=42)
        result = uq.permutation_test(
            group1, group2,
            statistic=lambda x, y: np.mean(x) - np.mean(y)
        )

        # Should NOT detect significant difference
        assert result['p_value'] > 0.05

    def test_fdr_correction(self):
        """Test FDR correction."""
        # Mix of significant and non-significant p-values
        p_values = np.array([0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9])

        uq = UncertaintyQuantifier()
        reject, p_adjusted = uq.fdr_correction(p_values, alpha=0.05)

        # First few should be rejected (significant after FDR)
        assert reject[0] == True  # 0.001
        assert reject[1] == True  # 0.01

        # Last ones should not be rejected
        assert reject[-1] == False  # 0.9
        assert reject[-2] == False  # 0.7

        # Adjusted p-values should be >= original
        assert all(p_adjusted >= p_values)

    def test_cluster_stability_basic(self):
        """Test cluster stability metric."""
        np.random.seed(42)
        # Create clearly separable clusters
        X1 = np.random.normal([0, 0], 0.5, (100, 2))
        X2 = np.random.normal([5, 5], 0.5, (100, 2))
        X = np.vstack([X1, X2])

        def clustering_func(data):
            km = KMeans(n_clusters=2, random_state=42, n_init=10)
            return km.fit_predict(data)

        uq = UncertaintyQuantifier(random_state=42)
        stability = uq.cluster_stability(
            X, clustering_func,
            n_iterations=50,
            sample_fraction=0.8
        )

        # Well-separated clusters should have high ARI
        assert stability['mean_ari'] > 0.7
        assert 'ci_ari_low' in stability
        assert 'ci_ari_high' in stability

    def test_consensus_clustering(self):
        """Test consensus clustering."""
        np.random.seed(42)
        # Create 3 clear clusters
        X1 = np.random.normal([0, 0], 0.5, (50, 2))
        X2 = np.random.normal([5, 0], 0.5, (50, 2))
        X3 = np.random.normal([2.5, 4], 0.5, (50, 2))
        X = np.vstack([X1, X2, X3])

        def clustering_func(data):
            km = KMeans(n_clusters=3, random_state=42, n_init=10)
            return km.fit_predict(data)

        uq = UncertaintyQuantifier(random_state=42)
        consensus_labels = uq.consensus_clustering(
            X, clustering_func,
            n_iterations=30,
            sample_fraction=0.8
        )

        # Check that we get 3 clusters
        assert len(np.unique(consensus_labels)) == 3

        # Check that labels are reasonable
        assert len(consensus_labels) == len(X)

    def test_quantify_metric_basic(self):
        """Test comprehensive metric quantification."""
        np.random.seed(42)
        data = np.random.normal(50, 10, 200)

        uq = UncertaintyQuantifier(n_bootstrap=200, random_state=42)
        result = uq.quantify_metric(
            np.mean, data,
            metric_name="test_mean",
            include_permutation=False
        )

        assert result['metric'] == 'test_mean'
        assert 'value' in result
        assert 'ci_low' in result
        assert 'ci_high' in result
        assert result['ci_low'] < result['value'] < result['ci_high']

    def test_quantify_metric_with_permutation(self):
        """Test metric quantification with permutation test."""
        np.random.seed(42)
        data = np.random.normal(50, 10, 200)

        uq = UncertaintyQuantifier(
            n_bootstrap=200,
            n_permutations=200,
            random_state=42
        )
        result = uq.quantify_metric(
            np.mean, data,
            metric_name="test_mean",
            include_permutation=True
        )

        assert 'p_value' in result
        assert 'null_mean' in result
        assert 0 <= result['p_value'] <= 1

    def test_reproducibility(self):
        """Test that results are reproducible with same random seed."""
        np.random.seed(42)
        data = np.random.normal(50, 10, 200)

        uq1 = UncertaintyQuantifier(n_bootstrap=100, random_state=42)
        v1, l1, h1 = uq1.bootstrap_ci(data, np.mean)

        uq2 = UncertaintyQuantifier(n_bootstrap=100, random_state=42)
        v2, l2, h2 = uq2.bootstrap_ci(data, np.mean)

        # Should be identical
        assert v1 == v2
        np.testing.assert_allclose([l1, h1], [l2, h2], rtol=1e-10)


class TestApplyDefaultUncertainty:
    """Tests for apply_default_uncertainty function."""

    def test_basic_application(self):
        """Test applying uncertainty to DataFrame."""
        df = pd.DataFrame({
            'cluster': [1, 1, 2, 2, 3, 3],
            'value': [10, 12, 20, 22, 30, 32]
        })

        df_result = apply_default_uncertainty(
            df,
            value_cols=['value'],
            group_col='cluster',
            n_bootstrap=100
        )

        assert 'value_ci_low' in df_result.columns
        assert 'value_ci_high' in df_result.columns

    def test_multiple_columns(self):
        """Test with multiple value columns."""
        df = pd.DataFrame({
            'metric1': np.random.normal(10, 2, 50),
            'metric2': np.random.normal(20, 3, 50)
        })

        df_result = apply_default_uncertainty(
            df,
            value_cols=['metric1', 'metric2'],
            n_bootstrap=100
        )

        assert 'metric1_ci_low' in df_result.columns
        assert 'metric1_ci_high' in df_result.columns
        assert 'metric2_ci_low' in df_result.columns
        assert 'metric2_ci_high' in df_result.columns

    def test_missing_column_warning(self):
        """Test warning for missing column."""
        df = pd.DataFrame({'value': [1, 2, 3]})

        with pytest.warns(UserWarning, match="Column 'nonexistent' not found"):
            df_result = apply_default_uncertainty(
                df,
                value_cols=['nonexistent'],
                n_bootstrap=100
            )


class TestStatisticalProperties:
    """Tests for statistical properties of uncertainty estimates."""

    def test_ci_coverage(self):
        """Test that 95% CI actually covers true parameter 95% of times."""
        true_mean = 100
        n_simulations = 100
        n_samples = 200
        coverage_count = 0

        uq = UncertaintyQuantifier(n_bootstrap=200, random_state=42)

        for seed in range(n_simulations):
            np.random.seed(seed)
            data = np.random.normal(true_mean, 15, n_samples)
            _, ci_low, ci_high = uq.bootstrap_ci(data, np.mean)

            if ci_low <= true_mean <= ci_high:
                coverage_count += 1

        coverage_rate = coverage_count / n_simulations

        # 95% CI should cover true parameter approximately 95% of times
        # Allow some tolerance (90-100%)
        assert 0.85 < coverage_rate < 1.0

    def test_permutation_null_uniform(self):
        """Test that p-values under null hypothesis are uniform."""
        n_simulations = 50
        p_values = []

        uq = UncertaintyQuantifier(n_permutations=100, random_state=42)

        for seed in range(n_simulations):
            np.random.seed(seed)
            # Two groups from same distribution (null hypothesis true)
            group1 = np.random.normal(0, 1, 50)
            group2 = np.random.normal(0, 1, 50)

            result = uq.permutation_test(
                group1, group2,
                statistic=lambda x, y: np.mean(x) - np.mean(y)
            )
            p_values.append(result['p_value'])

        # Under null, p-values should be approximately uniform
        # Check that mean is around 0.5
        assert 0.3 < np.mean(p_values) < 0.7
