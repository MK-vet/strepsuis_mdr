#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uncertainty Quantification Module
==================================

Provides comprehensive uncertainty quantification for clustering results,
including confidence intervals, p-values, and stability metrics.

Features:
    - Bootstrap confidence intervals (default for all metrics)
    - Permutation p-values for hypothesis testing
    - FDR correction (Benjamini-Hochberg)
    - Cluster stability metrics (ARI, Jaccard)
    - Consensus clustering across bootstrap samples

Scientific Rationale:
    - Point estimates without uncertainty are insufficient for publication
    - Bootstrap CI captures sampling variability
    - Permutation tests provide distribution-free p-values
    - FDR correction controls false discovery rate in multiple testing
    - Stability metrics assess robustness of clustering solutions

Author: MK-vet
Version: 1.0.0
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Callable, Any
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score, jaccard_score
from sklearn.utils import resample
import warnings


class UncertaintyQuantifier:
    """
    Comprehensive uncertainty quantification for clustering analysis.

    Provides bootstrap CI, permutation tests, stability metrics, and
    consensus clustering.

    Example:
        >>> uq = UncertaintyQuantifier(n_bootstrap=1000, random_state=42)
        >>> result = uq.quantify_metric(
        ...     metric_func=lambda x: np.mean(x),
        ...     data=cluster_sizes,
        ...     metric_name="mean_cluster_size"
        ... )
        >>> print(f"Mean: {result['value']:.2f} [{result['ci_low']:.2f}, {result['ci_high']:.2f}]")
    """

    def __init__(
        self,
        n_bootstrap: int = 1000,
        n_permutations: int = 1000,
        confidence_level: float = 0.95,
        random_state: Optional[int] = 42
    ):
        """
        Initialize uncertainty quantifier.

        Args:
            n_bootstrap: Number of bootstrap samples
            n_permutations: Number of permutation samples for p-values
            confidence_level: Confidence level for intervals (default: 0.95)
            random_state: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.n_permutations = n_permutations
        self.confidence_level = confidence_level
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def bootstrap_ci(
        self,
        data: np.ndarray,
        statistic: Callable,
        **kwargs
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence interval for any statistic.

        Args:
            data: Input data array
            statistic: Function that computes the statistic
            **kwargs: Additional arguments to statistic function

        Returns:
            Tuple of (point_estimate, ci_lower, ci_upper)

        Example:
            >>> data = np.array([1, 2, 3, 4, 5])
            >>> value, low, high = uq.bootstrap_ci(data, np.mean)
            >>> print(f"{value:.2f} [{low:.2f}, {high:.2f}]")
        """
        # Compute point estimate
        point_estimate = statistic(data, **kwargs)

        # Bootstrap resampling
        bootstrap_estimates = []
        for _ in range(self.n_bootstrap):
            sample = resample(data, random_state=self.rng.integers(0, 1e9))
            bootstrap_estimates.append(statistic(sample, **kwargs))

        bootstrap_estimates = np.array(bootstrap_estimates)

        # Compute percentile CI
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_estimates, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_estimates, (1 - alpha/2) * 100)

        return point_estimate, ci_lower, ci_upper

    def permutation_test(
        self,
        data1: np.ndarray,
        data2: Optional[np.ndarray] = None,
        statistic: Callable = None,
        observed_stat: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Permutation test for hypothesis testing.

        Computes p-value by comparing observed statistic to distribution
        under null hypothesis (random permutation of labels).

        Args:
            data1: First group data (or combined data if data2 is None)
            data2: Second group data (optional)
            statistic: Function to compute test statistic
            observed_stat: Pre-computed observed statistic (optional)

        Returns:
            Dictionary with keys: 'p_value', 'observed_stat', 'null_mean', 'null_std'

        Example:
            >>> # Two-sample test
            >>> result = uq.permutation_test(
            ...     group1, group2,
            ...     statistic=lambda x, y: np.mean(x) - np.mean(y)
            ... )
            >>> print(f"p-value: {result['p_value']:.4f}")
        """
        if statistic is None:
            raise ValueError("Must provide statistic function")

        # Compute observed statistic
        if observed_stat is None:
            if data2 is not None:
                observed_stat = statistic(data1, data2)
            else:
                observed_stat = statistic(data1)

        # Generate null distribution
        null_distribution = []

        if data2 is not None:
            # Two-sample permutation test
            combined = np.concatenate([data1, data2])
            n1 = len(data1)

            for _ in range(self.n_permutations):
                perm = self.rng.permutation(combined)
                perm_stat = statistic(perm[:n1], perm[n1:])
                null_distribution.append(perm_stat)
        else:
            # One-sample permutation test (e.g., label permutation)
            for _ in range(self.n_permutations):
                perm_data = self.rng.permutation(data1)
                perm_stat = statistic(perm_data)
                null_distribution.append(perm_stat)

        null_distribution = np.array(null_distribution)

        # Compute p-value (two-tailed)
        p_value = np.mean(np.abs(null_distribution) >= np.abs(observed_stat))

        return {
            'p_value': p_value,
            'observed_stat': observed_stat,
            'null_mean': np.mean(null_distribution),
            'null_std': np.std(null_distribution)
        }

    def fdr_correction(
        self,
        p_values: np.ndarray,
        method: str = 'fdr_bh',
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        FDR correction for multiple testing.

        Args:
            p_values: Array of p-values
            method: Correction method ('fdr_bh' for Benjamini-Hochberg)
            alpha: Significance level

        Returns:
            Tuple of (reject_mask, corrected_p_values)

        Example:
            >>> p_vals = np.array([0.001, 0.01, 0.05, 0.1])
            >>> reject, p_adj = uq.fdr_correction(p_vals)
            >>> print(f"Significant: {reject}")
            >>> print(f"Adjusted p-values: {p_adj}")
        """
        from statsmodels.stats.multitest import multipletests

        reject, p_adjusted, _, _ = multipletests(
            p_values,
            alpha=alpha,
            method=method,
            is_sorted=False,
            returnsorted=False
        )

        return reject, p_adjusted

    def cluster_stability(
        self,
        X: np.ndarray,
        clustering_func: Callable,
        n_iterations: int = 100,
        sample_fraction: float = 0.8
    ) -> Dict[str, float]:
        """
        Assess clustering stability using bootstrap resampling.

        Measures how consistent cluster assignments are across bootstrap
        samples using ARI (Adjusted Rand Index).

        Args:
            X: Data matrix
            clustering_func: Function that takes X and returns cluster labels
            n_iterations: Number of bootstrap iterations
            sample_fraction: Fraction of data to sample

        Returns:
            Dictionary with stability metrics

        Example:
            >>> def kmodes_func(X):
            ...     km = KModes(n_clusters=3)
            ...     return km.fit_predict(X)
            >>> stability = uq.cluster_stability(X, kmodes_func)
            >>> print(f"Mean ARI: {stability['mean_ari']:.3f}")
        """
        n_samples = len(X)
        sample_size = int(n_samples * sample_fraction)

        # Reference clustering (full data)
        labels_reference = clustering_func(X)

        ari_scores = []
        jaccard_scores = []

        for _ in range(n_iterations):
            # Bootstrap sample
            indices = self.rng.choice(n_samples, size=sample_size, replace=True)
            X_sample = X[indices]

            # Cluster bootstrap sample
            labels_sample = clustering_func(X_sample)

            # Map back to full data (only for sampled indices)
            labels_full_sample = np.full(n_samples, -1)
            labels_full_sample[indices] = labels_sample

            # Compute ARI (only on sampled indices)
            sampled_mask = labels_full_sample != -1
            if np.sum(sampled_mask) > 0:
                ari = adjusted_rand_score(
                    labels_reference[sampled_mask],
                    labels_full_sample[sampled_mask]
                )
                ari_scores.append(ari)

                # Jaccard (binary: same cluster or not)
                # Convert to binary problem for each cluster
                for cluster_id in np.unique(labels_reference[sampled_mask]):
                    ref_binary = (labels_reference[sampled_mask] == cluster_id).astype(int)
                    sample_binary = (labels_full_sample[sampled_mask] == cluster_id).astype(int)
                    if ref_binary.sum() > 0:  # Avoid empty clusters
                        jacc = jaccard_score(ref_binary, sample_binary, average='binary')
                        jaccard_scores.append(jacc)

        return {
            'mean_ari': np.mean(ari_scores),
            'std_ari': np.std(ari_scores),
            'ci_ari_low': np.percentile(ari_scores, 2.5),
            'ci_ari_high': np.percentile(ari_scores, 97.5),
            'mean_jaccard': np.mean(jaccard_scores) if jaccard_scores else np.nan,
            'std_jaccard': np.std(jaccard_scores) if jaccard_scores else np.nan
        }

    def consensus_clustering(
        self,
        X: np.ndarray,
        clustering_func: Callable,
        n_iterations: int = 100,
        sample_fraction: float = 0.8
    ) -> np.ndarray:
        """
        Consensus clustering via bootstrap aggregation.

        Runs clustering on bootstrap samples and aggregates results
        using co-occurrence matrix and hierarchical clustering.

        Args:
            X: Data matrix
            clustering_func: Function that takes X and returns cluster labels
            n_iterations: Number of bootstrap iterations
            sample_fraction: Fraction of data to sample

        Returns:
            Consensus cluster labels

        Example:
            >>> consensus_labels = uq.consensus_clustering(X, kmodes_func)
            >>> print(f"Consensus clusters: {np.unique(consensus_labels)}")
        """
        n_samples = len(X)
        sample_size = int(n_samples * sample_fraction)

        # Co-occurrence matrix: how often do pairs cluster together?
        cooccurrence = np.zeros((n_samples, n_samples))
        counts = np.zeros((n_samples, n_samples))

        for _ in range(n_iterations):
            # Bootstrap sample
            indices = self.rng.choice(n_samples, size=sample_size, replace=False)
            X_sample = X[indices]

            # Cluster
            labels_sample = clustering_func(X_sample)

            # Update co-occurrence for sampled pairs
            for i, idx_i in enumerate(indices):
                for j, idx_j in enumerate(indices):
                    if i < j:
                        counts[idx_i, idx_j] += 1
                        counts[idx_j, idx_i] += 1
                        if labels_sample[i] == labels_sample[j]:
                            cooccurrence[idx_i, idx_j] += 1
                            cooccurrence[idx_j, idx_i] += 1

        # Consensus matrix: proportion of times pairs cluster together
        consensus_matrix = np.divide(
            cooccurrence,
            counts,
            out=np.zeros_like(cooccurrence),
            where=counts > 0
        )

        # Convert to distance matrix (1 - consensus)
        distance_matrix = 1 - consensus_matrix

        # Hierarchical clustering on consensus matrix
        # Use average linkage
        np.fill_diagonal(distance_matrix, 0)  # Ensure diagonal is 0
        condensed_dist = distance_matrix[np.triu_indices(n_samples, k=1)]

        linkage_matrix = linkage(condensed_dist, method='average')

        # Determine optimal number of clusters from reference
        labels_reference = clustering_func(X)
        n_clusters = len(np.unique(labels_reference))

        # Cut dendrogram
        consensus_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

        return consensus_labels - 1  # Convert to 0-indexed

    def quantify_metric(
        self,
        metric_func: Callable,
        data: Any,
        metric_name: str = "metric",
        include_permutation: bool = False,
        null_data: Optional[Any] = None
    ) -> Dict[str, float]:
        """
        Comprehensive quantification of a single metric.

        Computes point estimate, bootstrap CI, and optionally permutation p-value.

        Args:
            metric_func: Function to compute metric
            data: Input data
            metric_name: Name of metric (for output)
            include_permutation: Whether to compute permutation p-value
            null_data: Data for permutation test (if different from input)

        Returns:
            Dictionary with all uncertainty metrics

        Example:
            >>> result = uq.quantify_metric(
            ...     np.mean, cluster_sizes,
            ...     metric_name="mean_cluster_size",
            ...     include_permutation=True
            ... )
        """
        # Bootstrap CI
        value, ci_low, ci_high = self.bootstrap_ci(data, metric_func)

        result = {
            'metric': metric_name,
            'value': value,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'ci_level': self.confidence_level
        }

        # Permutation test
        if include_permutation:
            perm_result = self.permutation_test(
                data if null_data is None else null_data,
                statistic=metric_func,
                observed_stat=value
            )
            result.update({
                'p_value': perm_result['p_value'],
                'null_mean': perm_result['null_mean'],
                'null_std': perm_result['null_std']
            })

        return result


def apply_default_uncertainty(
    df: pd.DataFrame,
    value_cols: List[str],
    group_col: Optional[str] = None,
    n_bootstrap: int = 1000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Apply default uncertainty quantification to DataFrame.

    Adds bootstrap CI columns for specified value columns.

    Args:
        df: Input DataFrame
        value_cols: List of columns to quantify
        group_col: Optional grouping column
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed

    Returns:
        DataFrame with added CI columns

    Example:
        >>> df_with_ci = apply_default_uncertainty(
        ...     df, value_cols=['mean', 'std'], group_col='cluster'
        ... )
    """
    uq = UncertaintyQuantifier(n_bootstrap=n_bootstrap, random_state=random_state)

    df_result = df.copy()

    for col in value_cols:
        if col not in df.columns:
            warnings.warn(f"Column '{col}' not found in DataFrame")
            continue

        ci_low_col = f"{col}_ci_low"
        ci_high_col = f"{col}_ci_high"

        if group_col is not None and group_col in df.columns:
            # Group-wise CI - compute for each row based on its group
            ci_lows = []
            ci_highs = []

            # Precompute CI for each group
            group_cis = {}
            for group_name, group_df in df.groupby(group_col):
                data = group_df[col].values
                _, ci_low, ci_high = uq.bootstrap_ci(data, np.mean)
                group_cis[group_name] = (ci_low, ci_high)

            # Assign CI to each row based on its group
            for _, row in df.iterrows():
                group_val = row[group_col]
                ci_low, ci_high = group_cis[group_val]
                ci_lows.append(ci_low)
                ci_highs.append(ci_high)

            df_result[ci_low_col] = ci_lows
            df_result[ci_high_col] = ci_highs
        else:
            # Single CI for all data
            data = df[col].values
            _, ci_low, ci_high = uq.bootstrap_ci(data, np.mean)
            df_result[ci_low_col] = ci_low
            df_result[ci_high_col] = ci_high

    return df_result
