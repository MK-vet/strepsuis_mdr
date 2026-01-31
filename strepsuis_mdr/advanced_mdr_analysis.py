#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced MDR Analysis Module for StrepSuis-MDR
==============================================

This module integrates advanced statistical methods for robust
antimicrobial resistance pattern analysis.

Features:
    - Confidence-aware Rules: Association rules with bootstrap CI
    - Edge Confidence Network: Robust co-resistance networks
    - Entropy-weighted Importance: Adjusted resistance gene importance
    - Bootstrap Stability: Test robustness of resistance patterns
    - Redundancy Pruning: Detect redundant resistance genes

Author: MK-vet
Version: 1.0.0
License: MIT
"""

from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np
import logging

try:
    from shared.advanced_statistics import (
        confidence_aware_rules,
        edge_confidence_network,
        entropy_weighted_importance,
        bootstrap_stability_matrix,
        redundancy_pruning,
    )
    HAS_ADVANCED_STATS = True
except ImportError:
    HAS_ADVANCED_STATS = False
    logging.warning(
        "Advanced statistics module not available. "
        "Install with: pip install -e ../shared"
    )

logger = logging.getLogger(__name__)


def robust_association_rules(
    resistance_data: pd.DataFrame,
    min_support: float = 0.05,
    min_confidence: float = 0.7,
    n_bootstrap: int = 500,
    ci_level: float = 0.95,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Find resistance gene association rules with bootstrap confidence intervals.

    Provides robust assessment of co-resistance patterns with uncertainty
    quantification for the lift metric.

    Parameters
    ----------
    resistance_data : pd.DataFrame
        Binary resistance gene presence/absence matrix (samples × genes)
    min_support : float, default=0.05
        Minimum support threshold (5%)
    min_confidence : float, default=0.7
        Minimum confidence threshold (70%)
    n_bootstrap : int, default=500
        Number of bootstrap iterations
    ci_level : float, default=0.95
        Confidence interval level
    n_jobs : int, default=-1
        Number of parallel jobs

    Returns
    -------
    pd.DataFrame
        Association rules with CI bounds and significance flags

    Example
    -------
    >>> from strepsuis_mdr.advanced_mdr_analysis import robust_association_rules
    >>>
    >>> # Load resistance data
    >>> resistance_df = pd.read_csv("amr_genes.csv", index_col=0)
    >>>
    >>> # Find robust rules
    >>> rules = robust_association_rules(
    ...     resistance_data=resistance_df,
    ...     min_support=0.05,
    ...     min_confidence=0.7,
    ...     n_bootstrap=500
    ... )
    >>>
    >>> # Filter significant rules (CI excludes 1)
    >>> significant = rules[rules['is_significant']]
    >>> print(f"Found {len(significant)} significant co-resistance patterns")
    """
    if not HAS_ADVANCED_STATS:
        raise ImportError(
            "Advanced statistics module is required. "
            "Install shared module: pip install -e ../shared"
        )

    logger.info("Computing association rules with bootstrap confidence intervals...")

    rules_df = confidence_aware_rules(
        data=resistance_data,
        min_support=min_support,
        min_confidence=min_confidence,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        n_jobs=n_jobs
    )

    if not rules_df.empty:
        n_sig = rules_df['is_significant'].sum()
        logger.info(f"Found {len(rules_df)} total rules, {n_sig} significant")

    return rules_df


def build_robust_coresistance_network(
    resistance_data: pd.DataFrame,
    edge_function: Callable,
    n_bootstrap: int = 500,
    confidence_threshold: float = 0.8,
    n_jobs: int = -1
) -> Dict:
    """
    Build co-resistance network with bootstrap-validated edges.

    Only edges (gene-gene associations) that appear consistently across
    bootstrap iterations are retained.

    Parameters
    ----------
    resistance_data : pd.DataFrame
        Binary resistance matrix
    edge_function : callable
        Function that takes DataFrame and returns list of (gene1, gene2, weight) tuples
    n_bootstrap : int, default=500
        Number of bootstrap iterations
    confidence_threshold : float, default=0.8
        Minimum edge confidence (0.8 = must appear in 80% of bootstraps)
    n_jobs : int, default=-1
        Number of parallel jobs

    Returns
    -------
    dict
        Dictionary with edge statistics and high-confidence network

    Example
    -------
    >>> def calc_cooccurrence(df):
    ...     edges = []
    ...     for g1 in df.columns:
    ...         for g2 in df.columns:
    ...             if g1 < g2:
    ...                 cooccur = ((df[g1] == 1) & (df[g2] == 1)).sum() / len(df)
    ...                 if cooccur > 0.1:
    ...                     edges.append((g1, g2, cooccur))
    ...     return edges
    >>>
    >>> network = build_robust_coresistance_network(
    ...     resistance_data=amr_df,
    ...     edge_function=calc_cooccurrence,
    ...     confidence_threshold=0.8
    ... )
    >>> print(f"Network: {network['n_high_confidence']} high-confidence edges")
    """
    if not HAS_ADVANCED_STATS:
        raise ImportError(
            "Advanced statistics module is required. "
            "Install shared module: pip install -e ../shared"
        )

    logger.info("Building robust co-resistance network with bootstrap validation...")

    network_stats = edge_confidence_network(
        data=resistance_data,
        edge_func=edge_function,
        n_bootstrap=n_bootstrap,
        confidence_threshold=confidence_threshold,
        n_jobs=n_jobs
    )

    return network_stats


def adjust_resistance_gene_importance(
    resistance_data: pd.DataFrame,
    base_importance: pd.Series,
    entropy_penalty: float = 0.5
) -> pd.DataFrame:
    """
    Adjust resistance gene importance by entropy.

    Genes that are present in nearly all or nearly no samples (low entropy)
    are down-weighted as they provide less discriminatory power.

    Parameters
    ----------
    resistance_data : pd.DataFrame
        Binary resistance matrix
    base_importance : pd.Series
        Initial importance scores (e.g., from Random Forest)
    entropy_penalty : float, default=0.5
        How much to penalize low-entropy genes (0-1)

    Returns
    -------
    pd.DataFrame
        DataFrame with original and entropy-adjusted importance

    Example
    -------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>>
    >>> # Get base importance
    >>> rf = RandomForestClassifier(n_estimators=100, random_state=42)
    >>> rf.fit(resistance_genes, mdr_labels)
    >>> base_imp = pd.Series(rf.feature_importances_, index=resistance_genes.columns)
    >>>
    >>> # Adjust by entropy
    >>> adjusted = adjust_resistance_gene_importance(
    ...     resistance_data=resistance_genes,
    ...     base_importance=base_imp
    ... )
    >>>
    >>> # Compare rankings
    >>> print("Top 10 genes (base):", adjusted.nlargest(10, 'base_importance').index.tolist())
    >>> print("Top 10 genes (adjusted):", adjusted.nlargest(10, 'adjusted_importance').index.tolist())
    """
    if not HAS_ADVANCED_STATS:
        raise ImportError(
            "Advanced statistics module is required. "
            "Install shared module: pip install -e ../shared"
        )

    logger.info("Adjusting gene importance by entropy...")

    adjusted_df = entropy_weighted_importance(
        data=resistance_data,
        base_importance=base_importance,
        entropy_penalty=entropy_penalty
    )

    return adjusted_df


def test_resistance_pattern_stability(
    resistance_data: pd.DataFrame,
    pattern_analysis_func: Callable,
    n_bootstrap: int = 500,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Test stability of resistance patterns across bootstrap samples.

    Parameters
    ----------
    resistance_data : pd.DataFrame
        Binary resistance matrix
    pattern_analysis_func : callable
        Function that takes DataFrame and returns list of significant genes
    n_bootstrap : int, default=500
        Number of bootstrap iterations
    n_jobs : int, default=-1
        Number of parallel jobs

    Returns
    -------
    pd.DataFrame
        Stability percentage for each resistance gene

    Example
    -------
    >>> def find_significant_genes(df):
    ...     # Your analysis to identify significant resistance genes
    ...     from scipy.stats import chi2_contingency
    ...     significant = []
    ...     for gene in df.columns:
    ...         if df[gene].sum() > 5:  # At least 5 positive samples
    ...             significant.append(gene)
    ...     return significant
    >>>
    >>> stability = test_resistance_pattern_stability(
    ...     resistance_data=amr_genes,
    ...     pattern_analysis_func=find_significant_genes,
    ...     n_bootstrap=500
    ... )
    >>>
    >>> # Most stable genes
    >>> print(stability.sort_values(ascending=False).head(10))
    """
    if not HAS_ADVANCED_STATS:
        raise ImportError(
            "Advanced statistics module is required. "
            "Install shared module: pip install -e ../shared"
        )

    logger.info(f"Testing pattern stability across {n_bootstrap} bootstrap samples...")

    stability_df = bootstrap_stability_matrix(
        data=resistance_data,
        analysis_func=pattern_analysis_func,
        n_bootstrap=n_bootstrap,
        n_jobs=n_jobs
    )

    return stability_df


def detect_redundant_resistance_genes(
    resistance_data: pd.DataFrame,
    jaccard_threshold: float = 0.9
) -> Dict:
    """
    Identify highly redundant resistance genes.

    Genes with Jaccard similarity > threshold have nearly identical
    presence/absence patterns and may be functionally redundant.

    Parameters
    ----------
    resistance_data : pd.DataFrame
        Binary resistance matrix
    jaccard_threshold : float, default=0.9
        Jaccard similarity threshold for redundancy (0.9 = 90% similar)

    Returns
    -------
    dict
        Dictionary with redundant gene pairs and statistics

    Example
    -------
    >>> redundancy = detect_redundant_resistance_genes(
    ...     resistance_data=amr_genes,
    ...     jaccard_threshold=0.9
    ... )
    >>>
    >>> print(f"Found {redundancy['n_redundant_pairs']} redundant gene pairs")
    >>> for pair in redundancy['redundant_pairs'][:10]:
    ...     print(f"{pair['feature1']} ↔ {pair['feature2']}: "
    ...           f"Jaccard = {pair['jaccard']:.3f}")
    """
    if not HAS_ADVANCED_STATS:
        raise ImportError(
            "Advanced statistics module is required. "
            "Install shared module: pip install -e ../shared"
        )

    logger.info("Detecting redundant resistance genes...")

    redundancy_report = redundancy_pruning(
        data=resistance_data,
        jaccard_threshold=jaccard_threshold
    )

    n_pairs = redundancy_report['n_redundant_pairs']
    logger.info(f"Found {n_pairs} redundant gene pairs")

    return redundancy_report


def comprehensive_mdr_analysis(
    resistance_data: pd.DataFrame,
    base_importance: Optional[pd.Series] = None,
    edge_function: Optional[Callable] = None,
    pattern_func: Optional[Callable] = None,
    min_rule_support: float = 0.05,
    min_rule_confidence: float = 0.7,
    n_bootstrap: int = 500,
    n_jobs: int = -1
) -> Dict:
    """
    Run comprehensive advanced MDR analysis pipeline.

    Combines all advanced statistical methods for robust resistance analysis.

    Parameters
    ----------
    resistance_data : pd.DataFrame
        Binary resistance gene matrix
    base_importance : pd.Series, optional
        Initial gene importance scores
    edge_function : callable, optional
        Function for network edge calculation
    pattern_func : callable, optional
        Function for pattern analysis
    min_rule_support : float, default=0.05
        Minimum support for association rules
    min_rule_confidence : float, default=0.7
        Minimum confidence for association rules
    n_bootstrap : int, default=500
        Number of bootstrap iterations
    n_jobs : int, default=-1
        Number of parallel jobs

    Returns
    -------
    dict
        Dictionary with all analysis results

    Example
    -------
    >>> results = comprehensive_mdr_analysis(
    ...     resistance_data=amr_genes,
    ...     base_importance=rf_importance,
    ...     min_rule_support=0.05,
    ...     n_bootstrap=500
    ... )
    >>>
    >>> # Access different components
    >>> significant_rules = results['rules'][results['rules']['is_significant']]
    >>> redundant_genes = results['redundancy']['redundant_pairs']
    >>> top_genes = results['entropy_adjusted'].head(10)
    """
    if not HAS_ADVANCED_STATS:
        raise ImportError(
            "Advanced statistics module is required. "
            "Install shared module: pip install -e ../shared"
        )

    logger.info("Running comprehensive advanced MDR analysis...")

    results = {}

    # 1. Robust association rules
    logger.info("Step 1/5: Computing robust association rules...")
    results['rules'] = robust_association_rules(
        resistance_data=resistance_data,
        min_support=min_rule_support,
        min_confidence=min_rule_confidence,
        n_bootstrap=n_bootstrap,
        n_jobs=n_jobs
    )

    # 2. Redundancy detection
    logger.info("Step 2/5: Detecting redundant genes...")
    results['redundancy'] = detect_redundant_resistance_genes(
        resistance_data=resistance_data,
        jaccard_threshold=0.9
    )

    # 3. Entropy-weighted importance (if provided)
    if base_importance is not None:
        logger.info("Step 3/5: Adjusting gene importance by entropy...")
        results['entropy_adjusted'] = adjust_resistance_gene_importance(
            resistance_data=resistance_data,
            base_importance=base_importance,
            entropy_penalty=0.5
        )
    else:
        logger.info("Step 3/5: Skipped (no base importance provided)")

    # 4. Network analysis (if edge function provided)
    if edge_function is not None:
        logger.info("Step 4/5: Building robust co-resistance network...")
        results['network'] = build_robust_coresistance_network(
            resistance_data=resistance_data,
            edge_function=edge_function,
            n_bootstrap=n_bootstrap,
            n_jobs=n_jobs
        )
    else:
        logger.info("Step 4/5: Skipped (no edge function provided)")

    # 5. Pattern stability (if pattern function provided)
    if pattern_func is not None:
        logger.info("Step 5/5: Testing pattern stability...")
        results['stability'] = test_resistance_pattern_stability(
            resistance_data=resistance_data,
            pattern_analysis_func=pattern_func,
            n_bootstrap=n_bootstrap,
            n_jobs=n_jobs
        )
    else:
        logger.info("Step 5/5: Skipped (no pattern function provided)")

    logger.info("Comprehensive analysis complete!")
    return results


def generate_mdr_advanced_report(
    results: Dict,
    output_file: str = "mdr_advanced_report.html"
) -> None:
    """
    Generate HTML report for advanced MDR analysis.

    Parameters
    ----------
    results : dict
        Results dictionary from comprehensive_mdr_analysis()
    output_file : str
        Output HTML file path
    """
    html = ["<html><head><title>Advanced MDR Analysis</title>"]
    html.append("<style>")
    html.append("body { font-family: Arial, sans-serif; margin: 20px; }")
    html.append("table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
    html.append("th, td { border: 1px solid #ddd; padding: 8px; }")
    html.append("th { background-color: #C62828; color: white; }")
    html.append("h1 { color: #B71C1C; }")
    html.append("h2 { color: #D32F2F; }")
    html.append(".metric { background-color: #FFEBEE; padding: 10px; margin: 10px 0; border-radius: 5px; }")
    html.append("</style></head><body>")

    html.append("<h1>Advanced Multidrug Resistance Analysis Report</h1>")

    # Association rules
    if 'rules' in results and not results['rules'].empty:
        html.append("<h2>Robust Association Rules (Top 15 Significant)</h2>")
        sig_rules = results['rules'][results['rules']['is_significant']]
        html.append(f"<p>Found <strong>{len(sig_rules)}</strong> significant co-resistance rules</p>")
        if len(sig_rules) > 0:
            display_cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift', 'lift_ci_lower', 'lift_ci_upper']
            html.append(sig_rules[display_cols].head(15).to_html(index=False))

    # Redundancy
    if 'redundancy' in results:
        html.append("<h2>Redundant Resistance Genes</h2>")
        n_pairs = results['redundancy']['n_redundant_pairs']
        html.append(f"<p>Found <strong>{n_pairs}</strong> highly redundant gene pairs</p>")
        if n_pairs > 0:
            html.append("<ul>")
            for pair in results['redundancy']['redundant_pairs'][:10]:
                html.append(
                    f"<li>{pair['feature1']} ↔ {pair['feature2']}: "
                    f"Jaccard = {pair['jaccard']:.3f}</li>"
                )
            html.append("</ul>")

    # Entropy-adjusted importance
    if 'entropy_adjusted' in results:
        html.append("<h2>Top Resistance Genes (Entropy-Adjusted Importance)</h2>")
        html.append(results['entropy_adjusted'].head(15).to_html())

    # Network
    if 'network' in results:
        html.append("<h2>Co-Resistance Network Statistics</h2>")
        net = results['network']
        html.append(f"<div class='metric'>")
        html.append(f"<strong>Total edges:</strong> {net['n_total_edges']}<br>")
        html.append(f"<strong>High-confidence edges:</strong> {net['n_high_confidence']}<br>")
        html.append(f"<strong>Edge retention rate:</strong> "
                   f"{net['n_high_confidence'] / max(net['n_total_edges'], 1):.1%}")
        html.append(f"</div>")

    # Stability
    if 'stability' in results:
        html.append("<h2>Most Stable Resistance Genes (Top 15)</h2>")
        top_stable = results['stability'].sort_values(ascending=False).head(15)
        html.append(top_stable.to_frame('Stability (%)').to_html())

    html.append("</body></html>")

    with open(output_file, 'w') as f:
        f.write('\n'.join(html))

    logger.info(f"Advanced MDR analysis report saved to: {output_file}")
