#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Profiling Script for strepsuis-mdr

This script profiles the performance of key functions in the MDR analysis pipeline
using cProfile and line_profiler to identify bottlenecks.
"""

import cProfile
import pstats
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from strepsuis_mdr.mdr_analysis_core import (
    compute_bootstrap_ci,
    pairwise_cooccurrence,
    build_hybrid_co_resistance_network,
    compute_network_mdr_risk_score,
    detect_sequential_resistance_patterns,
)


def generate_test_data(n_strains: int, n_features: int, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic binary test data."""
    np.random.seed(seed)
    data = np.random.binomial(1, 0.3, size=(n_strains, n_features))
    columns = [f"Feature_{i}" for i in range(n_features)]
    return pd.DataFrame(data, columns=columns)


def profile_bootstrap(n_strains: int = 100, n_features: int = 20, n_iter: int = 1000):
    """Profile bootstrap CI computation."""
    print(f"\n{'='*60}")
    print(f"Profiling: Bootstrap CI")
    print(f"Parameters: {n_strains} strains, {n_features} features, {n_iter} iterations")
    print(f"{'='*60}")
    
    data = generate_test_data(n_strains, n_features)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    result = compute_bootstrap_ci(data, n_iter=n_iter, confidence_level=0.95)
    elapsed = time.time() - start_time
    
    profiler.disable()
    
    print(f"Execution time: {elapsed:.3f} seconds")
    print(f"Result shape: {result.shape}")
    
    # Save profile
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
    
    return elapsed, result


def profile_cooccurrence(n_strains: int = 100, n_features: int = 20):
    """Profile pairwise co-occurrence analysis."""
    print(f"\n{'='*60}")
    print(f"Profiling: Pairwise Co-occurrence")
    print(f"Parameters: {n_strains} strains, {n_features} features")
    print(f"{'='*60}")
    
    data = generate_test_data(n_strains, n_features)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    result = pairwise_cooccurrence(data, alpha=0.05, method='fdr_bh')
    elapsed = time.time() - start_time
    
    profiler.disable()
    
    print(f"Execution time: {elapsed:.3f} seconds")
    print(f"Result shape: {result.shape}")
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    
    return elapsed, result


def profile_network_construction(n_strains: int = 100, n_features: int = 20):
    """Profile network construction."""
    print(f"\n{'='*60}")
    print(f"Profiling: Network Construction")
    print(f"Parameters: {n_strains} strains, {n_features} features")
    print(f"{'='*60}")
    
    data = generate_test_data(n_strains, n_features)
    
    # Split features into phenotypes and genes (half each)
    n_pheno = n_features // 2
    pheno_cols = [f"Feature_{i}" for i in range(n_pheno)]
    gene_cols = [f"Feature_{i}" for i in range(n_pheno, n_features)]
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    network = build_hybrid_co_resistance_network(
        data,
        pheno_cols,
        gene_cols,
        alpha=0.05,
        method='fdr_bh'
    )
    elapsed = time.time() - start_time
    
    profiler.disable()
    
    print(f"Execution time: {elapsed:.3f} seconds")
    print(f"Network nodes: {network.number_of_nodes()}")
    print(f"Network edges: {network.number_of_edges()}")
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    
    return elapsed, network


def profile_network_risk_scoring(n_strains: int = 100, n_features: int = 20):
    """Profile network risk scoring."""
    print(f"\n{'='*60}")
    print(f"Profiling: Network Risk Scoring")
    print(f"Parameters: {n_strains} strains, {n_features} features")
    print(f"{'='*60}")
    
    data = generate_test_data(n_strains, n_features)
    
    # Split features into phenotypes and genes (half each)
    n_pheno = n_features // 2
    pheno_cols = [f"Feature_{i}" for i in range(n_pheno)]
    gene_cols = [f"Feature_{i}" for i in range(n_pheno, n_features)]
    
    # Setup: get bootstrap CI and network
    bootstrap_ci = compute_bootstrap_ci(data, n_iter=1000, confidence_level=0.95)
    network = build_hybrid_co_resistance_network(
        data, pheno_cols, gene_cols, alpha=0.05, method='fdr_bh'
    )
    
    # Convert bootstrap CI to dict format
    ci_dict = {
        row['ColumnName']: (row['CI_Lower'] / 100, row['CI_Upper'] / 100)
        for _, row in bootstrap_ci.iterrows()
    }
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    result = compute_network_mdr_risk_score(
        network, 
        data, 
        ci_dict, 
        percentile_threshold=75.0
    )
    elapsed = time.time() - start_time
    
    profiler.disable()
    
    print(f"Execution time: {elapsed:.3f} seconds")
    print(f"Result shape: {result.shape}")
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    
    return elapsed, result


def profile_sequential_patterns(n_strains: int = 100, n_features: int = 20):
    """Profile sequential pattern detection."""
    print(f"\n{'='*60}")
    print(f"Profiling: Sequential Pattern Detection")
    print(f"Parameters: {n_strains} strains, {n_features} features")
    print(f"{'='*60}")
    
    data = generate_test_data(n_strains, n_features)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    result = detect_sequential_resistance_patterns(
        data,
        min_support=0.1,
        min_confidence=0.5,
        correlation_threshold=0.3
    )
    elapsed = time.time() - start_time
    
    profiler.disable()
    
    print(f"Execution time: {elapsed:.3f} seconds")
    print(f"Result shape: {result.shape}")
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    
    return elapsed, result


def run_scalability_tests():
    """Run scalability tests across different dataset sizes."""
    print(f"\n{'='*80}")
    print("SCALABILITY TESTS")
    print(f"{'='*80}")
    
    test_sizes = [
        (50, 10),
        (100, 20),
        (200, 20),
        (500, 50),
        (1000, 50),
    ]
    
    results = {
        'bootstrap': [],
        'cooccurrence': [],
        'network': [],
        'risk_scoring': [],
        'sequential': []
    }
    
    for n_strains, n_features in test_sizes:
        print(f"\n--- Testing: {n_strains} strains, {n_features} features ---")
        
        # Bootstrap
        try:
            elapsed, _ = profile_bootstrap(n_strains, n_features, n_iter=1000)
            results['bootstrap'].append((n_strains, n_features, elapsed))
        except Exception as e:
            print(f"Bootstrap failed: {e}")
        
        # Co-occurrence
        try:
            elapsed, _ = profile_cooccurrence(n_strains, n_features)
            results['cooccurrence'].append((n_strains, n_features, elapsed))
        except Exception as e:
            print(f"Co-occurrence failed: {e}")
        
        # Network (smaller sizes only)
        if n_strains <= 200:
            try:
                elapsed, _ = profile_network_construction(n_strains, n_features)
                results['network'].append((n_strains, n_features, elapsed))
            except Exception as e:
                print(f"Network construction failed: {e}")
    
    return results


def main():
    """Main profiling function."""
    print("="*80)
    print("PERFORMANCE PROFILING - strepsuis-mdr")
    print("="*80)
    
    # Individual function profiling
    print("\n" + "="*80)
    print("INDIVIDUAL FUNCTION PROFILING")
    print("="*80)
    
    profile_bootstrap(100, 20, n_iter=1000)
    profile_cooccurrence(100, 20)
    profile_network_construction(100, 20)
    profile_network_risk_scoring(100, 20)
    profile_sequential_patterns(100, 20)
    
    # Scalability tests
    results = run_scalability_tests()
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    for func_name, timings in results.items():
        if timings:
            print(f"\n{func_name.upper()}:")
            for n_strains, n_features, elapsed in timings:
                print(f"  {n_strains:4d} strains, {n_features:2d} features: {elapsed:6.3f}s")


if __name__ == "__main__":
    main()

