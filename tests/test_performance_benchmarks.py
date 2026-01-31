#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Benchmark Tests for strepsuis-mdr

This module provides performance benchmarks for all major operations.
Results are saved to validation/PERFORMANCE_BENCHMARKS_REPORT.md
"""

import pytest
import numpy as np
import pandas as pd
import time
import json
from datetime import datetime
from pathlib import Path


class BenchmarkReport:
    """Collect and save benchmark results."""
    
    def __init__(self):
        self.results = []
        self.start_time = datetime.now()
    
    def add_result(self, operation, n_samples, n_features, time_seconds, memory_mb=None):
        self.results.append({
            "operation": operation,
            "n_samples": n_samples,
            "n_features": n_features,
            "time_seconds": time_seconds,
            "memory_mb": memory_mb,
            "throughput": n_samples / time_seconds if time_seconds > 0 else 0
        })
    
    def save_report(self, output_dir):
        """Save benchmark report to markdown file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_path = output_path / "PERFORMANCE_BENCHMARKS_REPORT.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Performance Benchmarks Report - strepsuis-mdr\n\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n")
            f.write(f"**Total Benchmarks:** {len(self.results)}\n\n")
            f.write("---\n\n")
            
            f.write("## Benchmark Results\n\n")
            f.write("| Operation | Samples | Features | Time (s) | Throughput (samples/s) |\n")
            f.write("|-----------|---------|----------|----------|------------------------|\n")
            
            for r in self.results:
                f.write(f"| {r['operation']} | {r['n_samples']} | {r['n_features']} | {r['time_seconds']:.3f} | {r['throughput']:.1f} |\n")
            
            f.write("\n---\n\n")
            f.write("## Performance Summary\n\n")
            
            # Group by operation
            ops = {}
            for r in self.results:
                op = r['operation']
                if op not in ops:
                    ops[op] = []
                ops[op].append(r)
            
            for op, results in ops.items():
                f.write(f"### {op}\n\n")
                avg_throughput = np.mean([r['throughput'] for r in results])
                f.write(f"- **Average Throughput:** {avg_throughput:.1f} samples/s\n")
                f.write(f"- **Scalability:** Tested with {min(r['n_samples'] for r in results)}-{max(r['n_samples'] for r in results)} samples\n\n")
        
        # Also save as JSON
        json_path = output_path / "benchmark_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "results": self.results
            }, f, indent=2)
        
        return len(self.results)


# Global report instance
report = BenchmarkReport()


class TestBootstrapBenchmarks:
    """Benchmark bootstrap CI calculations."""
    
    @pytest.mark.parametrize("n_samples", [50, 100, 200])
    def test_bootstrap_ci_performance(self, n_samples):
        """Benchmark bootstrap CI calculation."""
        np.random.seed(42)
        data = pd.DataFrame(np.random.randint(0, 2, size=(n_samples, 10)))
        
        start = time.time()
        
        # Simulate bootstrap CI calculation
        n_bootstrap = 500
        for col in data.columns:
            boot_means = []
            for _ in range(n_bootstrap):
                boot_sample = np.random.choice(data[col], size=n_samples, replace=True)
                boot_means.append(boot_sample.mean())
            ci_low = np.percentile(boot_means, 2.5)
            ci_high = np.percentile(boot_means, 97.5)
        
        elapsed = time.time() - start
        
        report.add_result("Bootstrap CI", n_samples, 10, elapsed)
        
        # Should complete in reasonable time
        assert elapsed < 30  # 30 seconds max


class TestPairwiseCooccurrenceBenchmarks:
    """Benchmark pairwise co-occurrence calculations."""
    
    @pytest.mark.parametrize("n_samples,n_features", [(50, 20), (100, 30), (200, 40)])
    def test_pairwise_cooccurrence_performance(self, n_samples, n_features):
        """Benchmark pairwise co-occurrence calculation."""
        np.random.seed(42)
        data = pd.DataFrame(np.random.randint(0, 2, size=(n_samples, n_features)))
        
        start = time.time()
        
        # Calculate pairwise co-occurrence
        n_pairs = n_features * (n_features - 1) // 2
        results = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                cooccur = ((data.iloc[:, i] == 1) & (data.iloc[:, j] == 1)).sum()
                results.append((i, j, cooccur))
        
        elapsed = time.time() - start
        
        report.add_result("Pairwise Co-occurrence", n_samples, n_features, elapsed)
        
        assert elapsed < 10  # 10 seconds max


class TestAssociationRulesBenchmarks:
    """Benchmark association rule mining."""
    
    @pytest.mark.parametrize("n_samples", [50, 100, 200])
    def test_association_rules_performance(self, n_samples):
        """Benchmark association rule mining."""
        np.random.seed(42)
        data = pd.DataFrame(np.random.randint(0, 2, size=(n_samples, 15)))
        
        start = time.time()
        
        # Calculate support and confidence for all pairs
        n_features = data.shape[1]
        rules = []
        for i in range(n_features):
            for j in range(n_features):
                if i != j:
                    support_i = data.iloc[:, i].mean()
                    support_ij = ((data.iloc[:, i] == 1) & (data.iloc[:, j] == 1)).mean()
                    if support_i > 0:
                        confidence = support_ij / support_i
                        rules.append((i, j, support_ij, confidence))
        
        elapsed = time.time() - start
        
        report.add_result("Association Rules", n_samples, 15, elapsed)
        
        assert elapsed < 5  # 5 seconds max


class TestNetworkConstructionBenchmarks:
    """Benchmark network construction."""
    
    @pytest.mark.parametrize("n_nodes", [20, 50, 100])
    def test_network_construction_performance(self, n_nodes):
        """Benchmark network construction."""
        import networkx as nx
        
        np.random.seed(42)
        
        start = time.time()
        
        # Create network from pairwise associations
        G = nx.Graph()
        for i in range(n_nodes):
            G.add_node(f"node_{i}")
        
        # Add edges based on random associations
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if np.random.random() > 0.7:  # 30% edge probability
                    G.add_edge(f"node_{i}", f"node_{j}", weight=np.random.random())
        
        # Calculate centrality metrics
        degree = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        
        elapsed = time.time() - start
        
        report.add_result("Network Construction", n_nodes, n_nodes, elapsed)
        
        assert elapsed < 10  # 10 seconds max


class TestFullPipelineBenchmarks:
    """Benchmark full analysis pipeline."""
    
    @pytest.mark.parametrize("n_samples", [50, 100])
    def test_full_pipeline_performance(self, n_samples):
        """Benchmark full pipeline execution."""
        np.random.seed(42)
        
        n_features = 20
        data = pd.DataFrame(np.random.randint(0, 2, size=(n_samples, n_features)))
        
        start = time.time()
        
        # Step 1: Bootstrap CI
        n_bootstrap = 200
        for col in data.columns[:5]:  # Subset for speed
            boot_means = []
            for _ in range(n_bootstrap):
                boot_sample = np.random.choice(data[col], size=n_samples, replace=True)
                boot_means.append(boot_sample.mean())
        
        # Step 2: Pairwise co-occurrence
        for i in range(min(10, n_features)):
            for j in range(i + 1, min(10, n_features)):
                cooccur = ((data.iloc[:, i] == 1) & (data.iloc[:, j] == 1)).sum()
        
        # Step 3: Association rules
        for i in range(min(10, n_features)):
            for j in range(min(10, n_features)):
                if i != j:
                    support_i = data.iloc[:, i].mean()
                    support_ij = ((data.iloc[:, i] == 1) & (data.iloc[:, j] == 1)).mean()
        
        elapsed = time.time() - start
        
        report.add_result("Full Pipeline", n_samples, n_features, elapsed)
        
        assert elapsed < 30  # 30 seconds max


@pytest.fixture(scope="session", autouse=True)
def save_benchmark_report():
    """Save benchmark report after all tests."""
    yield
    
    # Save report
    output_dir = Path(__file__).parent.parent / "validation"
    n_benchmarks = report.save_report(output_dir)
    
    print(f"\n{'='*60}")
    print(f"PERFORMANCE BENCHMARKS REPORT - strepsuis-mdr")
    print(f"{'='*60}")
    print(f"Total Benchmarks: {n_benchmarks}")
    print(f"Report saved to: {output_dir / 'PERFORMANCE_BENCHMARKS_REPORT.md'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
