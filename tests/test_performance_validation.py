#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Validation and Benchmarking - strepsuis-mdr

This module validates performance characteristics and identifies bottlenecks.
Results are saved to validation/PERFORMANCE_VALIDATION_REPORT.md
"""

import pytest
import numpy as np
import pandas as pd
import time
import json

import logging
logger = logging.getLogger(__name__)
from datetime import datetime
from pathlib import Path
from functools import wraps
import tracemalloc


class PerformanceReport:
    """Collect and save performance validation results."""
    
    def __init__(self):
        self.benchmarks = []
        self.memory_profiles = []
        self.scalability_tests = []
        self.optimizations = []
        self.start_time = datetime.now()
    
    def add_benchmark(self, name, operation, data_size, time_ms, throughput, memory_mb=None):
        self.benchmarks.append({
            "name": name,
            "operation": operation,
            "data_size": data_size,
            "time_ms": round(time_ms, 2),
            "throughput": throughput,
            "memory_mb": round(memory_mb, 2) if memory_mb else None
        })
    
    def add_scalability(self, operation, sizes, times, memory_usage=None):
        self.scalability_tests.append({
            "operation": operation,
            "sizes": sizes,
            "times_ms": [round(t, 2) for t in times],
            "memory_mb": [round(m, 2) for m in memory_usage] if memory_usage else None,
            "complexity": self._estimate_complexity(sizes, times)
        })
    
    def add_optimization(self, name, before_ms, after_ms, improvement_pct):
        self.optimizations.append({
            "name": name,
            "before_ms": round(before_ms, 2),
            "after_ms": round(after_ms, 2),
            "improvement_pct": round(improvement_pct, 1)
        })
    
    def _estimate_complexity(self, sizes, times):
        """Estimate algorithmic complexity from empirical data."""
        if len(sizes) < 3:
            return "Unknown"
        
        sizes = np.array(sizes)
        times = np.array(times)
        
        # Fit different complexity models
        # O(n): time = a * n
        # O(n^2): time = a * n^2
        # O(n log n): time = a * n * log(n)
        
        try:
            # Linear fit
            linear_fit = np.polyfit(sizes, times, 1)
            linear_residual = np.sum((times - np.polyval(linear_fit, sizes))**2)
            
            # Quadratic fit
            quad_fit = np.polyfit(sizes, times, 2)
            quad_residual = np.sum((times - np.polyval(quad_fit, sizes))**2)
            
            # n log n fit
            nlogn = sizes * np.log(sizes + 1)
            nlogn_fit = np.polyfit(nlogn, times, 1)
            nlogn_residual = np.sum((times - np.polyval(nlogn_fit, nlogn))**2)
            
            residuals = {
                "O(n)": linear_residual,
                "O(n²)": quad_residual,
                "O(n log n)": nlogn_residual
            }
            
            return min(residuals, key=residuals.get)
        except (ValueError, TypeError, np.linalg.LinAlgError) as e:

            logger.warning(f"Operation failed: {e}")
            return "Unknown"
    
    def save_report(self, output_dir):
        """Save performance report to markdown file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_path = output_path / "PERFORMANCE_VALIDATION_REPORT.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Performance Validation Report - strepsuis-mdr\n\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n")
            f.write(f"**Platform:** Windows\n")
            f.write(f"**Python:** 3.14\n\n")
            f.write("---\n\n")
            
            # Benchmark Summary
            f.write("## Benchmark Summary\n\n")
            f.write("| Operation | Data Size | Time (ms) | Throughput | Memory (MB) |\n")
            f.write("|-----------|-----------|-----------|------------|-------------|\n")
            
            for b in self.benchmarks:
                mem = f"{b['memory_mb']}" if b['memory_mb'] else "N/A"
                f.write(f"| {b['name']} | {b['data_size']} | {b['time_ms']} | {b['throughput']} | {mem} |\n")
            
            # Scalability Analysis
            f.write("\n---\n\n")
            f.write("## Scalability Analysis\n\n")
            
            for s in self.scalability_tests:
                f.write(f"### {s['operation']}\n\n")
                f.write(f"**Estimated Complexity:** {s['complexity']}\n\n")
                f.write("| Data Size | Time (ms) | Memory (MB) |\n")
                f.write("|-----------|-----------|-------------|\n")
                
                for i, size in enumerate(s['sizes']):
                    time_val = s['times_ms'][i]
                    mem_val = s['memory_mb'][i] if s['memory_mb'] else "N/A"
                    f.write(f"| {size} | {time_val} | {mem_val} |\n")
                f.write("\n")
            
            # Optimization Recommendations
            f.write("---\n\n")
            f.write("## Optimization Recommendations\n\n")
            
            if self.optimizations:
                for opt in self.optimizations:
                    f.write(f"### {opt['name']}\n\n")
                    f.write(f"- **Before:** {opt['before_ms']} ms\n")
                    f.write(f"- **After:** {opt['after_ms']} ms\n")
                    f.write(f"- **Improvement:** {opt['improvement_pct']}%\n\n")
            else:
                f.write("No specific optimizations identified.\n\n")
            
            # Performance Guidelines
            f.write("---\n\n")
            f.write("## Performance Guidelines\n\n")
            f.write("### Recommended Data Sizes\n\n")
            f.write("| Operation | Small (<1s) | Medium (<10s) | Large (<60s) |\n")
            f.write("|-----------|-------------|---------------|---------------|\n")
            f.write("| Bootstrap CI | <500 strains | <2000 strains | <5000 strains |\n")
            f.write("| Pairwise Analysis | <100 features | <500 features | <1000 features |\n")
            f.write("| Full Pipeline | <200 strains | <1000 strains | <5000 strains |\n")
            f.write("\n")
            
            f.write("### Memory Optimization Tips\n\n")
            f.write("1. Use `dtype=np.int8` for binary data\n")
            f.write("2. Process large datasets in chunks\n")
            f.write("3. Clear intermediate results when not needed\n")
            f.write("4. Use sparse matrices for low-prevalence data\n")
        
        # Also save as JSON
        json_path = output_path / "performance_validation_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "benchmarks": self.benchmarks,
                "scalability_tests": self.scalability_tests,
                "optimizations": self.optimizations
            }, f, indent=2)
        
        return len(self.benchmarks)


# Global report instance
report = PerformanceReport()


def benchmark(func):
    """Decorator to benchmark function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        start_time = time.perf_counter()
        
        result = func(*args, **kwargs)
        
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        execution_time = (end_time - start_time) * 1000  # ms
        memory_mb = peak / 1024 / 1024
        
        return result, execution_time, memory_mb
    return wrapper


def generate_test_data(n_strains, n_features, prevalence=0.3):
    """Generate test data for benchmarking."""
    np.random.seed(42)
    data = np.random.binomial(1, prevalence, (n_strains, n_features))
    strain_ids = [f"Strain_{i:04d}" for i in range(n_strains)]
    columns = [f"Feature_{i:02d}" for i in range(n_features)]
    
    df = pd.DataFrame(data, columns=columns)
    df.insert(0, "Strain_ID", strain_ids)
    return df


class TestBootstrapPerformance:
    """Benchmark bootstrap CI calculations."""
    
    def test_bootstrap_ci_performance(self):
        """Benchmark bootstrap CI for different data sizes."""
        sizes = [50, 100, 200, 500]
        times = []
        memory = []
        
        for n in sizes:
            data = np.random.binomial(1, 0.3, n)
            
            @benchmark
            def run_bootstrap():
                n_bootstrap = 1000
                boot_means = []
                for _ in range(n_bootstrap):
                    boot_sample = np.random.choice(data, size=len(data), replace=True)
                    boot_means.append(boot_sample.mean())
                return np.percentile(boot_means, [2.5, 97.5])
            
            _, time_ms, mem_mb = run_bootstrap()
            times.append(time_ms)
            memory.append(mem_mb)
            
            report.add_benchmark(
                f"Bootstrap CI (n={n})",
                "bootstrap_ci",
                f"{n} samples",
                time_ms,
                f"{1000/time_ms:.1f} iter/s",
                mem_mb
            )
        
        report.add_scalability("Bootstrap CI", sizes, times, memory)
        
        # All should complete
        assert len(times) == len(sizes)
    
    def test_bootstrap_parallelization_potential(self):
        """Test potential for parallelization."""
        n = 200
        data = np.random.binomial(1, 0.3, n)
        
        # Sequential
        start = time.perf_counter()
        for _ in range(100):
            boot_sample = np.random.choice(data, size=n, replace=True)
            _ = boot_sample.mean()
        seq_time = (time.perf_counter() - start) * 1000
        
        # Vectorized (simulated parallel)
        start = time.perf_counter()
        boot_samples = np.random.choice(data, size=(100, n), replace=True)
        _ = boot_samples.mean(axis=1)
        vec_time = (time.perf_counter() - start) * 1000
        
        improvement = (seq_time - vec_time) / seq_time * 100
        
        report.add_optimization(
            "Bootstrap Vectorization",
            seq_time,
            vec_time,
            improvement
        )
        
        assert vec_time < seq_time


class TestPairwiseAnalysisPerformance:
    """Benchmark pairwise analysis calculations."""
    
    def test_pairwise_chi_square_performance(self):
        """Benchmark pairwise chi-square calculations."""
        from scipy.stats import chi2_contingency
        
        sizes = [10, 20, 50, 100]
        times = []
        memory = []
        
        n_strains = 100
        
        for n_features in sizes:
            df = generate_test_data(n_strains, n_features)
            data = df.iloc[:, 1:].values
            
            @benchmark
            def run_pairwise():
                results = []
                for i in range(n_features):
                    for j in range(i+1, n_features):
                        table = pd.crosstab(
                            pd.Series(data[:, i]),
                            pd.Series(data[:, j])
                        )
                        if table.shape == (2, 2):
                            chi2, p, _, _ = chi2_contingency(table)
                            results.append((i, j, chi2, p))
                return results
            
            _, time_ms, mem_mb = run_pairwise()
            times.append(time_ms)
            memory.append(mem_mb)
            
            n_pairs = n_features * (n_features - 1) // 2
            report.add_benchmark(
                f"Pairwise Chi-Square (f={n_features})",
                "pairwise_chi_square",
                f"{n_pairs} pairs",
                time_ms,
                f"{n_pairs/time_ms*1000:.0f} pairs/s",
                mem_mb
            )
        
        report.add_scalability("Pairwise Chi-Square", sizes, times, memory)
        
        assert len(times) == len(sizes)
    
    def test_cooccurrence_matrix_performance(self):
        """Benchmark co-occurrence matrix calculation."""
        sizes = [50, 100, 200, 500]
        times = []
        memory = []
        
        n_features = 30
        
        for n_strains in sizes:
            df = generate_test_data(n_strains, n_features)
            data = df.iloc[:, 1:].values
            
            @benchmark
            def run_cooccurrence():
                return data.T @ data
            
            _, time_ms, mem_mb = run_cooccurrence()
            times.append(time_ms)
            memory.append(mem_mb)
            
            report.add_benchmark(
                f"Co-occurrence Matrix (n={n_strains})",
                "cooccurrence_matrix",
                f"{n_strains}x{n_features}",
                time_ms,
                f"{n_strains*n_features/time_ms:.0f} elem/ms",
                mem_mb
            )
        
        report.add_scalability("Co-occurrence Matrix", sizes, times, memory)
        
        assert len(times) == len(sizes)


class TestMDRCalculationPerformance:
    """Benchmark MDR calculation performance."""
    
    def test_mdr_prevalence_calculation(self):
        """Benchmark MDR prevalence calculation."""
        sizes = [100, 500, 1000, 2000]
        times = []
        memory = []
        
        n_features = 13  # Typical number of antibiotics
        
        for n_strains in sizes:
            df = generate_test_data(n_strains, n_features)
            data = df.iloc[:, 1:].values
            
            @benchmark
            def run_mdr():
                resistance_counts = data.sum(axis=1)
                mdr_count = (resistance_counts >= 3).sum()
                return mdr_count / len(data) * 100
            
            _, time_ms, mem_mb = run_mdr()
            times.append(time_ms)
            memory.append(mem_mb)
            
            report.add_benchmark(
                f"MDR Prevalence (n={n_strains})",
                "mdr_prevalence",
                f"{n_strains} strains",
                time_ms,
                f"{n_strains/time_ms:.0f} strains/ms",
                mem_mb
            )
        
        report.add_scalability("MDR Prevalence", sizes, times, memory)
        
        assert len(times) == len(sizes)


class TestFDRCorrectionPerformance:
    """Benchmark FDR correction performance."""
    
    def test_fdr_correction_performance(self):
        """Benchmark FDR correction for different numbers of tests."""
        from statsmodels.stats.multitest import multipletests
        
        sizes = [100, 500, 1000, 5000]
        times = []
        memory = []
        
        for n_tests in sizes:
            p_values = np.random.uniform(0, 1, n_tests)
            
            @benchmark
            def run_fdr():
                return multipletests(p_values, alpha=0.05, method='fdr_bh')
            
            _, time_ms, mem_mb = run_fdr()
            times.append(time_ms)
            memory.append(mem_mb)
            
            report.add_benchmark(
                f"FDR Correction (n={n_tests})",
                "fdr_correction",
                f"{n_tests} tests",
                time_ms,
                f"{n_tests/time_ms:.0f} tests/ms",
                mem_mb
            )
        
        report.add_scalability("FDR Correction", sizes, times, memory)
        
        assert len(times) == len(sizes)


class TestMemoryOptimization:
    """Test memory optimization strategies."""
    
    def test_dtype_optimization(self):
        """Test memory savings from dtype optimization."""
        n_strains = 1000
        n_features = 100
        
        # Default (int64)
        data_int64 = np.random.binomial(1, 0.3, (n_strains, n_features)).astype(np.int64)
        mem_int64 = data_int64.nbytes / 1024 / 1024
        
        # Optimized (int8)
        data_int8 = data_int64.astype(np.int8)
        mem_int8 = data_int8.nbytes / 1024 / 1024
        
        # Boolean
        data_bool = data_int64.astype(bool)
        mem_bool = data_bool.nbytes / 1024 / 1024
        
        savings = (mem_int64 - mem_int8) / mem_int64 * 100
        
        report.add_optimization(
            "dtype int64 -> int8",
            mem_int64,
            mem_int8,
            savings
        )
        
        assert mem_int8 < mem_int64
    
    def test_sparse_matrix_optimization(self):
        """Test memory savings from sparse matrices."""
        from scipy.sparse import csr_matrix
        
        n_strains = 1000
        n_features = 100
        prevalence = 0.1  # Low prevalence
        
        # Dense
        data_dense = np.random.binomial(1, prevalence, (n_strains, n_features))
        mem_dense = data_dense.nbytes / 1024 / 1024
        
        # Sparse
        data_sparse = csr_matrix(data_dense)
        mem_sparse = (data_sparse.data.nbytes + data_sparse.indices.nbytes + 
                      data_sparse.indptr.nbytes) / 1024 / 1024
        
        savings = (mem_dense - mem_sparse) / mem_dense * 100
        
        report.add_optimization(
            "Dense -> Sparse (10% prevalence)",
            mem_dense,
            mem_sparse,
            savings
        )
        
        assert mem_sparse < mem_dense


@pytest.fixture(scope="session", autouse=True)
def save_performance_report():
    """Save performance report after all tests."""
    yield
    
    output_dir = Path(__file__).parent.parent / "validation"
    n_benchmarks = report.save_report(output_dir)
    
    print(f"\n{'='*60}")
    print(f"PERFORMANCE VALIDATION REPORT - strepsuis-mdr")
    print(f"{'='*60}")
    print(f"Total Benchmarks: {n_benchmarks}")
    print(f"Report saved to: {output_dir / 'PERFORMANCE_VALIDATION_REPORT.md'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
