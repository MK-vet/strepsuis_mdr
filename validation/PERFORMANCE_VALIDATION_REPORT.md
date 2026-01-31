# Performance Validation Report - strepsuis-mdr

**Generated:** 2026-01-31T10:02:24.781782
**Platform:** Windows
**Python:** 3.14

---

## Benchmark Summary

| Operation | Data Size | Time (ms) | Throughput | Memory (MB) |
|-----------|-----------|-----------|------------|-------------|
| Bootstrap CI (n=50) | 50 samples | 41.38 | 24.2 iter/s | 0.05 |
| Bootstrap CI (n=100) | 100 samples | 41.44 | 24.1 iter/s | 0.05 |
| Bootstrap CI (n=200) | 200 samples | 42.83 | 23.3 iter/s | 0.05 |
| Bootstrap CI (n=500) | 500 samples | 59.78 | 16.7 iter/s | 0.05 |
| Pairwise Chi-Square (f=10) | 45 pairs | 337.46 | 133 pairs/s | 0.08 |
| Pairwise Chi-Square (f=20) | 190 pairs | 1523.61 | 125 pairs/s | 0.09 |
| Pairwise Chi-Square (f=50) | 1225 pairs | 7071.03 | 173 pairs/s | 0.23 |
| Pairwise Chi-Square (f=100) | 4950 pairs | 21985.5 | 225 pairs/s | 0.71 |
| Co-occurrence Matrix (n=50) | 50x30 | 0.04 | 34803 elem/ms | N/A |
| Co-occurrence Matrix (n=100) | 100x30 | 0.04 | 75949 elem/ms | N/A |
| Co-occurrence Matrix (n=200) | 200x30 | 0.07 | 91047 elem/ms | N/A |
| Co-occurrence Matrix (n=500) | 500x30 | 0.17 | 86207 elem/ms | N/A |
| MDR Prevalence (n=100) | 100 strains | 0.03 | 2941 strains/ms | 0.01 |
| MDR Prevalence (n=500) | 500 strains | 0.02 | 25126 strains/ms | 0.05 |
| MDR Prevalence (n=1000) | 1000 strains | 0.03 | 32680 strains/ms | 0.07 |
| MDR Prevalence (n=2000) | 2000 strains | 0.03 | 62696 strains/ms | 0.08 |
| FDR Correction (n=100) | 100 tests | 0.11 | 909 tests/ms | 0.01 |
| FDR Correction (n=500) | 500 tests | 0.07 | 6793 tests/ms | 0.02 |
| FDR Correction (n=1000) | 1000 tests | 0.08 | 12594 tests/ms | 0.04 |
| FDR Correction (n=5000) | 5000 tests | 0.16 | 31270 tests/ms | 0.2 |

---

## Scalability Analysis

### Bootstrap CI

**Estimated Complexity:** O(n²)

| Data Size | Time (ms) | Memory (MB) |
|-----------|-----------|-------------|
| 50 | 41.38 | 0.05 |
| 100 | 41.44 | 0.05 |
| 200 | 42.83 | 0.05 |
| 500 | 59.78 | 0.05 |

### Pairwise Chi-Square

**Estimated Complexity:** O(n²)

| Data Size | Time (ms) | Memory (MB) |
|-----------|-----------|-------------|
| 10 | 337.46 | 0.08 |
| 20 | 1523.61 | 0.09 |
| 50 | 7071.03 | 0.23 |
| 100 | 21985.5 | 0.71 |

### Co-occurrence Matrix

**Estimated Complexity:** O(n²)

| Data Size | Time (ms) | Memory (MB) |
|-----------|-----------|-------------|
| 50 | 0.04 | 0.0 |
| 100 | 0.04 | 0.0 |
| 200 | 0.07 | 0.0 |
| 500 | 0.17 | 0.0 |

### MDR Prevalence

**Estimated Complexity:** O(n²)

| Data Size | Time (ms) | Memory (MB) |
|-----------|-----------|-------------|
| 100 | 0.03 | 0.01 |
| 500 | 0.02 | 0.05 |
| 1000 | 0.03 | 0.07 |
| 2000 | 0.03 | 0.08 |

### FDR Correction

**Estimated Complexity:** O(n²)

| Data Size | Time (ms) | Memory (MB) |
|-----------|-----------|-------------|
| 100 | 0.11 | 0.01 |
| 500 | 0.07 | 0.02 |
| 1000 | 0.08 | 0.04 |
| 5000 | 0.16 | 0.2 |

---

## Optimization Recommendations

### Bootstrap Vectorization

- **Before:** 1.76 ms
- **After:** 0.26 ms
- **Improvement:** 85.0%

### dtype int64 -> int8

- **Before:** 0.76 ms
- **After:** 0.1 ms
- **Improvement:** 87.5%

### Dense -> Sparse (10% prevalence)

- **Before:** 0.38 ms
- **After:** 0.08 ms
- **Improvement:** 78.6%

---

## Performance Guidelines

### Recommended Data Sizes

| Operation | Small (<1s) | Medium (<10s) | Large (<60s) |
|-----------|-------------|---------------|---------------|
| Bootstrap CI | <500 strains | <2000 strains | <5000 strains |
| Pairwise Analysis | <100 features | <500 features | <1000 features |
| Full Pipeline | <200 strains | <1000 strains | <5000 strains |

### Memory Optimization Tips

1. Use `dtype=np.int8` for binary data
2. Process large datasets in chunks
3. Clear intermediate results when not needed
4. Use sparse matrices for low-prevalence data
