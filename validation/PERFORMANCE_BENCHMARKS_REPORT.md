# Performance Benchmarks Report - strepsuis-mdr

**Generated:** 2026-01-31T10:02:24.783531
**Total Benchmarks:** 14

---

## Benchmark Results

| Operation | Samples | Features | Time (s) | Throughput (samples/s) |
|-----------|---------|----------|----------|------------------------|
| Bootstrap CI | 50 | 10 | 0.103 | 484.5 |
| Bootstrap CI | 100 | 10 | 0.106 | 945.8 |
| Bootstrap CI | 200 | 10 | 0.117 | 1715.5 |
| Pairwise Co-occurrence | 50 | 20 | 0.034 | 1474.5 |
| Pairwise Co-occurrence | 100 | 30 | 0.078 | 1284.4 |
| Pairwise Co-occurrence | 200 | 40 | 0.136 | 1470.7 |
| Association Rules | 50 | 15 | 0.048 | 1045.9 |
| Association Rules | 100 | 15 | 0.049 | 2042.2 |
| Association Rules | 200 | 15 | 0.048 | 4127.1 |
| Network Construction | 20 | 20 | 0.001 | 20641.3 |
| Network Construction | 50 | 50 | 0.007 | 6715.4 |
| Network Construction | 100 | 100 | 0.042 | 2393.7 |
| Full Pipeline | 50 | 20 | 0.047 | 1061.6 |
| Full Pipeline | 100 | 20 | 0.049 | 2042.3 |

---

## Performance Summary

### Bootstrap CI

- **Average Throughput:** 1048.6 samples/s
- **Scalability:** Tested with 50-200 samples

### Pairwise Co-occurrence

- **Average Throughput:** 1409.9 samples/s
- **Scalability:** Tested with 50-200 samples

### Association Rules

- **Average Throughput:** 2405.1 samples/s
- **Scalability:** Tested with 50-200 samples

### Network Construction

- **Average Throughput:** 9916.8 samples/s
- **Scalability:** Tested with 20-100 samples

### Full Pipeline

- **Average Throughput:** 1552.0 samples/s
- **Scalability:** Tested with 50-100 samples

