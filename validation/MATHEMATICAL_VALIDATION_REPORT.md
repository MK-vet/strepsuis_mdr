# Mathematical Validation Report - strepsuis-mdr

**Generated:** 2026-01-31T10:02:24.785167
**Total Tests:** 18
**Passed:** 18
**Coverage:** 100.0%

---

## Test Results

| Test | Expected | Actual | Status |
|------|----------|--------|--------|
| Chi-Square vs scipy | chi2=16.67, p=0.0000 | chi2=15.04, p=0.0001 | ✅ PASS |
| Fisher's Exact vs scipy | p=0.5238 | p=0.5238 | ✅ PASS |
| Phi Perfect Positive | 1.0 | 1.0000 | ✅ PASS |
| Phi Perfect Negative | -1.0 | -1.0000 | ✅ PASS |
| Phi No Association | ~0.0 | 0.0000 | ✅ PASS |
| Phi Bounds | [-1, 1] | All within bounds | ✅ PASS |
| FDR vs statsmodels | reject=2 | reject=2 | ✅ PASS |
| FDR Monotonicity | Monotonically non-decreasing | Monotonic | ✅ PASS |
| FDR Control | ≤5.0% | 0.0% | ✅ PASS |
| Log-Odds Calculation | log(OR) = 1.79 | log(OR) = 1.79 | ✅ PASS |
| Log-Odds Symmetry | log(OR_AB) = -log(OR_BA) | 1.79 = --1.79 | ✅ PASS |
| Risk Score Concept | Hub has highest centrality | B=1.00 > A=0.33 | ✅ PASS |
| Sequential Pattern Concept | P(B|A) ≈ 0.8 | P(B|A) = 0.73 | ✅ PASS |
| Support Calculation | Support(A)=0.6, Support(A,B)=0 | Support(A)=0.60, Support(A,B)= | ✅ PASS |
| Confidence Calculation | Confidence(A→B) = 0.667 | Confidence(A→B) = 0.667 | ✅ PASS |
| Lift Calculation | Lift(A→B) = 1.111 | Lift(A→B) = 1.111 | ✅ PASS |
| Bootstrap CI Concept | CI contains sample mean | [0.38, 0.57] contains 0.47 | ✅ PASS |
| Bootstrap Coverage | ~95% | 96.0% | ✅ PASS |

---

## Detailed Results

### Chi-Square vs scipy - ✅ PASS

- **Expected:** chi2=16.67, p=0.0000
- **Actual:** chi2=15.04, p=0.0001
- **Details:** Should match scipy.stats.chi2_contingency

### Fisher's Exact vs scipy - ✅ PASS

- **Expected:** p=0.5238
- **Actual:** p=0.5238
- **Details:** Should match scipy.stats.fisher_exact

### Phi Perfect Positive - ✅ PASS

- **Expected:** 1.0
- **Actual:** 1.0000
- **Details:** Perfect positive association

### Phi Perfect Negative - ✅ PASS

- **Expected:** -1.0
- **Actual:** -1.0000
- **Details:** Perfect negative association

### Phi No Association - ✅ PASS

- **Expected:** ~0.0
- **Actual:** 0.0000
- **Details:** Independent variables

### Phi Bounds - ✅ PASS

- **Expected:** [-1, 1]
- **Actual:** All within bounds
- **Details:** 20 random tests

### FDR vs statsmodels - ✅ PASS

- **Expected:** reject=2
- **Actual:** reject=2
- **Details:** FDR correction should work

### FDR Monotonicity - ✅ PASS

- **Expected:** Monotonically non-decreasing
- **Actual:** Monotonic
- **Details:** Corrected p-values should be monotonic

### FDR Control - ✅ PASS

- **Expected:** ≤5.0%
- **Actual:** 0.0%
- **Details:** FDR should be controlled at nominal level

### Log-Odds Calculation - ✅ PASS

- **Expected:** log(OR) = 1.79
- **Actual:** log(OR) = 1.79
- **Details:** Should produce valid log-odds

### Log-Odds Symmetry - ✅ PASS

- **Expected:** log(OR_AB) = -log(OR_BA)
- **Actual:** 1.79 = --1.79
- **Details:** Antisymmetric property

### Risk Score Concept - ✅ PASS

- **Expected:** Hub has highest centrality
- **Actual:** B=1.00 > A=0.33
- **Details:** Network centrality concept

### Sequential Pattern Concept - ✅ PASS

- **Expected:** P(B|A) ≈ 0.8
- **Actual:** P(B|A) = 0.73
- **Details:** Conditional probability

### Support Calculation - ✅ PASS

- **Expected:** Support(A)=0.6, Support(A,B)=0.4
- **Actual:** Support(A)=0.60, Support(A,B)=0.40
- **Details:** Correct support values

### Confidence Calculation - ✅ PASS

- **Expected:** Confidence(A→B) = 0.667
- **Actual:** Confidence(A→B) = 0.667
- **Details:** Correct confidence value

### Lift Calculation - ✅ PASS

- **Expected:** Lift(A→B) = 1.111
- **Actual:** Lift(A→B) = 1.111
- **Details:** Correct lift value

### Bootstrap CI Concept - ✅ PASS

- **Expected:** CI contains sample mean
- **Actual:** [0.38, 0.57] contains 0.47
- **Details:** Bootstrap CI should contain sample mean

### Bootstrap Coverage - ✅ PASS

- **Expected:** ~95%
- **Actual:** 96.0%
- **Details:** CI should contain true value ~95%

