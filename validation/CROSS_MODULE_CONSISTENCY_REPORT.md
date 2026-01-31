# Cross-Module Consistency Report

**Generated:** 2026-01-31T10:02:24.786829
**Modules Tested:** strepsuis-mdr, strepsuis-amrvirkm, strepsuis-genphennet, strepsuis-phylotrait
**Total Tests:** 7
**Passed:** 7
**Consistency Checks:** 7/7

---

## Module Consistency Matrix

| Check | Module 1 | Module 2 | Value 1 | Value 2 | Status |
|-------|----------|----------|---------|---------|--------|
| Chi-Square Calculation | scipy (reference) | manual implementation | 0.0000 | 0.0000 | ✅ Consistent |
| Phi Coefficient | from chi-square | direct formula | 0.0000 | 0.0000 | ✅ Consistent |
| Bootstrap Reproducibility | Run 1 (seed=42) | Run 2 (seed=42) | [0.0000, 0.0000] | [0.0000, 0.0000] | ✅ Consistent |
| FDR Correction | statsmodels | manual BH | [0.008      0.04     | [0.008      0.04     | ✅ Consistent |
| Prevalence Calculation | mean() | sum()/count() | 0.0000% | 0.0000% | ✅ Consistent |
| Data Type Handling | int | float/bool | 0.0000 | 0.0000 | ✅ Consistent |
| Pearson Correlation | numpy | scipy/pandas | 0.0000 | 0.0000 | ✅ Consistent |

---

## Test Results

| Test | Expected | Actual | Modules | Status |
|------|----------|--------|---------|--------|
| Chi-Square Consistency | scipy: 0.0000 | manual: 0.0000 | all modules | ✅ PASS |
| Phi Coefficient Consistency | from chi2: 0.0000 | direct: 0.0000 | mdr, amrvirkm, genphennet | ✅ PASS |
| Bootstrap Reproducibility | CI1: [0.0000, 0.0000] | CI2: [0.0000, 0.0000] | all modules | ✅ PASS |
| FDR BH Consistency | statsmodels: [0.008      0.04  | manual: [0.008      0.04       | all modules | ✅ PASS |
| Prevalence Calculation | mean: 0.00% | sum/count: 0.00%, vc: 0.00% | all modules | ✅ PASS |
| Binary Data Type Handling | int: 0.0000 | float: 0.0000, bool: 0.0000 | all modules | ✅ PASS |
| Pearson Correlation | numpy: 0.0000 | scipy: 0.0000, pandas: 0.0000 | all modules | ✅ PASS |

---

## Detailed Consistency Analysis

### Chi-Square Calculation - ✅ Consistent

- **scipy (reference):** 0.0000
- **manual implementation:** 0.0000
- **Interpretation:** All modules should use consistent chi-square calculation.

### Phi Coefficient - ✅ Consistent

- **from chi-square:** 0.0000
- **direct formula:** 0.0000
- **Interpretation:** Both methods should give same absolute phi value.

### Bootstrap Reproducibility - ✅ Consistent

- **Run 1 (seed=42):** [0.0000, 0.0000]
- **Run 2 (seed=42):** [0.0000, 0.0000]
- **Interpretation:** Bootstrap should be reproducible with fixed random seed.

### FDR Correction - ✅ Consistent

- **statsmodels:** [0.008      0.04       0.05333333]
- **manual BH:** [0.008      0.04       0.05333333]
- **Interpretation:** FDR correction should be consistent across implementations.

### Prevalence Calculation - ✅ Consistent

- **mean():** 0.0000%
- **sum()/count():** 0.0000%
- **Interpretation:** All prevalence calculation methods should give identical results.

### Data Type Handling - ✅ Consistent

- **int:** 0.0000
- **float/bool:** 0.0000
- **Interpretation:** Binary data should give same results regardless of type.

### Pearson Correlation - ✅ Consistent

- **numpy:** 0.0000
- **scipy/pandas:** 0.0000
- **Interpretation:** Correlation should be identical across libraries.

