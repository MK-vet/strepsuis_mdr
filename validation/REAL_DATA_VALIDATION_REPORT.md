# Real Data Validation Report - strepsuis-mdr

**Generated:** 2026-01-31T10:02:24.780184
**Data Source:** S. suis strains (MIC.csv, AMR_genes.csv, Virulence.csv)
**Total Tests:** 15
**Passed:** 15
**Coverage:** 100.0%

---

## Statistical Validation Results

| Test | Expected | Actual | Status |
|------|----------|--------|--------|
| Strain Count | ≥50 strains | 91 strains | ✅ PASS |
| Binary Values | All 0 or 1 | All binary | ✅ PASS |
| No Missing Values | No NaN/null | No missing | ✅ PASS |
| Strain ID Consistency | ≥50 common strains | 91 common strains | ✅ PASS |
| Prevalence Range | [0%, 100%] | Min=0.0%, Max=75.8% | ✅ PASS |
| Tetracycline Prevalence | Calculable | 72.0% | ✅ PASS |
| Chi-Square on Real Data | Valid chi2 and p-value | chi2=0.00, p=1.0000 | ✅ PASS |
| Multiple Testing Correction | FDR correction applied | 0/20 significant after FDR | ✅ PASS |
| MDR Prevalence | [0%, 100%] | 53.8% (49/91) | ✅ PASS |
| Resistance Distribution | Valid statistics | Mean=2.7±1.7, Max=7 | ✅ PASS |
| Gene Co-occurrence Matrix | Valid diagonal | Diagonal = column sums | ✅ PASS |
| Phi Coefficient Range | [-1, 1] | All in range | ✅ PASS |
| Bootstrap CI Real Data | CI contains observed | [0.0%, 0.0%] contains 0.0% | ✅ PASS |
| Virulence Prevalence | Valid range | Top 3:  LDH:100%,  manN:100%,  ArcD:100% | ✅ PASS |
| AMR-Virulence Correlation | Valid correlation | r=0.065, p=0.5375 | ✅ PASS |

---

## Biological Validation Results

### AMR Prevalence Distribution

**Description:** Distribution of antimicrobial resistance prevalence across antibiotics

**Result:** Range: 0.0% - 75.8%, Mean: 20.5%

**Interpretation:** Typical S. suis populations show variable resistance rates. High prevalence (>50%) for some antibiotics suggests selective pressure.

### Tetracycline Resistance

**Description:** Tetracycline resistance is commonly high in S. suis due to widespread use in pig farming

**Result:** Observed prevalence: 72.0%

**Interpretation:** Values >30% are typical for pig-associated S. suis. Lower values may indicate recent antibiotic stewardship.

### Genotype-Phenotype Associations

**Description:** Statistical associations between resistance phenotypes and AMR genes

**Result:** 0 significant associations found (FDR < 0.05)

**Interpretation:** Significant associations suggest functional relationships between genes and phenotypes. Expected for well-characterized resistance mechanisms.

### Multidrug Resistance

**Description:** Prevalence of strains resistant to 3 or more antimicrobial classes

**Result:** MDR prevalence: 53.8% (49 strains)

**Interpretation:** MDR prevalence >30% is concerning and suggests need for antibiotic stewardship. Values vary by geographic region and farm management.

### Resistance Burden

**Description:** Average number of antimicrobials each strain is resistant to

**Result:** Mean: 2.7 ± 1.7, Range: 0-7

**Interpretation:** Higher mean resistance burden indicates more challenging treatment options. Compare with regional surveillance data.

### Prevalence Confidence Interval

**Description:** 95% bootstrap CI for Amoxicillin_Clavulanic_acid prevalence

**Result:** Point estimate: 0.0%, 95% CI: [0.0%, 0.0%]

**Interpretation:** Narrow CI indicates precise estimate. Wide CI suggests need for larger sample size.

### Virulence Factor Distribution

**Description:** Prevalence of virulence factors in the strain collection

**Result:** Most common:  LDH:100%,  manN:100%,  ArcD:100%

**Interpretation:** High prevalence virulence factors may be essential for colonization. Low prevalence factors may be associated with invasive disease.

### AMR-Virulence Relationship

**Description:** Correlation between antimicrobial resistance gene count and virulence factor count

**Result:** Pearson r = 0.065, p = 0.5375

**Interpretation:** Positive correlation suggests co-selection. Negative suggests trade-off. No correlation indicates independent evolution.

---

## Detailed Test Results

### Strain Count - ✅ PASS

- **Category:** data_integrity
- **Expected:** ≥50 strains
- **Actual:** 91 strains
- **Details:** Dataset should have sufficient sample size

### Binary Values - ✅ PASS

- **Category:** data_integrity
- **Expected:** All 0 or 1
- **Actual:** All binary
- **Details:** Data should be binary presence/absence

### No Missing Values - ✅ PASS

- **Category:** data_integrity
- **Expected:** No NaN/null
- **Actual:** No missing
- **Details:** Data should be complete

### Strain ID Consistency - ✅ PASS

- **Category:** data_integrity
- **Expected:** ≥50 common strains
- **Actual:** 91 common strains
- **Details:** Strain IDs should match across files

### Prevalence Range - ✅ PASS

- **Category:** statistical
- **Expected:** [0%, 100%]
- **Actual:** Min=0.0%, Max=75.8%
- **Details:** All prevalences within valid range

### Tetracycline Prevalence - ✅ PASS

- **Category:** biological
- **Expected:** Calculable
- **Actual:** 72.0%
- **Details:** Tetracycline resistance prevalence

### Chi-Square on Real Data - ✅ PASS

- **Category:** statistical
- **Expected:** Valid chi2 and p-value
- **Actual:** chi2=0.00, p=1.0000
- **Details:** Testing Amoxicillin_Clavulanic_acid vs SAT-4

### Multiple Testing Correction - ✅ PASS

- **Category:** statistical
- **Expected:** FDR correction applied
- **Actual:** 0/20 significant after FDR
- **Details:** Benjamini-Hochberg FDR correction

### MDR Prevalence - ✅ PASS

- **Category:** biological
- **Expected:** [0%, 100%]
- **Actual:** 53.8% (49/91)
- **Details:** Strains resistant to ≥3 drug classes

### Resistance Distribution - ✅ PASS

- **Category:** biological
- **Expected:** Valid statistics
- **Actual:** Mean=2.7±1.7, Max=7
- **Details:** Distribution of resistance counts per strain

### Gene Co-occurrence Matrix - ✅ PASS

- **Category:** statistical
- **Expected:** Valid diagonal
- **Actual:** Diagonal = column sums
- **Details:** Co-occurrence matrix validation

### Phi Coefficient Range - ✅ PASS

- **Category:** statistical
- **Expected:** [-1, 1]
- **Actual:** All in range
- **Details:** Tested 10 gene pairs

### Bootstrap CI Real Data - ✅ PASS

- **Category:** statistical
- **Expected:** CI contains observed
- **Actual:** [0.0%, 0.0%] contains 0.0%
- **Details:** Bootstrap CI for Amoxicillin_Clavulanic_acid

### Virulence Prevalence - ✅ PASS

- **Category:** biological
- **Expected:** Valid range
- **Actual:** Top 3:  LDH:100%,  manN:100%,  ArcD:100%
- **Details:** Virulence factor prevalence

### AMR-Virulence Correlation - ✅ PASS

- **Category:** biological
- **Expected:** Valid correlation
- **Actual:** r=0.065, p=0.5375
- **Details:** Correlation between AMR and virulence burden

