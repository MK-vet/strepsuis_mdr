# Synthetic Data Generation Methodology

## Overview

This document describes the statistical methodology used to generate synthetic
antimicrobial resistance (AMR) data for validation and testing purposes.

## Generation Parameters

- **Number of strains**: 200
- **Number of antibiotics**: 13
- **Number of AMR genes**: 30
- **Number of virulence factors**: 20
- **Base prevalence mean**: 0.35
- **Base prevalence std**: 0.20
- **Correlation strength**: 0.70
- **Noise level**: 0.05
- **Random seed**: 42

## Statistical Distributions Used

### 1. Beta Distribution for Prevalence Rates

Prevalence rates for each feature were generated using a Beta distribution,
which is bounded between 0 and 1 and can capture the typical distribution
of resistance gene prevalences in bacterial populations.

**Parameters:**
- Mean (μ) = 0.35
- Standard deviation (σ) = 0.20

**Beta parameters calculated as:**
- α = μ × (μ(1-μ)/σ² - 1)
- β = (1-μ) × (μ(1-μ)/σ² - 1)

### 2. Binomial Distribution for Binary Data

Individual strain observations were generated using a Binomial distribution
with n=1 (Bernoulli trials), where the success probability is the feature's
prevalence rate.

**Mathematical representation:**
- P(X=1) = p (prevalence rate)
- P(X=0) = 1 - p

### 3. Gaussian Noise for Measurement Error

A small proportion of observations (5.0%) were randomly
flipped to simulate measurement error, contamination, or threshold effects
commonly observed in experimental data.

### 4. Induced Correlations

Known correlations between specific gene-phenotype pairs were induced by
modifying conditional probabilities:
- For correlation ρ > 0: P(Y=1|X=1) > P(Y=1|X=0)
- For correlation ρ < 0: P(Y=1|X=1) < P(Y=1|X=0)

## Ground Truth Values

### Known Correlations
- Oxytetracycline ↔ AMR_Gene_01: ρ = 0.70
- Spectinomycin ↔ AMR_Gene_04: ρ = 0.70
- Trimethoprim_Sulphamethoxazole ↔ AMR_Gene_07: ρ = 0.70
- Ampicillin ↔ AMR_Gene_10: ρ = 0.70
- Florfenicol ↔ AMR_Gene_13: ρ = 0.70

### MDR Status
- MDR isolates: 195 (97.5%)
- MDR definition: Resistance to ≥3 antibiotic classes

## References

1. Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall.
2. Fleiss, J. L. (1981). Statistical Methods for Rates and Proportions. Wiley.
3. Forbes, C., et al. (2011). Statistical Distributions. Wiley.

## Generation Timestamp

2026-01-19T23:45:21.010733

---
*This data was generated for validation and testing purposes only.*
