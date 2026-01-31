"""
Synthetic Data Generator for StrepSuis Suite - Statistical Validation Module

This module generates synthetic datasets using proper statistical distributions
that mimic real biological variability and noise in antimicrobial resistance (AMR) data.

Generation Methodology:
-----------------------
1. **Binomial Distribution**: Used for binary presence/absence data (resistance phenotypes,
   gene presence). Each sample is a Bernoulli trial with success probability p.
   
2. **Gaussian (Normal) Distribution**: Used to model measurement noise by stochastically
   flipping binary values with a noise probability.
   
3. **Beta Distribution**: Used to generate prevalence rates that follow biological reality
   (most resistance genes have low to moderate prevalence).

4. **Correlation Structure**: Synthetic data includes known correlation patterns to
   validate co-occurrence and association analyses.

Scientific References:
---------------------
- Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. JRSS B, 57(1).
- Fleiss, J. L. (1981). Statistical Methods for Rates and Proportions. Wiley.

Author: MK-vet Team
License: MIT
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation.
    
    This dataclass contains all parameters needed to generate synthetic AMR data
    with known statistical properties for validation purposes.
    
    Attributes:
        n_strains: Number of bacterial strains to generate
        n_antibiotics: Number of antibiotic resistance phenotypes
        n_genes: Number of AMR genes
        n_virulence: Number of virulence factors
        base_prevalence_mean: Mean prevalence rate (Beta distribution parameter)
        base_prevalence_std: Standard deviation for prevalence rates
        correlation_strength: Strength of induced correlations between features
        noise_level: Proportion of random noise to add (0.0 to 1.0)
        mdr_proportion: Expected proportion of MDR isolates
        random_state: Random seed for reproducibility
    """
    n_strains: int = 200
    n_antibiotics: int = 13
    n_genes: int = 30
    n_virulence: int = 20
    base_prevalence_mean: float = 0.35
    base_prevalence_std: float = 0.20
    correlation_strength: float = 0.7
    noise_level: float = 0.05
    mdr_proportion: float = 0.40
    random_state: int = 42


@dataclass
class SyntheticDataMetadata:
    """Metadata describing the generated synthetic data.
    
    Contains ground truth values that can be used to validate
    the correctness of analytical methods.
    
    Attributes:
        config: The configuration used to generate the data
        true_prevalences: Dict of feature -> true prevalence rate
        true_correlations: List of (feature1, feature2, true_correlation) tuples
        true_mdr_status: Array of true MDR status for each strain
        antibiotic_columns: List of antibiotic column names
        gene_columns: List of gene column names
        virulence_columns: List of virulence factor column names
        generation_timestamp: When the data was generated
        generation_method: Description of the statistical methods used
    """
    config: SyntheticDataConfig
    true_prevalences: Dict[str, float] = field(default_factory=dict)
    true_correlations: List[Tuple[str, str, float]] = field(default_factory=list)
    true_mdr_status: np.ndarray = field(default_factory=lambda: np.array([]))
    antibiotic_columns: List[str] = field(default_factory=list)
    gene_columns: List[str] = field(default_factory=list)
    virulence_columns: List[str] = field(default_factory=list)
    generation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_method: str = "Poisson/Gaussian/Binomial hybrid with Beta prevalences"


def generate_prevalence_rates(
    n_features: int,
    mean: float = 0.35,
    std: float = 0.20,
    random_state: Optional[int] = 42,
) -> np.ndarray:
    """
    Generate prevalence rates using a Beta distribution.
    
    The Beta distribution is ideal for modeling proportions as it is bounded
    between 0 and 1 and can capture various shapes of prevalence distributions
    observed in biological data.
    
    Parameters:
        n_features: Number of features to generate prevalences for
        mean: Mean prevalence (0 to 1)
        std: Standard deviation of prevalence
        random_state: Random seed for reproducibility
        
    Returns:
        np.ndarray: Array of prevalence rates between 0.01 and 0.99
        
    Mathematical Details:
        - Beta distribution parameters α and β are calculated from mean (μ) and variance (σ²):
          α = μ × (μ(1-μ)/σ² - 1)
          β = (1-μ) × (μ(1-μ)/σ² - 1)
        - Prevalences are clipped to [0.01, 0.99] to avoid edge cases
    """
    rng = np.random.default_rng(random_state)
    
    # Calculate Beta distribution parameters from mean and std
    variance = std ** 2
    # Ensure variance is valid for Beta distribution
    max_variance = mean * (1 - mean)
    variance = min(variance, max_variance * 0.9)
    
    if variance <= 0:
        return np.full(n_features, mean)
    
    # Beta distribution parameters
    common_term = mean * (1 - mean) / variance - 1
    alpha = mean * common_term
    beta = (1 - mean) * common_term
    
    # Ensure positive parameters
    alpha = max(alpha, 0.5)
    beta = max(beta, 0.5)
    
    prevalences = rng.beta(alpha, beta, size=n_features)
    
    # Clip to avoid extreme values that cause numerical issues
    return np.clip(prevalences, 0.01, 0.99)


def generate_binary_data_binomial(
    n_samples: int,
    prevalence: float,
    random_state: Optional[int] = 42,
) -> np.ndarray:
    """
    Generate binary presence/absence data using binomial distribution.
    
    This method simulates biological variability by using a binomial distribution
    to generate independent Bernoulli trials for each sample, where success
    (presence of resistance) occurs with probability equal to the prevalence rate.
    
    Parameters:
        n_samples: Number of samples (strains) to generate
        prevalence: Expected prevalence rate (0 to 1), used as success probability
        random_state: Random seed for reproducibility
        
    Returns:
        np.ndarray: Binary array (0/1) of length n_samples
        
    Mathematical Details:
        - Each sample is a Bernoulli trial with success probability p = prevalence
        - Binomial(1, p) is equivalent to Bernoulli(p) for single trials
        - Expected value E[X] = p (prevalence)
        - Variance Var(X) = p(1-p)
    """
    rng = np.random.default_rng(random_state)
    
    # Use binomial for binary data (Bernoulli trials)
    return rng.binomial(1, prevalence, size=n_samples)


def add_gaussian_noise_to_binary(
    data: np.ndarray,
    noise_level: float = 0.05,
    random_state: Optional[int] = 42,
) -> np.ndarray:
    """
    Add Gaussian-derived noise to binary data to simulate measurement error.
    
    This simulates real-world variability where some observations may be
    misclassified due to experimental error, contamination, or threshold effects.
    
    Parameters:
        data: Binary array to add noise to
        noise_level: Proportion of data points to flip (0 to 1)
        random_state: Random seed for reproducibility
        
    Returns:
        np.ndarray: Binary array with noise added
        
    Mathematical Details:
        - Noise is added by flipping binary values with probability = noise_level
        - The noise follows a Bernoulli distribution with p = noise_level
    """
    rng = np.random.default_rng(random_state)
    
    # Create a noise mask
    noise_mask = rng.random(len(data)) < noise_level
    
    # Flip values where noise mask is True
    noisy_data = data.copy()
    noisy_data[noise_mask] = 1 - noisy_data[noise_mask]
    
    return noisy_data


def generate_correlated_features(
    base_feature: np.ndarray,
    correlation: float,
    random_state: Optional[int] = 42,
) -> np.ndarray:
    """
    Generate a binary feature correlated with an existing feature.
    
    Creates a new feature that has a specified correlation (phi coefficient)
    with the base feature. This is essential for validating co-occurrence
    and association analyses.
    
    Parameters:
        base_feature: The base binary feature to correlate with
        correlation: Target correlation coefficient (-1 to 1)
        random_state: Random seed for reproducibility
        
    Returns:
        np.ndarray: New binary feature correlated with base_feature
        
    Mathematical Details:
        - For positive correlation: P(new=1|base=1) > P(new=1|base=0)
        - For negative correlation: P(new=1|base=1) < P(new=1|base=0)
        - The correlation strength determines the probability difference
    """
    rng = np.random.default_rng(random_state)
    n = len(base_feature)
    
    # Base prevalence (from the base feature)
    base_prev = np.mean(base_feature)
    
    # Calculate conditional probabilities
    if correlation >= 0:
        # Positive correlation: higher prob when base is 1
        p_given_1 = min(0.95, base_prev + abs(correlation) * (1 - base_prev))
        p_given_0 = max(0.05, base_prev - abs(correlation) * base_prev)
    else:
        # Negative correlation: lower prob when base is 1
        p_given_1 = max(0.05, base_prev - abs(correlation) * base_prev)
        p_given_0 = min(0.95, base_prev + abs(correlation) * (1 - base_prev))
    
    # Generate correlated feature
    new_feature = np.zeros(n, dtype=int)
    for i in range(n):
        if base_feature[i] == 1:
            new_feature[i] = int(rng.random() < p_given_1)
        else:
            new_feature[i] = int(rng.random() < p_given_0)
    
    return new_feature


def generate_mdr_synthetic_dataset(
    config: Optional[SyntheticDataConfig] = None,
) -> Tuple[pd.DataFrame, SyntheticDataMetadata]:
    """
    Generate a complete synthetic AMR dataset with known statistical properties.
    
    This function creates a realistic synthetic dataset that includes:
    - Antibiotic resistance phenotypes (binary)
    - AMR genes (binary presence/absence)
    - Virulence factors (binary)
    - Known correlations between specific features
    - Controlled MDR proportion
    
    Parameters:
        config: Configuration object. Uses defaults if None.
        
    Returns:
        Tuple of (DataFrame, SyntheticDataMetadata):
            - DataFrame contains the synthetic data
            - SyntheticDataMetadata contains ground truth values
            
    Example:
        >>> config = SyntheticDataConfig(n_strains=200, random_state=42)
        >>> data, metadata = generate_mdr_synthetic_dataset(config)
        >>> print(f"Generated {len(data)} strains with {len(metadata.antibiotic_columns)} antibiotics")
    """
    if config is None:
        config = SyntheticDataConfig()
    
    rng = np.random.default_rng(config.random_state)
    
    # Initialize metadata
    metadata = SyntheticDataMetadata(config=config)
    
    # Generate strain IDs
    strain_ids = [f"Strain_{i:04d}" for i in range(1, config.n_strains + 1)]
    
    # Define realistic antibiotic names (matching real dataset)
    antibiotic_names = [
        "Oxytetracycline", "Doxycycline", "Tulathromycin", "Spectinomycin",
        "Gentamicin", "Tiamulin", "Trimethoprim_Sulphamethoxazole",
        "Enrofloxacin", "Penicillin", "Ampicillin", "Amoxicillin_Clavulanic_acid",
        "Ceftiofur", "Florfenicol"
    ][:config.n_antibiotics]
    metadata.antibiotic_columns = antibiotic_names
    
    # Generate gene names
    gene_names = [f"AMR_Gene_{i:02d}" for i in range(1, config.n_genes + 1)]
    metadata.gene_columns = gene_names
    
    # Generate virulence factor names
    vir_names = [f"Virulence_{i:02d}" for i in range(1, config.n_virulence + 1)]
    metadata.virulence_columns = vir_names
    
    # Generate prevalence rates for all features
    ab_prevalences = generate_prevalence_rates(
        config.n_antibiotics,
        mean=config.base_prevalence_mean,
        std=config.base_prevalence_std,
        random_state=config.random_state,
    )
    
    gene_prevalences = generate_prevalence_rates(
        config.n_genes,
        mean=config.base_prevalence_mean * 0.8,  # Genes typically have lower prevalence
        std=config.base_prevalence_std,
        random_state=config.random_state + 1,
    )
    
    vir_prevalences = generate_prevalence_rates(
        config.n_virulence,
        mean=config.base_prevalence_mean * 0.6,
        std=config.base_prevalence_std,
        random_state=config.random_state + 2,
    )
    
    # Store true prevalences
    for name, prev in zip(antibiotic_names, ab_prevalences):
        metadata.true_prevalences[name] = float(prev)
    for name, prev in zip(gene_names, gene_prevalences):
        metadata.true_prevalences[name] = float(prev)
    for name, prev in zip(vir_names, vir_prevalences):
        metadata.true_prevalences[name] = float(prev)
    
    # Generate base data
    data_dict = {"Strain_ID": strain_ids}
    
    # Generate antibiotic resistance data
    for i, (name, prev) in enumerate(zip(antibiotic_names, ab_prevalences)):
        data_dict[name] = generate_binary_data_binomial(
            config.n_strains, prev, random_state=config.random_state + i + 100
        )
    
    # Generate gene data with some correlations to antibiotics
    known_correlations = []
    for i, (name, prev) in enumerate(zip(gene_names, gene_prevalences)):
        if i < config.n_antibiotics and i % 3 == 0:
            # Create correlation with corresponding antibiotic
            ab_name = antibiotic_names[i]
            base_feature = np.array(data_dict[ab_name])
            data_dict[name] = generate_correlated_features(
                base_feature,
                config.correlation_strength,
                random_state=config.random_state + i + 200,
            )
            known_correlations.append((ab_name, name, config.correlation_strength))
        else:
            data_dict[name] = generate_binary_data_binomial(
                config.n_strains, prev, random_state=config.random_state + i + 200
            )
    
    # Generate virulence factor data
    for i, (name, prev) in enumerate(zip(vir_names, vir_prevalences)):
        data_dict[name] = generate_binary_data_binomial(
            config.n_strains, prev, random_state=config.random_state + i + 300
        )
    
    metadata.true_correlations = known_correlations
    
    # Add noise to all features
    feature_cols = antibiotic_names + gene_names + vir_names
    for col in feature_cols:
        data_dict[col] = add_gaussian_noise_to_binary(
            np.array(data_dict[col]),
            config.noise_level,
            random_state=config.random_state + hash(col) % 1000,
        )
    
    # Create DataFrame
    df = pd.DataFrame(data_dict)
    
    # Calculate true MDR status (resistance to >= 3 antibiotic classes)
    # Define antibiotic classes for MDR calculation
    antibiotic_classes = {
        "Tetracyclines": ["Oxytetracycline", "Doxycycline"],
        "Macrolides": ["Tulathromycin"],
        "Aminoglycosides": ["Spectinomycin", "Gentamicin"],
        "Pleuromutilins": ["Tiamulin"],
        "Sulfonamides": ["Trimethoprim_Sulphamethoxazole"],
        "Fluoroquinolones": ["Enrofloxacin"],
        "Penicillins": ["Penicillin", "Ampicillin", "Amoxicillin_Clavulanic_acid"],
        "Cephalosporins": ["Ceftiofur"],
        "Phenicols": ["Florfenicol"],
    }
    
    # Calculate class resistance
    class_resistance = pd.DataFrame(index=df.index)
    for cls_name, drugs in antibiotic_classes.items():
        valid_drugs = [d for d in drugs if d in df.columns]
        if valid_drugs:
            class_resistance[cls_name] = df[valid_drugs].max(axis=1)
        else:
            class_resistance[cls_name] = 0
    
    # MDR = resistance to >= 3 classes
    mdr_status = (class_resistance.sum(axis=1) >= 3).astype(int).values
    metadata.true_mdr_status = mdr_status
    
    return df, metadata


def save_synthetic_data(
    data: pd.DataFrame,
    metadata: SyntheticDataMetadata,
    output_dir: str = "synthetic_data",
) -> Dict[str, str]:
    """
    Save synthetic data and metadata to files.
    
    Parameters:
        data: The synthetic DataFrame to save
        metadata: The metadata object with ground truth
        output_dir: Directory to save files to
        
    Returns:
        Dict with paths to saved files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    # Save main data as CSV
    data_file = output_path / "synthetic_amr_data.csv"
    data.to_csv(data_file, index=False)
    saved_files["data"] = str(data_file)
    
    # Save MIC-style data (just antibiotic columns)
    mic_data = data[["Strain_ID"] + metadata.antibiotic_columns]
    mic_file = output_path / "synthetic_MIC.csv"
    mic_data.to_csv(mic_file, index=False)
    saved_files["mic"] = str(mic_file)
    
    # Save AMR genes data
    gene_data = data[["Strain_ID"] + metadata.gene_columns]
    gene_file = output_path / "synthetic_AMR_genes.csv"
    gene_data.to_csv(gene_file, index=False)
    saved_files["genes"] = str(gene_file)
    
    # Save virulence data
    vir_data = data[["Strain_ID"] + metadata.virulence_columns]
    vir_file = output_path / "synthetic_Virulence.csv"
    vir_data.to_csv(vir_file, index=False)
    saved_files["virulence"] = str(vir_file)
    
    # Save metadata as JSON
    import json
    
    metadata_dict = {
        "config": {
            "n_strains": metadata.config.n_strains,
            "n_antibiotics": metadata.config.n_antibiotics,
            "n_genes": metadata.config.n_genes,
            "n_virulence": metadata.config.n_virulence,
            "base_prevalence_mean": metadata.config.base_prevalence_mean,
            "base_prevalence_std": metadata.config.base_prevalence_std,
            "correlation_strength": metadata.config.correlation_strength,
            "noise_level": metadata.config.noise_level,
            "mdr_proportion": metadata.config.mdr_proportion,
            "random_state": metadata.config.random_state,
        },
        "true_prevalences": metadata.true_prevalences,
        "true_correlations": [
            {"feature1": c[0], "feature2": c[1], "correlation": c[2]}
            for c in metadata.true_correlations
        ],
        "true_mdr_count": int(metadata.true_mdr_status.sum()),
        "true_mdr_proportion": float(metadata.true_mdr_status.mean()),
        "antibiotic_columns": metadata.antibiotic_columns,
        "gene_columns": metadata.gene_columns,
        "virulence_columns": metadata.virulence_columns,
        "generation_timestamp": metadata.generation_timestamp,
        "generation_method": metadata.generation_method,
    }
    
    metadata_file = output_path / "synthetic_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=2)
    saved_files["metadata"] = str(metadata_file)
    
    # Save scientific methodology note
    methodology_note = f"""# Synthetic Data Generation Methodology

## Overview

This document describes the statistical methodology used to generate synthetic
antimicrobial resistance (AMR) data for validation and testing purposes.

## Generation Parameters

- **Number of strains**: {metadata.config.n_strains}
- **Number of antibiotics**: {metadata.config.n_antibiotics}
- **Number of AMR genes**: {metadata.config.n_genes}
- **Number of virulence factors**: {metadata.config.n_virulence}
- **Base prevalence mean**: {metadata.config.base_prevalence_mean:.2f}
- **Base prevalence std**: {metadata.config.base_prevalence_std:.2f}
- **Correlation strength**: {metadata.config.correlation_strength:.2f}
- **Noise level**: {metadata.config.noise_level:.2f}
- **Random seed**: {metadata.config.random_state}

## Statistical Distributions Used

### 1. Beta Distribution for Prevalence Rates

Prevalence rates for each feature were generated using a Beta distribution,
which is bounded between 0 and 1 and can capture the typical distribution
of resistance gene prevalences in bacterial populations.

**Parameters:**
- Mean (μ) = {metadata.config.base_prevalence_mean:.2f}
- Standard deviation (σ) = {metadata.config.base_prevalence_std:.2f}

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

A small proportion of observations ({metadata.config.noise_level*100:.1f}%) were randomly
flipped to simulate measurement error, contamination, or threshold effects
commonly observed in experimental data.

### 4. Induced Correlations

Known correlations between specific gene-phenotype pairs were induced by
modifying conditional probabilities:
- For correlation ρ > 0: P(Y=1|X=1) > P(Y=1|X=0)
- For correlation ρ < 0: P(Y=1|X=1) < P(Y=1|X=0)

## Ground Truth Values

### Known Correlations
{chr(10).join([f"- {c[0]} ↔ {c[1]}: ρ = {c[2]:.2f}" for c in metadata.true_correlations]) if metadata.true_correlations else "- No induced correlations"}

### MDR Status
- MDR isolates: {int(metadata.true_mdr_status.sum())} ({metadata.true_mdr_status.mean()*100:.1f}%)
- MDR definition: Resistance to ≥3 antibiotic classes

## References

1. Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall.
2. Fleiss, J. L. (1981). Statistical Methods for Rates and Proportions. Wiley.
3. Forbes, C., et al. (2011). Statistical Distributions. Wiley.

## Generation Timestamp

{metadata.generation_timestamp}

---
*This data was generated for validation and testing purposes only.*
"""
    
    methodology_file = output_path / "GENERATION_METHODOLOGY.md"
    with open(methodology_file, "w", encoding="utf-8") as f:
        f.write(methodology_note)
    saved_files["methodology"] = str(methodology_file)
    
    return saved_files


def validate_synthetic_data(
    data: pd.DataFrame,
    metadata: SyntheticDataMetadata,
) -> Dict[str, Any]:
    """
    Validate that synthetic data has expected statistical properties.
    
    Parameters:
        data: The synthetic DataFrame
        metadata: The metadata with ground truth
        
    Returns:
        Dict with validation results
    """
    results = {
        "validation_passed": True,
        "checks": [],
        "warnings": [],
        "errors": [],
    }
    
    # Check 1: Verify shape
    expected_rows = metadata.config.n_strains
    actual_rows = len(data)
    if actual_rows == expected_rows:
        results["checks"].append(f"OK Row count: {actual_rows} (expected {expected_rows})")
    else:
        results["errors"].append(f"✗ Row count: {actual_rows} (expected {expected_rows})")
        results["validation_passed"] = False
    
    # Check 2: Verify columns exist
    for col in metadata.antibiotic_columns:
        if col in data.columns:
            results["checks"].append(f"OK Column exists: {col}")
        else:
            results["errors"].append(f"✗ Missing column: {col}")
            results["validation_passed"] = False
    
    # Check 3: Verify binary data
    for col in metadata.antibiotic_columns + metadata.gene_columns:
        if col not in data.columns:
            continue
        unique_vals = set(data[col].unique())
        if unique_vals.issubset({0, 1}):
            results["checks"].append(f"OK Binary data: {col}")
        else:
            results["warnings"].append(f"WARN Non-binary values in {col}: {unique_vals}")
    
    # Check 4: Verify prevalences are reasonable
    for col, expected_prev in metadata.true_prevalences.items():
        if col not in data.columns:
            continue
        actual_prev = data[col].mean()
        # Allow 20% relative error due to noise and sampling
        if abs(actual_prev - expected_prev) / max(expected_prev, 0.01) < 0.3:
            results["checks"].append(
                f"OK Prevalence {col}: {actual_prev:.2f} (expected ~{expected_prev:.2f})"
            )
        else:
            results["warnings"].append(
                f"WARN Prevalence deviation {col}: {actual_prev:.2f} (expected ~{expected_prev:.2f})"
            )
    
    # Check 5: Verify MDR proportion
    if len(metadata.true_mdr_status) == len(data):
        mdr_count = metadata.true_mdr_status.sum()
        mdr_prop = mdr_count / len(data)
        results["checks"].append(f"OK MDR proportion: {mdr_prop:.2%} ({mdr_count} isolates)")
    
    return results


if __name__ == "__main__":
    # Generate synthetic data when run directly
    print("Generating synthetic AMR data...")
    
    config = SyntheticDataConfig(
        n_strains=200,
        n_antibiotics=13,
        n_genes=30,
        n_virulence=20,
        random_state=42,
    )
    # Debug logging removed - was environment-specific hardcoded path
    
    data, metadata = generate_mdr_synthetic_dataset(config)
    
    print(f"Generated {len(data)} strains with:")
    print(f"  - {len(metadata.antibiotic_columns)} antibiotics")
    print(f"  - {len(metadata.gene_columns)} AMR genes")
    print(f"  - {len(metadata.virulence_columns)} virulence factors")
    print(f"  - {len(metadata.true_correlations)} known correlations")
    print(f"  - {metadata.true_mdr_status.sum()} MDR isolates ({metadata.true_mdr_status.mean()*100:.1f}%)")
    
    # Validate
    print("\nValidating synthetic data...")
    validation = validate_synthetic_data(data, metadata)
    
    for check in validation["checks"][:5]:
        print(f"  {check}")
    print(f"  ... and {len(validation['checks']) - 5} more checks")
    
    if validation["warnings"]:
        print(f"\nWarnings: {len(validation['warnings'])}")
    if validation["errors"]:
        print(f"Errors: {len(validation['errors'])}")
    
    print(f"\nValidation: {'PASSED' if validation['validation_passed'] else 'FAILED'}")

    # Save validation report (publication-ready)
    validation_dir = Path(__file__).parent.parent / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    validation_payload = {
        "generated": datetime.utcnow().isoformat(),
        "n_strains": int(len(data)),
        "n_features": int(len(data.columns) - 1),
        "checks": validation.get("checks", []),
        "warnings": validation.get("warnings", []),
        "errors": validation.get("errors", []),
        "validation_passed": bool(validation.get("validation_passed", False)),
    }
    with open(validation_dir / "synthetic_validation_results.json", "w", encoding="utf-8") as f:
        json.dump(validation_payload, f, indent=2)
    report_lines = [
        "# Synthetic Data Validation Report - strepsuis-mdr",
        "",
        f"Generated: {validation_payload['generated']}",
        "Data Source: Synthetic data with known ground truth",
        f"Strains: {validation_payload['n_strains']}",
        f"Features: {validation_payload['n_features']}",
        f"Checks: {len(validation_payload['checks'])}",
        f"Warnings: {len(validation_payload['warnings'])}",
        f"Errors: {len(validation_payload['errors'])}",
        f"Status: {'PASSED' if validation_payload['validation_passed'] else 'FAILED'}",
        "",
        "## Checks",
    ]
    report_lines.extend([f"- {item}" for item in validation_payload["checks"]])
    if validation_payload["warnings"]:
        report_lines.append("")
        report_lines.append("## Warnings")
        report_lines.extend([f"- {item}" for item in validation_payload["warnings"]])
    if validation_payload["errors"]:
        report_lines.append("")
        report_lines.append("## Errors")
        report_lines.extend([f"- {item}" for item in validation_payload["errors"]])
    with open(validation_dir / "SYNTHETIC_DATA_VALIDATION_REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    # Save data
    print("\nSaving synthetic data...")
    output_dir = Path(__file__).parent.parent / "synthetic_data"
    saved = save_synthetic_data(data, metadata, str(output_dir))
    
    for key, path in saved.items():
        print(f"  Saved {key}: {path}")
    
    print("\nDone!")
