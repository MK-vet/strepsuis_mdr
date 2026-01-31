#!/usr/bin/env python3
"""
Synthetic Data Generator for StrepSuis-MDR Mathematical Validation

Generates three types of synthetic datasets:
1. clean_dataset.csv - Perfect data, no noise
2. noisy_dataset.csv - Realistic noise (10% random flips)
3. adversarial_dataset.csv - Edge cases (all zeros, all ones, etc.)

These datasets are used to validate mathematical functions against ground truth.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)


def generate_clean_dataset(n_strains: int = 200, n_features: int = 20) -> pd.DataFrame:
    """
    Generate clean synthetic dataset with known properties.
    
    Args:
        n_strains: Number of strains/isolates
        n_features: Number of binary features
        
    Returns:
        DataFrame with clean binary data
    """
    set_seed(42)
    
    # Create strain IDs
    strain_ids = [f"STRAIN_{i:04d}" for i in range(n_strains)]
    
    # Generate binary features with different prevalences
    data = {}
    data['Strain_ID'] = strain_ids
    
    # Features with varying prevalence rates
    prevalences = np.linspace(0.1, 0.9, n_features)
    
    for i, prev in enumerate(prevalences):
        feature_name = f"Feature_{i:02d}"
        data[feature_name] = np.random.binomial(1, prev, n_strains)
    
    # Add some perfectly correlated features for testing
    data['Perfect_Corr_A'] = data['Feature_05'].copy()
    data['Perfect_Corr_B'] = 1 - data['Feature_05']  # Perfect negative correlation
    
    # Add some moderately correlated features
    data['Mod_Corr_A'] = data['Feature_10'].copy()
    # Flip 20% of values for moderate correlation
    flip_indices = np.random.choice(n_strains, size=int(0.2 * n_strains), replace=False)
    data['Mod_Corr_A'][flip_indices] = 1 - data['Mod_Corr_A'][flip_indices]
    
    df = pd.DataFrame(data)
    return df


def generate_noisy_dataset(clean_df: pd.DataFrame, noise_rate: float = 0.1) -> pd.DataFrame:
    """
    Generate noisy dataset by adding random bit flips.
    
    Args:
        clean_df: Clean dataset to add noise to
        noise_rate: Proportion of bits to flip (default 10%)
        
    Returns:
        DataFrame with noisy data
    """
    set_seed(43)  # Different seed for noise
    
    noisy_df = clean_df.copy()
    
    # Don't add noise to Strain_ID column
    feature_cols = [col for col in noisy_df.columns if col != 'Strain_ID']
    
    for col in feature_cols:
        n_flips = int(len(noisy_df) * noise_rate)
        flip_indices = np.random.choice(len(noisy_df), size=n_flips, replace=False)
        noisy_df.loc[flip_indices, col] = 1 - noisy_df.loc[flip_indices, col]
    
    return noisy_df


def generate_adversarial_dataset(n_strains: int = 200) -> pd.DataFrame:
    """
    Generate adversarial dataset with edge cases.
    
    Contains:
    - All zeros columns
    - All ones columns
    - Single positive/negative cases
    - Perfect correlations
    - Nearly constant columns
    
    Returns:
        DataFrame with edge case data
    """
    set_seed(44)
    
    strain_ids = [f"EDGE_{i:03d}" for i in range(n_strains)]
    
    data = {'Strain_ID': strain_ids}
    
    # All zeros
    data['All_Zeros'] = np.zeros(n_strains, dtype=int)
    
    # All ones
    data['All_Ones'] = np.ones(n_strains, dtype=int)
    
    # Single positive
    single_pos = np.zeros(n_strains, dtype=int)
    single_pos[0] = 1
    data['Single_Positive'] = single_pos
    
    # Single negative
    single_neg = np.ones(n_strains, dtype=int)
    single_neg[0] = 0
    data['Single_Negative'] = single_neg
    
    # Nearly constant (98% zeros)
    nearly_zero = np.zeros(n_strains, dtype=int)
    nearly_zero[0] = 1
    data['Nearly_Zero'] = nearly_zero
    
    # Nearly constant (98% ones)
    nearly_one = np.ones(n_strains, dtype=int)
    nearly_one[0] = 0
    data['Nearly_One'] = nearly_one
    
    # Perfect positive correlation pair
    perfect_a = np.random.binomial(1, 0.5, n_strains)
    data['Perfect_Pos_A'] = perfect_a
    data['Perfect_Pos_B'] = perfect_a.copy()
    
    # Perfect negative correlation pair
    perfect_neg_a = np.random.binomial(1, 0.5, n_strains)
    data['Perfect_Neg_A'] = perfect_neg_a
    data['Perfect_Neg_B'] = 1 - perfect_neg_a
    
    # Moderately variable features
    data['Moderate_50'] = np.random.binomial(1, 0.5, n_strains)
    data['Moderate_25'] = np.random.binomial(1, 0.25, n_strains)
    data['Moderate_75'] = np.random.binomial(1, 0.75, n_strains)
    
    # Alternating pattern
    data['Alternating'] = np.array([i % 2 for i in range(n_strains)])
    
    df = pd.DataFrame(data)
    return df


def compute_ground_truth_stats(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Compute ground truth statistics for validation.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dictionary of statistics per column
    """
    stats = {}
    
    feature_cols = [col for col in df.columns if col != 'Strain_ID']
    
    for col in feature_cols:
        values = df[col].values
        stats[col] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'prevalence': float(np.sum(values) / len(values)),
            'n_positive': int(np.sum(values)),
            'n_negative': int(len(values) - np.sum(values)),
            'is_constant': bool(np.std(values) == 0),
            'is_binary': bool(set(values).issubset({0, 1}))
        }
    
    return stats


def save_datasets_with_metadata(output_dir: Path):
    """
    Generate and save all three datasets with metadata.
    
    Args:
        output_dir: Directory to save datasets
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating synthetic datasets...")
    # #region agent log
    try:
        import json
        import time

        payload = {
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": "H2",
            "location": "synthetic_data/generate_synthetic_data.py",
            "message": "mdr_math_synthetic_config",
            "data": {"clean_n_strains": 200, "clean_n_features": 20},
            "timestamp": int(time.time() * 1000),
        }
        with open(r"c:\Users\ABC\Documents\GitHub\.cursor\debug.log", "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass
    # #endregion
    
    # Generate clean dataset
    print("  1. Generating clean_dataset.csv...")
    clean_df = generate_clean_dataset(n_strains=200, n_features=20)
    clean_df.to_csv(output_dir / "clean_dataset.csv", index=False)
    clean_stats = compute_ground_truth_stats(clean_df)
    
    # Generate noisy dataset
    print("  2. Generating noisy_dataset.csv...")
    noisy_df = generate_noisy_dataset(clean_df, noise_rate=0.1)
    noisy_df.to_csv(output_dir / "noisy_dataset.csv", index=False)
    noisy_stats = compute_ground_truth_stats(noisy_df)
    
    # Generate adversarial dataset
    print("  3. Generating adversarial_dataset.csv...")
    adversarial_df = generate_adversarial_dataset(n_strains=200)
    adversarial_df.to_csv(output_dir / "adversarial_dataset.csv", index=False)
    adversarial_stats = compute_ground_truth_stats(adversarial_df)
    
    # Save metadata
    print("  4. Saving metadata...")
    metadata = {
        'clean_dataset': {
            'description': 'Perfect data with no noise',
            'n_strains': len(clean_df),
            'n_features': len(clean_df.columns) - 1,
            'statistics': clean_stats
        },
        'noisy_dataset': {
            'description': 'Realistic noise (10% random flips)',
            'n_strains': len(noisy_df),
            'n_features': len(noisy_df.columns) - 1,
            'noise_rate': 0.1,
            'statistics': noisy_stats
        },
        'adversarial_dataset': {
            'description': 'Edge cases (zeros, ones, constants, etc.)',
            'n_strains': len(adversarial_df),
            'n_features': len(adversarial_df.columns) - 1,
            'statistics': adversarial_stats
        }
    }
    
    import json
    with open(output_dir / "metadata.json", 'w', encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    # Create README
    readme_content = f"""# Synthetic Data for Mathematical Validation

This directory contains three synthetic datasets used for validating mathematical functions:

## Datasets

### 1. clean_dataset.csv
- **Purpose**: Perfect data with known properties, no noise
- **Strains**: {len(clean_df)}
- **Features**: {len(clean_df.columns) - 1} (20 random + 3 engineered correlations)
- **Use Case**: Validate basic statistical calculations

### 2. noisy_dataset.csv
- **Purpose**: Realistic noise (10% random bit flips)
- **Strains**: {len(noisy_df)}
- **Features**: {len(noisy_df.columns) - 1} (same as clean dataset + noise)
- **Use Case**: Validate robustness to noise

### 3. adversarial_dataset.csv
- **Purpose**: Edge cases for stress testing
- **Strains**: {len(adversarial_df)}
- **Features**: {len(adversarial_df.columns) - 1} engineered edge cases
- **Edge Cases**:
  - All zeros columns
  - All ones columns
  - Single positive/negative cases
  - Nearly constant columns (98% same value)
  - Perfect correlations (positive and negative)
  - Alternating patterns

## Ground Truth Statistics

See `metadata.json` for complete ground truth statistics including:
- Mean, std, prevalence for each feature
- Number of positive/negative cases
- Constant and binary flags

## Usage

```python
import pandas as pd

# Load datasets
clean_df = pd.read_csv("clean_dataset.csv")
noisy_df = pd.read_csv("noisy_dataset.csv")
adversarial_df = pd.read_csv("adversarial_dataset.csv")

# Load metadata
import json
with open("metadata.json") as f:
    metadata = json.load(f)
```

## Validation Protocol

1. Run algorithms on all three datasets
2. Compare results against ground truth in metadata.json
3. Verify error metrics (MAE, RMSE) are within acceptable bounds
4. Check that edge cases are handled gracefully (no crashes, NaN handling)

## Regeneration

To regenerate datasets:
```bash
python generate_synthetic_data.py
```

Uses fixed random seed (42-44) for reproducibility.
"""
    
    with open(output_dir / "README.md", 'w') as f:
        f.write(readme_content)
    
    print(f"\nOK Datasets generated successfully in {output_dir}")
    print(f"  - clean_dataset.csv: {len(clean_df)} strains")
    print(f"  - noisy_dataset.csv: {len(noisy_df)} strains")
    print(f"  - adversarial_dataset.csv: {len(adversarial_df)} strains")
    print(f"  - metadata.json: ground truth statistics")
    print(f"  - README.md: documentation")


if __name__ == "__main__":
    import sys
    
    # Get output directory from command line or use default
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    
    save_datasets_with_metadata(output_dir)
