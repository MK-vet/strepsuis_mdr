# Synthetic Data for Mathematical Validation

This directory contains three synthetic datasets used for validating mathematical functions:

## Datasets

### 1. clean_dataset.csv
- **Purpose**: Perfect data with known properties, no noise
- **Strains**: 200
- **Features**: 23 (20 random + 3 engineered correlations)
- **Use Case**: Validate basic statistical calculations

### 2. noisy_dataset.csv
- **Purpose**: Realistic noise (10% random bit flips)
- **Strains**: 200
- **Features**: 23 (same as clean dataset + noise)
- **Use Case**: Validate robustness to noise

### 3. adversarial_dataset.csv
- **Purpose**: Edge cases for stress testing
- **Strains**: 200
- **Features**: 14 engineered edge cases
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
