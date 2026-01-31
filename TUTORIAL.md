# StrepSuisMDR Tutorial

## Quick Start Guide

This tutorial will guide you through using StrepSuisMDR for multidrug resistance analysis.

## Table of Contents

1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Running Analysis](#running-analysis)
4. [Understanding Results](#understanding-results)
5. [Advanced Usage](#advanced-usage)

## Installation

### Option 1: pip install (recommended)

```bash
pip install strepsuis-mdr
```

### Option 2: From source

```bash
git clone https://github.com/MK-vet/MKrep.git
cd MKrep/separated_repos/strepsuis-mdr
pip install -e .
```

### Verify installation

```python
import strepsuis_mdr
print(strepsuis_mdr.__version__)
```

## Data Preparation

### Required Files

Your data directory should contain:

1. **MIC.csv** - Minimum Inhibitory Concentration data
2. **AMR_genes.csv** - Antimicrobial resistance genes

### File Format

All CSV files must have:
- First column: `Strain_ID`
- Binary values: 0 (absent) or 1 (present)
- UTF-8 encoding

Example `MIC.csv`:
```csv
Strain_ID,Penicillin_R,Tetracycline_R,Erythromycin_R
Strain001,1,0,1
Strain002,0,1,1
Strain003,1,1,0
```

## Running Analysis

### Command Line Interface

```bash
# Basic usage
strepsuis-mdr --data-dir ./data --output ./results

# With custom parameters
strepsuis-mdr \
  --data-dir ./data \
  --output ./results \
  --bootstrap 1000 \
  --fdr-alpha 0.05
```

### Python API

```python
from strepsuis_mdr import MDRAnalyzer

# Initialize
analyzer = MDRAnalyzer(
    data_dir="./data",
    output_dir="./results"
)

# Run analysis
results = analyzer.run()

# Check results
print(f"Status: {results['status']}")
print(f"Output: {results['output_dir']}")
```

## Understanding Results

### Output Files

1. **HTML Report** - Interactive tables and visualizations
2. **Excel Report** - Multi-sheet workbook with all results
3. **PNG Charts** - Publication-ready figures

### Key Metrics

- **Bootstrap CI**: 95% confidence intervals for prevalence
- **Phi Coefficient**: Association strength (-1 to 1)
- **FDR-corrected p-values**: Multiple testing correction
- **Network Risk Score**: Novel MDR risk metric

## Advanced Usage

### Custom Configuration

```python
from strepsuis_mdr import Config, MDRAnalyzer

config = Config(
    bootstrap_iterations=1000,
    fdr_alpha=0.05,
    min_support=0.1,
    min_confidence=0.5
)

analyzer = MDRAnalyzer(
    data_dir="./data",
    output_dir="./results",
    config=config
)
```

### Using Innovations

#### Network Risk Scoring

```python
from strepsuis_mdr.mdr_analysis_core import compute_network_mdr_risk_score

# Calculate risk scores
risk_scores = compute_network_mdr_risk_score(
    network=G,
    strain_features=features_df,
    bootstrap_ci=ci_df
)
```

#### Sequential Pattern Detection

```python
from strepsuis_mdr.mdr_analysis_core import detect_sequential_resistance_patterns

# Detect patterns
patterns = detect_sequential_resistance_patterns(
    data=binary_df,
    min_support=0.1,
    min_confidence=0.5
)
```

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Check that all required CSV files exist
2. **ValueError**: Ensure data contains only 0/1 values
3. **MemoryError**: Reduce bootstrap iterations for large datasets

### Getting Help

- GitHub Issues: https://github.com/MK-vet/strepsuis-mdr/issues
- Documentation: See README.md and USER_GUIDE.md

## Performance Tips

Based on our benchmarks:

| Dataset Size | Recommended Bootstrap | Expected Time |
|--------------|----------------------|---------------|
| < 100 strains | 1000 | < 1 min |
| 100-500 strains | 500 | 1-5 min |
| > 500 strains | 200 | 5-15 min |

## Next Steps

- Read [ALGORITHMS.md](ALGORITHMS.md) for details on novel features and algorithms
- See [VALIDATION.md](VALIDATION.md) for statistical validation
- Check [BENCHMARKS.md](BENCHMARKS.md) for performance data
