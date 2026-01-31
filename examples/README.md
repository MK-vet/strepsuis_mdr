# Example Usage Scripts

This directory contains example usage scripts and expected outputs for testing and learning StrepSuis-AMRPat.

## Data Location

**Important:** The example datasets are now located in the main repository's data directory:
```
../../data/
```

All CSV files previously stored here have been moved to eliminate duplication.

## Available Examples

### 1. Basic Example (`basic/`)

**Purpose:** Quick test and learning

**Files needed:**
- `../../data/AMR_genes.csv`
- `../../data/MIC.csv`
- `../../data/Virulence.csv`

**Dataset size:** ~91 strains, ~21 features per file

**Expected runtime:** ~1-2 minutes

**What you'll see:**
- Bootstrap-based prevalence estimation and co-occurrence patterns
- Summary statistics and visualizations
- Interactive HTML reports

**Use case:** First-time users, testing installation

### 2. Advanced Example (`advanced/`)

**Purpose:** Comprehensive analysis with all data types

**Files needed:**
- `../../data/AMR_genes.csv`
- `../../data/MGE.csv`
- `../../data/MIC.csv`
- `../../data/MLST.csv`
- `../../data/Plasmid.csv`
- `../../data/Serotype.csv`
- `../../data/Virulence.csv`

**Expected runtime:** ~5-8 minutes

**What you'll see:**
- Complete resistance network analysis with all metadata
- More detailed associations and patterns
- Complete metadata integration

**Use case:** Publication-ready analysis, exploring all features

## Using These Examples

### Command Line

```bash
# Basic example - use main data directory
strepsuis-mdr --data-dir ../../data/ --output results_basic/

# Advanced example
strepsuis-mdr --data-dir ../../data/ --output results_advanced/
```

### Python API

```python
from strepsuis_mdr import Analyzer

# Basic example
analyzer = Analyzer(
    data_dir='../../data/',
    output_dir='results_basic/'
)
results = analyzer.run()

# Advanced example with custom parameters
analyzer = Analyzer(
    data_dir='../../data/',
    output_dir='results_advanced/',
    bootstrap_iterations=1000,
    fdr_alpha=0.05
)
results = analyzer.run()
```

### Google Colab
Reference the main data directory or upload the needed CSV files from `../../data/`.

## Data Format

All example files follow the required format:
- First column: `Strain_ID`
- Binary values: 0 (absent) / 1 (present)
- UTF-8 encoding
- No missing values

## Creating Your Own Data

Use the files in `../../data/` as templates:
1. Keep the same column structure
2. Replace `Strain_ID` values with your strain names
3. Update binary values (0/1) based on your data
4. Ensure no missing values

## Expected Output

Both `basic/` and `advanced/` directories contain `expected_output.txt` files describing what results you should see.

## Questions?

See [USER_GUIDE.md](../USER_GUIDE.md) for detailed data format requirements.
