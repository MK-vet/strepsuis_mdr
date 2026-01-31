# Copilot Instructions for StrepSuis-AMRPat

Automated detection of antimicrobial resistance patterns with bootstrap resampling and network analysis for bacterial genomics.

## Project Overview

- **Language**: Python 3.8+
- **Domain**: Bioinformatics, genomics, AMR pattern detection
- **Package name**: `strepsuis-mdr`
- **CLI command**: `strepsuis-mdr`

## Code Style and Conventions

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Include docstrings for all public functions and classes
- Use descriptive variable names reflecting biological terminology
- Line length: 100 characters (black formatter)

## Repository Structure

- **`strepsuis_mdr/`**: Main package source code
  - `cli.py`: Command-line interface
  - `config.py`: Configuration and validation
  - `analyzer.py`: Core analysis logic
- **`tests/`**: Test suite (pytest)
- **`data/`**: Example datasets
- **`notebooks/`**: Jupyter notebooks for Google Colab
- **`examples/`**: Usage examples

## Building and Testing

### Install Package
```bash
pip install -e .
pip install -e .[dev]  # With development dependencies
```

### Run Tests
```bash
pytest
pytest --cov --cov-report=html
pytest -m "not slow"  # Fast tests only
```

### Code Quality
```bash
pre-commit run --all-files
black .
isort .
ruff check .
```

## Key Features

- Bootstrap resampling for prevalence estimation
- Co-occurrence analysis for phenotypes and genes
- Association rule mining (Apriori algorithm)
- Hybrid co-resistance network construction
- Louvain community detection

## Data Format Conventions

- **Input files**: CSV format with `Strain_ID` column
- **Binary data**: `0 = Absence`, `1 = Presence`
- **Required files**: `MIC.csv`, `AMR_genes.csv`
- **Optional files**: `Virulence.csv`, `MLST.csv`, `Serotype.csv`

## Output Files

- HTML report with Bootstrap 5 styling
- Excel workbook with multiple sheets
- PNG charts (150+ DPI) in `png_charts/` subfolder

## Important Guidelines

- Maintain statistical method documentation
- Use fixed random seeds for reproducibility
- Ensure publication-quality visualizations
- Test across Python 3.8-3.12
- Update CHANGELOG.md for significant changes
