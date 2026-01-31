# Data Directory

This module uses the shared example datasets from the main MKrep repository.

## Data Location

The example datasets are located in the main repository at:
```
../../data/
```

## Available Datasets

The following files are available in the main data directory:

- **AMR_genes.csv** - Antimicrobial resistance gene profiles
- **MGE.csv** - Mobile genetic elements data
- **MIC.csv** - Minimum inhibitory concentration values
- **MLST.csv** - Multi-locus sequence typing data
- **Plasmid.csv** - Plasmid information
- **Serotype.csv** - Serotype classifications
- **Virulence.csv** - Virulence factor profiles
- **Snp_tree.newick** - Phylogenetic tree (for modules that need it)
- **merged_resistance_data.csv** - Pre-merged resistance data

## Usage

When running analyses, use the `--data-dir` option to point to the main data directory:

```bash
# From the module directory
strepsuis-mdr analyze --data-dir ../../data/ --output ./results/

# Or from the main repository root
strepsuis-mdr analyze --data-dir ./data/ --output ./separated_repos/strepsuis-mdr/results/
```

## Generating Results

To generate example analysis results:

```bash
# Install the module
pip install -e .

# Run analysis with example data
strepsuis-mdr analyze \
  --amr-file ../../data/AMR_genes.csv \
  --mic-file ../../data/MIC.csv \
  --output ./results/ \
  --bootstrap-iterations 500 \
  --random-seed 42
```

Results will be saved to `./results/` and will include:
- HTML interactive report
- Excel workbook with multiple sheets
- PNG charts directory with publication-quality figures

## Note

This structure eliminates data duplication across modules while maintaining independence.
Each module can still be deployed standalone by copying the necessary data files from the main repository.
