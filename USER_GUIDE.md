# User Guide for StrepSuisMDR

Advanced multidrug resistance pattern detection with bootstrap resampling and network analysis

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
  - [Method 1: Python Package](#method-1-python-package)
  - [Method 2: Docker Container](#method-2-docker-container)
  - [Method 3: Google Colab](#method-3-google-colab)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Input Data Format](#input-data-format)
- [Output Files](#output-files)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

## Introduction

StrepSuisMDR is a professional bioinformatics tool for analyzing bacterial genomic data. This guide provides step-by-step instructions for using the tool across three different deployment methods.

### Who Should Use This Tool?

- **Researchers** studying antimicrobial resistance and virulence
- **Bioinformaticians** analyzing bacterial genomic data
- **Public health professionals** tracking resistance patterns
- **Students** learning bioinformatics analysis

### What You Need

- Input data in CSV format (see [Input Data Format](#input-data-format))
- Basic knowledge of bacterial genomics (recommended)
- Python 3.8+ OR Docker OR Google account (for Colab)

## Installation

Choose ONE of the three installation methods below based on your preference and technical expertise.

### Method 1: Python Package

**Best for:** Users comfortable with Python and command-line tools

**Requirements:**
- Python 3.8 or higher
- pip package manager

**Steps:**

1. Install directly from GitHub:
   ```bash
   pip install git+https://github.com/MK-vet/strepsuis-mdr.git
   ```

2. Verify installation:
   ```bash
   strepsuis-mdr --version
   strepsuis-mdr --help
   ```

3. You're ready to use the tool!

### Method 2: Docker Container

**Best for:** Users who want a consistent environment across different systems

**Requirements:**
- Docker installed on your system
- Basic knowledge of Docker commands

**Steps:**

1. Pull the Docker image (when published):
   ```bash
   docker pull ghcr.io/mk-vet/strepsuis-mdr:latest
   ```

   OR build locally:
   ```bash
   git clone https://github.com/MK-vet/strepsuis-mdr.git
   cd strepsuis-mdr
   docker build -t strepsuis-mdr:latest .
   ```

2. Test the container:
   ```bash
   docker run --rm strepsuis-mdr:latest --help
   ```

3. Run with your data:
   ```bash
   docker run -v $(pwd)/data:/data -v $(pwd)/output:/output \
       strepsuis-mdr:latest --data-dir /data --output /output
   ```

### Method 3: Google Colab

**Best for:** Users who want to run analysis without installing anything

**Requirements:**
- Google account
- Web browser
- Internet connection

**Steps:**

1. Open the Colab notebook:
   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MK-vet/strepsuis-mdr/blob/main/notebooks/AMRPat_Analysis.ipynb)

2. Follow the instructions in the notebook:
   - Run the first cell to install the package
   - Upload your data files when prompted
   - Configure parameters using the provided interface
   - Run the analysis
   - Download results

3. No installation needed!

## Quick Start

### Using Python Package

```bash
# Navigate to your data directory
cd /path/to/your/data

# Run analysis with default parameters
strepsuis-mdr --data-dir . --output ./results

# Run with custom parameters
strepsuis-mdr \
  --data-dir . \
  --output ./results \
  --bootstrap 1000 \
  --fdr-alpha 0.05
```

### Using Docker

```bash
# Run with current directory as data source
docker run -v $(pwd):/data -v $(pwd)/output:/output \
    strepsuis-mdr:latest --data-dir /data --output /output
```

### Using Docker Compose

```bash
# Clone the repository
git clone https://github.com/MK-vet/strepsuis-mdr.git
cd strepsuis-mdr

# Place your data files in the examples/ directory

# Run the analysis
docker-compose up
```

## Detailed Usage

### Command-Line Options

```
strepsuis-mdr [OPTIONS]

Required:
  --data-dir PATH       Directory containing input CSV files
  --output PATH         Directory for output files

Optional:
  --bootstrap INT       Number of bootstrap iterations (default: 500)
  --fdr-alpha FLOAT     FDR significance level (default: 0.05)
  --max-clusters INT    Maximum number of clusters (default: 8)
  --random-seed INT     Random seed for reproducibility (default: 42)
  --verbose            Enable verbose output
  --version            Show version and exit
  --help               Show this help message
```

### Python API

For programmatic access:

```python
from strepsuis_mdr import MDRAnalyzer

# Initialize analyzer
analyzer = MDRAnalyzer(
    data_dir="./data",
    output_dir="./results",
    bootstrap_iterations=500,
    fdr_alpha=0.05
)

# Run analysis
results = analyzer.run()

# Access results
print(results['summary'])

# Generate reports
analyzer.generate_html_report(results)
analyzer.generate_excel_report(results)
```

## Input Data Format

### Required Files

Your data directory must contain CSV files with a `Strain_ID` column:

- `MIC.csv` - Minimum Inhibitory Concentration data
- `AMR_genes.csv` - Antimicrobial resistance genes

### Binary Data Format

**IMPORTANT:** Binary data must be encoded as:
- `0` = Absence of feature/gene/resistance
- `1` = Presence of feature/gene/resistance

Do NOT use missing values (NaN) - encode as 0 or 1.

### Example Data Structure

```csv
Strain_ID,Gene1,Gene2,Gene3
Strain001,1,0,1
Strain002,0,1,1
Strain003,1,1,0
```

### Data Requirements

- **Format:** CSV (comma-separated values)
- **Encoding:** UTF-8
- **First column:** Must be named `Strain_ID`
- **Data columns:** Binary (0/1) or categorical
- **Missing data:** Not allowed - encode as 0 or 1
- **Strain IDs:** Must be unique

### Example Datasets

Example datasets are provided in the `examples/` directory. You can use these to test the tool:

```bash
strepsuis-mdr --data-dir examples/ --output test_output/
```

## Output Files

The tool generates multiple output files:

### 1. HTML Report

**File:** `MDR_Analysis_report.html`

Interactive HTML report with:
- Summary statistics
- Interactive visualizations
- Detailed results tables
- Interpretation guide

### 2. Excel Report

**File:** `MDR_Analysis_Report_<timestamp>.xlsx`

Multi-sheet Excel workbook with:
- Metadata sheet with analysis parameters
- Summary statistics
- Detailed results tables
- Methodology description

### 3. PNG Charts

**Directory:** `png_charts/`

High-resolution (150+ DPI) charts:
- Network visualizations
- Clustering plots
- Statistical charts
- Heatmaps

### Output Directory Structure

```
output/
├── MDR_Analysis_report.html
├── MDR_Analysis_Report_20250114_120000.xlsx
└── png_charts/
    ├── network_visualization.png
    ├── cluster_plot.png
    ├── heatmap.png
    └── ...
```

## Examples

### Example 1: Basic Analysis

```bash
# Using example data
strepsuis-mdr --data-dir examples/ --output results/

# Check results
ls results/
```

### Example 2: Custom Parameters

```bash
# Higher bootstrap iterations for more robust results
strepsuis-mdr \
  --data-dir my_data/ \
  --output my_results/ \
  --bootstrap 1000 \
  --fdr-alpha 0.01 \
  --verbose
```

### Example 3: Batch Processing

```bash
# Process multiple datasets
for dataset in data_*/; do
  output="results_$(basename $dataset)"
  strepsuis-mdr --data-dir "$dataset" --output "$output"
done
```

### Example 4: Docker with Local Data

```bash
# Process data in current directory
docker run \
  -v $(pwd)/my_data:/data \
  -v $(pwd)/my_results:/output \
  strepsuis-mdr:latest \
  --data-dir /data \
  --output /output \
  --bootstrap 1000
```

## Troubleshooting

### Common Issues

#### Issue: "FileNotFoundError: No such file or directory"

**Solution:** Ensure all required CSV files are in the data directory.

```bash
# Check files in data directory
ls -la data/

# Required files: MIC.csv, AMR_genes.csv
```

#### Issue: "ValueError: Strain_ID column not found"

**Solution:** Ensure your CSV files have a `Strain_ID` column as the first column.

```bash
# Check CSV structure
head -n 5 data/MIC.csv
```

#### Issue: "ModuleNotFoundError: No module named 'strepsuis_mdr'"

**Solution:** Reinstall the package.

```bash
pip uninstall -y strepsuis_mdr
pip install git+https://github.com/MK-vet/strepsuis-mdr.git
```

#### Issue: Docker container exits immediately

**Solution:** Check Docker logs and mount paths.

```bash
# View container logs
docker logs <container_id>

# Ensure correct mount paths
docker run -v $(pwd)/data:/data strepsuis-mdr:latest --data-dir /data --output /output
```

#### Issue: Low memory errors

**Solution:** 
- Close other applications
- Use a machine with more RAM
- Reduce dataset size for testing
- Use Google Colab Pro for larger datasets

### Getting Help

If you encounter issues not covered here:

1. Check the [FAQ](#faq) section below
2. Review the [CONTRIBUTING.md](CONTRIBUTING.md) file
3. Search existing [GitHub Issues](https://github.com/MK-vet/strepsuis-mdr/issues)
4. Create a new issue with:
   - Error message
   - Steps to reproduce
   - System information (OS, Python version, etc.)

## FAQ

### General Questions

**Q: Can I use this tool with my own bacterial species?**

A: Yes! While developed for *Streptococcus suis*, the tool works with any bacterial species data in the correct format.

**Q: How long does the analysis take?**

A: Depends on dataset size and bootstrap iterations. Typical runtime: 1-5 minutes for small datasets (<100 strains), 10-30 minutes for larger datasets (>500 strains).

**Q: Can I run this on Windows?**

A: Yes! All three methods (Python package, Docker, Google Colab) work on Windows. Python and Docker require installation on Windows.

**Q: Is this tool free to use?**

A: Yes! The tool is open-source under the MIT License. You can use it freely for research and commercial purposes.

### Technical Questions

**Q: What Python version do I need?**

A: Python 3.8 or higher. Tested on Python 3.8, 3.9, 3.10, 3.11, and 3.12.

**Q: Can I modify the source code?**

A: Yes! The tool is open-source. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Q: How do I cite this tool?**

A: See the citation information in the main [README.md](README.md).

**Q: What statistical methods are used?**

A: See the Methodology section in the HTML/Excel reports for detailed information about statistical methods.

### Data Questions

**Q: What if my data has missing values?**

A: The tool requires complete data. Encode missing values as 0 (absence) or 1 (presence) based on your biological interpretation.

**Q: Can I use non-binary data?**

A: The tool is designed for binary data (0/1). Categorical data may work but is not officially supported.

**Q: What's the maximum dataset size?**

A: Limited by available memory. Tested with up to 1000 strains and 500 features. For larger datasets, use a high-memory system or Google Colab Pro.

### Output Questions

**Q: Can I customize the output format?**

A: The tool generates standard HTML and Excel reports. You can modify the source code for custom outputs.

**Q: How do I interpret the results?**

A: See the Interpretation section in the HTML report and the separate interpretation guide in the documentation.

**Q: Can I use the figures in publications?**

A: Yes! All PNG charts are publication-quality (150+ DPI). Ensure to cite the tool appropriately.

---

## Additional Resources

- **Main Documentation:** [README.md](README.md)
- **Contributing Guide:** [CONTRIBUTING.md](CONTRIBUTING.md)
- **Example Data:** [examples/](examples/)
- **Source Code:** [GitHub Repository](https://github.com/MK-vet/strepsuis-mdr)
- **Issues & Support:** [GitHub Issues](https://github.com/MK-vet/strepsuis-mdr/issues)

## Version Information

- **Current Version:** 1.0.0
- **Last Updated:** 2025-01-14
- **License:** MIT

---

**Need help?** Open an issue on [GitHub](https://github.com/MK-vet/strepsuis-mdr/issues) or refer to the troubleshooting section above.
