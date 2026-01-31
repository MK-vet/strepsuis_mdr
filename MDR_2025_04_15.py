#!/usr/bin/env python3
"""
MDR_2025_04_15.py - Canonical Entry Point
==========================================

StrepSuis-AMRPat: Multidrug resistance pattern analysis

This is the canonical entry point for workflow orchestration (Nextflow/Snakemake).
It wraps the strepsuis_mdr package with standardized I/O following the
4-layer architecture.

Usage:
    python MDR_2025_04_15.py --config config.yaml
    python MDR_2025_04_15.py --data-dir input/raw_data --output out/run_20260131

Architecture Compliance:
    Input:  input/raw_data/*.csv + config.yaml
    Output: out/run_<ID>/
            ├── manifest.json
            ├── summary.json
            ├── results/*.parquet
            ├── figures/*.png
            ├── exports/*.csv
            ├── report.pdf
            └── site/

Module ID: StrepSuis-AMRPat
Canonical Name: MDR_2025_04_15.py
Date: 2025-04-15
"""

import sys
import argparse
from pathlib import Path

# Import the actual implementation from the package
from strepsuis_mdr.cli import main as cli_main

def main():
    """
    Canonical entry point with architecture-compliant defaults.
    """
    parser = argparse.ArgumentParser(
        description='StrepSuis-AMRPat: Multidrug resistance pattern analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--config',
        type=Path,
        help='Path to config.yaml (default: input/config.yaml)'
    )

    parser.add_argument(
        '--data-dir',
        type=Path,
        help='Input data directory (default: input/raw_data)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        help='Output directory (default: out/run_<timestamp>)'
    )

    parser.add_argument(
        '--run-id',
        type=str,
        help='Run identifier for output directory (default: timestamp)'
    )

    # Pass through to underlying CLI
    sys.exit(cli_main())


if __name__ == '__main__':
    main()
