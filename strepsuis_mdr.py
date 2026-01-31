#!/usr/bin/env python3
"""
strepsuis_mdr.py - Canonical Entry Point
=========================================

Multidrug resistance pattern analysis

Usage:
    python strepsuis_mdr.py --data-dir input/raw_data --output out/run_20260131
    python strepsuis_mdr.py --config config.yaml

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
"""

import sys
from strepsuis_mdr.cli import main

if __name__ == '__main__':
    sys.exit(main())
