"""
StrepSuisMDR: Integrated Phenotype-Genotype Analysis of Multidrug Resistance
=============================================================================

A bioinformatics tool for comprehensive analysis of antimicrobial resistance
patterns in bacterial genomics, with focus on Streptococcus suis.

Features:
    - Bootstrap resampling for prevalence estimation with confidence intervals
    - Statistical association testing (Chi-square, Fisher's exact, Phi coefficient)
    - Multiple testing correction using Benjamini-Hochberg FDR
    - Network-based co-resistance pattern analysis with Louvain clustering
    - Association rule mining for resistance gene co-occurrence

Example:
    >>> from strepsuis_mdr import MDRAnalyzer, Config
    >>> config = Config(data_dir="./data", output_dir="./output")
    >>> analyzer = MDRAnalyzer(config)
    >>> results = analyzer.run()

Author: MK-vet
License: MIT
"""

__version__ = "1.0.0"
__author__ = "MK-vet"
__license__ = "MIT"

from .analyzer import MDRAnalyzer
from .config import Config, AnalysisConfig
from .validation_utils import validate_input_data, validate_mdr_output, get_data_summary
from .synthetic_data_utils import (
    generate_synthetic_amr_data,
    generate_cooccurrence_data,
    run_synthetic_smoke_test,
)
from .generate_synthetic_data import (
    SyntheticDataConfig,
    SyntheticDataMetadata,
    generate_mdr_synthetic_dataset,
    save_synthetic_data,
    validate_synthetic_data,
)

# High-performance data backend (Parquet + DuckDB)
from .data_backend import DataBackend, load_data_efficient, get_backend_status

# Uncertainty quantification (Bootstrap CI + Permutation tests)
from .uncertainty import UncertaintyQuantifier, apply_default_uncertainty

# Parallel network analysis (Community detection with consensus)
from .parallel_network import (
    parallel_community_detection,
    compute_parallel_modularity,
    export_community_dataframe,
)

# Advanced statistical features from shared module
try:
    from shared.advanced_statistics import (
        confidence_aware_rules,
        edge_confidence_network,
        entropy_weighted_importance,
        bootstrap_stability_matrix,
        redundancy_pruning,
    )
    _HAS_ADVANCED_STATS = True
except ImportError:
    _HAS_ADVANCED_STATS = False

__all__ = [
    "MDRAnalyzer",
    "Config",
    "AnalysisConfig",
    "validate_input_data",
    "validate_mdr_output",
    "get_data_summary",
    "generate_synthetic_amr_data",
    "generate_cooccurrence_data",
    "run_synthetic_smoke_test",
    "SyntheticDataConfig",
    "SyntheticDataMetadata",
    "generate_mdr_synthetic_dataset",
    "save_synthetic_data",
    "validate_synthetic_data",
    "DataBackend",
    "load_data_efficient",
    "get_backend_status",
    "UncertaintyQuantifier",
    "apply_default_uncertainty",
    "parallel_community_detection",
    "compute_parallel_modularity",
    "export_community_dataframe",
    "__version__",
]

# Add advanced statistics if available
if _HAS_ADVANCED_STATS:
    __all__.extend([
        "confidence_aware_rules",
        "edge_confidence_network",
        "entropy_weighted_importance",
        "bootstrap_stability_matrix",
        "redundancy_pruning",
    ])
