"""
Comprehensive Unit Tests for CLI and Utility Functions

This module tests:
- CLI argument parsing
- Config validation
- Excel report generation
- Analyzer workflow

Target: 80%+ coverage for cli.py, config.py, excel_report_utils.py
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock


# ============================================================================
# Test CLI functions
# ============================================================================
class TestCLI:
    """Test CLI functionality."""

    def test_setup_logging_calls_basicconfig(self):
        """Test that setup_logging configures logging."""
        from strepsuis_mdr.cli import setup_logging
        import logging
        
        # Just verify it doesn't raise exceptions
        setup_logging(verbose=False)
        setup_logging(verbose=True)

    def test_main_without_args_fails(self):
        """Test that main() fails without required arguments."""
        from strepsuis_mdr.cli import main
        
        with patch('sys.argv', ['strepsuis-mdr']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2  # argparse exit code

    def test_main_with_nonexistent_data_dir(self):
        """Test main() with non-existent data directory."""
        from strepsuis_mdr.cli import main
        
        with patch('sys.argv', ['strepsuis-mdr', '--data-dir', '/nonexistent/path']):
            result = main()
            assert result == 1  # Should fail

    def test_main_with_valid_args(self, tmp_path):
        """Test main() with valid arguments but missing files."""
        from strepsuis_mdr.cli import main
        
        # Create a temp directory as data-dir
        with patch('sys.argv', ['strepsuis-mdr', '--data-dir', str(tmp_path)]):
            result = main()
            # Should fail because MIC.csv and AMR_genes.csv are missing
            assert result == 1


# ============================================================================
# Test Config validation
# ============================================================================
class TestConfigValidation:
    """Test configuration validation."""

    def test_config_with_valid_params(self, tmp_path):
        """Test Config with valid parameters."""
        from strepsuis_mdr.config import Config
        
        config = Config(
            data_dir=str(tmp_path),
            output_dir=str(tmp_path / 'output'),
            bootstrap_iterations=500,
            fdr_alpha=0.05,
            random_seed=42,
        )
        
        assert config.data_dir == str(tmp_path)
        assert config.bootstrap_iterations == 500
        assert config.fdr_alpha == 0.05
        assert config.random_seed == 42

    def test_config_with_minimum_bootstrap(self, tmp_path):
        """Test Config with minimum valid bootstrap iterations."""
        from strepsuis_mdr.config import Config
        
        config = Config(
            data_dir=str(tmp_path),
            bootstrap_iterations=100,
        )
        
        assert config.bootstrap_iterations == 100

    def test_config_with_high_fdr_alpha(self, tmp_path):
        """Test Config with high but valid fdr_alpha."""
        from strepsuis_mdr.config import Config
        
        config = Config(
            data_dir=str(tmp_path),
            fdr_alpha=0.99,
        )
        
        assert config.fdr_alpha == 0.99

    def test_config_creates_nested_output_dir(self, tmp_path):
        """Test Config creates nested output directory."""
        from strepsuis_mdr.config import Config
        
        nested_dir = tmp_path / 'level1' / 'level2' / 'output'
        
        config = Config(
            data_dir=str(tmp_path),
            output_dir=str(nested_dir),
        )
        
        assert os.path.exists(str(nested_dir))


# ============================================================================
# Test Analyzer
# ============================================================================
class TestAnalyzer:
    """Test MDRAnalyzer class."""

    def test_analyzer_initialization(self, tmp_path):
        """Test MDRAnalyzer initialization."""
        from strepsuis_mdr.analyzer import MDRAnalyzer
        from strepsuis_mdr.config import Config
        
        config = Config(data_dir=str(tmp_path))
        analyzer = MDRAnalyzer(config=config)
        
        assert analyzer.config == config
        assert analyzer.results is None


# ============================================================================
# Test Excel Report Utils
# ============================================================================
class TestExcelReportUtils:
    """Test Excel report generation utilities."""

    def test_sanitize_sheet_name(self):
        """Test sheet name sanitization."""
        from strepsuis_mdr.excel_report_utils import sanitize_sheet_name
        
        # Test long name truncation
        long_name = "A" * 50
        result = sanitize_sheet_name(long_name)
        assert len(result) <= 31
        
        # Test special character removal
        result = sanitize_sheet_name("Test:Name*Here?")
        assert ":" not in result
        assert "*" not in result
        assert "?" not in result

    def test_excel_report_generator_creation(self, tmp_path):
        """Test ExcelReportGenerator initialization."""
        from strepsuis_mdr.excel_report_utils import ExcelReportGenerator
        
        generator = ExcelReportGenerator(str(tmp_path))
        
        assert generator is not None
        assert os.path.exists(generator.png_folder)


# ============================================================================
# Test synthetic data utils
# ============================================================================
class TestSyntheticDataUtils:
    """Test synthetic data generation utilities."""

    def test_generate_synthetic_amr_data(self):
        """Test synthetic AMR data generation."""
        from strepsuis_mdr.synthetic_data_utils import generate_synthetic_amr_data
        
        data, metadata = generate_synthetic_amr_data(n_strains=50, random_state=42)
        
        assert len(data) == 50
        assert 'Strain_ID' in data.columns
        assert 'true_clusters' in metadata
        assert len(metadata['true_clusters']) == 50


# ============================================================================
# Test validation utils
# ============================================================================
class TestValidationUtils:
    """Test validation utility functions."""

    def test_validate_input_data(self):
        """Test input data validation."""
        from strepsuis_mdr.validation_utils import validate_input_data
        
        df = pd.DataFrame({
            'Strain_ID': ['S1', 'S2', 'S3'],
            'Gene_A': [1, 0, 1],
            'Gene_B': [0, 1, 1],
        })
        
        # Should not raise any exception for valid data
        validate_input_data(df)

    def test_get_data_summary(self):
        """Test data summary generation."""
        from strepsuis_mdr.validation_utils import get_data_summary
        
        df = pd.DataFrame({
            'Strain_ID': ['S1', 'S2', 'S3', 'S4'],
            'Gene_A': [1, 0, 1, 0],
            'Gene_B': [0, 1, 1, 1],
        })
        
        summary = get_data_summary(df)
        
        assert summary is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
