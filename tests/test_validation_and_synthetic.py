"""
Tests for validation_utils and synthetic_data_utils modules.
"""

import numpy as np
import pandas as pd
import pytest


class TestValidationUtils:
    """Test validation utility functions."""

    def test_validate_input_data_valid(self):
        """Test validation passes for valid data."""
        from strepsuis_mdr.validation_utils import validate_input_data
        
        df = pd.DataFrame({
            'Strain_ID': ['A', 'B', 'C'],
            'Gene1': [1, 0, 1],
            'Gene2': [0, 1, 0],
        })
        
        is_valid, msg = validate_input_data(df)
        assert is_valid, f"Expected valid, got: {msg}"
        assert "passed" in msg.lower()

    def test_validate_input_data_empty(self):
        """Test validation fails for empty DataFrame."""
        from strepsuis_mdr.validation_utils import validate_input_data
        
        df = pd.DataFrame()
        is_valid, msg = validate_input_data(df)
        assert not is_valid
        assert "empty" in msg.lower()

    def test_validate_input_data_missing_id_col(self):
        """Test validation fails when ID column is missing."""
        from strepsuis_mdr.validation_utils import validate_input_data
        
        df = pd.DataFrame({
            'NotStrainID': ['A', 'B'],
            'Gene1': [1, 0],
        })
        
        is_valid, msg = validate_input_data(df, id_col='Strain_ID')
        assert not is_valid
        assert "Strain_ID" in msg

    def test_validate_input_data_duplicate_ids(self):
        """Test validation fails for duplicate IDs."""
        from strepsuis_mdr.validation_utils import validate_input_data
        
        df = pd.DataFrame({
            'Strain_ID': ['A', 'A', 'B'],  # Duplicate 'A'
            'Gene1': [1, 0, 1],
        })
        
        is_valid, msg = validate_input_data(df)
        assert not is_valid
        assert "duplicate" in msg.lower()

    def test_validate_input_data_high_missing(self):
        """Test validation warns for high missing values."""
        from strepsuis_mdr.validation_utils import validate_input_data
        
        df = pd.DataFrame({
            'Strain_ID': ['A', 'B', 'C', 'D', 'E'],
            'Gene1': [1, 0, np.nan, np.nan, np.nan],  # 60% missing
        })
        
        is_valid, msg = validate_input_data(df, max_missing_pct=0.5)
        assert is_valid  # Still valid, but with warning
        assert "missing" in msg.lower()

    def test_validate_input_data_non_binary(self):
        """Test validation warns for non-binary values."""
        from strepsuis_mdr.validation_utils import validate_input_data
        
        df = pd.DataFrame({
            'Strain_ID': ['A', 'B', 'C'],
            'Gene1': [0, 1, 2],  # Non-binary
        })
        
        is_valid, msg = validate_input_data(df, require_binary=True)
        assert is_valid  # Warning only, not error
        assert "non-binary" in msg.lower() or "warning" in msg.lower()

    def test_check_binary_encoding(self):
        """Test binary encoding check."""
        from strepsuis_mdr.validation_utils import check_binary_encoding
        
        # Binary data
        df_binary = pd.DataFrame({
            'col1': [0, 1, 0, 1],
            'col2': [1, 1, 0, 0],
        })
        all_binary, non_binary = check_binary_encoding(df_binary)
        assert all_binary
        assert len(non_binary) == 0
        
        # Non-binary data
        df_non_binary = pd.DataFrame({
            'col1': [0, 1, 0, 1],
            'col2': [0, 1, 2, 3],
        })
        all_binary, non_binary = check_binary_encoding(df_non_binary)
        assert not all_binary
        assert 'col2' in non_binary

    def test_get_data_summary(self):
        """Test data summary generation."""
        from strepsuis_mdr.validation_utils import get_data_summary
        
        df = pd.DataFrame({
            'Strain_ID': ['A', 'B', 'C', 'D'],
            'Gene1': [1, 0, 1, 0],
            'Gene2': [0, 1, 0, 1],
        })
        
        summary = get_data_summary(df)
        
        assert summary['n_samples'] == 4
        assert summary['n_features'] == 2
        assert summary['id_column'] == 'Strain_ID'
        assert summary['missing_total'] == 0
        assert summary['all_binary'] == True

    def test_validate_mdr_output_valid(self):
        """Test MDR output validation for valid output."""
        from strepsuis_mdr.validation_utils import validate_mdr_output
        
        overview = pd.DataFrame({
            'Metric': ['Total Isolates', 'MDR Isolates'],
            'Value': [100, 50],
        })
        
        is_valid, msg = validate_mdr_output(overview)
        assert is_valid

    def test_validate_mdr_output_empty(self):
        """Test MDR output validation for empty output."""
        from strepsuis_mdr.validation_utils import validate_mdr_output
        
        overview = pd.DataFrame()
        is_valid, msg = validate_mdr_output(overview)
        assert not is_valid
        assert "empty" in msg.lower()


class TestSyntheticDataUtils:
    """Test synthetic data generation utilities."""

    def test_generate_synthetic_amr_data_basic(self):
        """Test basic synthetic data generation."""
        from strepsuis_mdr.synthetic_data_utils import generate_synthetic_amr_data
        
        data, metadata = generate_synthetic_amr_data(n_strains=50, random_state=42)
        
        assert len(data) == 50
        assert 'Strain_ID' in data.columns
        assert len(metadata['antibiotic_cols']) > 0
        assert len(metadata['gene_cols']) > 0

    def test_generate_synthetic_amr_data_reproducibility(self):
        """Test that synthetic data is reproducible with same seed."""
        from strepsuis_mdr.synthetic_data_utils import generate_synthetic_amr_data
        
        data1, _ = generate_synthetic_amr_data(n_strains=30, random_state=42)
        data2, _ = generate_synthetic_amr_data(n_strains=30, random_state=42)
        
        pd.testing.assert_frame_equal(data1, data2)

    def test_generate_synthetic_amr_data_different_seeds(self):
        """Test that different seeds produce different data."""
        from strepsuis_mdr.synthetic_data_utils import generate_synthetic_amr_data
        
        data1, _ = generate_synthetic_amr_data(n_strains=30, random_state=42)
        data2, _ = generate_synthetic_amr_data(n_strains=30, random_state=123)
        
        # Data should differ (at least in some values)
        # We can't use assert_frame_not_equal, so check some values differ
        feature_cols = [c for c in data1.columns if c != 'Strain_ID']
        values1 = data1[feature_cols].values.flatten()
        values2 = data2[feature_cols].values.flatten()
        
        # At least some values should differ
        assert not np.array_equal(values1, values2)

    def test_generate_synthetic_amr_data_binary(self):
        """Test that generated data is binary."""
        from strepsuis_mdr.synthetic_data_utils import generate_synthetic_amr_data
        
        data, metadata = generate_synthetic_amr_data(n_strains=50, random_state=42)
        
        feature_cols = metadata['antibiotic_cols'] + metadata['gene_cols']
        for col in feature_cols:
            unique_vals = data[col].unique()
            assert set(unique_vals).issubset({0, 1}), f"Column {col} has non-binary values"

    def test_generate_synthetic_amr_data_clusters(self):
        """Test that cluster assignments are generated."""
        from strepsuis_mdr.synthetic_data_utils import generate_synthetic_amr_data
        
        data, metadata = generate_synthetic_amr_data(
            n_strains=100, n_clusters=3, random_state=42
        )
        
        assert 'true_clusters' in metadata
        assert len(metadata['true_clusters']) == 100
        assert len(np.unique(metadata['true_clusters'])) == 3

    def test_generate_cooccurrence_data(self):
        """Test co-occurrence data generation."""
        from strepsuis_mdr.synthetic_data_utils import generate_cooccurrence_data
        
        data, associations = generate_cooccurrence_data(
            n_samples=100, n_features=6, random_state=42
        )
        
        assert len(data) == 100
        assert len(data.columns) == 6
        assert len(associations) > 0

    def test_run_synthetic_smoke_test(self):
        """Test the smoke test runner."""
        from strepsuis_mdr.synthetic_data_utils import run_synthetic_smoke_test
        
        results = run_synthetic_smoke_test(n_strains=20, random_state=42, verbose=False)
        
        assert results['data_generated'] == True
        assert results['validation_passed'] == True
        assert results['analysis_ran'] == True
        assert results['success'] == True
        assert len(results['errors']) == 0


class TestAnalysisConfig:
    """Test the Pydantic AnalysisConfig model."""

    def test_analysis_config_defaults(self):
        """Test AnalysisConfig with default values."""
        from strepsuis_mdr.config import AnalysisConfig
        
        config = AnalysisConfig()
        
        assert config.n_bootstrap == 5000
        assert config.confidence_level == 0.95
        assert config.fdr_alpha == 0.05

    def test_analysis_config_custom_values(self):
        """Test AnalysisConfig with custom values."""
        from strepsuis_mdr.config import AnalysisConfig
        
        config = AnalysisConfig(
            n_bootstrap=10000,
            confidence_level=0.99,
            fdr_alpha=0.01,
        )
        
        assert config.n_bootstrap == 10000
        assert config.confidence_level == 0.99
        assert config.fdr_alpha == 0.01

    def test_analysis_config_validation_n_bootstrap(self):
        """Test that n_bootstrap validation works."""
        from strepsuis_mdr.config import AnalysisConfig
        import pydantic
        
        # Too low
        with pytest.raises(pydantic.ValidationError):
            AnalysisConfig(n_bootstrap=100)  # Below 1000 minimum
        
        # Too high
        with pytest.raises(pydantic.ValidationError):
            AnalysisConfig(n_bootstrap=200000)  # Above 100000 maximum

    def test_analysis_config_validation_confidence_level(self):
        """Test confidence level validation."""
        from strepsuis_mdr.config import AnalysisConfig
        import pydantic
        
        # Out of range
        with pytest.raises(pydantic.ValidationError):
            AnalysisConfig(confidence_level=0)
        
        with pytest.raises(pydantic.ValidationError):
            AnalysisConfig(confidence_level=1.0)

    def test_analysis_config_to_legacy_config(self):
        """Test conversion to legacy Config."""
        from strepsuis_mdr.config import AnalysisConfig, Config
        
        analysis_config = AnalysisConfig(n_bootstrap=5000)
        legacy_config = analysis_config.to_config()
        
        assert isinstance(legacy_config, Config)
        assert legacy_config.bootstrap_iterations == 5000


class TestIntegrationWithExistingCode:
    """Test integration of new utilities with existing analysis code."""

    def test_synthetic_data_with_mdr_analysis(self):
        """Test that synthetic data works with MDR analysis functions."""
        from strepsuis_mdr.synthetic_data_utils import generate_synthetic_amr_data
        from strepsuis_mdr.mdr_analysis_core import (
            build_class_resistance,
            identify_mdr_isolates,
            compute_bootstrap_ci,
        )
        
        # Generate data
        data, metadata = generate_synthetic_amr_data(n_strains=50, random_state=42)
        
        # Build class resistance
        pheno_cols = metadata['antibiotic_cols']
        class_res = build_class_resistance(data, pheno_cols)
        
        assert len(class_res) == 50
        
        # Identify MDR
        mdr_mask = identify_mdr_isolates(class_res, threshold=3)
        assert mdr_mask.dtype == bool
        
        # Compute bootstrap CI
        ci_result = compute_bootstrap_ci(class_res, n_iter=100, confidence_level=0.95)
        assert not ci_result.empty

    def test_validation_before_analysis(self):
        """Test validation workflow before running analysis."""
        from strepsuis_mdr.synthetic_data_utils import generate_synthetic_amr_data
        from strepsuis_mdr.validation_utils import validate_input_data, get_data_summary
        from strepsuis_mdr.mdr_analysis_core import pairwise_cooccurrence
        
        # Generate data
        data, metadata = generate_synthetic_amr_data(n_strains=50, random_state=42)
        
        # Validate
        is_valid, msg = validate_input_data(data)
        assert is_valid, f"Validation failed: {msg}"
        
        # Get summary
        summary = get_data_summary(data)
        assert summary['n_samples'] == 50
        
        # Run analysis on valid data
        feature_cols = metadata['antibiotic_cols']
        result = pairwise_cooccurrence(data[feature_cols], alpha=0.5)
        # Result may be empty if no significant co-occurrences, but should not error
        assert isinstance(result, pd.DataFrame)
