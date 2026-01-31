"""
Tests for generate_synthetic_data module.

Tests synthetic data generation for MDR analysis.
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os
from pathlib import Path

from strepsuis_mdr.generate_synthetic_data import (
    SyntheticDataConfig,
    SyntheticDataMetadata,
    generate_prevalence_rates,
    generate_binary_data_binomial,
    add_gaussian_noise_to_binary,
    generate_correlated_features,
    generate_mdr_synthetic_dataset,
    save_synthetic_data,
    validate_synthetic_data,
)


class TestSyntheticDataConfig:
    """Tests for SyntheticDataConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = SyntheticDataConfig()
        
        assert config.n_strains == 200
        assert config.n_antibiotics == 13
        assert config.n_genes == 30
        assert config.n_virulence == 20
        assert 0 <= config.base_prevalence_mean <= 1
        assert config.random_state == 42
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SyntheticDataConfig(
            n_strains=50,
            n_antibiotics=10,
            n_genes=20,
            n_virulence=15,
            base_prevalence_mean=0.5,
            random_state=123
        )
        
        assert config.n_strains == 50
        assert config.n_antibiotics == 10
        assert config.n_genes == 20
        assert config.n_virulence == 15
        assert config.base_prevalence_mean == 0.5
        assert config.random_state == 123
    
    def test_config_mdr_proportion(self):
        """Test MDR proportion configuration."""
        config = SyntheticDataConfig(mdr_proportion=0.6)
        
        assert config.mdr_proportion == 0.6
    
    def test_config_noise_level(self):
        """Test noise level configuration."""
        config = SyntheticDataConfig(noise_level=0.1)
        
        assert config.noise_level == 0.1
    
    def test_config_correlation_strength(self):
        """Test correlation strength configuration."""
        config = SyntheticDataConfig(correlation_strength=0.8)
        
        assert config.correlation_strength == 0.8


class TestSyntheticDataMetadata:
    """Tests for SyntheticDataMetadata class."""
    
    def test_metadata_creation(self):
        """Test metadata creation."""
        config = SyntheticDataConfig()
        metadata = SyntheticDataMetadata(config=config)
        
        assert metadata.config == config
        assert isinstance(metadata.true_prevalences, dict)
        assert isinstance(metadata.true_correlations, list)
        assert metadata.generation_timestamp is not None
    
    def test_metadata_columns(self):
        """Test metadata column lists."""
        config = SyntheticDataConfig()
        metadata = SyntheticDataMetadata(
            config=config,
            antibiotic_columns=['AB1', 'AB2'],
            gene_columns=['Gene1', 'Gene2'],
            virulence_columns=['Vir1']
        )
        
        assert len(metadata.antibiotic_columns) == 2
        assert len(metadata.gene_columns) == 2
        assert len(metadata.virulence_columns) == 1


class TestGeneratePrevalenceRates:
    """Tests for prevalence rate generation."""
    
    def test_basic_prevalence(self):
        """Test basic prevalence generation."""
        rates = generate_prevalence_rates(10, mean=0.35, std=0.2, random_state=42)
        
        assert len(rates) == 10
        assert all(0.01 <= r <= 0.99 for r in rates)
    
    def test_prevalence_bounds(self):
        """Test prevalence bounds."""
        rates = generate_prevalence_rates(5, mean=0.5, std=0.1, random_state=42)
        
        assert all(0.01 <= r <= 0.99 for r in rates)
    
    def test_prevalence_reproducibility(self):
        """Test reproducibility with seed."""
        rates1 = generate_prevalence_rates(10, random_state=42)
        rates2 = generate_prevalence_rates(10, random_state=42)
        
        np.testing.assert_array_equal(rates1, rates2)
    
    def test_prevalence_mean_approximate(self):
        """Test that mean is approximately correct."""
        rates = generate_prevalence_rates(1000, mean=0.5, std=0.1, random_state=42)
        
        assert abs(np.mean(rates) - 0.5) < 0.1
    
    def test_prevalence_low_variance(self):
        """Test with very low variance."""
        rates = generate_prevalence_rates(10, mean=0.5, std=0.01, random_state=42)
        
        assert all(0.01 <= r <= 0.99 for r in rates)


class TestGenerateBinaryData:
    """Tests for binary data generation."""
    
    def test_basic_binary_generation(self):
        """Test basic binary data generation."""
        data = generate_binary_data_binomial(100, 0.3, random_state=42)
        
        assert len(data) == 100
        assert set(np.unique(data)).issubset({0, 1})
    
    def test_binary_prevalence_approximate(self):
        """Test that generated prevalences are approximately correct."""
        data = generate_binary_data_binomial(10000, 0.3, random_state=42)
        
        actual_prevalence = np.mean(data)
        assert abs(actual_prevalence - 0.3) < 0.05
    
    def test_binary_reproducibility(self):
        """Test reproducibility."""
        data1 = generate_binary_data_binomial(100, 0.5, random_state=42)
        data2 = generate_binary_data_binomial(100, 0.5, random_state=42)
        
        np.testing.assert_array_equal(data1, data2)
    
    def test_binary_extreme_prevalence(self):
        """Test extreme prevalence values."""
        data_low = generate_binary_data_binomial(100, 0.01, random_state=42)
        data_high = generate_binary_data_binomial(100, 0.99, random_state=42)
        
        assert np.mean(data_low) < 0.2
        assert np.mean(data_high) > 0.8


class TestAddGaussianNoise:
    """Tests for Gaussian noise addition."""
    
    def test_noise_addition(self):
        """Test noise addition to binary data."""
        data = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        
        noisy = add_gaussian_noise_to_binary(data, noise_level=0.1, random_state=42)
        
        assert len(noisy) == len(data)
        assert set(np.unique(noisy)).issubset({0, 1})
    
    def test_no_noise(self):
        """Test with zero noise level."""
        data = np.array([1, 0, 1, 0])
        
        noisy = add_gaussian_noise_to_binary(data, noise_level=0.0, random_state=42)
        
        np.testing.assert_array_equal(noisy, data)
    
    def test_high_noise(self):
        """Test with high noise level."""
        data = np.zeros(1000)
        
        noisy = add_gaussian_noise_to_binary(data, noise_level=0.5, random_state=42)
        
        # About half should be flipped
        assert 0.3 < np.mean(noisy) < 0.7
    
    def test_noise_reproducibility(self):
        """Test noise reproducibility."""
        data = np.array([1, 0, 1, 0, 1])
        
        noisy1 = add_gaussian_noise_to_binary(data, noise_level=0.2, random_state=42)
        noisy2 = add_gaussian_noise_to_binary(data, noise_level=0.2, random_state=42)
        
        np.testing.assert_array_equal(noisy1, noisy2)


class TestGenerateCorrelatedFeatures:
    """Tests for correlated feature generation."""
    
    def test_positive_correlation(self):
        """Test positive correlation generation."""
        base = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0] * 100)
        
        correlated = generate_correlated_features(base, correlation=0.8, random_state=42)
        
        assert len(correlated) == len(base)
        assert set(np.unique(correlated)).issubset({0, 1})
        
        # Check correlation is positive
        corr = np.corrcoef(base, correlated)[0, 1]
        assert corr > 0
    
    def test_negative_correlation(self):
        """Test negative correlation generation."""
        base = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0] * 100)
        
        correlated = generate_correlated_features(base, correlation=-0.8, random_state=42)
        
        # Check correlation is negative
        corr = np.corrcoef(base, correlated)[0, 1]
        assert corr < 0
    
    def test_zero_correlation(self):
        """Test near-zero correlation."""
        base = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0] * 100)
        
        correlated = generate_correlated_features(base, correlation=0.0, random_state=42)
        
        # Correlation should be close to 0
        corr = np.corrcoef(base, correlated)[0, 1]
        assert abs(corr) < 0.3


class TestGenerateMDRDataset:
    """Tests for full MDR dataset generation."""
    
    def test_basic_dataset_generation(self):
        """Test basic dataset generation."""
        config = SyntheticDataConfig(
            n_strains=50,
            n_antibiotics=5,
            n_genes=10,
            n_virulence=5,
            random_state=42
        )
        
        data, metadata = generate_mdr_synthetic_dataset(config)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 50
        assert 'Strain_ID' in data.columns
    
    def test_dataset_has_all_columns(self):
        """Test that dataset has all expected columns."""
        config = SyntheticDataConfig(
            n_strains=30,
            n_antibiotics=3,
            n_genes=5,
            n_virulence=2,
            random_state=42
        )
        
        data, metadata = generate_mdr_synthetic_dataset(config)
        
        # Should have antibiotic columns
        assert len(metadata.antibiotic_columns) == 3
        # Should have gene columns
        assert len(metadata.gene_columns) == 5
        # Should have virulence columns
        assert len(metadata.virulence_columns) == 2
    
    def test_dataset_binary_values(self):
        """Test that all feature columns are binary."""
        config = SyntheticDataConfig(
            n_strains=50,
            n_antibiotics=5,
            n_genes=10,
            n_virulence=5,
            random_state=42
        )
        
        data, metadata = generate_mdr_synthetic_dataset(config)
        
        # Check all feature columns are binary
        feature_cols = metadata.antibiotic_columns + metadata.gene_columns + metadata.virulence_columns
        for col in feature_cols:
            if col in data.columns:
                assert set(data[col].unique()).issubset({0, 1})
    
    def test_default_config_generation(self):
        """Test generation with default config."""
        data, metadata = generate_mdr_synthetic_dataset(None)
        
        assert len(data) == 200  # Default n_strains
        assert metadata.config.n_strains == 200
    
    def test_metadata_prevalences(self):
        """Test that metadata contains prevalences."""
        config = SyntheticDataConfig(n_strains=100, random_state=42)
        
        data, metadata = generate_mdr_synthetic_dataset(config)
        
        assert len(metadata.true_prevalences) > 0


class TestSaveSyntheticData:
    """Tests for saving synthetic data."""
    
    def test_save_to_csv(self):
        """Test saving data to CSV."""
        config = SyntheticDataConfig(n_strains=20, n_antibiotics=3, n_genes=5, n_virulence=2, random_state=42)
        data, metadata = generate_mdr_synthetic_dataset(config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = save_synthetic_data(data, metadata, output_dir=tmpdir)
            
            assert "data" in result
            assert os.path.exists(result["data"])
            
            # Verify data can be read back
            loaded = pd.read_csv(result["data"])
            assert len(loaded) == len(data)
    
    def test_save_with_metadata(self):
        """Test saving data with metadata."""
        config = SyntheticDataConfig(n_strains=20, random_state=42)
        data, metadata = generate_mdr_synthetic_dataset(config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = save_synthetic_data(data, metadata, output_dir=tmpdir)
            
            assert "metadata" in result
            assert os.path.exists(result["metadata"])
    
    def test_save_creates_directory(self):
        """Test that save creates directory if needed."""
        config = SyntheticDataConfig(n_strains=10, random_state=42)
        data, metadata = generate_mdr_synthetic_dataset(config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "subdir")
            result = save_synthetic_data(data, metadata, output_dir=output_path)
            
            assert os.path.exists(output_path)


class TestValidateSyntheticData:
    """Tests for synthetic data validation."""
    
    def test_valid_data(self):
        """Test validation of valid data."""
        config = SyntheticDataConfig(n_strains=50, n_antibiotics=5, n_genes=10, random_state=42)
        data, metadata = generate_mdr_synthetic_dataset(config)
        
        report = validate_synthetic_data(data, metadata)
        
        assert isinstance(report, dict)
        assert len(report) > 0
    
    def test_validation_returns_dict(self):
        """Test that validation returns a dictionary."""
        config = SyntheticDataConfig(n_strains=30, random_state=42)
        data, metadata = generate_mdr_synthetic_dataset(config)
        
        report = validate_synthetic_data(data, metadata)
        
        assert isinstance(report, dict)


class TestIntegration:
    """Integration tests for synthetic data generation."""
    
    def test_full_pipeline(self):
        """Test full synthetic data pipeline."""
        # Create config
        config = SyntheticDataConfig(
            n_strains=100,
            n_antibiotics=8,
            n_genes=15,
            n_virulence=10,
            mdr_proportion=0.6,
            random_state=42
        )
        
        # Generate data
        data, metadata = generate_mdr_synthetic_dataset(config)
        
        # Validate
        report = validate_synthetic_data(data, metadata)
        
        assert isinstance(report, dict)
        assert len(data) == 100
        
        # Save and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            result = save_synthetic_data(data, metadata, output_dir=tmpdir)
            
            loaded = pd.read_csv(result["data"])
            assert len(loaded) == len(data)
    
    def test_reproducibility(self):
        """Test that generation is reproducible."""
        config = SyntheticDataConfig(n_strains=50, random_state=42)
        
        data1, _ = generate_mdr_synthetic_dataset(config)
        data2, _ = generate_mdr_synthetic_dataset(config)
        
        pd.testing.assert_frame_equal(data1, data2)
    
    def test_different_seeds_different_data(self):
        """Test that different seeds produce different data."""
        config1 = SyntheticDataConfig(n_strains=50, random_state=42)
        config2 = SyntheticDataConfig(n_strains=50, random_state=123)
        
        data1, _ = generate_mdr_synthetic_dataset(config1)
        data2, _ = generate_mdr_synthetic_dataset(config2)
        
        # Data should be different
        assert not data1.equals(data2)
