"""
Tests for config module.

Tests configuration management for MDR analysis.
"""

import os
import tempfile
import pytest

from strepsuis_mdr.config import Config, AnalysisConfig, PYDANTIC_AVAILABLE


class TestConfigBasic:
    """Tests for basic Config functionality."""
    
    def test_default_config(self):
        """Test default configuration values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(data_dir=tmpdir)
            
            assert config.bootstrap_iterations > 0
            assert config.fdr_alpha > 0
            assert config.fdr_alpha < 1
            assert config.random_seed >= 0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(
                data_dir=tmpdir,
                bootstrap_iterations=1000,
                fdr_alpha=0.01,
                random_seed=123
            )
            
            assert config.bootstrap_iterations == 1000
            assert config.fdr_alpha == 0.01
            assert config.random_seed == 123
    
    def test_config_data_dir(self):
        """Test data directory configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(data_dir=tmpdir)
            
            assert config.data_dir == tmpdir
    
    def test_config_output_dir(self):
        """Test output directory configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output")
            config = Config(data_dir=tmpdir, output_dir=output_path)
            
            assert config.output_dir == output_path
            assert os.path.exists(output_path)


class TestConfigValidation:
    """Tests for Config validation."""
    
    def test_invalid_data_dir(self):
        """Test invalid data directory."""
        with pytest.raises(ValueError):
            Config(data_dir="/nonexistent/path/12345")
    
    def test_invalid_fdr_alpha(self):
        """Test invalid FDR alpha value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError):
                Config(data_dir=tmpdir, fdr_alpha=1.5)
    
    def test_invalid_bootstrap(self):
        """Test invalid bootstrap iterations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError):
                Config(data_dir=tmpdir, bootstrap_iterations=50)


class TestConfigFromDict:
    """Tests for Config.from_dict method."""
    
    def test_from_dict_basic(self):
        """Test creating config from dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = {
                'data_dir': tmpdir,
                'bootstrap_iterations': 1000,
                'fdr_alpha': 0.01
            }
            
            config = Config.from_dict(d)
            
            assert config.data_dir == tmpdir
            assert config.bootstrap_iterations == 1000
            assert config.fdr_alpha == 0.01
    
    def test_from_dict_ignores_unknown(self):
        """Test that from_dict ignores unknown keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = {
                'data_dir': tmpdir,
                'unknown_key': 'value'
            }
            
            config = Config.from_dict(d)
            
            assert config.data_dir == tmpdir
            assert not hasattr(config, 'unknown_key')


class TestAnalysisConfig:
    """Tests for AnalysisConfig class."""
    
    def test_default_analysis_config(self):
        """Test default AnalysisConfig values."""
        config = AnalysisConfig()
        
        assert config.n_bootstrap == 5000
        assert config.confidence_level == 0.95
        assert config.fdr_alpha == 0.05
    
    def test_custom_analysis_config(self):
        """Test custom AnalysisConfig values."""
        config = AnalysisConfig(
            n_bootstrap=2000,
            confidence_level=0.99,
            fdr_alpha=0.01
        )
        
        assert config.n_bootstrap == 2000
        assert config.confidence_level == 0.99
        assert config.fdr_alpha == 0.01
    
    def test_analysis_config_validation(self):
        """Test AnalysisConfig validation."""
        with pytest.raises(ValueError):
            AnalysisConfig(n_bootstrap=100)  # Too low
        
        with pytest.raises(ValueError):
            AnalysisConfig(confidence_level=1.5)  # Out of range
    
    def test_to_config(self):
        """Test conversion to legacy Config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analysis_config = AnalysisConfig(output_dir=tmpdir)
            
            legacy_config = analysis_config.to_config()
            
            assert isinstance(legacy_config, Config)
            assert legacy_config.output_dir == tmpdir


class TestConfigRepr:
    """Tests for Config string representation."""
    
    def test_str_representation(self):
        """Test string representation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(data_dir=tmpdir)
            
            str_repr = str(config)
            assert isinstance(str_repr, str)
    
    def test_repr_representation(self):
        """Test repr representation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(data_dir=tmpdir)
            
            repr_str = repr(config)
            assert isinstance(repr_str, str)
            assert 'Config' in repr_str


class TestConfigReportingOptions:
    """Tests for Config reporting options."""
    
    def test_html_generation_option(self):
        """Test HTML generation option."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(data_dir=tmpdir, generate_html=False)
            
            assert config.generate_html is False
    
    def test_excel_generation_option(self):
        """Test Excel generation option."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(data_dir=tmpdir, generate_excel=False)
            
            assert config.generate_excel is False
    
    def test_png_charts_option(self):
        """Test PNG charts option."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(data_dir=tmpdir, save_png_charts=False)
            
            assert config.save_png_charts is False
    
    def test_dpi_option(self):
        """Test DPI option."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(data_dir=tmpdir, dpi=300)
            
            assert config.dpi == 300


class TestConfigParallelProcessing:
    """Tests for Config parallel processing options."""
    
    def test_n_jobs_default(self):
        """Test default n_jobs value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(data_dir=tmpdir)
            
            assert config.n_jobs == -1  # Use all cores
    
    def test_n_jobs_custom(self):
        """Test custom n_jobs value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(data_dir=tmpdir, n_jobs=4)
            
            assert config.n_jobs == 4
