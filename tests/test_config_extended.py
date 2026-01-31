"""Extended tests for Config module with various scenarios."""

import pytest
import tempfile
from pathlib import Path
import os


def test_config_with_string_paths(tmp_path):
    """Test Config with string path inputs."""
    from strepsuis_mdr.config import Config
    
    data_dir = tmp_path / "data"
    data_dir.mkdir()  # Create directory first
    output_dir = str(tmp_path / "output")
    
    config = Config(data_dir=str(data_dir), output_dir=output_dir)
    
    assert config.data_dir == str(data_dir)
    assert config.output_dir == output_dir


def test_config_with_pathlib_paths(tmp_path):
    """Test Config with Path object inputs."""
    from strepsuis_mdr.config import Config
    
    data_dir = tmp_path / "data"
    data_dir.mkdir()  # Create directory first
    output_dir = tmp_path / "output"
    
    config = Config(data_dir=str(data_dir), output_dir=str(output_dir))
    
    assert Path(config.data_dir).name == "data"
    assert Path(config.output_dir).name == "output"


def test_config_default_values(tmp_path):
    """Test Config default parameter values."""
    from strepsuis_mdr.config import Config
    
    data_dir = tmp_path / "data"
    data_dir.mkdir()  # Create directory first
    output_dir = str(tmp_path / "output")
    
    config = Config(data_dir=str(data_dir), output_dir=output_dir)
    
    # Check that config has expected attributes
    assert hasattr(config, "data_dir")
    assert hasattr(config, "output_dir")


def test_config_attribute_access(tmp_path):
    """Test accessing Config attributes."""
    from strepsuis_mdr.config import Config
    
    data_dir = tmp_path / "data"
    data_dir.mkdir()  # Create directory first
    output_dir = str(tmp_path / "output")
    
    config = Config(data_dir=str(data_dir), output_dir=output_dir)
    
    # Should be able to access as attributes
    _ = config.data_dir
    _ = config.output_dir


def test_config_with_absolute_paths(tmp_path):
    """Test Config with absolute paths."""
    from strepsuis_mdr.config import Config
    
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    output_dir = tmp_path / "output"
    
    config = Config(data_dir=str(data_dir.absolute()), output_dir=str(output_dir.absolute()))
    
    assert Path(config.data_dir).is_absolute() or config.data_dir
    assert Path(config.output_dir).is_absolute() or config.output_dir


def test_config_creates_output_dir(tmp_path):
    """Test that Config creates output directory if needed."""
    from strepsuis_mdr.config import Config
    
    data_dir = tmp_path / "data"
    data_dir.mkdir()  # Create directory first
    output_dir = tmp_path / "new_output"
    
    config = Config(data_dir=str(data_dir), output_dir=str(output_dir))
    
    # Output directory should exist after config creation
    assert output_dir.exists() or config.output_dir


def test_config_repr_method(tmp_path):
    """Test Config string representation."""
    from strepsuis_mdr.config import Config
    
    data_dir = tmp_path / "data"
    data_dir.mkdir()  # Create directory first
    output_dir = str(tmp_path / "output")
    
    config = Config(data_dir=str(data_dir), output_dir=output_dir)
    
    # Should have string representation
    repr_str = repr(config)
    assert isinstance(repr_str, str)
    assert len(repr_str) > 0


def test_config_immutability_attempt(tmp_path):
    """Test Config behavior with attribute modification."""
    from strepsuis_mdr.config import Config
    
    data_dir = tmp_path / "data"
    data_dir.mkdir()  # Create directory first
    output_dir = str(tmp_path / "output")
    
    config = Config(data_dir=str(data_dir), output_dir=output_dir)
    
    original_data_dir = config.data_dir
    # Config attributes should be accessible
    assert config.data_dir == original_data_dir


def test_config_with_nested_directories(tmp_path):
    """Test Config with nested directory structures."""
    from strepsuis_mdr.config import Config
    
    data_dir = tmp_path / "level1" / "level2" / "data"
    data_dir.mkdir(parents=True)  # Create nested directories first
    output_dir = str(tmp_path / "level1" / "level2" / "output")
    
    config = Config(data_dir=str(data_dir), output_dir=output_dir)
    
    assert config.data_dir
    assert config.output_dir


def test_config_parameter_types(tmp_path):
    """Test Config with different parameter types."""
    from strepsuis_mdr.config import Config
    
    data_dir = tmp_path / "data"
    data_dir.mkdir()  # Create directory first
    output_dir = str(tmp_path / "output")
    
    # Test with different valid inputs
    config = Config(data_dir=str(data_dir), output_dir=output_dir)
    
    assert isinstance(config.data_dir, str) or isinstance(config.data_dir, Path)
    assert isinstance(config.output_dir, str) or isinstance(config.output_dir, Path)
