"""
Tests for strepsuis_mdr

Basic test suite to ensure the package is functional.
"""

import pytest

from strepsuis_mdr import __version__


def test_version():
    """Test that version is defined."""
    assert __version__ == "1.0.0"


def test_imports():
    """Test that main classes can be imported."""
    from strepsuis_mdr import Config, MDRAnalyzer

    assert MDRAnalyzer is not None
    assert Config is not None


def test_config_initialization():
    """Test Config class initialization."""
    from strepsuis_mdr import Config

    config = Config(data_dir="./data", output_dir="./output")

    assert config.data_dir == "./data"
    assert config.output_dir == "./output"


def test_analyzer_initialization():
    """Test MDRAnalyzer initialization."""
    from strepsuis_mdr import MDRAnalyzer

    analyzer = MDRAnalyzer(data_dir="./data", output_dir="./output")

    assert analyzer.data_dir == "./data"
    assert analyzer.output_dir == "./output"


@pytest.mark.integration
def test_example_data_exists():
    """Test that example data files exist."""
    from pathlib import Path

    examples_dir = Path("examples")
    if examples_dir.exists():
        csv_files = list(examples_dir.glob("*.csv"))
        assert len(csv_files) > 0, "No CSV files found in examples directory"
