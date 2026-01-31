"""
Pytest configuration and fixtures.
"""

import shutil
from pathlib import Path

import pytest

# Configure matplotlib to use non-interactive backend for testing
import matplotlib
matplotlib.use('Agg')


@pytest.fixture
def data_dir(tmp_path):
    """Create a temporary data directory with example files."""
    data = tmp_path / "data"
    data.mkdir()

    # Copy example data files from main repository data directory
    # Check both local data/examples (for backward compatibility) and main repo data
    example_dir = Path(__file__).parent.parent / "data" / "examples"
    main_data_dir = Path(__file__).parent.parent.parent.parent / "data"
    
    source_dir = example_dir if example_dir.exists() else main_data_dir
    
    if source_dir.exists():
        for csv_file in source_dir.glob("*.csv"):
            shutil.copy(csv_file, data)
        for newick_file in source_dir.glob("*.newick"):
            shutil.copy(newick_file, data)

    return data


@pytest.fixture
def output_dir(tmp_path):
    """Create a temporary output directory."""
    output = tmp_path / "output"
    output.mkdir()
    return output


@pytest.fixture
def sample_csv_data():
    """Provide sample CSV data."""
    return """Strain_ID,Feature1,Feature2,Feature3
Strain001,1,0,1
Strain002,0,1,1
Strain003,1,1,0
Strain004,0,0,1
Strain005,1,1,1"""
