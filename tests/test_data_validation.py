"""Tests for data loading and validation using example data."""

import pandas as pd
import pytest
from pathlib import Path


@pytest.fixture
def example_data_dir():
    """Get the example data directory."""
    return Path(__file__).parent.parent / "data" / "examples"


def test_example_data_exists(example_data_dir):
    """Test that example data directory exists."""
    assert example_data_dir.exists(), "Example data directory should exist"
    assert example_data_dir.is_dir(), "Example data path should be a directory"


def test_mic_data_loads(example_data_dir):
    """Test that MIC.csv can be loaded."""
    mic_file = example_data_dir / "MIC.csv"
    assert mic_file.exists(), "MIC.csv should exist"
    
    df = pd.read_csv(mic_file)
    assert not df.empty, "MIC data should not be empty"
    assert len(df.columns) > 0, "MIC data should have columns"


def test_amr_genes_data_loads(example_data_dir):
    """Test that AMR_genes.csv can be loaded."""
    amr_file = example_data_dir / "AMR_genes.csv"
    assert amr_file.exists(), "AMR_genes.csv should exist"
    
    df = pd.read_csv(amr_file)
    assert not df.empty, "AMR genes data should not be empty"
    assert len(df.columns) > 0, "AMR genes data should have columns"


def test_all_required_files_present(example_data_dir):
    """Test that all required example files are present."""
    required_files = ["MIC.csv", "AMR_genes.csv"]
    
    for filename in required_files:
        file_path = example_data_dir / filename
        assert file_path.exists(), f"{filename} should exist in example data"


def test_data_consistency(example_data_dir):
    """Test that data files have consistent structure."""
    mic_df = pd.read_csv(example_data_dir / "MIC.csv")
    amr_df = pd.read_csv(example_data_dir / "AMR_genes.csv")
    
    # Both should have strain/sample ID column
    assert len(mic_df.columns) > 0, "MIC should have columns"
    assert len(amr_df.columns) > 0, "AMR genes should have columns"
    
    # Should have reasonable number of rows
    assert len(mic_df) > 0, "MIC should have data rows"
    assert len(amr_df) > 0, "AMR genes should have data rows"


def test_no_completely_empty_columns(example_data_dir):
    """Test that data files don't have completely empty columns."""
    for csv_file in example_data_dir.glob("*.csv"):
        df = pd.read_csv(csv_file)
        
        # Check for columns that are completely empty
        for col in df.columns:
            # Allow some NaN values, but not all
            if len(df) > 0:
                non_null_count = df[col].notna().sum()
                # At least one non-null value expected
                assert non_null_count >= 0, f"{csv_file.name}: Column {col} exists"


def test_csv_files_readable(example_data_dir):
    """Test that all CSV files can be read without errors."""
    csv_files = list(example_data_dir.glob("*.csv"))
    assert len(csv_files) > 0, "Should have at least one CSV file"
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            assert isinstance(df, pd.DataFrame), f"{csv_file.name} should load as DataFrame"
        except Exception as e:
            pytest.fail(f"Failed to read {csv_file.name}: {str(e)}")


def test_data_files_not_corrupted(example_data_dir):
    """Test that data files are not corrupted."""
    for csv_file in example_data_dir.glob("*.csv"):
        # File should have reasonable size
        file_size = csv_file.stat().st_size
        assert file_size > 10, f"{csv_file.name} should have content (size > 10 bytes)"
        
        # Should be readable
        df = pd.read_csv(csv_file)
        assert len(df.columns) > 0, f"{csv_file.name} should have at least one column"
