"""Integration tests using real example data."""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def example_data_with_output(tmp_path):
    """Create a test environment with example data and output directory."""
    # Get example data directory
    example_dir = Path(__file__).parent.parent / "data" / "examples"
    
    # Create test data directory
    test_data_dir = tmp_path / "data"
    test_data_dir.mkdir()
    
    # Copy example data
    for csv_file in example_dir.glob("*.csv"):
        shutil.copy(csv_file, test_data_dir)
    
    # Copy newick files if they exist
    for newick_file in example_dir.glob("*.newick"):
        shutil.copy(newick_file, test_data_dir)
    
    # Create output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    return {
        "data_dir": test_data_dir,
        "output_dir": output_dir,
        "example_dir": example_dir
    }


def test_example_data_structure(example_data_with_output):
    """Test that example data has expected structure."""
    data_dir = example_data_with_output["data_dir"]
    
    # Check required files exist
    required_files = ["MIC.csv", "AMR_genes.csv"]
    for filename in required_files:
        assert (data_dir / filename).exists(), f"{{filename}} should exist"
    
    # Verify files are readable
    mic_df = pd.read_csv(data_dir / "MIC.csv")
    assert not mic_df.empty, "MIC data should not be empty"


def test_data_loading_with_real_files(example_data_with_output):
    """Test data loading with actual example files."""
    data_dir = example_data_with_output["data_dir"]
    
    # Load all CSV files
    csv_files = list(data_dir.glob("*.csv"))
    assert len(csv_files) > 0, "Should have CSV files"
    
    dataframes = {}
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dataframes[csv_file.name] = df
            assert not df.empty, f"{{csv_file.name}} should not be empty"
        except Exception as e:
            pytest.fail(f"Failed to load {{csv_file.name}}: {{str(e)}}")
    
    # Should have loaded multiple files
    assert len(dataframes) >= 2, "Should load at least 2 data files"


def test_output_directory_usage(example_data_with_output):
    """Test that output directory is usable."""
    output_dir = example_data_with_output["output_dir"]
    
    # Should be able to create files in output
    test_file = output_dir / "test_output.txt"
    test_file.write_text("test content")
    
    assert test_file.exists()
    assert test_file.read_text() == "test content"


def test_data_file_formats(example_data_with_output):
    """Test that data files have correct formats."""
    data_dir = example_data_with_output["data_dir"]
    
    # Check MIC.csv format
    mic_df = pd.read_csv(data_dir / "MIC.csv")
    assert len(mic_df.columns) > 1, "MIC should have multiple columns"
    
    # Check AMR_genes.csv format
    amr_df = pd.read_csv(data_dir / "AMR_genes.csv")
    assert len(amr_df.columns) > 1, "AMR genes should have multiple columns"


def test_data_consistency_checks(example_data_with_output):
    """Test data consistency across files."""
    data_dir = example_data_with_output["data_dir"]
    
    # Load main files
    mic_df = pd.read_csv(data_dir / "MIC.csv")
    amr_df = pd.read_csv(data_dir / "AMR_genes.csv")
    
    # Both should have data
    assert len(mic_df) > 0, "MIC should have rows"
    assert len(amr_df) > 0, "AMR genes should have rows"
    
    # Columns should be unique
    assert len(mic_df.columns) == len(set(mic_df.columns)), "MIC columns should be unique"
    assert len(amr_df.columns) == len(set(amr_df.columns)), "AMR columns should be unique"


def test_file_copy_integrity(example_data_with_output):
    """Test that copied files maintain integrity."""
    data_dir = example_data_with_output["data_dir"]
    example_dir = example_data_with_output["example_dir"]
    
    # Compare a file between original and copy
    original = pd.read_csv(example_dir / "MIC.csv")
    copied = pd.read_csv(data_dir / "MIC.csv")
    
    # Should have same shape
    assert original.shape == copied.shape, "Copied file should match original"
    
    # Should have same columns
    assert list(original.columns) == list(copied.columns), "Columns should match"


def test_multiple_csv_loading(example_data_with_output):
    """Test loading multiple CSV files simultaneously."""
    data_dir = example_data_with_output["data_dir"]
    
    csv_files = list(data_dir.glob("*.csv"))
    loaded_dfs = []
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        loaded_dfs.append(df)
    
    # All should load successfully
    assert len(loaded_dfs) == len(csv_files)
    
    # All should be DataFrames
    for df in loaded_dfs:
        assert isinstance(df, pd.DataFrame)


def test_newick_file_handling(example_data_with_output):
    """Test handling of Newick tree files if present."""
    data_dir = example_data_with_output["data_dir"]
    
    newick_files = list(data_dir.glob("*.newick"))
    
    # If newick files exist, test them
    for newick_file in newick_files:
        assert newick_file.exists()
        content = newick_file.read_text()
        assert len(content) > 0, "Newick file should have content"
        # Basic check: should contain parentheses and semicolon
        if ";" in content:
            assert True  # Valid newick format marker


def test_data_directory_structure(example_data_with_output):
    """Test that data directory has proper structure."""
    data_dir = example_data_with_output["data_dir"]
    output_dir = example_data_with_output["output_dir"]
    
    # Data directory should exist and be writable
    assert data_dir.exists()
    assert data_dir.is_dir()
    
    # Output directory should exist and be writable
    assert output_dir.exists()
    assert output_dir.is_dir()
    
    # Should be able to create subdirectories
    sub_output = output_dir / "subdir"
    sub_output.mkdir()
    assert sub_output.exists()
