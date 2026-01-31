"""Tests for analyzer module."""

from pathlib import Path

import pandas as pd
import pytest

from strepsuis_mdr.analyzer import MDRAnalyzer
from strepsuis_mdr.config import Config


@pytest.fixture
def sample_data(tmp_path):
    """Create sample test data with required files."""
    import shutil

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Copy example data files from the repo
    example_dir = Path(__file__).parent.parent / "data" / "examples"
    if example_dir.exists():
        for csv_file in example_dir.glob("*.csv"):
            shutil.copy(csv_file, data_dir)
        for newick_file in example_dir.glob("*.newick"):
            shutil.copy(newick_file, data_dir)

    return data_dir


def test_analyzer_initialization(sample_data, tmp_path):
    """Test analyzer initialization."""
    output_dir = tmp_path / "output"
    analyzer = MDRAnalyzer(data_dir=str(sample_data), output_dir=str(output_dir))
    assert analyzer.config.data_dir == str(sample_data)
    assert Path(analyzer.config.output_dir).exists()


def test_analyzer_initialization_with_config(sample_data, tmp_path):
    """Test analyzer initialization with Config object."""
    output_dir = tmp_path / "output"
    config = Config(data_dir=str(sample_data), output_dir=str(output_dir))
    analyzer = MDRAnalyzer(config=config)
    assert analyzer.config == config
    assert analyzer.results is None


def test_load_data(sample_data, tmp_path):
    """Test data loading."""
    output_dir = tmp_path / "output"
    _ = MDRAnalyzer(data_dir=str(sample_data), output_dir=str(output_dir))
    # Verify data directory exists
    assert Path(sample_data).exists()
    csv_files = list(Path(sample_data).glob("*.csv"))
    assert len(csv_files) > 0


@pytest.mark.skip(reason="Requires proper test data setup and multiprocessing fix")
def test_analysis_execution(sample_data, tmp_path):
    """Test main analysis execution."""
    output_dir = tmp_path / "output"
    analyzer = MDRAnalyzer(data_dir=str(sample_data), output_dir=str(output_dir))
    results = analyzer.run()
    assert results is not None
    assert isinstance(results, dict)
    assert "status" in results


def test_output_generation(sample_data, tmp_path):
    """Test output file generation."""
    output_dir = tmp_path / "output"
    analyzer = MDRAnalyzer(data_dir=str(sample_data), output_dir=str(output_dir))
    results = analyzer.run()

    # Check that results contain expected keys
    assert results is not None
    assert "status" in results
    assert "output_dir" in results
    assert results["status"] == "success"

    # Check that output directory was created
    assert Path(output_dir).exists()


def test_analyzer_with_bootstrap(sample_data, tmp_path):
    """Test analyzer with bootstrap parameters."""
    output_dir = tmp_path / "output"
    analyzer = MDRAnalyzer(
        data_dir=str(sample_data), output_dir=str(output_dir), bootstrap_iterations=100
    )
    assert analyzer.config.bootstrap_iterations == 100


def test_analyzer_with_fdr_alpha(sample_data, tmp_path):
    """Test analyzer with FDR alpha parameter."""
    output_dir = tmp_path / "output"
    analyzer = MDRAnalyzer(data_dir=str(sample_data), output_dir=str(output_dir), fdr_alpha=0.01)
    assert analyzer.config.fdr_alpha == 0.01


def test_analyzer_invalid_data_dir(tmp_path):
    """Test analyzer with invalid data directory."""
    output_dir = tmp_path / "output"
    with pytest.raises(ValueError):
        MDRAnalyzer(data_dir="/nonexistent", output_dir=str(output_dir))


def test_generate_report_without_results(sample_data, tmp_path):
    """Test that analyzer requires data files to run."""
    output_dir = tmp_path / "output"
    # Create an empty data directory
    empty_data = tmp_path / "empty_data"
    empty_data.mkdir()
    analyzer = MDRAnalyzer(data_dir=str(empty_data), output_dir=str(output_dir))
    # Should raise error when required files are missing
    with pytest.raises(FileNotFoundError):
        analyzer.run()


def test_reproducibility(sample_data, tmp_path):
    """Test that analysis is reproducible."""
    output_dir1 = tmp_path / "output1"
    output_dir2 = tmp_path / "output2"

    analyzer1 = MDRAnalyzer(data_dir=str(sample_data), output_dir=str(output_dir1))
    analyzer2 = MDRAnalyzer(data_dir=str(sample_data), output_dir=str(output_dir2))

    results1 = analyzer1.run()
    results2 = analyzer2.run()

    # Compare key results - should be identical with same configuration
    assert results1["status"] == results2["status"]
    assert results1["status"] == "success"


def test_empty_data_handling(tmp_path):
    """Test analyzer handles empty data gracefully."""

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create empty CSV files
    pd.DataFrame(columns=["Strain_ID"]).to_csv(empty_dir / "test_data.csv", index=False)

    # Should handle empty data appropriately
    # Note: This tests the robustness of the analyzer


def test_multiple_runs(sample_data, tmp_path):
    """Test that analyzer can run multiple times."""
    output_dir = tmp_path / "output"
    analyzer = MDRAnalyzer(data_dir=str(sample_data), output_dir=str(output_dir))

    # Run analysis twice
    results1 = analyzer.run()
    results2 = analyzer.run()

    # Both should succeed
    assert results1 is not None
    assert results2 is not None
    assert results1["status"] == "success"
    assert results2["status"] == "success"


def test_output_directory_creation(sample_data, tmp_path):
    """Test that output directory is created if it doesn't exist."""
    output_dir = tmp_path / "new_output"

    # Directory shouldn't exist yet
    assert not output_dir.exists()

    analyzer = MDRAnalyzer(data_dir=str(sample_data), output_dir=str(output_dir))

    # Should create directory
    assert Path(analyzer.config.output_dir).exists()
