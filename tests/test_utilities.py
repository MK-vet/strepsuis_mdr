"""Tests for utility functions and helper modules."""

import pytest
import tempfile
from pathlib import Path
import os


def test_output_directory_creation():
    """Test that output directories can be created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        assert output_dir.exists()
        assert output_dir.is_dir()


def test_path_handling():
    """Test path handling and resolution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        
        # Test absolute path
        abs_path = base_path / "subdir" / "file.txt"
        assert abs_path.is_absolute()
        
        # Test parent directory
        parent = abs_path.parent
        assert parent.name == "subdir"


def test_file_permissions():
    """Test file creation and permissions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("test content")
        
        assert test_file.exists()
        assert test_file.is_file()
        assert test_file.read_text() == "test content"


def test_directory_listing():
    """Test directory operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        
        # Create some test files
        (base / "file1.txt").write_text("content1")
        (base / "file2.csv").write_text("content2")
        
        # List files
        txt_files = list(base.glob("*.txt"))
        csv_files = list(base.glob("*.csv"))
        
        assert len(txt_files) == 1
        assert len(csv_files) == 1


def test_temp_file_cleanup():
    """Test temporary file handling."""
    temp_path = None
    
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        assert temp_path.exists()
    
    # After context, directory should be cleaned up
    assert not temp_path.exists()


def test_path_concatenation():
    """Test path joining and manipulation."""
    base = Path("/base/path")
    subpath = base / "subdir" / "file.txt"
    
    # Use as_posix for cross-platform consistency
    assert subpath.as_posix() == "/base/path/subdir/file.txt"
    assert subpath.name == "file.txt"
    assert subpath.suffix == ".txt"
    assert subpath.stem == "file"


def test_relative_path_conversion():
    """Test relative path handling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        subdir = base / "subdir"
        subdir.mkdir()
        
        file_path = subdir / "test.txt"
        file_path.write_text("content")
        
        # Get relative path - use as_posix for cross-platform
        rel_path = file_path.relative_to(base)
        assert rel_path.as_posix() == "subdir/test.txt"


def test_environment_variables():
    """Test environment variable handling."""
    # Set and get environment variable
    os.environ["TEST_VAR"] = "test_value"
    assert os.environ.get("TEST_VAR") == "test_value"
    
    # Clean up
    del os.environ["TEST_VAR"]
    assert os.environ.get("TEST_VAR") is None


def test_current_working_directory():
    """Test current directory operations."""
    original_dir = os.getcwd()
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            current = os.getcwd()
            # Restore BEFORE exiting context to avoid PermissionError on Windows
            os.chdir(original_dir)
            assert Path(current).samefile(Path(tmpdir))
    finally:
        os.chdir(original_dir)
    
    assert os.getcwd() == original_dir


def test_file_extension_handling():
    """Test file extension operations."""
    test_files = [
        ("data.csv", ".csv"),
        ("report.html", ".html"),
        ("tree.newick", ".newick"),
        ("results.xlsx", ".xlsx"),
    ]
    
    for filename, expected_ext in test_files:
        path = Path(filename)
        assert path.suffix == expected_ext
        assert path.stem == filename.replace(expected_ext, "")
