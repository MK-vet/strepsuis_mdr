"""Tests for CLI module."""

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from strepsuis_mdr.cli import main


def setup_test_data(data_dir):
    """Copy example data files to test directory."""
    example_dir = Path(__file__).parent.parent / "data" / "examples"
    if example_dir.exists():
        for csv_file in example_dir.glob("*.csv"):
            shutil.copy(csv_file, data_dir)
        for newick_file in example_dir.glob("*.newick"):
            shutil.copy(newick_file, data_dir)


def test_cli_help(capsys, monkeypatch):
    """Test CLI help command."""
    monkeypatch.setattr(sys, "argv", ["strepsuis-mdr", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out.lower() or "Usage:" in captured.out


def test_cli_version(capsys, monkeypatch):
    """Test CLI version command."""
    monkeypatch.setattr(sys, "argv", ["strepsuis-mdr", "--version"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "1.0.0" in captured.out


def test_cli_missing_required_args(monkeypatch):
    """Test CLI with missing required arguments."""
    monkeypatch.setattr(sys, "argv", ["strepsuis-mdr"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code != 0


def test_cli_with_valid_args(monkeypatch):
    """Test CLI with valid arguments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()
        # Copy example data files
        setup_test_data(data_dir)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "strepsuis-mdr",
                "--data-dir",
                str(data_dir),
                "--output",
                str(Path(tmpdir) / "output"),
            ],
        )
        result = main()
        assert result == 0


def test_cli_with_bootstrap_option(monkeypatch):
    """Test CLI with bootstrap iterations option."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()
        # Copy example data files
        setup_test_data(data_dir)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "strepsuis-mdr",
                "--data-dir",
                str(data_dir),
                "--output",
                str(Path(tmpdir) / "output"),
                "--bootstrap",
                "100",
            ],
        )
        result = main()
        assert result == 0


def test_cli_with_fdr_alpha_option(monkeypatch):
    """Test CLI with FDR alpha option."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()
        # Copy example data files
        setup_test_data(data_dir)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "strepsuis-mdr",
                "--data-dir",
                str(data_dir),
                "--output",
                str(Path(tmpdir) / "output"),
                "--fdr-alpha",
                "0.01",
            ],
        )
        result = main()
        assert result == 0


def test_cli_with_verbose_option(monkeypatch):
    """Test CLI with verbose option."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()
        # Copy example data files
        setup_test_data(data_dir)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "strepsuis-mdr",
                "--data-dir",
                str(data_dir),
                "--output",
                str(Path(tmpdir) / "output"),
                "--verbose",
            ],
        )
        result = main()
        assert result == 0


def test_cli_with_invalid_data_dir(monkeypatch):
    """Test CLI with non-existent data directory."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["strepsuis-mdr", "--data-dir", "/nonexistent/path", "--output", str(Path(tempfile.mkdtemp()) / "output")],
    )
    result = main()
    assert result != 0
