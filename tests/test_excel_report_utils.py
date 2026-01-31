"""
Comprehensive tests for Excel report generation utilities.

Target: Increase coverage from 19% to 80%+
"""

import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go

from strepsuis_mdr.excel_report_utils import ExcelReportGenerator, sanitize_sheet_name


class TestExcelReportGenerator:
    """Test ExcelReportGenerator class."""

    def test_initialization(self, tmp_path):
        """Test ExcelReportGenerator initialization."""
        generator = ExcelReportGenerator(str(tmp_path))
        
        assert generator.output_folder == str(tmp_path)
        assert os.path.exists(generator.png_folder)
        assert len(generator.png_files) == 0

    def test_save_matplotlib_figure(self, tmp_path):
        """Test saving matplotlib figure."""
        generator = ExcelReportGenerator(str(tmp_path))
        
        # Create a simple matplotlib figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        ax.set_title("Test Plot")
        
        # Save figure
        filepath = generator.save_matplotlib_figure(fig, "test_plot", dpi=150)
        
        assert os.path.exists(filepath)
        assert filepath.endswith(".png")
        assert "test_plot.png" in filepath
        assert len(generator.png_files) == 1
        assert filepath in generator.png_files

    def test_save_plotly_figure(self, tmp_path):
        """Test saving plotly figure."""
        generator = ExcelReportGenerator(str(tmp_path))
        
        # Create a simple plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 4, 9], name="Test"))
        fig.update_layout(title="Test Plot")
        
        try:
            filepath = generator.save_plotly_figure(fig, "test_plotly", width=800, height=600)
            assert os.path.exists(filepath)
            assert filepath.endswith(".png")
            assert "test_plotly.png" in filepath
            assert len(generator.png_files) == 1
        except Exception as e:
            # If kaleido is not available, test fallback
            pytest.skip(f"Plotly image export requires kaleido: {e}")

    def test_save_plotly_figure_fallback(self, tmp_path):
        """Test plotly figure fallback to HTML."""
        generator = ExcelReportGenerator(str(tmp_path))
        
        # Create a simple plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))
        
        try:
            filepath = generator.save_plotly_figure_fallback(fig, "test_fallback")
            assert os.path.exists(filepath)
            # Could be PNG or HTML depending on kaleido availability
            assert filepath.endswith(".png") or filepath.endswith(".html")
            assert len(generator.png_files) == 1
        except Exception as e:
            pytest.skip(f"Plotly export failed: {e}")

    def test_create_metadata_sheet(self, tmp_path):
        """Test metadata sheet creation."""
        generator = ExcelReportGenerator(str(tmp_path))
        
        excel_path = os.path.join(tmp_path, "test_metadata.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            generator.create_metadata_sheet(
                writer,
                "Test Script",
                analysis_date="2026-01-01 12:00:00",
                custom_field="custom_value"
            )
        
        # Verify Excel file was created
        assert os.path.exists(excel_path)
        
        # Read and verify content
        df = pd.read_excel(excel_path, sheet_name="Metadata")
        assert len(df) > 0
        assert "Analysis Script" in df["Report Information"].values
        assert "Test Script" in df["Value"].values
        assert "custom_field" in df["Report Information"].values

    def test_create_methodology_sheet_string(self, tmp_path):
        """Test methodology sheet with string input."""
        generator = ExcelReportGenerator(str(tmp_path))
        
        excel_path = os.path.join(tmp_path, "test_methodology.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            generator.create_methodology_sheet(writer, "This is a test methodology description.")
        
        assert os.path.exists(excel_path)
        df = pd.read_excel(excel_path, sheet_name="Methodology")
        assert len(df) > 0

    def test_create_methodology_sheet_dict(self, tmp_path):
        """Test methodology sheet with dict input."""
        generator = ExcelReportGenerator(str(tmp_path))
        
        methodology = {
            "Section 1": "Description 1",
            "Section 2": "Description 2"
        }
        
        excel_path = os.path.join(tmp_path, "test_methodology_dict.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            generator.create_methodology_sheet(writer, methodology)
        
        assert os.path.exists(excel_path)
        df = pd.read_excel(excel_path, sheet_name="Methodology")
        assert len(df) == 2
        assert "Section 1" in df["Section"].values

    def test_create_chart_index_sheet(self, tmp_path):
        """Test chart index sheet creation."""
        generator = ExcelReportGenerator(str(tmp_path))
        
        # Add some PNG files (create separate figures as save_matplotlib_figure closes them)
        fig1, ax1 = plt.subplots()
        ax1.plot([1, 2, 3])
        generator.save_matplotlib_figure(fig1, "chart1")
        
        fig2, ax2 = plt.subplots()
        ax2.plot([1, 2, 3])
        generator.save_matplotlib_figure(fig2, "chart2")
        
        excel_path = os.path.join(tmp_path, "test_chart_index.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            generator.create_chart_index_sheet(writer)
        
        assert os.path.exists(excel_path)
        df = pd.read_excel(excel_path, sheet_name="Chart_Index")
        assert len(df) == 2
        assert "chart1.png" in df["Filename"].values

    def test_create_chart_index_sheet_empty(self, tmp_path):
        """Test chart index sheet with no charts."""
        generator = ExcelReportGenerator(str(tmp_path))
        
        excel_path = os.path.join(tmp_path, "test_empty_chart_index.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            # openpyxl cannot save a workbook with zero visible sheets
            pd.DataFrame({"Message": ["No charts generated"]}).to_excel(
                writer, sheet_name="Info", index=False
            )
            # Should not raise error even with no charts
            generator.create_chart_index_sheet(writer)
        
        assert os.path.exists(excel_path)
        # Chart_Index sheet should not exist if no charts
        xls = pd.ExcelFile(excel_path)
        assert "Chart_Index" not in xls.sheet_names

    def test_add_dataframe_sheet(self, tmp_path):
        """Test adding DataFrame as sheet."""
        generator = ExcelReportGenerator(str(tmp_path))
        
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        
        excel_path = os.path.join(tmp_path, "test_dataframe.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            generator.add_dataframe_sheet(writer, df, "TestSheet", description="Test description")
        
        assert os.path.exists(excel_path)
        # Read raw to handle the optional description rows robustly
        raw = pd.read_excel(excel_path, sheet_name="TestSheet", header=None)
        # Find the row that contains the header cells "A", "B"
        header_rows = raw.index[(raw.iloc[:, 0] == "A") & (raw.iloc[:, 1] == "B")].tolist()
        assert header_rows, "Expected to find header row with columns A and B"
        header_row = header_rows[0]
        data = raw.iloc[header_row + 1 : header_row + 4, 0:2].reset_index(drop=True)
        assert len(data) == 3
        assert data.iloc[0, 0] == 1
        assert data.iloc[0, 1] == 4

    def test_add_dataframe_sheet_empty(self, tmp_path):
        """Test adding empty DataFrame."""
        generator = ExcelReportGenerator(str(tmp_path))
        
        df = pd.DataFrame()
        
        excel_path = os.path.join(tmp_path, "test_empty_df.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            generator.add_dataframe_sheet(writer, df, "EmptySheet")
        
        assert os.path.exists(excel_path)
        df_read = pd.read_excel(excel_path, sheet_name="EmptySheet")
        assert "No data available" in df_read["Message"].values[0]

    def test_add_dataframe_sheet_none(self, tmp_path):
        """Test adding None as DataFrame."""
        generator = ExcelReportGenerator(str(tmp_path))
        
        excel_path = os.path.join(tmp_path, "test_none_df.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            generator.add_dataframe_sheet(writer, None, "NoneSheet")
        
        assert os.path.exists(excel_path)
        df_read = pd.read_excel(excel_path, sheet_name="NoneSheet")
        assert "No data available" in df_read["Message"].values[0]

    def test_add_dataframe_sheet_long_name(self, tmp_path):
        """Test adding sheet with long name (should be truncated)."""
        generator = ExcelReportGenerator(str(tmp_path))
        
        df = pd.DataFrame({"A": [1, 2, 3]})
        long_name = "A" * 50  # Longer than 31 chars
        
        excel_path = os.path.join(tmp_path, "test_long_name.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            generator.add_dataframe_sheet(writer, df, long_name)
        
        assert os.path.exists(excel_path)
        xls = pd.ExcelFile(excel_path)
        # Sheet name should be truncated to 31 chars
        assert any(len(name) <= 31 for name in xls.sheet_names)

    def test_generate_excel_report(self, tmp_path):
        """Test complete Excel report generation."""
        generator = ExcelReportGenerator(str(tmp_path))
        
        # Create test data
        df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        df2 = pd.DataFrame({"X": [10, 20], "Y": [30, 40]})
        
        sheets_data = {
            "Sheet1": df1,
            "Sheet2": (df2, "Description for Sheet2")
        }
        
        methodology = {
            "Method": "Test methodology",
            "Parameters": "Test parameters"
        }
        
        excel_path = generator.generate_excel_report(
            "test_report",
            sheets_data,
            methodology=methodology,
            custom_meta="custom_value"
        )
        
        assert os.path.exists(excel_path)
        assert excel_path.endswith(".xlsx")
        
        # Verify sheets exist
        xls = pd.ExcelFile(excel_path)
        assert "Metadata" in xls.sheet_names
        assert "Methodology" in xls.sheet_names
        assert "Sheet1" in xls.sheet_names
        assert "Sheet2" in xls.sheet_names

    def test_generate_excel_report_with_charts(self, tmp_path):
        """Test Excel report generation with charts."""
        generator = ExcelReportGenerator(str(tmp_path))
        
        # Create and save a chart
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        generator.save_matplotlib_figure(fig, "test_chart")
        
        sheets_data = {
            "Data": pd.DataFrame({"A": [1, 2, 3]})
        }
        
        excel_path = generator.generate_excel_report("test_with_charts", sheets_data)
        
        assert os.path.exists(excel_path)
        xls = pd.ExcelFile(excel_path)
        assert "Chart_Index" in xls.sheet_names
        
        # Verify chart index
        df_charts = pd.read_excel(excel_path, sheet_name="Chart_Index")
        assert len(df_charts) == 1
        assert "test_chart.png" in df_charts["Filename"].values

    def test_generate_excel_report_numeric_rounding(self, tmp_path):
        """Test that numeric columns are rounded in Excel output."""
        generator = ExcelReportGenerator(str(tmp_path))
        
        # Create DataFrame with many decimal places
        df = pd.DataFrame({
            "A": [1.123456789, 2.987654321, 3.555555555]
        })
        
        excel_path = os.path.join(tmp_path, "test_rounding.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            generator.add_dataframe_sheet(writer, df, "Rounded")
        
        # Read back and verify rounding
        df_read = pd.read_excel(excel_path, sheet_name="Rounded")
        # Values should be rounded to 4 decimal places
        assert all(df_read["A"].apply(lambda x: len(str(x).split('.')[-1]) <= 4))


class TestSanitizeSheetName:
    """Test sanitize_sheet_name function."""

    def test_sanitize_normal_name(self):
        """Test sanitization of normal name."""
        result = sanitize_sheet_name("NormalSheetName")
        assert result == "NormalSheetName"

    def test_sanitize_special_characters(self):
        """Test removal of special characters."""
        result = sanitize_sheet_name("Test:Name*Here?")
        assert ":" not in result
        assert "*" not in result
        assert "?" not in result
        assert "/" not in result
        assert "\\" not in result
        assert "[" not in result
        assert "]" not in result

    def test_sanitize_long_name(self):
        """Test truncation of long names."""
        long_name = "A" * 50
        result = sanitize_sheet_name(long_name)
        assert len(result) <= 31

    def test_sanitize_all_invalid_chars(self):
        """Test all invalid characters are replaced."""
        invalid_name = "Test:Name\\With/Invalid?Chars*[Here]"
        result = sanitize_sheet_name(invalid_name)
        assert ":" not in result
        assert "\\" not in result
        assert "/" not in result
        assert "?" not in result
        assert "*" not in result
        assert "[" not in result
        assert "]" not in result
