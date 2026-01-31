"""
Shared utility module for generating Excel reports with PNG charts.
This module provides common functions for creating detailed Excel reports
with consistent structure across all analysis scripts.
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd


class ExcelReportGenerator:
    """Generate comprehensive Excel reports with multiple sheets and embedded/referenced PNG charts."""

    def __init__(self, output_folder="output"):
        """
        Initialize Excel report generator.

        Args:
            output_folder: Directory where reports and PNG files will be saved
        """
        self.output_folder = output_folder
        self.png_folder = os.path.join(output_folder, "png_charts")
        os.makedirs(self.png_folder, exist_ok=True)
        self.png_files = []

    def save_matplotlib_figure(self, fig, filename, dpi=150, bbox_inches="tight"):
        """
        Save a matplotlib figure as PNG.

        Args:
            fig: matplotlib figure object
            filename: Name for the PNG file (without extension)
            dpi: Resolution in dots per inch
            bbox_inches: Bounding box specification

        Returns:
            Path to saved PNG file
        """
        filepath = os.path.join(self.png_folder, f"{filename}.png")
        fig.savefig(filepath, format="png", dpi=dpi, bbox_inches=bbox_inches)
        plt.close(fig)
        self.png_files.append(filepath)
        print(f"Saved matplotlib chart: {filepath}")
        return filepath

    def save_plotly_figure(self, fig, filename, width=1200, height=800, scale=2):
        """
        Save a plotly figure as PNG.

        Args:
            fig: plotly figure object
            filename: Name for the PNG file (without extension)
            width: Image width in pixels
            height: Image height in pixels
            scale: Scale factor for image quality

        Returns:
            Path to saved PNG file
        """
        filepath = os.path.join(self.png_folder, f"{filename}.png")
        fig.write_image(filepath, format="png", width=width, height=height, scale=scale)
        self.png_files.append(filepath)
        print(f"Saved plotly chart: {filepath}")
        return filepath

    def save_plotly_figure_fallback(self, fig, filename, width=1200, height=800):
        """
        Save a plotly figure as PNG using kaleido (fallback if write_image fails).
        If kaleido is not available, saves as static HTML.

        Args:
            fig: plotly figure object
            filename: Name for the PNG file (without extension)
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            Path to saved file (PNG or HTML)
        """
        try:
            return self.save_plotly_figure(fig, filename, width, height)
        except Exception as e:
            print(f"Could not save as PNG (kaleido might not be installed): {e}")
            # Fallback: save as static HTML
            filepath = os.path.join(self.png_folder, f"{filename}.html")
            fig.write_html(filepath, include_plotlyjs="cdn")
            self.png_files.append(filepath)
            print(f"Saved plotly chart as HTML: {filepath}")
            return filepath

    def create_metadata_sheet(self, writer, script_name, analysis_date=None, **kwargs):
        """
        Create a metadata sheet with report information.

        Args:
            writer: pandas ExcelWriter object
            script_name: Name of the analysis script
            analysis_date: Date of analysis (default: current timestamp)
            **kwargs: Additional metadata key-value pairs
        """
        if analysis_date is None:
            analysis_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        metadata = {
            "Report Information": [
                "Analysis Script",
                "Analysis Date",
                "Output Folder",
                "Number of PNG Charts Generated",
            ],
            "Value": [script_name, analysis_date, self.output_folder, len(self.png_files)],
        }

        # Add custom metadata
        for key, value in kwargs.items():
            metadata["Report Information"].append(key)
            metadata["Value"].append(str(value))

        df_metadata = pd.DataFrame(metadata)
        df_metadata.to_excel(writer, sheet_name="Metadata", index=False)

        # Format the sheet
        worksheet = writer.sheets["Metadata"]
        worksheet.column_dimensions["A"].width = 40
        worksheet.column_dimensions["B"].width = 60

    def create_methodology_sheet(self, writer, methodology_text):
        """
        Create a methodology sheet with detailed analysis description.

        Args:
            writer: pandas ExcelWriter object
            methodology_text: Text or dict describing the methodology
        """
        if isinstance(methodology_text, dict):
            # Convert dict to DataFrame
            rows = []
            for section, content in methodology_text.items():
                rows.append({"Section": section, "Description": content})
            df_method = pd.DataFrame(rows)
        elif isinstance(methodology_text, str):
            # Create simple text-based sheet
            df_method = pd.DataFrame({"Methodology": [methodology_text]})
        else:
            df_method = pd.DataFrame({"Methodology": ["Detailed methodology not provided"]})

        df_method.to_excel(writer, sheet_name="Methodology", index=False)

        # Format the sheet
        worksheet = writer.sheets["Methodology"]
        for col in worksheet.columns:
            max_length = 0
            column = col[0].column_letter
            for cell in col:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except (TypeError, AttributeError):
                    pass
            adjusted_width = min(max_length + 2, 100)
            worksheet.column_dimensions[column].width = adjusted_width

    def create_chart_index_sheet(self, writer):
        """
        Create a sheet listing all generated PNG charts.

        Args:
            writer: pandas ExcelWriter object
        """
        if not self.png_files:
            return

        chart_data = {"Chart Number": [], "Filename": [], "Full Path": []}

        for idx, filepath in enumerate(self.png_files, 1):
            chart_data["Chart Number"].append(idx)
            chart_data["Filename"].append(os.path.basename(filepath))
            chart_data["Full Path"].append(filepath)

        df_charts = pd.DataFrame(chart_data)
        df_charts.to_excel(writer, sheet_name="Chart_Index", index=False)

        # Format the sheet
        worksheet = writer.sheets["Chart_Index"]
        worksheet.column_dimensions["A"].width = 15
        worksheet.column_dimensions["B"].width = 50
        worksheet.column_dimensions["C"].width = 80

    def add_dataframe_sheet(self, writer, df, sheet_name, description=None):
        """
        Add a DataFrame as a sheet with optional description.

        Args:
            writer: pandas ExcelWriter object
            df: pandas DataFrame to add
            sheet_name: Name for the sheet (max 31 chars, will be truncated)
            description: Optional description text
        """
        # Excel sheet names must be <= 31 characters
        sheet_name = sheet_name[:31]

        if df is None or (hasattr(df, "empty") and df.empty):
            # Create empty sheet with message
            df_empty = pd.DataFrame({"Message": ["No data available for this analysis"]})
            df_empty.to_excel(writer, sheet_name=sheet_name, index=False)
            return

        # Round numeric columns for better readability
        df_copy = df.copy()
        numeric_cols = df_copy.select_dtypes(include=["float64", "float32"]).columns
        for col in numeric_cols:
            df_copy[col] = df_copy[col].round(4)

        # Write to Excel
        start_row = 0
        if description:
            # Add description at the top
            df_desc = pd.DataFrame({"Description": [description]})
            df_desc.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
            start_row = 3

        df_copy.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row)

        # Auto-adjust column widths
        worksheet = writer.sheets[sheet_name]
        for col in worksheet.columns:
            max_length = 0
            column = col[0].column_letter
            for cell in col:
                try:
                    if cell.value and len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except (TypeError, AttributeError):
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column].width = adjusted_width

    def generate_excel_report(self, report_name, sheets_data, methodology=None, **metadata):
        """
        Generate complete Excel report with multiple sheets.

        Args:
            report_name: Base name for the Excel file (without extension)
            sheets_data: Dict of {sheet_name: DataFrame} or {sheet_name: (DataFrame, description)}
            methodology: Methodology text or dict
            **metadata: Additional metadata key-value pairs

        Returns:
            Path to generated Excel file
        """
        # Create output file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_filename = f"{report_name}_{timestamp}.xlsx"
        excel_path = os.path.join(self.output_folder, excel_filename)

        # Create Excel writer with openpyxl engine
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            # Add metadata sheet
            self.create_metadata_sheet(writer, report_name, **metadata)

            # Add methodology sheet if provided
            if methodology:
                self.create_methodology_sheet(writer, methodology)

            # Add data sheets
            for sheet_name, sheet_content in sheets_data.items():
                if isinstance(sheet_content, tuple):
                    # (DataFrame, description)
                    df, description = sheet_content
                    self.add_dataframe_sheet(writer, df, sheet_name, description)
                else:
                    # Just DataFrame
                    self.add_dataframe_sheet(writer, sheet_content, sheet_name)

            # Add chart index sheet
            self.create_chart_index_sheet(writer)

        print(f"\n{'='*80}")
        print(f"Excel report generated: {excel_path}")
        print(f"Total sheets: {len(sheets_data) + 2}")  # +2 for metadata and chart index
        print(f"Total PNG charts: {len(self.png_files)}")
        print(f"Charts saved in: {self.png_folder}")
        print(f"{'='*80}\n")

        return excel_path


def sanitize_sheet_name(name):
    r"""
    Sanitize sheet name to comply with Excel requirements.
    - Max 31 characters
    - No special characters: : \ / ? * [ ]

    Args:
        name: Original sheet name

    Returns:
        Sanitized sheet name
    """
    # Remove invalid characters
    invalid_chars = [":", "\\", "/", "?", "*", "[", "]"]
    sanitized = name
    for char in invalid_chars:
        sanitized = sanitized.replace(char, "_")

    # Truncate to 31 characters
    if len(sanitized) > 31:
        sanitized = sanitized[:31]

    return sanitized
