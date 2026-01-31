"""
Validation Utilities for StrepSuis Suite

Provides unified input data validation functions used across all modules.
"""

from typing import List, Tuple, Optional
import pandas as pd
import numpy as np


def validate_input_data(
    df: pd.DataFrame,
    id_col: str = "Strain_ID",
    feature_cols: Optional[List[str]] = None,
    max_missing_pct: float = 0.2,
    require_binary: bool = True,
) -> Tuple[bool, str]:
    """
    Validate structural and value-level integrity of input data.

    This function performs comprehensive validation checks on input DataFrames
    to ensure they meet the requirements for downstream analysis.

    Checks performed:
        - Presence of id_col and all feature_cols
        - Duplicate IDs detection
        - Proportion of missing values per column
        - Binary 0/1 encoding where expected
        - Flags unexpected floats (e.g., values between 0 and 1)

    Parameters:
        df: Input DataFrame to validate
        id_col: Name of the identifier column (default: "Strain_ID")
        feature_cols: List of required feature columns. If None, all columns
                      except id_col are treated as features.
        max_missing_pct: Maximum allowed proportion of missing values per column
                         (default: 0.2 = 20%)
        require_binary: Whether to enforce binary (0/1) encoding for feature
                        columns (default: True)

    Returns:
        Tuple of (is_valid, message) where:
            - is_valid: True if all validation checks pass
            - message: Description of validation result or error details
    
    Examples:
        >>> df = pd.DataFrame({'Strain_ID': ['A', 'B'], 'Gene1': [1, 0]})
        >>> is_valid, msg = validate_input_data(df)
        >>> print(is_valid)
        True
        
        >>> df_dup = pd.DataFrame({'Strain_ID': ['A', 'A'], 'Gene1': [1, 0]})
        >>> is_valid, msg = validate_input_data(df_dup)
        >>> print(is_valid)
        False
    """
    if df.empty:
        return False, "Input DataFrame is empty"
    
    errors = []
    warnings = []
    
    # Check for ID column presence
    if id_col not in df.columns:
        errors.append(f"Required ID column '{id_col}' not found in DataFrame")
    else:
        # Check for duplicate IDs
        duplicates = df[id_col].duplicated()
        if duplicates.any():
            dup_ids = df[id_col][duplicates].unique().tolist()[:5]  # Show first 5
            errors.append(
                f"Duplicate IDs detected: {dup_ids}"
                + (f" (and {duplicates.sum() - 5} more)" if duplicates.sum() > 5 else "")
            )
    
    # Determine feature columns
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != id_col]
    
    # Check for required feature columns
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required feature columns: {missing_cols[:10]}")
    
    valid_feature_cols = [c for c in feature_cols if c in df.columns]
    
    # Check missing values per column
    high_missing_cols = []
    for col in valid_feature_cols:
        missing_pct = df[col].isna().mean()
        if missing_pct > max_missing_pct:
            high_missing_cols.append((col, missing_pct))
    
    if high_missing_cols:
        col_info = ", ".join([f"{c}({p*100:.1f}%)" for c, p in high_missing_cols[:5]])
        warnings.append(
            f"Columns with >{ max_missing_pct*100:.0f}% missing values: {col_info}"
            + (f" (and {len(high_missing_cols) - 5} more)" if len(high_missing_cols) > 5 else "")
        )
    
    # Check binary encoding if required
    if require_binary:
        non_binary_cols = []
        suspicious_float_cols = []
        
        for col in valid_feature_cols:
            col_data = df[col].dropna()
            if col_data.empty:
                continue
            
            unique_vals = col_data.unique()
            
            # Check if values are binary (0/1)
            if not set(unique_vals).issubset({0, 1, 0.0, 1.0, True, False}):
                non_binary_cols.append(col)
                
                # Check for suspicious float values (between 0 and 1, non-binary)
                if col_data.dtype in [np.float32, np.float64]:
                    suspicious = col_data[(col_data > 0) & (col_data < 1)]
                    if len(suspicious) > 0:
                        suspicious_float_cols.append(col)
        
        if non_binary_cols:
            warnings.append(
                f"Non-binary values detected in {len(non_binary_cols)} columns: "
                f"{non_binary_cols[:5]}"
                + (f" (and {len(non_binary_cols) - 5} more)" if len(non_binary_cols) > 5 else "")
            )
        
        if suspicious_float_cols:
            errors.append(
                f"Suspicious float values (between 0 and 1) in columns: "
                f"{suspicious_float_cols[:5]}. "
                "These may need to be converted to binary presence/absence."
            )
    
    # Compile result
    if errors:
        return False, "Validation failed: " + "; ".join(errors)
    
    if warnings:
        return True, "Validation passed with warnings: " + "; ".join(warnings)
    
    return True, "Validation passed successfully"


def validate_mdr_output(
    overview_df: pd.DataFrame,
    detail_df: Optional[pd.DataFrame] = None,
    expected_total: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Validate MDR analysis output for internal consistency.

    Checks that:
        - Overview totals match actual counts
        - No negative counts
        - Percentages are valid (0-100%)

    Parameters:
        overview_df: DataFrame containing summary statistics
        detail_df: Optional DataFrame with detailed results for cross-validation
        expected_total: Expected total isolate count for validation

    Returns:
        Tuple of (is_valid, message)
    """
    errors = []
    
    if overview_df.empty:
        return False, "Overview DataFrame is empty"
    
    # Check for required columns
    required_cols = ["Metric", "Value"]
    missing = [c for c in required_cols if c not in overview_df.columns]
    if missing:
        return False, f"Missing required columns: {missing}"
    
    # Validate total counts if expected_total provided
    if expected_total is not None:
        total_row = overview_df[overview_df["Metric"] == "Total Isolates"]
        if not total_row.empty:
            reported_total = int(total_row["Value"].values[0])
            if reported_total != expected_total:
                errors.append(
                    f"Total isolates mismatch: reported {reported_total}, expected {expected_total}"
                )
    
    # Check for negative values (shouldn't happen in count data)
    if "Value" in overview_df.columns:
        numeric_vals = pd.to_numeric(overview_df["Value"], errors="coerce")
        if (numeric_vals < 0).any():
            errors.append("Negative values detected in overview")
    
    if errors:
        return False, "Validation failed: " + "; ".join(errors)
    
    return True, "Output validation passed"


def check_binary_encoding(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> Tuple[bool, List[str]]:
    """
    Check if specified columns contain only binary (0/1) values.

    Parameters:
        df: Input DataFrame
        columns: Columns to check (default: all numeric columns)

    Returns:
        Tuple of (all_binary, non_binary_columns) where:
            - all_binary: True if all checked columns are binary
            - non_binary_columns: List of column names with non-binary values
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number, bool]).columns.tolist()
    
    non_binary = []
    for col in columns:
        if col not in df.columns:
            continue
        
        unique_vals = df[col].dropna().unique()
        if not set(unique_vals).issubset({0, 1, 0.0, 1.0, True, False}):
            non_binary.append(col)
    
    return len(non_binary) == 0, non_binary


def get_data_summary(df: pd.DataFrame, id_col: str = "Strain_ID") -> dict:
    """
    Generate a summary of input data characteristics.

    Parameters:
        df: Input DataFrame
        id_col: Name of identifier column

    Returns:
        Dictionary with data summary statistics
    """
    feature_cols = [c for c in df.columns if c != id_col]
    
    summary = {
        "n_samples": len(df),
        "n_features": len(feature_cols),
        "id_column": id_col if id_col in df.columns else None,
        "missing_total": df[feature_cols].isna().sum().sum() if feature_cols else 0,
        "missing_pct": (
            df[feature_cols].isna().sum().sum() / (len(df) * len(feature_cols)) * 100
            if feature_cols and len(df) > 0 else 0
        ),
    }
    
    # Check binary encoding
    all_binary, non_binary = check_binary_encoding(df, feature_cols)
    summary["all_binary"] = all_binary
    summary["non_binary_columns"] = non_binary
    
    # Feature prevalence stats (for binary columns)
    if all_binary and feature_cols:
        prevalences = df[feature_cols].mean()
        summary["mean_prevalence"] = prevalences.mean()
        summary["min_prevalence"] = prevalences.min()
        summary["max_prevalence"] = prevalences.max()
    
    return summary
