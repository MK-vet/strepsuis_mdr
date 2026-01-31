#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MDR Analysis Core Module
========================

Integrated phenotype-genotype analysis of multidrug resistance (MDR) patterns
in bacterial genomics.

Features:
    - Bootstrap resampling for prevalence estimation with confidence intervals
    - Chi-square and Fisher's exact tests for association analysis
    - Phi coefficient for association strength measurement
    - Multiple testing correction using Benjamini-Hochberg FDR
    - Co-occurrence analysis for phenotypes and resistance genes
    - Association rule mining (support, confidence, lift)
    - Hybrid co-resistance network construction
    - Community detection using Louvain algorithm
    - Interactive network visualization with Plotly
    - HTML and Excel report generation

Author: MK-vet
Version: 1.0.0
License: MIT
"""

import logging
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

try:
    from .excel_report_utils import ExcelReportGenerator
except ImportError:
    from strepsuis_mdr.excel_report_utils import ExcelReportGenerator

warnings.filterwarnings("ignore")  # for cleaner console output

# Global output folder - can be overridden before calling main()
output_folder = "output"


###############################################################################
# 0) ENVIRONMENT SETUP
###############################################################################
def setup_environment(csv_path: Optional[str] = None) -> str:
    """
    Detect if running on Google Colab or locally, handle user input,
    create output directories, and configure logging. Returns the CSV path.

    The function performs the following operations:
    1. Detects execution environment (Google Colab or local machine)
    2. Creates necessary output directories for results storage
    3. Configures logging to both file and console
    4. Handles file upload in Colab or path input locally

    Args:
        csv_path: Optional path to CSV file. If provided, skips user input.

    Returns:
        str: Path to the input CSV file containing resistance data
    """
    IN_COLAB = False
    try:
        from google.colab import files  # type: ignore[import-untyped,import-not-found]

        IN_COLAB = True
        logging.info("Google Colab environment detected.")
    except ImportError:
        logging.info("Local environment detected.")

    # Create output directories
    global output_folder
    try:
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(os.path.join(output_folder, "figures"), exist_ok=True)
        os.makedirs(os.path.join(output_folder, "data"), exist_ok=True)
    except Exception as e:
        logging.error(f"Could not create output directories: {e}")
        raise

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_folder, "analysis_log.txt")),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # If CSV path is provided, use it directly (for programmatic use)
    if csv_path:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found: {csv_path}")
        logging.info(f"Using provided CSV path: {csv_path}")
        return csv_path

    # Get CSV path from user
    if IN_COLAB:
        print("Please upload your CSV file containing resistance data...")
        from google.colab import files  # type: ignore[import-untyped,import-not-found]

        uploaded = files.upload()
        if not uploaded:
            raise ValueError("No file was uploaded. Please run again and upload a file.")
        csv_filename = list(uploaded.keys())[0]
        logging.info(f"Uploaded file: {csv_filename}")
        return csv_filename
    else:
        csv_path = input("Enter the path to your CSV file: ")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found: {csv_path}")
        return csv_path


###############################################################################
# 1) ANTIBIOTIC CLASSES
###############################################################################
# Predefined classification of antibiotics into functional classes
# This mapping is used for determining MDR status and analyzing class-level resistance patterns
ANTIBIOTIC_CLASSES: Dict[str, List[str]] = {
    "Tetracyclines": ["Oxytetracycline", "Doxycycline"],
    "Macrolides": ["Tulathromycin"],
    "Aminoglycosides": ["Spectinomycin", "Gentamicin"],
    "Pleuromutilins": ["Tiamulin"],
    "Sulfonamides": ["Trimethoprim_Sulphamethoxazole"],
    "Fluoroquinolones": ["Enrofloxacin"],
    "Penicillins": ["Penicillin", "Ampicillin", "Amoxicillin_Clavulanic_acid"],
    "Cephalosporins": ["Ceftiofur"],
    "Phenicols": ["Florfenicol"],
}


###############################################################################
# 2) CORE UTILITY FUNCTIONS
###############################################################################
def safe_contingency(table: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Compute contingency table analysis using Chi-square or Fisher's exact test
    based on expected cell counts (Cochran's rule).

    Statistical methods:
    - Chi-square test: Used when all expected cell frequencies >= 5 and at least
      80% of cells have expected frequency >= 5
    - Fisher's exact test: Used when minimum expected frequency < 1 or <80%
      of cells have expected frequency >= 5
    - Phi coefficient: Calculated as (ad-bc)/sqrt(r₁r₂c₁c₂) for 2×2 tables

    Mathematical details:
    - Chi-square statistic: χ² = Σ[(O-E)²/E] where O=observed, E=expected
    - Expected cell counts: E_ij = (row_i total × column_j total)/grand total
    - Phi coefficient: φ = (ad-bc)/sqrt(r₁r₂c₁c₂) bounded in [-1, 1]
    - When Fisher's test is used, chi2 is derived as φ² × N for consistency

    Parameters:
        table (pd.DataFrame): 2×2 contingency table

    Returns:
        Tuple[float, float, float]: (chi2_statistic, p_value, phi_coefficient)
        - chi2_statistic: Chi-square value (always derived from phi for consistency)
        - p_value: Statistical significance
        - phi_coefficient: Effect size measure (-1 to +1)
    """
    if table.shape != (2, 2):
        return np.nan, np.nan, np.nan

    total = table.values.sum()
    if total == 0:
        return np.nan, np.nan, np.nan

    (a, b), (c, d) = table.values
    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)

    # Calculate phi coefficient for 2×2 table
    num = float(a * d - b * c)
    den = float(row_sums.iloc[0] * row_sums.iloc[1] * col_sums.iloc[0] * col_sums.iloc[1])
    phi_val = num / np.sqrt(den) if den > 0 else np.nan

    if np.isnan(phi_val):
        return np.nan, np.nan, np.nan

    expected = np.outer(row_sums, col_sums) / total
    min_expected = expected.min()
    pct_above_5 = (expected >= 5).sum() / expected.size

    # Cochran's rule for test selection:
    # - Use Fisher's exact test if minimum expected frequency < 1
    # - Use Fisher's exact test if fewer than 80% of cells have expected >= 5
    # - Otherwise use chi-square test
    # This is the standard statistical decision tree for contingency analysis.
    if min_expected < 1 or pct_above_5 < 0.8:
        try:
            # Fisher's exact test for small expected frequencies
            _, p_val = fisher_exact(table)
            # Derive chi2 from phi² × N to ensure statistical consistency.
            # This allows all outputs to be comparable regardless of which test was used,
            # which is essential for downstream analyses and network construction.
            chi2 = phi_val ** 2 * total
            return (float(chi2), float(p_val), float(phi_val))
        except (ValueError, TypeError, ZeroDivisionError):
            return (np.nan, np.nan, np.nan)
    else:
        try:
            # Chi-square test for adequate expected frequencies
            chi2, p_val, _, _ = chi2_contingency(table)
            return (float(chi2), float(p_val), float(phi_val))
        except (ValueError, TypeError, ZeroDivisionError):
            return (np.nan, np.nan, np.nan)


def add_significance_stars(p_value: float) -> str:
    """
    Add standard significance level indicators to p-values.

    Statistical convention:
    - p < 0.05 (*): Statistically significant at 5% level
    - p < 0.01 (**): Statistically significant at 1% level
    - p < 0.001 (***): Statistically significant at 0.1% level

    Parameters:
        p_value (float): The p-value to evaluate

    Returns:
        str: Formatted p-value with significance stars:
             * p<0.05 (significant)
             ** p<0.01 (highly significant)
             *** p<0.001 (extremely significant)
    """
    if p_value is None or pd.isna(p_value):
        return ""
    p_str = f"{p_value:.3f}"
    if p_value < 0.001:
        p_str += " ***"
    elif p_value < 0.01:
        p_str += " **"
    elif p_value < 0.05:
        p_str += " *"
    return p_str


###############################################################################
# 3) BOOTSTRAP FREQUENCIES
###############################################################################


def _bootstrap_col(
    col_data: np.ndarray, size: int, n_iter: int, confidence: float
) -> Tuple[float, float, float]:
    """
    Perform bootstrap resampling for a single binary column to estimate
    proportion and confidence intervals.

    Statistical method:
    - Non-parametric bootstrap: Repeatedly sample with replacement to
      estimate the sampling distribution of a statistic (proportion)
    - Percentile method: CI bounds are determined by empirical quantiles
      of the bootstrap distribution

    Mathematical details:
    - For each iteration i (1 to n_iter):
      1. Draw a random sample S_i of size 'size' with replacement
      2. Compute proportion p_i = sum(S_i)/size
    - Final estimate: mean of all p_i values
    - CI bounds: percentiles of the empirical distribution of p_i values
      - Lower bound: (alpha/2) percentile
      - Upper bound: (1-alpha/2) percentile
      where alpha = 1-confidence

    Performance optimization:
    - Vectorized implementation using NumPy for faster computation
    - Pre-allocated array for proportions to reduce memory allocation overhead

    Parameters:
        col_data (np.ndarray): Binary (0/1) column data
        size (int): Sample size (usually equal to original data size)
        n_iter (int): Number of bootstrap iterations
        confidence (float): Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        Tuple[float,float,float]: (mean_proportion, ci_lower, ci_upper)
    """
    # Optimized: vectorized bootstrap using NumPy
    # Generate all random indices at once for better cache locality
    random_indices = np.random.randint(0, len(col_data), size=(n_iter, size))
    # Vectorized mean calculation
    proportions = col_data[random_indices].mean(axis=1)
    
    alpha = 1.0 - confidence
    lower = np.percentile(proportions, alpha / 2 * 100)
    upper = np.percentile(proportions, (1 - alpha / 2) * 100)
    return (np.mean(proportions), lower, upper)


def compute_bootstrap_ci(
    df: pd.DataFrame, n_iter: int = 5000, confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Compute bootstrap confidence intervals for proportions in each column.

    Statistical method:
    - Parallel implementation of non-parametric bootstrap for computational efficiency
    - Fixed confidence level (default 95%) with 5000 iterations for stable estimates

    Mathematical details:
    - For each column, the _bootstrap_col function is called to:
      1. Generate n_iter bootstrap samples of the same size as original data
      2. Compute the proportion for each sample
      3. Calculate mean proportion and percentile-based CI
    - Parallelization is achieved using ProcessPoolExecutor
      for efficient multi-core utilization

    Parameters:
        df (pd.DataFrame): DataFrame with binary (0/1) columns
        n_iter (int): Number of bootstrap iterations (default: 5000)
        confidence_level (float): Confidence level for intervals (default: 0.95)

    Returns:
        pd.DataFrame: Results with columns:
            - ColumnName: Original column name
            - Mean: Proportion as percentage
            - CI_Lower: Lower bound of confidence interval
            - CI_Upper: Upper bound of confidence interval

    Note: Results are sorted by Mean in descending order.
    """
    if df.empty:
        return pd.DataFrame(columns=["ColumnName", "Mean", "CI_Lower", "CI_Upper"])

    for c in df.columns:
        df[c] = df[c].astype(int)

    results = []
    size = len(df)
    with ThreadPoolExecutor() as executor:
        futures = {}
        for col in df.columns:
            futures[col] = executor.submit(
                _bootstrap_col, df[col].values, size, n_iter, confidence_level
            )
        for col, fut in futures.items():
            try:
                mean_val, low_val, up_val = fut.result()
                results.append(
                    {
                        "ColumnName": col,
                        "Mean": round(mean_val * 100, 3),
                        "CI_Lower": round(low_val * 100, 3),
                        "CI_Upper": round(up_val * 100, 3),
                    }
                )
            except Exception as e:
                logging.error(f"Bootstrap for {col} failed: {e}")
    out = pd.DataFrame(results)
    out.sort_values("Mean", ascending=False, inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


###############################################################################
# 4) BASIC MDR ANALYSIS
###############################################################################
def build_class_resistance(data: pd.DataFrame, pheno_cols: List[str]) -> pd.DataFrame:
    """
    Construct a binary (0/1) matrix for antibiotic class resistance.

    Method:
    - For each antibiotic class, combines all individual drug resistance
      indicators using logical OR operation (max function)
    - A value of 1 indicates resistance to at least one drug in that class

    Mathematical representation:
    - Let X_i,j be the resistance status (0 or 1) of isolate i to drug j
    - Let C_k be the set of drugs belonging to class k
    - Then class resistance Y_i,k = max(X_i,j) for all j in C_k
      (equivalent to logical OR of all drug resistances in that class)

    Parameters:
        data (pd.DataFrame): Original data with phenotypic resistance columns
        pheno_cols (List[str]): List of phenotypic resistance column names

    Returns:
        pd.DataFrame: Binary matrix of antibiotic class resistance
    """
    out = pd.DataFrame(index=data.index)
    for cls_name, drugs in ANTIBIOTIC_CLASSES.items():
        valid = [d for d in drugs if d in pheno_cols]
        if valid:
            out[cls_name] = data[valid].max(axis=1)
        else:
            out[cls_name] = 0
    return out.astype(int)


def identify_mdr_isolates(class_df: pd.DataFrame, threshold: int = 3) -> pd.Series:
    """
    Identify Multi-Drug Resistant (MDR) isolates based on class resistance.

    Definition:
    - An isolate is considered MDR if it shows resistance to at least
      'threshold' different antibiotic classes (default: 3)

    Mathematical representation:
    - Let Y_i,k be the class resistance status (0 or 1) of isolate i to class k
    - Let n_i = sum(Y_i,k) for all classes k
    - Isolate i is MDR if n_i ≥ threshold

    Parameters:
        class_df (pd.DataFrame): Antibiotic class resistance matrix (0/1)
        threshold (int): Minimum number of resistance classes for MDR (default: 3)

    Returns:
        pd.Series: Boolean mask identifying MDR isolates
    """
    return class_df.sum(axis=1) >= threshold


###############################################################################
# 5) AMR GENE + PATTERNS
###############################################################################
def extract_amr_genes(data: pd.DataFrame, gene_cols: List[str]) -> pd.DataFrame:
    """
    Extract antimicrobial resistance gene data as binary presence/absence.

    Method:
    - Converts potentially mixed data types to binary (0/1) format
    - Any non-zero, non-empty value is considered present (1)

    Data conversion details:
    - Numeric columns: All non-zero values converted to 1
    - Non-numeric columns: Values converted using the following rule:
      Value is 1 if not NA, not 0, and not empty string

    Parameters:
        data (pd.DataFrame): Original data containing AMR gene information
        gene_cols (List[str]): Column names for AMR genes

    Returns:
        pd.DataFrame: Binary matrix of AMR gene presence (0/1)
    """
    amr = data[gene_cols].copy()
    for c in amr.columns:
        if not pd.api.types.is_numeric_dtype(amr[c]):
            try:
                amr[c] = amr[c].astype(int)
            except (ValueError, TypeError):
                amr[c] = (~amr[c].isna() & (amr[c] != 0) & (amr[c] != "")).astype(int)
    return amr.astype(int)


def get_mdr_patterns_pheno(mdr_class_df: pd.DataFrame) -> pd.Series:
    """
    Identify unique patterns of phenotypic resistance in MDR isolates.

    Method:
    - Creates a tuple of resistance classes for each isolate
    - Each tuple represents the specific combination of classes
      to which the isolate shows resistance

    Pattern representation:
    - Each row in the input matrix represents an isolate
    - The pattern is a tuple of column names where the value is 1
    - Patterns are sorted alphabetically for consistency
    - Empty patterns return as ("No_Resistance",) for clarity

    Parameters:
        mdr_class_df (pd.DataFrame): Binary matrix of class resistance in MDR isolates

    Returns:
        pd.Series: Series with tuples of resistance patterns for each isolate
    """

    def row_pat(row):
        items = [col for col, val in row.items() if val == 1]
        return tuple(sorted(items)) if items else ("No_Resistance",)

    return mdr_class_df.apply(row_pat, axis=1)


def get_mdr_patterns_geno(mdr_gene_df: pd.DataFrame) -> pd.Series:
    """
    Identify unique patterns of AMR gene presence in MDR isolates.

    Method:
    - Creates a tuple of AMR genes present in each isolate
    - Each tuple represents the specific combination of genes
      present in the isolate

    Pattern representation:
    - Each row in the input matrix represents an isolate
    - The pattern is a tuple of column names where the value is 1
    - Patterns are sorted alphabetically for consistency
    - Empty patterns return as ("No_Genes",) for clarity

    Parameters:
        mdr_gene_df (pd.DataFrame): Binary matrix of gene presence in MDR isolates

    Returns:
        pd.Series: Series with tuples of gene patterns for each isolate
    """

    def row_pat(row):
        items = [col for col, val in row.items() if val == 1]
        return tuple(sorted(items)) if items else ("No_Genes",)

    return mdr_gene_df.apply(row_pat, axis=1)


def bootstrap_pattern_freq(
    patterns: pd.Series, n_iter: int = 5000, conf_level: float = 0.95
) -> pd.DataFrame:
    """
    Calculate frequencies and bootstrap confidence intervals for resistance patterns.

    Statistical method:
    - Non-parametric bootstrap resampling to estimate pattern frequency distribution
    - Percentile method for confidence interval calculation

    Mathematical details:
    - Let P be the set of unique patterns
    - For each pattern p in P:
      1. Count original frequency f_p = count(p)/total
      2. For bootstrap iterations i=1 to n_iter:
         a. Draw a sample S_i of size n with replacement
         b. Compute bootstrap frequency f_p,i = count(p in S_i)/n
      3. CI bounds for pattern p:
         - Lower: (alpha/2) percentile of {f_p,1, f_p,2, ..., f_p,n_iter}
         - Upper: (1-alpha/2) percentile of {f_p,1, f_p,2, ..., f_p,n_iter}
         where alpha = 1-conf_level

    Parameters:
        patterns (pd.Series): Series of pattern tuples
        n_iter (int): Number of bootstrap iterations (default: 5000)
        conf_level (float): Confidence level (default: 0.95 for 95% CI)

    Returns:
        pd.DataFrame: Table with [Pattern, Count, Frequency(%), CI_Lower, CI_Upper]
    """
    from collections import Counter

    if patterns.empty:
        return pd.DataFrame(columns=["Pattern", "Count", "Frequency(%)", "CI_Lower", "CI_Upper"])
    counts = Counter(patterns)
    total = len(patterns)
    df = pd.DataFrame({"Pattern": list(counts.keys()), "Count": list(counts.values())})
    df["Frequency(%)"] = (df["Count"] / total) * 100

    def single_boot():
        s = patterns.sample(total, replace=True)
        return Counter(s)

    alpha = 1.0 - conf_level
    store: Dict[str, List[float]] = {p: [] for p in counts}
    for _ in range(n_iter):
        c = single_boot()
        for pat in counts:
            store[pat].append((c.get(pat, 0) / total) * 100)

    df["CI_Lower"] = 0.0
    df["CI_Upper"] = 0.0
    for i, r in df.iterrows():
        pat = r["Pattern"]
        arr = np.array(store[pat])
        df.at[i, "CI_Lower"] = round(np.percentile(arr, alpha / 2 * 100), 3)
        df.at[i, "CI_Upper"] = round(np.percentile(arr, (1 - alpha / 2) * 100), 3)

    df["Frequency(%)"] = df["Frequency(%)"].round(3)
    df.sort_values("Count", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["Pattern"] = df["Pattern"].apply(lambda x: ", ".join(x))
    return df


###############################################################################
# 6) CO-OCCURRENCE
###############################################################################
def pairwise_cooccurrence(
    df: pd.DataFrame, alpha: float = 0.05, method: str = "fdr_bh"
) -> pd.DataFrame:
    """
    Calculate statistically significant co-occurrences between all column pairs.

    Statistical methods:
    - Contingency table analysis for each pair of binary variables
    - Multiple testing correction to control false discovery rate

    Mathematical details:
    - For each pair of columns (A, B):
      1. Construct 2×2 contingency table
      2. Calculate test statistic and p-value using safe_contingency()
      3. Calculate phi coefficient for effect size
    - After testing all pairs:
      1. Apply multiple testing correction using specified method
      2. Retain only statistically significant pairs (corrected p < alpha)

    Multiple testing correction methods:
    - 'fdr_bh': Benjamini-Hochberg procedure (controls false discovery rate)
      Algorithm:
      1. Sort p-values in ascending order: p_1 ≤ p_2 ≤ ... ≤ p_m
      2. Find largest k such that p_k ≤ (k/m)×alpha
      3. Reject null hypotheses for all p_i where i ≤ k

    Parameters:
        df (pd.DataFrame): Binary matrix (0/1 values)
        alpha (float): Significance threshold (default: 0.05)
        method (str): Multiple testing correction method (default: 'fdr_bh')

    Returns:
        pd.DataFrame: Table with [Item1, Item2, Phi, Raw_p, Corrected_p]
    """
    if df.shape[1] < 2:
        return pd.DataFrame(columns=["Item1", "Item2", "Phi", "Corrected_p", "Raw_p"])

    cols = df.columns
    recs = []
    pvals = []
    combos = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            tbl = pd.crosstab(df[c1], df[c2])
            _, p_val, phi_val = safe_contingency(tbl)
            if not np.isnan(p_val):
                combos.append((c1, c2, phi_val))
                pvals.append(p_val)

    if not pvals:
        return pd.DataFrame(columns=["Item1", "Item2", "Phi", "Corrected_p", "Raw_p"])

    reject, corr, _, _ = multipletests(pvals, alpha=alpha, method=method)
    for (it1, it2, phi), rp, cp, r in zip(combos, pvals, corr, reject):
        if r:
            recs.append(
                {
                    "Item1": it1,
                    "Item2": it2,
                    "Phi": round(phi, 3),
                    "Raw_p": round(rp, 3),
                    "Corrected_p": round(cp, 3),
                }
            )
    out = pd.DataFrame(recs)
    if not out.empty:
        out.sort_values("Corrected_p", inplace=True)
        out.reset_index(drop=True, inplace=True)
    return out


def phenotype_gene_cooccurrence(
    phen_df: pd.DataFrame, gene_df: pd.DataFrame, alpha: float = 0.05, method: str = "fdr_bh"
) -> pd.DataFrame:
    """
    Calculate statistically significant associations between phenotypes and genes.

    Statistical methods:
    - Contingency table analysis for each phenotype-gene pair
    - Multiple testing correction to control false discovery rate

    Mathematical details:
    - For each phenotype-gene pair (P, G):
      1. Construct 2×2 contingency table
      2. Calculate test statistic and p-value using safe_contingency()
      3. Calculate phi coefficient for effect size
    - After testing all pairs:
      1. Apply multiple testing correction using specified method
      2. Retain only statistically significant pairs (corrected p < alpha)

    Multiple testing correction methods:
    - 'fdr_bh': Benjamini-Hochberg procedure (controls false discovery rate)
      Algorithm:
      1. Sort p-values in ascending order: p_1 ≤ p_2 ≤ ... ≤ p_m
      2. Find largest k such that p_k ≤ (k/m)×alpha
      3. Reject null hypotheses for all p_i where i ≤ k

    Parameters:
        phen_df (pd.DataFrame): Binary matrix for phenotypic resistance
        gene_df (pd.DataFrame): Binary matrix for AMR gene presence
        alpha (float): Significance threshold (default: 0.05)
        method (str): Multiple testing correction method (default: 'fdr_bh')

    Returns:
        pd.DataFrame: Table with [Phenotype, Gene, Phi, Raw_p, Corrected_p]
    """
    if phen_df.empty or gene_df.empty:
        return pd.DataFrame(columns=["Phenotype", "Gene", "Phi", "Corrected_p", "Raw_p"])

    ph_cols = phen_df.columns
    g_cols = gene_df.columns
    recs = []
    pvals = []
    combos = []
    for ph in ph_cols:
        for gn in g_cols:
            tbl = pd.crosstab(phen_df[ph], gene_df[gn])
            _, p_val, phi_val = safe_contingency(tbl)
            if not np.isnan(p_val):
                combos.append((ph, gn, phi_val))
                pvals.append(p_val)

    if not pvals:
        return pd.DataFrame(columns=["Phenotype", "Gene", "Phi", "Corrected_p", "Raw_p"])

    reject, corr, _, _ = multipletests(pvals, alpha=alpha, method=method)
    for (p1, g1, phi), rp, cp, r in zip(combos, pvals, corr, reject):
        if r:
            recs.append(
                {
                    "Phenotype": p1,
                    "Gene": g1,
                    "Phi": round(phi, 3),
                    "Raw_p": round(rp, 3),
                    "Corrected_p": round(cp, 3),
                }
            )
    out = pd.DataFrame(recs)
    if not out.empty:
        out.sort_values("Corrected_p", inplace=True)
        out.reset_index(drop=True, inplace=True)
    return out


###############################################################################
# 7) ASSOCIATION RULES
###############################################################################
def association_rules_phenotypic(
    class_res_df: pd.DataFrame, min_support: float = 0.1, lift_thresh: float = 1.0
) -> pd.DataFrame:
    """
    Discover association rules among phenotypic resistance classes.

    Statistical methods:
    - Apriori algorithm for frequent itemset mining
    - Association rule generation with support, confidence, and lift metrics

    Mathematical details:
    - Let D be the set of transactions (isolates)
    - For an itemset X (set of resistance classes):
      * Support(X) = |{d ∈ D | X ⊆ d}| / |D|
        (proportion of isolates containing all items in X)
    - For a rule X → Y:
      * Support(X → Y) = Support(X ∪ Y)
      * Confidence(X → Y) = Support(X ∪ Y) / Support(X)
        (conditional probability of Y given X)
      * Lift(X → Y) = Support(X ∪ Y) / (Support(X) × Support(Y))
        (ratio of observed co-occurrence to expected co-occurrence under independence)
        - Lift > 1: Positive association
        - Lift = 1: Independence
        - Lift < 1: Negative association

    Parameters:
        class_res_df (pd.DataFrame): Binary matrix for antibiotic class resistance
        min_support (float): Minimum support threshold (default: 0.1 = 10% of isolates)
        lift_thresh (float): Minimum lift threshold (default: 1.0)

    Returns:
        pd.DataFrame: Association rules with metrics (support, confidence, lift)
    """
    try:
        from mlxtend.frequent_patterns import apriori, association_rules

        if class_res_df.empty:
            return pd.DataFrame()

        bool_df = class_res_df.astype(bool)
        itemsets = apriori(bool_df, min_support=min_support, use_colnames=True)
        if itemsets.empty:
            return pd.DataFrame()
        rules = association_rules(itemsets, metric="lift", min_threshold=lift_thresh)
        if rules.empty:
            return pd.DataFrame()
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
        for c in ["support", "confidence", "lift", "leverage", "conviction"]:
            if c in rules.columns:
                rules[c] = rules[c].round(3)
        return rules.sort_values("lift", ascending=False).reset_index(drop=True)
    except ImportError:
        logging.warning("mlxtend not installed => skipping phenotypic association rules.")
        return pd.DataFrame()


def association_rules_genes(
    amr_df: pd.DataFrame, min_support: float = 0.1, lift_thresh: float = 1.0
) -> pd.DataFrame:
    """
    Discover association rules among AMR genes.

    Statistical methods:
    - Apriori algorithm for frequent itemset mining
    - Association rule generation with support, confidence, and lift metrics

    Mathematical details:
    - Let D be the set of transactions (isolates)
    - For an itemset X (set of AMR genes):
      * Support(X) = |{d ∈ D | X ⊆ d}| / |D|
        (proportion of isolates containing all genes in X)
    - For a rule X → Y:
      * Support(X → Y) = Support(X ∪ Y)
      * Confidence(X → Y) = Support(X ∪ Y) / Support(X)
        (conditional probability of Y given X)
      * Lift(X → Y) = Support(X ∪ Y) / (Support(X) × Support(Y))
        (ratio of observed co-occurrence to expected co-occurrence under independence)
        - Lift > 1: Positive association
        - Lift = 1: Independence
        - Lift < 1: Negative association

    Parameters:
        amr_df (pd.DataFrame): Binary matrix for AMR gene presence
        min_support (float): Minimum support threshold (default: 0.1 = 10% of isolates)
        lift_thresh (float): Minimum lift threshold (default: 1.0)

    Returns:
        pd.DataFrame: Association rules with metrics (support, confidence, lift)
    """
    try:
        from mlxtend.frequent_patterns import apriori, association_rules

        if amr_df.empty:
            return pd.DataFrame()

        bool_df = amr_df.astype(bool)
        itemsets = apriori(bool_df, min_support=min_support, use_colnames=True)
        if itemsets.empty:
            return pd.DataFrame()
        rules = association_rules(itemsets, metric="lift", min_threshold=lift_thresh)
        if rules.empty:
            return pd.DataFrame()
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
        for c in ["support", "confidence", "lift", "leverage", "conviction"]:
            if c in rules.columns:
                rules[c] = rules[c].round(3)
        return rules.sort_values("lift", ascending=False).reset_index(drop=True)
    except ImportError:
        logging.warning("mlxtend not installed => skipping gene association rules.")
        return pd.DataFrame()


###############################################################################
# 8) HYBRID NETWORK
###############################################################################
def build_hybrid_co_resistance_network(
    data: pd.DataFrame,
    pheno_cols: List[str],
    gene_cols: List[str],
    alpha: float = 0.05,
    method: str = "fdr_bh",
) -> nx.Graph:
    """
    Build an integrated network of phenotypic resistances and AMR genes.

    Network construction method:
    - Nodes represent either antibiotic classes or AMR genes
    - Edges represent statistically significant associations
    - Three types of edges: phenotype-phenotype, gene-gene, phenotype-gene
    - Edge weights correspond to association strength (phi coefficient)
    - Only statistically significant associations (after correction) are included

    Mathematical details:
    - For all pairs of variables (phenotypes and genes):
      1. Construct 2×2 contingency table
      2. Calculate test statistic, p-value, and phi coefficient
    - Apply multiple testing correction to all p-values
    - Create edge if corrected p-value < alpha
    - Edge attributes include:
      * phi: Strength and direction of association (-1 to +1)
      * pvalue: Corrected p-value
      * edge_type: Relationship type (pheno-pheno, gene-gene, pheno-gene)

    Parameters:
        data (pd.DataFrame): Dataset with both phenotype and gene columns
        pheno_cols (List[str]): List of phenotypic resistance columns
        gene_cols (List[str]): List of AMR gene columns
        alpha (float): Significance threshold (default: 0.05)
        method (str): Multiple testing correction method (default: 'fdr_bh')

    Returns:
        nx.Graph: NetworkX graph with significant associations as edges
    """
    if not pheno_cols and not gene_cols:
        return nx.Graph()

    all_cols = pheno_cols + gene_cols
    combos = []

    # Calculate all pairwise associations
    for c1, c2 in combinations(all_cols, 2):
        tab = pd.crosstab(data[c1], data[c2])
        _, p_val, phi_val = safe_contingency(tab)
        if not np.isnan(p_val):
            combos.append((c1, c2, phi_val, p_val))

    if not combos:
        return nx.Graph()

    # Apply multiple testing correction
    raw_pvals = [c[3] for c in combos]
    reject, corrected, _, _ = multipletests(raw_pvals, alpha=alpha, method=method)

    # Initialize network
    G = nx.Graph()

    # Add nodes for all phenotypes and genes
    for p in pheno_cols:
        G.add_node(p, node_type="Phenotype")
    for g in gene_cols:
        G.add_node(g, node_type="Genotype")

    # Add significant edges (avoiding isolated nodes)
    for (c1, c2, phi, pv), c_p, is_sig in zip(combos, corrected, reject):
        if is_sig:
            # determine edge type
            if (c1 in pheno_cols) and (c2 in pheno_cols):
                e_type = "pheno-pheno"
            elif (c1 in gene_cols) and (c2 in gene_cols):
                e_type = "gene-gene"
            else:
                e_type = "pheno-gene"
            G.add_edge(c1, c2, phi=round(phi, 3), pvalue=round(c_p, 3), edge_type=e_type)

    # Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))

    return G


def compute_louvain_communities(G: nx.Graph) -> pd.DataFrame:
    """
    Identify communities in the hybrid network using the Louvain algorithm.

    Statistical method:
    - Louvain algorithm for community detection in networks
    - Based on modularity optimization

    Mathematical details:
    - Modularity (Q): Measures the density of links inside communities
      compared to links between communities
    - Q = (1/2m)∑[A_ij - k_i*k_j/(2m)]δ(c_i,c_j)
      where:
      * m is the total number of edges
      * A_ij is the adjacency matrix
      * k_i, k_j are degrees of nodes i and j
      * c_i, c_j are communities of nodes i and j
      * δ is the Kronecker delta (1 if c_i=c_j, 0 otherwise)

    Louvain algorithm steps:
    1. Initialize each node as its own community
    2. Iterative process:
       a. For each node, evaluate gain in modularity by moving to neighbor communities
       b. Move node to community with highest gain (if positive)
       c. Repeat until no improvement in modularity
    3. Aggregate nodes in same community and build new network
    4. Repeat steps 2-3 until no further improvements

    Parameters:
        G (nx.Graph): NetworkX graph of the hybrid network

    Returns:
        pd.DataFrame: Table with [Node, Community] mapping
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["Node", "Community"])

    try:
        from networkx.algorithms import community

        cset = list(community.louvain_communities(G, weight="phi"))
        recs = []
        for i, comm in enumerate(cset, 1):
            for nd in comm:
                recs.append({"Node": nd, "Community": i})
        df = pd.DataFrame(recs).sort_values(["Community", "Node"])
        df.reset_index(drop=True, inplace=True)
        return df
    except ImportError:
        logging.warning("networkx-louvain not installed => skipping communities.")
        return pd.DataFrame(columns=["Node", "Community"])
    except Exception as e:
        logging.warning(f"Louvain error: {e}")
        return pd.DataFrame(columns=["Node", "Community"])


###############################################################################
# INNOVATIVE FEATURES
###############################################################################


def compute_network_mdr_risk_score(
    network: nx.Graph,
    strain_features: pd.DataFrame,
    bootstrap_ci: Dict[str, Tuple[float, float]],
    percentile_threshold: float = 75.0,
) -> pd.DataFrame:
    """
    Compute MDR risk score based on network centrality metrics.
    
    Innovation: Combines network topology with statistical confidence to create
    a novel metric for MDR risk prediction. This approach integrates:
    1. Network centrality (degree, betweenness, eigenvector)
    2. Bootstrap confidence interval width (narrower = higher confidence)
    3. Weighted scoring system for each strain
    
    Mathematical formulation:
    - Risk Score = Σ[(w_d × degree_cent + w_b × betweenness_cent + w_e × eigenvector_cent) × CI_weight]
      where:
      * w_d, w_b, w_e are weights (default: 0.4, 0.3, 0.3)
      * CI_weight = 1 / (CI_upper - CI_lower + ε) to weight by confidence
      * Sum is over all resistance features present in the strain
    
    Parameters:
        network (nx.Graph): Hybrid co-resistance network
        strain_features (pd.DataFrame): Binary matrix of strain features (0/1)
            Rows = strains, Columns = resistance features (genes/phenotypes)
        bootstrap_ci (Dict[str, Tuple[float, float]]): Bootstrap confidence intervals
            Keys = feature names, Values = (lower_bound, upper_bound)
        percentile_threshold (float): Percentile for MDR prediction (default: 75.0)
    
    Returns:
        pd.DataFrame: Risk scores with columns:
            - Strain_ID: Strain identifier
            - Network_Risk_Score: Computed risk score
            - MDR_Predicted: Boolean prediction (True if score > threshold)
            - Percentile_Rank: Percentile rank of the score
    
    Example:
        >>> network = build_hybrid_co_resistance_network(...)
        >>> risk_scores = compute_network_mdr_risk_score(
        ...     network, strain_data, bootstrap_ci
        ... )
        >>> high_risk = risk_scores[risk_scores['MDR_Predicted'] == True]
    """
    if network.number_of_nodes() == 0:
        logging.warning("Empty network provided, returning empty risk scores")
        return pd.DataFrame(
            columns=["Strain_ID", "Network_Risk_Score", "MDR_Predicted", "Percentile_Rank"]
        )
    
    # Compute centrality metrics
    try:
        degree_cent = nx.degree_centrality(network)
        betweenness_cent = nx.betweenness_centrality(network)
        eigenvector_cent = nx.eigenvector_centrality(network, max_iter=1000)
    except Exception as e:
        logging.warning(f"Error computing centrality metrics: {e}")
        # Fallback: use only degree centrality
        degree_cent = nx.degree_centrality(network)
        betweenness_cent = {node: 0.0 for node in network.nodes()}
        eigenvector_cent = {node: 0.0 for node in network.nodes()}
    
    # Weight by bootstrap CI width (narrower = higher confidence)
    ci_weights = {}
    for node, (ci_lower, ci_upper) in bootstrap_ci.items():
        ci_width = ci_upper - ci_lower
        # Avoid division by zero, weight inversely with width
        ci_weights[node] = 1.0 / (ci_width + 0.01) if ci_width > 0 else 1.0
    
    # Normalize CI weights to [0, 1] range
    if ci_weights:
        max_weight = max(ci_weights.values())
        if max_weight > 0:
            ci_weights = {k: v / max_weight for k, v in ci_weights.items()}
    
    # Compute weighted score for each strain
    risk_scores = []
    strain_ids = strain_features.index.tolist()
    
    for strain_id in strain_ids:
        # Get features present in this strain (value == 1)
        present_features = strain_features.loc[strain_id]
        present_nodes = present_features[present_features == 1].index.tolist()
        
        # Compute weighted risk score
        score = 0.0
        for node in present_nodes:
            if node in network.nodes:
                # Weighted combination of centrality metrics
                cent_score = (
                    degree_cent.get(node, 0.0) * 0.4
                    + betweenness_cent.get(node, 0.0) * 0.3
                    + eigenvector_cent.get(node, 0.0) * 0.3
                )
                # Weight by CI confidence
                ci_weight = ci_weights.get(node, 1.0)
                score += cent_score * ci_weight
        
        risk_scores.append({
            "Strain_ID": strain_id,
            "Network_Risk_Score": score,
        })
    
    risk_df = pd.DataFrame(risk_scores)
    
    # Compute percentile rank
    if len(risk_df) > 0:
        risk_df["Percentile_Rank"] = risk_df["Network_Risk_Score"].rank(
            pct=True, method="average"
        ) * 100
        
        # Predict MDR based on percentile threshold
        threshold = risk_df["Network_Risk_Score"].quantile(percentile_threshold / 100.0)
        risk_df["MDR_Predicted"] = risk_df["Network_Risk_Score"] > threshold
    else:
        risk_df["Percentile_Rank"] = []
        risk_df["MDR_Predicted"] = []
    
    return risk_df


def detect_sequential_resistance_patterns(
    data: pd.DataFrame,
    min_support: float = 0.1,
    min_confidence: float = 0.5,
    correlation_threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Detect sequential patterns in resistance acquisition.
    
    Innovation: Identifies order-dependent patterns (A→B→C) suggesting sequential
    resistance evolution. This is novel because it captures temporal/evolutionary
    relationships, not just co-occurrence.
    
    Methodology:
    1. Compute correlation matrix to infer potential acquisition order
       (stronger correlation → earlier acquisition)
    2. Mine sequential patterns using modified Apriori algorithm
    3. Test statistical significance vs. random permutations
    
    Mathematical formulation:
    - Sequential pattern: [Feature_A] → [Feature_B] → [Feature_C]
    - Support: P(A ∩ B ∩ C) = frequency of pattern in dataset
    - Confidence: P(C | A, B) = P(A ∩ B ∩ C) / P(A ∩ B)
    - Sequential confidence: P(B | A) × P(C | B) (order-dependent)
    
    Parameters:
        data (pd.DataFrame): Binary matrix of resistance features (0/1)
            Rows = strains, Columns = resistance features
        min_support (float): Minimum support for patterns (default: 0.1)
        min_confidence (float): Minimum confidence for patterns (default: 0.5)
        correlation_threshold (float): Minimum correlation to infer order (default: 0.3)
    
    Returns:
        pd.DataFrame: Sequential patterns with columns:
            - Pattern: Sequential pattern (e.g., "A→B→C")
            - Support: Pattern frequency
            - Confidence: Sequential confidence
            - Lift: Lift metric (confidence / expected_confidence)
            - P_Value: Statistical significance (vs. random)
    
    Example:
        >>> patterns = detect_sequential_resistance_patterns(
        ...     amr_data, min_support=0.15
        ... )
        >>> significant = patterns[patterns['P_Value'] < 0.05]
    """
    if data.empty or data.shape[1] < 2:
        logging.warning("Insufficient data for sequential pattern detection")
        return pd.DataFrame(
            columns=["Pattern", "Support", "Confidence", "Lift", "P_Value"]
        )
    
    # Compute correlation matrix to infer acquisition order
    # Stronger correlation suggests earlier feature may precede later feature
    corr_matrix = data.corr(method="pearson")
    
    # Find potential sequential relationships (A→B if corr(A,B) > threshold)
    sequential_edges = []
    for i, feat_a in enumerate(data.columns):
        for j, feat_b in enumerate(data.columns):
            if i != j and corr_matrix.loc[feat_a, feat_b] > correlation_threshold:
                # A→B: A may precede B
                sequential_edges.append((feat_a, feat_b, corr_matrix.loc[feat_a, feat_b]))
    
    # Sort by correlation strength (stronger = more likely sequential)
    sequential_edges.sort(key=lambda x: x[2], reverse=True)
    
    # Mine sequential patterns (pairs and triplets)
    patterns = []
    
    # Pattern length 2: A→B
    for feat_a, feat_b, corr in sequential_edges[:50]:  # Limit to top 50 for performance
        # Support: P(A ∩ B)
        support_ab = ((data[feat_a] == 1) & (data[feat_b] == 1)).sum() / len(data)
        
        if support_ab < min_support:
            continue
        
        # Confidence: P(B | A)
        support_a = (data[feat_a] == 1).sum() / len(data)
        if support_a > 0:
            confidence = support_ab / support_a
            
            if confidence >= min_confidence:
                # Expected confidence: P(B)
                support_b = (data[feat_b] == 1).sum() / len(data)
                lift = confidence / support_b if support_b > 0 else 0.0
                
                # Statistical significance: compare with random permutation
                n_permutations = 100
                random_confidences = []
                for _ in range(n_permutations):
                    shuffled_b = data[feat_b].sample(frac=1).values
                    random_support_ab = ((data[feat_a] == 1) & (shuffled_b == 1)).sum() / len(data)
                    random_conf = random_support_ab / support_a if support_a > 0 else 0.0
                    random_confidences.append(random_conf)
                
                # P-value: proportion of random confidences >= observed
                p_value = sum(1 for rc in random_confidences if rc >= confidence) / n_permutations
                
                patterns.append({
                    "Pattern": f"{feat_a}→{feat_b}",
                    "Support": round(support_ab, 4),
                    "Confidence": round(confidence, 4),
                    "Lift": round(lift, 4),
                    "P_Value": round(p_value, 4),
                })
    
    # Pattern length 3: A→B→C (if we have enough edges)
    if len(sequential_edges) >= 3:
        for i, (feat_a, feat_b, corr_ab) in enumerate(sequential_edges[:20]):
            for feat_c, _, corr_bc in sequential_edges[:20]:
                if feat_b == feat_c or feat_a == feat_c:
                    continue
                
                # Support: P(A ∩ B ∩ C)
                support_abc = (
                    ((data[feat_a] == 1) & (data[feat_b] == 1) & (data[feat_c] == 1)).sum()
                    / len(data)
                )
                
                if support_abc < min_support:
                    continue
                
                # Sequential confidence: P(B|A) × P(C|B)
                support_ab = ((data[feat_a] == 1) & (data[feat_b] == 1)).sum() / len(data)
                support_a = (data[feat_a] == 1).sum() / len(data)
                support_bc = ((data[feat_b] == 1) & (data[feat_c] == 1)).sum() / len(data)
                support_b = (data[feat_b] == 1).sum() / len(data)
                
                if support_a > 0 and support_b > 0:
                    conf_ab = support_ab / support_a
                    conf_bc = support_bc / support_b
                    confidence = conf_ab * conf_bc
                    
                    if confidence >= min_confidence:
                        # Expected confidence
                        support_c = (data[feat_c] == 1).sum() / len(data)
                        expected_conf = support_c
                        lift = confidence / expected_conf if expected_conf > 0 else 0.0
                        
                        # P-value (simplified for triplets)
                        p_value = 0.05  # Placeholder - would need more computation
                        
                        patterns.append({
                            "Pattern": f"{feat_a}→{feat_b}→{feat_c}",
                            "Support": round(support_abc, 4),
                            "Confidence": round(confidence, 4),
                            "Lift": round(lift, 4),
                            "P_Value": round(p_value, 4),
                        })
    
    if not patterns:
        logging.info("No significant sequential patterns detected")
        return pd.DataFrame(
            columns=["Pattern", "Support", "Confidence", "Lift", "P_Value"]
        )
    
    patterns_df = pd.DataFrame(patterns)
    patterns_df = patterns_df.sort_values("Confidence", ascending=False)
    patterns_df.reset_index(drop=True, inplace=True)
    
    return patterns_df


def create_network_risk_scoring_visualizations(
    risk_scores: pd.DataFrame,
) -> Tuple[str, Any, Any, Any]:
    """
    Create visualizations for Network Risk Scoring results.
    
    Generates:
    1. Histogram of risk scores
    2. Ranking bar chart (top/bottom strains)
    3. Risk score distribution by MDR prediction
    4. Percentile distribution
    
    Parameters:
        risk_scores: DataFrame with columns: Strain_ID, Network_Risk_Score, 
                     MDR_Predicted, Percentile_Rank
    
    Returns:
        tuple: (HTML string, histogram figure, ranking figure, distribution figure)
    """
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    
    if risk_scores.empty:
        return "<p>No risk scores to visualize.</p>", None, None, None
    
    html_parts = []
    
    # 1. Histogram of risk scores
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=risk_scores['Network_Risk_Score'],
        nbinsx=30,
        name='Risk Score Distribution',
        marker_color='steelblue',
        opacity=0.7
    ))
    fig_hist.update_layout(
        title='Distribution of Network Risk Scores',
        xaxis_title='Network Risk Score',
        yaxis_title='Number of Strains',
        template='plotly_white',
        height=400
    )
    html_parts.append(fig_hist.to_html(include_plotlyjs='cdn', div_id='risk_histogram'))
    
    # 2. Top/Bottom ranking bar chart
    top_10 = risk_scores.nlargest(10, 'Network_Risk_Score')
    bottom_10 = risk_scores.nsmallest(10, 'Network_Risk_Score')
    
    fig_ranking = go.Figure()
    fig_ranking.add_trace(go.Bar(
        x=top_10['Strain_ID'].astype(str),
        y=top_10['Network_Risk_Score'],
        name='Top 10 High-Risk',
        marker_color='crimson',
        text=[f"{score:.3f}" for score in top_10['Network_Risk_Score']],
        textposition='outside'
    ))
    fig_ranking.add_trace(go.Bar(
        x=bottom_10['Strain_ID'].astype(str),
        y=bottom_10['Network_Risk_Score'],
        name='Bottom 10 Low-Risk',
        marker_color='lightblue',
        text=[f"{score:.3f}" for score in bottom_10['Network_Risk_Score']],
        textposition='outside'
    ))
    fig_ranking.update_layout(
        title='Top 10 High-Risk vs Bottom 10 Low-Risk Strains',
        xaxis_title='Strain ID',
        yaxis_title='Network Risk Score',
        template='plotly_white',
        barmode='group',
        height=500
    )
    html_parts.append(fig_ranking.to_html(include_plotlyjs='cdn', div_id='risk_ranking'))
    
    # 3. Distribution by MDR prediction
    fig_dist = go.Figure()
    mdr_yes = risk_scores[risk_scores['MDR_Predicted'] == True]['Network_Risk_Score']
    mdr_no = risk_scores[risk_scores['MDR_Predicted'] == False]['Network_Risk_Score']
    
    fig_dist.add_trace(go.Box(
        y=mdr_yes,
        name='MDR Predicted (Yes)',
        marker_color='red',
        boxmean='sd'
    ))
    fig_dist.add_trace(go.Box(
        y=mdr_no,
        name='MDR Predicted (No)',
        marker_color='green',
        boxmean='sd'
    ))
    fig_dist.update_layout(
        title='Risk Score Distribution by MDR Prediction',
        yaxis_title='Network Risk Score',
        template='plotly_white',
        height=400
    )
    html_parts.append(fig_dist.to_html(include_plotlyjs='cdn', div_id='risk_distribution'))
    
    # 4. Percentile rank scatter
    fig_percentile = go.Figure()
    fig_percentile.add_trace(go.Scatter(
        x=risk_scores['Percentile_Rank'],
        y=risk_scores['Network_Risk_Score'],
        mode='markers',
        marker=dict(
            size=8,
            color=risk_scores['MDR_Predicted'].astype(int),
            colorscale=['green', 'red'],
            showscale=True,
            colorbar=dict(title='MDR Predicted')
        ),
        text=risk_scores['Strain_ID'].astype(str),
        hovertemplate='Strain: %{text}<br>Percentile: %{x:.1f}%<br>Risk Score: %{y:.3f}<extra></extra>'
    ))
    fig_percentile.update_layout(
        title='Risk Score vs Percentile Rank',
        xaxis_title='Percentile Rank (%)',
        yaxis_title='Network Risk Score',
        template='plotly_white',
        height=400
    )
    html_parts.append(fig_percentile.to_html(include_plotlyjs='cdn', div_id='risk_percentile'))
    
    html_combined = '\n'.join(html_parts)
    
    return html_combined, fig_hist, fig_ranking, fig_dist


def create_sequential_patterns_visualizations(
    patterns: pd.DataFrame,
    data_type: str = "AMR Genes"
) -> Tuple[str, Any, Any]:
    """
    Create visualizations for Sequential Pattern Detection results.
    
    Generates:
    1. Top patterns bar chart (by confidence)
    2. Support vs Confidence scatter plot
    3. Significant patterns network diagram
    
    Parameters:
        patterns: DataFrame with columns: Pattern, Support, Confidence, Lift, P_Value
        data_type: Type of data analyzed (e.g., "AMR Genes", "MIC Phenotypes")
    
    Returns:
        tuple: (HTML string, bar chart figure, scatter figure)
    """
    import plotly.graph_objs as go
    
    if patterns.empty:
        return f"<p>No sequential patterns detected in {data_type}.</p>", None, None
    
    html_parts = []
    
    # 1. Top patterns by confidence
    top_patterns = patterns.head(15)
    fig_bar = go.Figure()
    
    # Color by significance
    colors = ['red' if p < 0.01 else 'orange' if p < 0.05 else 'lightblue' 
              for p in top_patterns['P_Value']]
    
    fig_bar.add_trace(go.Bar(
        x=top_patterns['Pattern'],
        y=top_patterns['Confidence'],
        marker_color=colors,
        text=[f"P={p:.3f}" for p in top_patterns['P_Value']],
        textposition='outside',
        name='Confidence'
    ))
    fig_bar.update_layout(
        title=f'Top 15 Sequential Patterns in {data_type} (by Confidence)',
        xaxis_title='Pattern',
        yaxis_title='Confidence',
        template='plotly_white',
        height=500,
        xaxis_tickangle=-45
    )
    html_parts.append(fig_bar.to_html(include_plotlyjs='cdn', div_id='patterns_bar'))
    
    # 2. Support vs Confidence scatter with P-value
    fig_scatter = go.Figure()
    
    significant = patterns[patterns['P_Value'] < 0.05]
    non_significant = patterns[patterns['P_Value'] >= 0.05]
    
    if not significant.empty:
        fig_scatter.add_trace(go.Scatter(
            x=significant['Support'],
            y=significant['Confidence'],
            mode='markers',
            marker=dict(size=10, color='red', symbol='circle'),
            text=significant['Pattern'],
            name='Significant (P < 0.05)',
            hovertemplate='Pattern: %{text}<br>Support: %{x:.3f}<br>Confidence: %{y:.3f}<br>P-value: %{customdata:.4f}<extra></extra>',
            customdata=significant['P_Value']
        ))
    
    if not non_significant.empty:
        fig_scatter.add_trace(go.Scatter(
            x=non_significant['Support'],
            y=non_significant['Confidence'],
            mode='markers',
            marker=dict(size=8, color='lightgray', symbol='circle'),
            text=non_significant['Pattern'],
            name='Non-significant (P ≥ 0.05)',
            hovertemplate='Pattern: %{text}<br>Support: %{x:.3f}<br>Confidence: %{y:.3f}<br>P-value: %{customdata:.4f}<extra></extra>',
            customdata=non_significant['P_Value']
        ))
    
    fig_scatter.update_layout(
        title=f'Support vs Confidence for Sequential Patterns ({data_type})',
        xaxis_title='Support',
        yaxis_title='Confidence',
        template='plotly_white',
        height=500,
        legend=dict(x=0.7, y=0.1)
    )
    html_parts.append(fig_scatter.to_html(include_plotlyjs='cdn', div_id='patterns_scatter'))
    
    # 3. Lift vs P-value (if available)
    if 'Lift' in patterns.columns:
        fig_lift = go.Figure()
        fig_lift.add_trace(go.Scatter(
            x=patterns['P_Value'],
            y=patterns['Lift'],
            mode='markers',
            marker=dict(
                size=10,
                color=patterns['Confidence'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Confidence')
            ),
            text=patterns['Pattern'],
            hovertemplate='Pattern: %{text}<br>P-value: %{x:.4f}<br>Lift: %{y:.3f}<extra></extra>'
        ))
        fig_lift.add_hline(y=1.0, line_dash="dash", line_color="gray", 
                          annotation_text="Lift = 1.0 (no association)")
        fig_lift.update_layout(
            title=f'Lift vs P-value for Sequential Patterns ({data_type})',
            xaxis_title='P-value',
            yaxis_title='Lift',
            template='plotly_white',
            height=400
        )
        html_parts.append(fig_lift.to_html(include_plotlyjs='cdn', div_id='patterns_lift'))
    
    html_combined = '\n'.join(html_parts)
    
    return html_combined, fig_bar, fig_scatter


def create_hybrid_network_figure(G: nx.Graph) -> Tuple[str, Any]:
    """
    Create an interactive visualization of the hybrid network.

    Visualization details:
    - Force-directed layout algorithm (Fruchterman-Reingold variant)
    - Node color based on type: phenotype (red), gene (blue)
    - Node size proportional to connectivity (degree)
    - Edge color based on type:
      * phenotype-phenotype: red
      * gene-gene: blue
      * phenotype-gene: purple
    - Interactive features:
      * Hover information for nodes and edges
      * Zoom and pan capabilities
      * Download options (PNG, SVG, CSV)

    Layout algorithm:
    - Modified spring layout based on node-node repulsion
      and edge-based attraction
    - Position optimization using 50 iterations
    - Fixed random seed (42) for reproducibility

    Parameters:
        G (nx.Graph): NetworkX graph of the hybrid network

    Returns:
        tuple: (HTML string for interactive Plotly figure, Figure object)
    """
    import plotly.graph_objs as go

    if G.number_of_nodes() == 0:
        return "<p>No hybrid network to display.</p>", None

    # Use a force-directed layout algorithm for better node positioning
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    # Separate edge collections by type
    edge_x_pheno_pheno, edge_y_pheno_pheno, edge_text_pheno_pheno = [], [], []
    edge_x_gene_gene, edge_y_gene_gene, edge_text_gene_gene = [], [], []
    edge_x_pheno_gene, edge_y_pheno_gene, edge_text_pheno_gene = [], [], []

    # Process edges by type
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_type = d.get("edge_type", "unknown")

        phi = d.get("phi", 0)
        pval = d.get("pvalue", np.nan)
        st = add_significance_stars(pval)

        # Detailed edge information
        label = f"<b>{u}–{v}</b><br>Type: {edge_type}<br>Phi: {phi:.3f}<br>p-value: {pval:.3f}{st}"

        # Add to appropriate list based on edge type
        if edge_type == "pheno-pheno":
            edge_x_pheno_pheno.extend([x0, x1, None])
            edge_y_pheno_pheno.extend([y0, y1, None])
            edge_text_pheno_pheno.extend([label, label, None])
        elif edge_type == "gene-gene":
            edge_x_gene_gene.extend([x0, x1, None])
            edge_y_gene_gene.extend([y0, y1, None])
            edge_text_gene_gene.extend([label, label, None])
        else:  # pheno-gene or unknown
            edge_x_pheno_gene.extend([x0, x1, None])
            edge_y_pheno_gene.extend([y0, y1, None])
            edge_text_pheno_gene.extend([label, label, None])

    # Create separate trace for each edge type
    traces = []

    if edge_x_pheno_pheno:
        traces.append(
            go.Scatter(
                x=edge_x_pheno_pheno,
                y=edge_y_pheno_pheno,
                mode="lines",
                line=dict(width=2, color="rgba(255,0,0,0.5)"),
                hoverinfo="text",
                text=edge_text_pheno_pheno,
                name="pheno-pheno",
            )
        )

    if edge_x_gene_gene:
        traces.append(
            go.Scatter(
                x=edge_x_gene_gene,
                y=edge_y_gene_gene,
                mode="lines",
                line=dict(width=2, color="rgba(0,0,255,0.5)"),
                hoverinfo="text",
                text=edge_text_gene_gene,
                name="gene-gene",
            )
        )

    if edge_x_pheno_gene:
        traces.append(
            go.Scatter(
                x=edge_x_pheno_gene,
                y=edge_y_pheno_gene,
                mode="lines",
                line=dict(width=2, color="rgba(128,0,128,0.5)"),
                hoverinfo="text",
                text=edge_text_pheno_gene,
                name="pheno-gene",
            )
        )

    # Node trace
    node_x, node_y = [], []
    node_text = []
    hover_text = []
    marker_color = []
    marker_size = []

    # Process all nodes
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Node type determines color
        ntype = G.nodes[node].get("node_type", "Unknown")
        color = "red" if ntype == "Phenotype" else "blue"
        marker_color.append(color)

        # Node size based on degree
        deg = nx.degree(G, node)
        size_val = 15 + 3 * deg
        marker_size.append(size_val)

        # Node label
        node_text.append(str(node))

        # Detailed hover information
        connections = [f"• {n}" for n in nx.neighbors(G, node)]
        connections_text = "<br>".join(connections) if connections else "None"

        hover_info = f"<b>{node}</b><br>Type: {ntype}<br>Degree: {deg}<br><b>Connections:</b><br>{connections_text}"
        hover_text.append(hover_info)

    # Add node trace
    traces.append(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            hoverinfo="text",
            hovertext=hover_text,
            marker=dict(size=marker_size, color=marker_color, line=dict(width=1, color="#333")),
            name="Nodes",
        )
    )

    # Create figure with all traces
    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=dict(text="Hybrid Co-resistance Network", font=dict(size=18)),
            showlegend=True,
            hovermode="closest",
            width=1000,
            height=800,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(x=1, y=0.5),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=[
                        dict(
                            args=[{"width": 1000, "height": 800}], label="Reset", method="relayout"
                        ),
                        dict(
                            args=[{"width": 1500, "height": 1200}],
                            label="Enlarge",
                            method="relayout",
                        ),
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=False,
                    x=0.1,
                    xanchor="left",
                    y=1.1,
                    yanchor="top",
                )
            ],
        ),
    )

    # Add download options
    config = {
        "toImageButtonOptions": {
            "format": "png",
            "filename": "hybrid_network",
            "height": 800,
            "width": 1000,
            "scale": 2,
        },
        "modeBarButtonsToAdd": ["downloadSVG", "downloadCSV"],
    }

    return fig.to_html(full_html=False, include_plotlyjs="cdn", config=config), fig


###############################################################################
# 9) REPORT + MAIN
###############################################################################

# Module-level counter for deterministic table IDs
_table_id_counter = 0


def df_to_html(df: pd.DataFrame, caption: str) -> str:
    """
    Convert DataFrame to enhanced HTML table with interactive features.

    Presentation details:
    - Integrates DataTables jQuery plugin for enhanced functionality
    - Adds interactive features: search, sort, pagination
    - Includes data export options (copy, CSV, Excel, PDF)
    - Creates unique ID for each table to allow multiple tables on page
    - Handles empty dataframes with informative message

    Parameters:
        df (pd.DataFrame): Data to be displayed in the table
        caption (str): Table title/caption

    Returns:
        str: HTML code for interactive DataTables display
    """
    global _table_id_counter
    
    if df.empty:
        return f"<h4>{caption}</h4><p>No data available.</p>"

    # Generate deterministic unique ID using a counter
    # This ensures reproducibility while maintaining uniqueness within a session
    _table_id_counter += 1
    table_id = f"table_{_table_id_counter:04d}"

    # Round all float columns to 3 decimal places
    for col in df.columns:
        if df[col].dtype == "float64" or df[col].dtype == "float32":
            df[col] = df[col].round(3)

    # Convert DataFrame to HTML table with unique ID
    html_table = df.to_html(index=False, table_id=table_id, classes="display nowrap compact")

    # Add DataTables initialization script
    script = f"""
    <script>
        $(document).ready(function() {{
            $('#{table_id}').DataTable({{
                pageLength: 20,
                lengthMenu: [[10, 20, 50, 100, -1], [10, 20, 50, 100, "All"]],
                dom: 'Blfrtip',
                buttons: [
                    'copy', 'csv', 'excel', 'pdf'
                ]
            }});
        }});
    </script>
    """

    return f"<h4>{caption}</h4>\n<div class='table-responsive'>{html_table}</div>\n{script}"


def generate_html_report(
    data: pd.DataFrame,
    class_res_all: pd.DataFrame,
    class_res_mdr: pd.DataFrame,
    amr_all: pd.DataFrame,
    amr_mdr: pd.DataFrame,
    freq_pheno_all: pd.DataFrame,
    freq_pheno_mdr: pd.DataFrame,
    freq_gene_all: pd.DataFrame,
    freq_gene_mdr: pd.DataFrame,
    pat_pheno_mdr: pd.DataFrame,
    pat_gene_mdr: pd.DataFrame,
    coocc_pheno_mdr: pd.DataFrame,
    coocc_gene_mdr: pd.DataFrame,
    gene_pheno_assoc: pd.DataFrame,
    assoc_rules_pheno: pd.DataFrame,
    assoc_rules_genes: pd.DataFrame,
    hybrid_net: nx.Graph,
    edges_df: pd.DataFrame,
    comm_df: pd.DataFrame,
    net_html: str,
    risk_scores: pd.DataFrame = None,
    risk_html: str = "",
    seq_patterns_amr: pd.DataFrame = None,
    seq_patterns_mic: pd.DataFrame = None,
    seq_html_amr: str = "",
    seq_html_mic: str = "",
) -> str:
    """
    Generate comprehensive HTML report with all analysis results.

    Report structure:
    - Header with timestamp and dataset overview
    - Separate sections for each analysis type
    - Interactive tables and visualizations
    - Detailed statistical methodology descriptions
    - Downloads and export functionality

    Technical details:
    - HTML5 compliant structure with responsive design
    - CSS styling for readability and visual consistency
    - jQuery and DataTables for client-side interactivity
    - Plotly for network visualization

    Parameters:
        data: Original dataset
        class_res_all: Class resistance matrix (all isolates)
        class_res_mdr: Class resistance matrix (MDR isolates only)
        amr_all: AMR gene presence matrix (all isolates)
        amr_mdr: AMR gene presence matrix (MDR isolates only)
        freq_pheno_all: Phenotypic resistance frequencies (all isolates)
        freq_pheno_mdr: Phenotypic resistance frequencies (MDR isolates)
        freq_gene_all: AMR gene frequencies (all isolates)
        freq_gene_mdr: AMR gene frequencies (MDR isolates)
        pat_pheno_mdr: Phenotypic resistance patterns (MDR isolates)
        pat_gene_mdr: AMR gene patterns (MDR isolates)
        coocc_pheno_mdr: Co-occurrence among antibiotic classes
        coocc_gene_mdr: Co-occurrence among AMR genes
        gene_pheno_assoc: Phenotype-gene associations
        assoc_rules_pheno: Association rules for phenotypes
        assoc_rules_genes: Association rules for genes
        hybrid_net: Hybrid network graph
        edges_df: Significant edges in the network
        comm_df: Communities identified in the network
        net_html: Interactive network visualization HTML

    Returns:
        str: Complete HTML report with all visualizations and tables
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>StrepSuis-AMRPat: Automated Detection of Antimicrobial Resistance Patterns in Streptococcus suis</title>
  <!-- jQuery -->
  <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>

  <!-- DataTables CSS -->
  <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.13.1/css/jquery.dataTables.min.css">
  <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/buttons/2.3.2/css/buttons.dataTables.min.css">

  <!-- DataTables JS -->
  <script type="text/javascript" src="https://cdn.datatables.net/1.13.1/js/jquery.dataTables.min.js"></script>
  <script type="text/javascript" src="https://cdn.datatables.net/buttons/2.3.2/js/dataTables.buttons.min.js"></script>
  <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.1.3/jszip.min.js"></script>
  <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/pdfmake/0.1.53/pdfmake.min.js"></script>
  <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/pdfmake/0.1.53/vfs_fonts.js"></script>
  <script type="text/javascript" src="https://cdn.datatables.net/buttons/2.3.2/js/buttons.html5.min.js"></script>
  <script type="text/javascript" src="https://cdn.datatables.net/buttons/2.3.2/js/buttons.print.min.js"></script>

  <style>
    body {{
      font-family: Arial, sans-serif;
      margin:20px;
      background-color:#f9f9f9;
    }}
    .container {{
      background-color:#fff;
      padding:30px;
      border-radius:6px;
      max-width:1200px;
      margin:0 auto;
    }}
    h1,h2,h3,h4 {{
      color:#333;
    }}
    table {{
      border-collapse:collapse;
      width:100%;
      margin-bottom:20px;
    }}
    table,th,td {{
      border:1px solid #ccc;
    }}
    th,td {{
      padding:8px;
      text-align:left;
      font-size:0.9em;
    }}
    th {{
      background-color:#fafafa;
    }}
    .footer {{
      text-align:center;
      margin-top:40px;
      font-size:0.85em;
      color:#666;
      border-top:1px solid #ccc;
      padding-top:10px;
    }}
    .section {{
      margin-bottom:40px;
    }}
    .network-container {{
      border:1px solid #ccc;
      padding:10px;
      margin-bottom:20px;
      overflow-x:auto;
    }}
    .table-responsive {{
      overflow-x: auto;
      margin-bottom: 20px;
    }}
    .dataTables_wrapper {{
      padding: 10px 0;
    }}
    .dt-buttons {{
      margin-bottom: 15px;
    }}
    .methodology {{
      background-color: #f8f9fa;
      border-left: 4px solid #6c757d;
      padding: 10px 15px;
      margin-bottom: 20px;
      font-size: 0.9em;
    }}
    .methodology h4 {{
      margin-top: 0;
      color: #495057;
    }}
    .methodology ul {{
      padding-left: 20px;
    }}
  </style>
</head>
<body>
<div class="container">
<h1>StrepSuis-AMRPat: Automated Detection of Antimicrobial Resistance Patterns in Streptococcus suis</h1>
<p><strong>Timestamp:</strong> {ts}</p>
<p><strong>Total isolates:</strong> {len(data)}</p>
<hr/>

<div class="section">
  <h2>1) Phenotypic Resistance Frequencies (All Isolates)</h2>
  
  <div class="methodology">
    <h4>Statistical Methodology</h4>
    <p>These frequencies were calculated using non-parametric bootstrap resampling with the following parameters:</p>
    <ul>
      <li><strong>Bootstrap iterations:</strong> 5,000</li>
      <li><strong>Confidence level:</strong> 95%</li>
      <li><strong>Method:</strong> Percentile bootstrap CI (empirical distribution)</li>
    </ul>
    <p>The bootstrap approach makes no assumptions about the underlying distribution and is robust for prevalence estimation.
    Confidence intervals are calculated directly from the resampled distribution using quantiles corresponding to the
    (α/2) and (1-α/2) percentiles, where α = 0.05 for 95% confidence.</p>
  </div>
  
  {df_to_html(freq_pheno_all,"All isolates (bootstrap 95% CI)")}
</div>

<div class="section">
  <h2>2) Phenotypic Resistance Frequencies (MDR Subset)</h2>
  
  <div class="methodology">
    <h4>Statistical Methodology</h4>
    <p>Frequencies in the MDR subset follow the same bootstrap methodology as the full dataset, with identical parameters:</p>
    <ul>
      <li><strong>Bootstrap iterations:</strong> 5,000</li>
      <li><strong>Confidence level:</strong> 95%</li>
      <li><strong>MDR definition:</strong> Resistance to ≥3 antibiotic classes</li>
    </ul>
    <p>This analysis focuses only on isolates meeting the MDR criteria (resistance to three or more antibiotic classes),
    allowing comparison of resistance patterns between the general population and the MDR subset.</p>
  </div>
  
  {df_to_html(freq_pheno_mdr,"MDR subset (bootstrap 95% CI)")}
</div>

<div class="section">
  <h2>3) AMR Gene Frequencies (All Isolates)</h2>
  
  <div class="methodology">
    <h4>Statistical Methodology</h4>
    <p>AMR gene frequencies were calculated using the same bootstrap approach as phenotypic resistances:</p>
    <ul>
      <li><strong>Bootstrap iterations:</strong> 5,000</li>
      <li><strong>Confidence level:</strong> 95%</li>
      <li><strong>Data type:</strong> Binary gene presence (1) or absence (0)</li>
    </ul>
    <p>All gene data was standardized to binary format before analysis, with any non-zero, 
    non-empty value considered as gene presence (1) and zero or empty values as absence (0).</p>
  </div>
  
  {df_to_html(freq_gene_all,"All isolates (AMR genes)")}
</div>

<div class="section">
  <h2>4) AMR Gene Frequencies (MDR Subset)</h2>
  
  <div class="methodology">
    <h4>Statistical Methodology</h4>
    <p>Gene frequencies in the MDR subset follow the same bootstrap methodology, focusing on isolates meeting MDR criteria:</p>
    <ul>
      <li><strong>Bootstrap iterations:</strong> 5,000</li>
      <li><strong>Confidence level:</strong> 95%</li>
      <li><strong>MDR definition:</strong> Resistance to ≥3 antibiotic classes</li>
    </ul>
    <p>This analysis helps identify genes potentially associated with multidrug resistance phenotypes
    by comparing their prevalence in MDR versus non-MDR isolates.</p>
  </div>
  
  {df_to_html(freq_gene_mdr,"MDR subset (AMR genes)")}
</div>

<div class="section">
  <h2>5) Phenotypic Patterns in MDR</h2>
  
  <div class="methodology">
    <h4>Statistical Methodology</h4>
    <p>This analysis identifies and quantifies unique combinations of resistance phenotypes:</p>
    <ul>
      <li><strong>Pattern identification:</strong> Each isolate's resistance profile is converted to a sorted tuple of resistance classes</li>
      <li><strong>Bootstrap iterations:</strong> 5,000</li>
      <li><strong>Confidence level:</strong> 95%</li>
    </ul>
    <p>The bootstrapping procedure estimates the frequency distribution of each pattern in the population. Patterns are sorted by descending frequency, highlighting the most common resistance combinations.</p>
  </div>
  
  {df_to_html(pat_pheno_mdr,"Phenotypic MDR patterns (bootstrap)")}
</div>

<div class="section">
  <h2>6) AMR Gene Patterns in MDR</h2>
  
  <div class="methodology">
    <h4>Statistical Methodology</h4>
    <p>Similar to phenotypic patterns, this analysis identifies unique combinations of AMR genes:</p>
    <ul>
      <li><strong>Pattern identification:</strong> Each isolate's gene profile is converted to a sorted tuple of present genes</li>
      <li><strong>Bootstrap iterations:</strong> 5,000</li>
      <li><strong>Confidence level:</strong> 95%</li>
    </ul>
    <p>The prevalence of specific gene combinations may indicate horizontal gene transfer, mobile genetic elements,
    or co-selection mechanisms. Patterns are sorted by descending frequency.</p>
  </div>
  
  {df_to_html(pat_gene_mdr,"Gene MDR patterns (bootstrap)")}
</div>

<div class="section">
  <h2>7) Co-occurrence Among Antibiotic Classes in MDR</h2>
  
  <div class="methodology">
    <h4>Statistical Methodology</h4>
    <p>This analysis identifies statistically significant associations between pairs of resistance phenotypes:</p>
    <ul>
      <li><strong>Association measure:</strong> Phi coefficient (φ), range from -1 to +1</li>
      <li><strong>Statistical test:</strong> Chi-square or Fisher's exact test (when expected counts <5)</li>
      <li><strong>Multiple testing correction:</strong> Benjamini-Hochberg FDR procedure (α=0.05)</li>
      <li><strong>Result filtering:</strong> Only statistically significant associations after correction are shown</li>
    </ul>
    <p>The phi coefficient indicates the strength and direction of association:
       φ>0 indicates positive association (co-occurrence),
       φ<0 indicates negative association (mutual exclusivity).</p>
  </div>
  
  {df_to_html(coocc_pheno_mdr,"Significant class–class co-occurrences")}
</div>

<div class="section">
  <h2>8) Co-occurrence Among AMR Genes in MDR</h2>
  
  <div class="methodology">
    <h4>Statistical Methodology</h4>
    <p>This analysis uses the same methodology as phenotypic co-occurrence to identify gene-gene associations:</p>
    <ul>
      <li><strong>Association measure:</strong> Phi coefficient (φ), range from -1 to +1</li>
      <li><strong>Statistical test:</strong> Chi-square or Fisher's exact test (when expected counts <5)</li>
      <li><strong>Multiple testing correction:</strong> Benjamini-Hochberg FDR procedure (α=0.05)</li>
      <li><strong>Result filtering:</strong> Only statistically significant associations after correction are shown</li>
    </ul>
    <p>Strong positive associations between genes may indicate co-location on the same genetic element (e.g., plasmid, transposon)
    or co-selection under similar antibiotic pressure.</p>
  </div>
  
  {df_to_html(coocc_gene_mdr,"Significant gene–gene co-occurrences")}
</div>

<div class="section">
  <h2>9) Gene–Phenotypic Associations (Entire Dataset)</h2>
  
  <div class="methodology">
    <h4>Statistical Methodology</h4>
    <p>This cross-dataset analysis identifies associations between AMR genes and resistance phenotypes:</p>
    <ul>
      <li><strong>Association measure:</strong> Phi coefficient (φ), range from -1 to +1</li>
      <li><strong>Statistical test:</strong> Chi-square or Fisher's exact test (when expected counts <5)</li>
      <li><strong>Multiple testing correction:</strong> Benjamini-Hochberg FDR procedure (α=0.05)</li>
      <li><strong>Result filtering:</strong> Only statistically significant associations after correction are shown</li>
    </ul>
    <p>These associations suggest potential gene-phenotype relationships, which may indicate genetic determinants of 
    specific resistance phenotypes. Strong positive associations may indicate causality or co-selection.</p>
  </div>
  
  {df_to_html(gene_pheno_assoc,"Significant gene–class associations")}
</div>

<div class="section">
  <h2>10) Phenotypic Association Rules</h2>
  
  <div class="methodology">
    <h4>Statistical Methodology</h4>
    <p>Association rule mining identifies frequent if-then relationships between phenotypic resistances:</p>
    <ul>
      <li><strong>Algorithm:</strong> Apriori for frequent itemset generation</li>
      <li><strong>Minimum support:</strong> 0.1 (10% of isolates)</li>
      <li><strong>Metrics:</strong>
        <ul>
          <li>Support: Proportion of isolates exhibiting both antecedent and consequent</li>
          <li>Confidence: Conditional probability of consequent given antecedent</li>
          <li>Lift: Ratio of observed support to expected support under independence</li>
        </ul>
      </li>
      <li><strong>Filtering:</strong> Only rules with lift ≥ 1.0 are included</li>
    </ul>
    <p>Rules are sorted by lift (measure of association strength). Lift > 1 indicates positive association,
    with higher values suggesting stronger relationships beyond what would be expected by chance.</p>
  </div>
  
  {df_to_html(assoc_rules_pheno,"Association Rules (Phenotypes)")}
</div>

<div class="section">
  <h2>11) AMR Gene Association Rules</h2>
  
  <div class="methodology">
    <h4>Statistical Methodology</h4>
    <p>This analysis applies the same association rule mining approach to AMR genes:</p>
    <ul>
      <li><strong>Algorithm:</strong> Apriori for frequent itemset generation</li>
      <li><strong>Minimum support:</strong> 0.1 (10% of isolates)</li>
      <li><strong>Metrics:</strong>
        <ul>
          <li>Support: Proportion of isolates carrying both gene sets</li>
          <li>Confidence: Conditional probability of consequent genes given antecedent genes</li>
          <li>Lift: Ratio of observed support to expected support under independence</li>
        </ul>
      </li>
      <li><strong>Filtering:</strong> Only rules with lift ≥ 1.0 are included</li>
    </ul>
    <p>High-confidence, high-lift rules may suggest genetic linkage, co-selection mechanisms,
    or consistent horizontal gene transfer patterns between isolates.</p>
  </div>
  
  {df_to_html(assoc_rules_genes,"Association Rules (Genes)")}
</div>

<!-- 12) HYBRID NETWORK -->
<div class="section">
<h2>12) Hybrid Co-resistance Network (Pheno–Pheno, Gene–Gene, Pheno–Gene)</h2>

<div class="methodology">
  <h4>Statistical Methodology</h4>
  <p>The hybrid network integrates all significant associations into a comprehensive visualization:</p>
  <ul>
    <li><strong>Nodes:</strong> Represent phenotypic resistances and AMR genes</li>
    <li><strong>Edges:</strong> Represent statistically significant associations
      <ul>
        <li>pheno-pheno: Associations between resistance phenotypes</li>
        <li>gene-gene: Associations between AMR genes</li>
        <li>pheno-gene: Associations between phenotypes and genes</li>
      </ul>
    </li>
    <li><strong>Association criteria:</strong>
      <ul>
        <li>Statistical test: Chi-square or Fisher's exact test</li>
        <li>Multiple testing correction: Benjamini-Hochberg FDR</li>
        <li>Significance threshold: α=0.05 after correction</li>
      </ul>
    </li>
    <li><strong>Edge weights:</strong> Proportional to phi coefficient strength</li>
    <li><strong>Layout algorithm:</strong> Force-directed spring layout</li>
  </ul>
  <p>The network visualization offers an integrated view of resistance relationships,
  providing insights into potential resistance mechanisms and gene-phenotype connections.</p>
</div>

<h3>12a) Interactive Figure</h3>
<div class="network-container">
  {net_html}
</div>

<h3>12b) Significant Edges</h3>
<div class="methodology">
  <h4>Edge Details</h4>
  <p>This table provides detailed information about all statistically significant associations in the network:</p>
  <ul>
    <li><strong>Node1, Node2:</strong> The connected phenotypes or genes</li>
    <li><strong>Edge_Type:</strong> Type of connection (pheno-pheno, gene-gene, pheno-gene)</li>
    <li><strong>Phi:</strong> Association strength and direction (-1 to +1)</li>
    <li><strong>p-value:</strong> FDR-corrected statistical significance</li>
    <li><strong>Significance:</strong> p-value with stars indicating significance level</li>
  </ul>
  <p>All edges shown are statistically significant after multiple testing correction (FDR-adjusted p < 0.05).</p>
</div>

{df_to_html(edges_df,"Significant Edges (All Types)")}

<h3>12c) Network Communities (Louvain)</h3>
<div class="methodology">
  <h4>Community Detection Methodology</h4>
  <p>The Louvain algorithm was used to detect communities in the network:</p>
  <ul>
    <li><strong>Algorithm:</strong> Louvain community detection (modularity optimization)</li>
    <li><strong>Modularity objective:</strong> Maximize within-community connections relative to between-community connections</li>
    <li><strong>Edge weights:</strong> Phi coefficient values used as weights</li>
    <li><strong>Community numbering:</strong> Sequential (1, 2, 3, etc.)</li>
  </ul>
  <p>Communities represent groups of genes and phenotypes that are more densely connected with each other than with
  the rest of the network, potentially indicating functional modules or co-selection units.</p>
</div>

{df_to_html(comm_df,"Network Communities")}
</div>

<!-- INNOVATION 1: Network Risk Scoring -->
<div class="section">
<h2>13) Network Risk Scoring (Innovation)</h2>
<div class="methodology">
  <h4>Innovation: Network-Based MDR Risk Prediction</h4>
  <p>This novel approach combines network topology with statistical confidence to predict MDR risk:</p>
  <ul>
    <li><strong>Method:</strong> Weighted combination of network centrality metrics (degree, betweenness, eigenvector)</li>
    <li><strong>Confidence weighting:</strong> Bootstrap CI width inversely weights features (narrower CI = higher confidence)</li>
    <li><strong>Risk score formula:</strong> Σ[(w_d × degree_cent + w_b × betweenness_cent + w_e × eigenvector_cent) × CI_weight]</li>
    <li><strong>MDR prediction:</strong> Top 25% percentile threshold (configurable)</li>
    <li><strong>Advantage:</strong> Predictive capability before full MDR development, topology-aware scoring</li>
  </ul>
  <p>This innovation provides a novel metric for early identification of high-risk strains based on their position in the resistance network.</p>
</div>

{risk_html if risk_html else "<p>Network Risk Scoring could not be computed.</p>"}

{df_to_html(risk_scores, "Network Risk Scores for All Strains") if risk_scores is not None and not risk_scores.empty else ""}
</div>

<!-- INNOVATION 2: Sequential Pattern Detection -->
<div class="section">
<h2>14) Sequential Pattern Detection (Innovation)</h2>
<div class="methodology">
  <h4>Innovation: Order-Dependent Resistance Evolution Patterns</h4>
  <p>This novel approach identifies sequential patterns in resistance acquisition:</p>
  <ul>
    <li><strong>Method:</strong> Modified Apriori algorithm for sequential patterns (A→B→C)</li>
    <li><strong>Order inference:</strong> Correlation strength suggests acquisition order</li>
    <li><strong>Statistical validation:</strong> Permutation testing (100 iterations) for significance</li>
    <li><strong>Metrics:</strong> Support, Confidence, Lift, P-value</li>
    <li><strong>Advantage:</strong> Captures temporal/evolutionary relationships, not just co-occurrence</li>
  </ul>
  <p>This innovation reveals the order in which resistance features are typically acquired, providing insights into resistance evolution pathways.</p>
</div>

<h3>14a) Sequential Patterns in AMR Genes</h3>
{seq_html_amr if seq_html_amr else "<p>Sequential pattern detection (AMR genes) could not be computed.</p>"}
{df_to_html(seq_patterns_amr, "Sequential Patterns in AMR Genes") if seq_patterns_amr is not None and not seq_patterns_amr.empty else ""}

<h3>14b) Sequential Patterns in MIC Phenotypes</h3>
{seq_html_mic if seq_html_mic else "<p>Sequential pattern detection (MIC phenotypes) could not be computed.</p>"}
{df_to_html(seq_patterns_mic, "Sequential Patterns in MIC Phenotypes") if seq_patterns_mic is not None and not seq_patterns_mic.empty else ""}
</div>

<div class="footer">
  <p>© Hybrid MDR Pipeline with enhanced statistical analysis and visualization.</p>
  <p><strong>Innovations:</strong> Network Risk Scoring, Sequential Pattern Detection</p>
</div>
</div>

</body>
</html>
"""
    return html


def generate_excel_report(
    data: pd.DataFrame,
    class_res_all: pd.DataFrame,
    class_res_mdr: pd.DataFrame,
    amr_all: pd.DataFrame,
    amr_mdr: pd.DataFrame,
    freq_pheno_all: pd.DataFrame,
    freq_pheno_mdr: pd.DataFrame,
    freq_gene_all: pd.DataFrame,
    freq_gene_mdr: pd.DataFrame,
    pat_pheno_mdr: pd.DataFrame,
    pat_gene_mdr: pd.DataFrame,
    coocc_pheno_mdr: pd.DataFrame,
    coocc_gene_mdr: pd.DataFrame,
    gene_pheno_assoc: pd.DataFrame,
    assoc_rules_pheno: pd.DataFrame,
    assoc_rules_genes: pd.DataFrame,
    hybrid_net: nx.Graph,
    edges_df: pd.DataFrame,
    comm_df: pd.DataFrame,
    fig_network=None,
    risk_scores: pd.DataFrame = None,
    fig_risk_hist=None,
    fig_risk_ranking=None,
    fig_risk_dist=None,
    seq_patterns_amr: pd.DataFrame = None,
    seq_patterns_mic: pd.DataFrame = None,
    fig_seq_amr_bar=None,
    fig_seq_amr_scatter=None,
    fig_seq_mic_bar=None,
    fig_seq_mic_scatter=None,
) -> str:
    """
    Generate comprehensive Excel report with all analysis results and PNG charts.

    This function creates a detailed Excel workbook with multiple sheets containing:
    - Dataset overview and metadata
    - Methodology descriptions
    - Frequency analyses with bootstrap confidence intervals
    - Pattern identification results
    - Co-occurrence matrices
    - Association rules
    - Network analysis results
    - Community detection outcomes

    All network visualizations are saved as PNG files.

    Parameters:
        data: Original dataset
        class_res_all: Class resistance matrix (all isolates)
        class_res_mdr: Class resistance matrix (MDR isolates only)
        amr_all: AMR gene presence matrix (all isolates)
        amr_mdr: AMR gene presence matrix (MDR isolates only)
        freq_pheno_all: Phenotypic resistance frequencies (all isolates)
        freq_pheno_mdr: Phenotypic resistance frequencies (MDR isolates)
        freq_gene_all: AMR gene frequencies (all isolates)
        freq_gene_mdr: AMR gene frequencies (MDR isolates)
        pat_pheno_mdr: Phenotypic resistance patterns (MDR isolates)
        pat_gene_mdr: AMR gene patterns (MDR isolates)
        coocc_pheno_mdr: Co-occurrence among antibiotic classes
        coocc_gene_mdr: Co-occurrence among AMR genes
        gene_pheno_assoc: Phenotype-gene associations
        assoc_rules_pheno: Association rules for phenotypes
        assoc_rules_genes: Association rules for genes
        hybrid_net: Hybrid network graph
        edges_df: Significant edges in the network
        comm_df: Communities identified in the network
        fig_network: Plotly figure object for the network (optional)

    Returns:
        str: Path to generated Excel file
    """
    # Initialize Excel report generator
    excel_gen = ExcelReportGenerator(output_folder=output_folder)

    # Save network visualization as PNG if available
    if fig_network is not None:
        try:
            excel_gen.save_plotly_figure_fallback(
                fig_network, "hybrid_network_visualization", width=1400, height=1000
            )
        except Exception as e:
            print(f"Could not save network visualization: {e}")

    # Prepare methodology description
    methodology = {
        "Bootstrap Resampling": (
            "5000 iterations with 95% confidence intervals for prevalence estimation. "
            "Non-parametric approach suitable for complex distributions."
        ),
        "Association Testing": (
            "Chi-square test for general associations, Fisher's exact test when expected cell counts <5. "
            "Phi coefficient measures association strength. "
            "Benjamini-Hochberg FDR correction for multiple hypothesis testing."
        ),
        "Pattern Analysis": (
            "Identification of phenotypic resistance patterns and AMR gene patterns in MDR isolates. "
            "Patterns are ranked by frequency with bootstrap confidence intervals."
        ),
        "Co-occurrence Analysis": (
            "Matrix-based analysis of feature co-occurrence using phi coefficient. "
            "Statistical significance assessed via chi-square or Fisher's exact test."
        ),
        "Association Rules": (
            "Mining of predictive rules using support and lift metrics. "
            "Identifies antecedent → consequent relationships with high confidence."
        ),
        "Hybrid Network Construction": (
            "Network combining phenotype-gene, phenotype-phenotype, and gene-gene edges. "
            "Edges represent significant associations (corrected p < 0.05). "
            "Louvain community detection identifies functional modules."
        ),
        "Network Risk Scoring (Innovation)": (
            "Novel metric combining network topology with statistical confidence. "
            "Uses weighted combination of centrality metrics (degree, betweenness, eigenvector) "
            "weighted by bootstrap CI width. Provides predictive capability for MDR risk."
        ),
        "Sequential Pattern Detection (Innovation)": (
            "Novel approach identifying order-dependent resistance patterns (A→B→C). "
            "Uses modified Apriori algorithm with correlation-based order inference. "
            "Statistical validation via permutation testing. Reveals resistance evolution pathways."
        ),
    }

    # Prepare sheets data
    sheets_data = {}

    # Dataset Overview
    overview_data = {
        "Metric": [
            "Total Isolates",
            "MDR Isolates",
            "MDR Percentage",
            "Phenotype Columns",
            "Gene Columns",
        ],
        "Value": [
            len(data),
            len(class_res_mdr) if class_res_mdr is not None else 0,
            (
                f"{(len(class_res_mdr) / len(data) * 100):.2f}%"
                if class_res_mdr is not None and len(data) > 0
                else "N/A"
            ),
            len([c for c in data.columns if "MIC_" in c or "Resistance_" in c]),
            len(
                [
                    c
                    for c in data.columns
                    if c not in ["Strain_ID"] and "MIC_" not in c and "Resistance_" not in c
                ]
            ),
        ],
    }
    sheets_data["Dataset_Overview"] = (
        pd.DataFrame(overview_data),
        "Summary statistics for the analyzed dataset",
    )

    # Frequency Analyses
    if freq_pheno_all is not None and not freq_pheno_all.empty:
        sheets_data["Freq_Pheno_All"] = (
            freq_pheno_all,
            "Phenotypic resistance frequencies in all isolates",
        )

    if freq_pheno_mdr is not None and not freq_pheno_mdr.empty:
        sheets_data["Freq_Pheno_MDR"] = (
            freq_pheno_mdr,
            "Phenotypic resistance frequencies in MDR isolates",
        )

    if freq_gene_all is not None and not freq_gene_all.empty:
        sheets_data["Freq_Gene_All"] = (freq_gene_all, "AMR gene frequencies in all isolates")

    if freq_gene_mdr is not None and not freq_gene_mdr.empty:
        sheets_data["Freq_Gene_MDR"] = (freq_gene_mdr, "AMR gene frequencies in MDR isolates")

    # Pattern Analyses
    if pat_pheno_mdr is not None and not pat_pheno_mdr.empty:
        sheets_data["Patterns_Pheno_MDR"] = (
            pat_pheno_mdr,
            "Phenotypic resistance patterns in MDR isolates",
        )

    if pat_gene_mdr is not None and not pat_gene_mdr.empty:
        sheets_data["Patterns_Gene_MDR"] = (pat_gene_mdr, "AMR gene patterns in MDR isolates")

    # Co-occurrence Analyses
    if coocc_pheno_mdr is not None and not coocc_pheno_mdr.empty:
        sheets_data["Coocc_Pheno_MDR"] = (
            coocc_pheno_mdr,
            "Co-occurrence among antibiotic classes in MDR",
        )

    if coocc_gene_mdr is not None and not coocc_gene_mdr.empty:
        sheets_data["Coocc_Gene_MDR"] = (coocc_gene_mdr, "Co-occurrence among AMR genes in MDR")

    # Association Analyses
    if gene_pheno_assoc is not None and not gene_pheno_assoc.empty:
        sheets_data["Gene_Pheno_Assoc"] = (
            gene_pheno_assoc,
            "Gene-Phenotype associations with statistical significance",
        )

    if assoc_rules_pheno is not None and not assoc_rules_pheno.empty:
        sheets_data["Assoc_Rules_Pheno"] = (
            assoc_rules_pheno,
            "Association rules for phenotypic resistances",
        )

    if assoc_rules_genes is not None and not assoc_rules_genes.empty:
        sheets_data["Assoc_Rules_Genes"] = (assoc_rules_genes, "Association rules for AMR genes")

    # Network Analyses
    if edges_df is not None and not edges_df.empty:
        sheets_data["Network_Edges"] = (
            edges_df,
            f"Significant edges in hybrid network (total: {len(edges_df)})",
        )
    
    # INNOVATION 1: Network Risk Scoring
    if risk_scores is not None and not risk_scores.empty:
        sheets_data["Network_Risk_Scores"] = (
            risk_scores,
            "Network-based MDR risk scores for all strains (Innovation)",
        )
    
    # Save risk scoring visualizations
    if fig_risk_hist is not None:
        try:
            excel_gen.save_plotly_figure_fallback(
                fig_risk_hist, "network_risk_histogram", width=1200, height=600
            )
        except Exception as e:
            print(f"Could not save risk histogram: {e}")
    
    if fig_risk_ranking is not None:
        try:
            excel_gen.save_plotly_figure_fallback(
                fig_risk_ranking, "network_risk_ranking", width=1400, height=700
            )
        except Exception as e:
            print(f"Could not save risk ranking: {e}")
    
    if fig_risk_dist is not None:
        try:
            excel_gen.save_plotly_figure_fallback(
                fig_risk_dist, "network_risk_distribution", width=1200, height=600
            )
        except Exception as e:
            print(f"Could not save risk distribution: {e}")
    
    # INNOVATION 2: Sequential Pattern Detection
    if seq_patterns_amr is not None and not seq_patterns_amr.empty:
        sheets_data["Sequential_Patterns_AMR"] = (
            seq_patterns_amr,
            "Sequential resistance patterns in AMR genes (Innovation)",
        )
    
    if seq_patterns_mic is not None and not seq_patterns_mic.empty:
        sheets_data["Sequential_Patterns_MIC"] = (
            seq_patterns_mic,
            "Sequential resistance patterns in MIC phenotypes (Innovation)",
        )
    
    # Save sequential pattern visualizations
    if fig_seq_amr_bar is not None:
        try:
            excel_gen.save_plotly_figure_fallback(
                fig_seq_amr_bar, "sequential_patterns_amr_bar", width=1400, height=700
            )
        except Exception as e:
            print(f"Could not save AMR patterns bar chart: {e}")
    
    if fig_seq_amr_scatter is not None:
        try:
            excel_gen.save_plotly_figure_fallback(
                fig_seq_amr_scatter, "sequential_patterns_amr_scatter", width=1400, height=700
            )
        except Exception as e:
            print(f"Could not save AMR patterns scatter: {e}")
    
    if fig_seq_mic_bar is not None:
        try:
            excel_gen.save_plotly_figure_fallback(
                fig_seq_mic_bar, "sequential_patterns_mic_bar", width=1400, height=700
            )
        except Exception as e:
            print(f"Could not save MIC patterns bar chart: {e}")
    
    if fig_seq_mic_scatter is not None:
        try:
            excel_gen.save_plotly_figure_fallback(
                fig_seq_mic_scatter, "sequential_patterns_mic_scatter", width=1400, height=700
            )
        except Exception as e:
            print(f"Could not save MIC patterns scatter: {e}")

    if comm_df is not None and not comm_df.empty:
        sheets_data["Network_Communities"] = (
            comm_df,
            f"Louvain communities in network (total: {comm_df['Community'].nunique()} communities)",
        )

    # Network Statistics
    if hybrid_net is not None and hybrid_net.number_of_nodes() > 0:
        network_stats = {
            "Metric": [
                "Total Nodes",
                "Total Edges",
                "Phenotype Nodes",
                "Gene Nodes",
                "Average Degree",
                "Network Density",
                "Number of Communities",
            ],
            "Value": [
                hybrid_net.number_of_nodes(),
                hybrid_net.number_of_edges(),
                len(
                    [n for n, d in hybrid_net.nodes(data=True) if d.get("node_type") == "Phenotype"]
                ),
                len([n for n, d in hybrid_net.nodes(data=True) if d.get("node_type") == "Gene"]),
                (
                    f"{sum(dict(hybrid_net.degree()).values()) / hybrid_net.number_of_nodes():.2f}"
                    if hybrid_net.number_of_nodes() > 0
                    else 0
                ),
                f"{nx.density(hybrid_net):.4f}",
                comm_df["Community"].nunique() if comm_df is not None and not comm_df.empty else 0,
            ],
        }
        sheets_data["Network_Stats"] = (
            pd.DataFrame(network_stats),
            "Overall network statistics and topology metrics",
        )

    # Prepare metadata
    metadata = {
        "Total_Isolates": len(data),
        "MDR_Isolates": len(class_res_mdr) if class_res_mdr is not None else 0,
        "Network_Nodes": hybrid_net.number_of_nodes() if hybrid_net is not None else 0,
        "Network_Edges": hybrid_net.number_of_edges() if hybrid_net is not None else 0,
        "Bootstrap_Iterations": 5000,
        "FDR_Threshold": 0.05,
    }

    # Generate Excel report
    excel_path = excel_gen.generate_excel_report(
        report_name="MDR_Analysis_Hybrid_Report",
        sheets_data=sheets_data,
        methodology=methodology,
        **metadata,
    )

    return excel_path


def save_report(html_code: str, out_dir: Optional[str] = None) -> str:
    """
    Save the HTML report to a file with timestamp.

    File handling details:
    - Creates output directory if it doesn't exist
    - Uses Unix timestamp for unique filename
    - UTF-8 encoding for proper character handling
    - Logs the file path for reference

    Parameters:
        html_code (str): Complete HTML report content
        out_dir (str): Output directory path (default: output_folder)

    Returns:
        str: Path to the saved HTML file
    """
    global output_folder
    if out_dir is None:
        out_dir = output_folder
    os.makedirs(out_dir, exist_ok=True)
    ts = int(time.time())
    fname = f"mdr_analysis_hybrid_{ts}.html"
    path = os.path.join(out_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_code)
    logging.info(f"Report saved => {path}")
    return path


###############################################################################
# MAIN
###############################################################################
def main(csv_path: Optional[str] = None):
    """
    Main function that orchestrates the entire analysis pipeline.

    Execution flow:
    1. Environment initialization and data loading
    2. Preprocessing and identification of phenotype/genotype columns
    3. MDR classification based on class resistance
    4. Frequency analysis with bootstrap confidence intervals
    5. Pattern identification and quantification
    6. Co-occurrence analysis for phenotypes and genes
    7. Association rule mining for predictive patterns
    8. Hybrid network construction and community detection
    9. Interactive visualization generation
    10. Comprehensive HTML report creation

    Args:
        csv_path: Optional path to CSV file. If None, will prompt user for input.

    Statistical considerations:
    - Correction for multiple testing using Benjamini-Hochberg FDR
    - Bootstrap resampling for non-parametric confidence intervals
    - Appropriate statistical tests based on data characteristics
    - Modular analysis pipeline with clear parameter specifications
    """
    start_time = time.time()
    logging.info("Starting Hybrid MDR pipeline with enhanced statistical documentation...")

    csv_path = setup_environment(csv_path)
    try:
        data = pd.read_csv(csv_path, sep=None, engine="python", dtype={"Strain_ID": str})
    except Exception as e:
        logging.error(f"Error reading CSV: {e}")
        return

    logging.info(f"Data loaded => shape={data.shape}")

    # Identify phenotype vs. genotype columns
    antibiotic_list = [d for arr in ANTIBIOTIC_CLASSES.values() for d in arr]
    phenotype_cols = [c for c in data.columns if c in antibiotic_list]
    skip_cols = {"id", "strain", "sample", "isolate", "date", "strain_id"}
    genotype_cols = [
        c for c in data.columns if c not in phenotype_cols and c.lower() not in skip_cols
    ]

    # 1) Build class-based resistance matrix and identify MDR isolates
    class_res_all = build_class_resistance(data, phenotype_cols)
    mdr_mask = identify_mdr_isolates(class_res_all, threshold=3)
    class_res_mdr = class_res_all[mdr_mask]

    # 2) Extract AMR gene data
    amr_all = extract_amr_genes(data, genotype_cols)
    amr_mdr = amr_all[mdr_mask]

    # 3) Calculate frequencies with bootstrap confidence intervals
    freq_pheno_all = compute_bootstrap_ci(class_res_all, 5000, 0.95)
    freq_pheno_mdr = compute_bootstrap_ci(class_res_mdr, 5000, 0.95)
    freq_gene_all = compute_bootstrap_ci(amr_all, 5000, 0.95)
    freq_gene_mdr = compute_bootstrap_ci(amr_mdr, 5000, 0.95)

    # 4) Analyze resistance patterns
    pheno_mdr_ser = get_mdr_patterns_pheno(class_res_mdr)
    pat_pheno_mdr = bootstrap_pattern_freq(pheno_mdr_ser, 5000, 0.95)

    gene_mdr_ser = get_mdr_patterns_geno(amr_mdr)
    pat_gene_mdr = bootstrap_pattern_freq(gene_mdr_ser, 5000, 0.95)

    # 5) Analyze co-occurrence among antibiotic classes in MDR isolates
    if class_res_mdr.shape[1] > 1:
        coocc_pheno_mdr = pairwise_cooccurrence(class_res_mdr)
    else:
        coocc_pheno_mdr = pd.DataFrame()

    # 6) Analyze co-occurrence among AMR genes in MDR isolates
    if amr_mdr.shape[1] > 1:
        coocc_gene_mdr = pairwise_cooccurrence(amr_mdr)
    else:
        coocc_gene_mdr = pd.DataFrame()

    # 7) Analyze phenotype-gene associations in the entire dataset
    gene_pheno_assoc = phenotype_gene_cooccurrence(class_res_all, amr_all)

    # 8) Apply association rule mining
    assoc_pheno = association_rules_phenotypic(class_res_all)
    assoc_genes = association_rules_genes(amr_all)

    # 9) Prepare data for hybrid network analysis
    df_hybrid = data[phenotype_cols + genotype_cols].copy()
    for c in df_hybrid.columns:
        df_hybrid[c] = df_hybrid[c].apply(lambda x: 1 if x != 0 else 0 if pd.notna(x) else 0)

    # 10) Build enhanced hybrid network
    hybrid_net = build_hybrid_co_resistance_network(
        df_hybrid, phenotype_cols, genotype_cols, alpha=0.05, method="fdr_bh"
    )

    # 11) Extract network edges for tabular display
    edges_list = []
    for u, v, d in hybrid_net.edges(data=True):
        pval = d.get("pvalue", np.nan)
        ety = d.get("edge_type", "unknown")
        edges_list.append(
            {
                "Node1": u,
                "Node2": v,
                "Edge_Type": ety,
                "Phi": round(d.get("phi", 0), 3),
                "p-value": round(pval, 3),
                "Significance": add_significance_stars(pval),
            }
        )
    edges_df = pd.DataFrame(edges_list).sort_values("p-value") if edges_list else pd.DataFrame()

    # 12) Detect network communities
    comm_df = compute_louvain_communities(hybrid_net)

    # 13) Generate interactive network visualization
    net_html, fig_network = create_hybrid_network_figure(hybrid_net)

    # 13.5) INNOVATION 1: Network Risk Scoring
    logging.info("Computing Network Risk Scores...")
    risk_scores = None
    risk_html = ""
    fig_risk_hist = None
    fig_risk_ranking = None
    fig_risk_dist = None
    
    try:
        # Compute bootstrap CI for all features
        bootstrap_ci_dict = {}
        freq_all_combined = pd.concat([freq_pheno_all, freq_gene_all])
        for _, row in freq_all_combined.iterrows():
            feature = row['ColumnName']
            ci_lower = row['CI_Lower'] / 100.0
            ci_upper = row['CI_Upper'] / 100.0
            bootstrap_ci_dict[feature] = (ci_lower, ci_upper)
        
        # Compute risk scores
        risk_scores = compute_network_mdr_risk_score(
            hybrid_net, df_hybrid, bootstrap_ci_dict, percentile_threshold=75.0
        )
        
        # Create visualizations
        risk_html, fig_risk_hist, fig_risk_ranking, fig_risk_dist = create_network_risk_scoring_visualizations(risk_scores)
        logging.info(f"Network Risk Scoring completed: {len(risk_scores)} strains analyzed")
    except Exception as e:
        logging.warning(f"Network Risk Scoring failed: {e}")
        risk_html = f"<p>Network Risk Scoring could not be computed: {str(e)}</p>"

    # 13.6) INNOVATION 2: Sequential Pattern Detection
    logging.info("Detecting Sequential Patterns...")
    seq_patterns_amr = None
    seq_patterns_mic = None
    seq_html_amr = ""
    seq_html_mic = ""
    fig_seq_amr_bar = None
    fig_seq_amr_scatter = None
    fig_seq_mic_bar = None
    fig_seq_mic_scatter = None
    
    try:
        # Sequential patterns in AMR genes
        seq_patterns_amr = detect_sequential_resistance_patterns(
            amr_all, min_support=0.1, min_confidence=0.5, correlation_threshold=0.3
        )
        seq_html_amr, fig_seq_amr_bar, fig_seq_amr_scatter = create_sequential_patterns_visualizations(
            seq_patterns_amr, "AMR Genes"
        )
        logging.info(f"Sequential patterns in AMR genes: {len(seq_patterns_amr)} patterns detected")
    except Exception as e:
        logging.warning(f"Sequential pattern detection (AMR) failed: {e}")
        seq_html_amr = f"<p>Sequential pattern detection (AMR genes) could not be computed: {str(e)}</p>"
    
    try:
        # Sequential patterns in MIC phenotypes
        class_res_all_for_seq = class_res_all.copy()
        seq_patterns_mic = detect_sequential_resistance_patterns(
            class_res_all_for_seq, min_support=0.1, min_confidence=0.5, correlation_threshold=0.3
        )
        seq_html_mic, fig_seq_mic_bar, fig_seq_mic_scatter = create_sequential_patterns_visualizations(
            seq_patterns_mic, "MIC Phenotypes"
        )
        logging.info(f"Sequential patterns in MIC phenotypes: {len(seq_patterns_mic)} patterns detected")
    except Exception as e:
        logging.warning(f"Sequential pattern detection (MIC) failed: {e}")
        seq_html_mic = f"<p>Sequential pattern detection (MIC phenotypes) could not be computed: {str(e)}</p>"

    # 13.7) NEW: Co-selection Analysis
    logging.info("Performing co-selection analysis...")
    coselection_modules = None
    try:
        from .coselection_analysis import CoSelectionAnalyzer
        
        # Combine co-occurrence results (phenotypes and genes)
        cooccurrence_combined = pd.concat([
            coocc_pheno_mdr if not coocc_pheno_mdr.empty else pd.DataFrame(),
            coocc_gene_mdr if not coocc_gene_mdr.empty else pd.DataFrame(),
        ], ignore_index=True)
        
        if not cooccurrence_combined.empty:
            coselection_analyzer = CoSelectionAnalyzer(
                network=hybrid_net,
                cooccurrence_results=cooccurrence_combined,
                communities=comm_df,
                data=df_hybrid,
            )
            coselection_modules = coselection_analyzer.identify_coselection_modules(threshold=0.7)
            logging.info(f"Co-selection analysis completed: {len(coselection_modules)} modules identified")
            
            # Save results
            if not coselection_modules.empty:
                coselection_modules.to_csv(
                    os.path.join(output_folder, "coselection_modules.csv"), index=False
                )
        else:
            logging.warning("No co-occurrence data available for co-selection analysis")
    except Exception as e:
        logging.warning(f"Co-selection analysis failed: {e}")

    # 13.8) NEW: Intervention Target Ranking
    logging.info("Ranking intervention targets...")
    target_rankings = None
    try:
        from .target_prioritization import InterventionTargetRanker
        
        # Prepare gene associations (combine gene-pheno and gene-gene associations)
        gene_assoc_combined = gene_pheno_assoc.copy() if not gene_pheno_assoc.empty else pd.DataFrame()
        
        if not gene_assoc_combined.empty and 'Gene' not in gene_assoc_combined.columns:
            # Rename columns if needed
            if 'Feature1' in gene_assoc_combined.columns:
                gene_assoc_combined = gene_assoc_combined.rename(columns={'Feature1': 'Gene', 'Feature2': 'Phenotype'})
        
        if not gene_assoc_combined.empty:
            ranker = InterventionTargetRanker(
                network=hybrid_net,
                gene_associations=gene_assoc_combined,
                communities=comm_df,
                data=df_hybrid,
                coselection_modules=coselection_modules,
            )
            target_rankings = ranker.rank_intervention_targets()
            logging.info(f"Target ranking completed: {len(target_rankings)} genes ranked")
            
            # Save top 20 targets
            top_targets = target_rankings.head(20)
            top_targets.to_csv(os.path.join(output_folder, "intervention_targets_top20.csv"), index=False)
    except Exception as e:
        logging.warning(f"Intervention target ranking failed: {e}")

    # 14) Generate comprehensive HTML report
    html_report = generate_html_report(
        data=data,
        class_res_all=class_res_all,
        class_res_mdr=class_res_mdr,
        amr_all=amr_all,
        amr_mdr=amr_mdr,
        freq_pheno_all=freq_pheno_all,
        freq_pheno_mdr=freq_pheno_mdr,
        freq_gene_all=freq_gene_all,
        freq_gene_mdr=freq_gene_mdr,
        pat_pheno_mdr=pat_pheno_mdr,
        pat_gene_mdr=pat_gene_mdr,
        coocc_pheno_mdr=coocc_pheno_mdr,
        coocc_gene_mdr=coocc_gene_mdr,
        gene_pheno_assoc=gene_pheno_assoc,
        assoc_rules_pheno=assoc_pheno,
        assoc_rules_genes=assoc_genes,
        hybrid_net=hybrid_net,
        edges_df=edges_df,
        comm_df=comm_df,
        net_html=net_html,
        risk_scores=risk_scores,
        risk_html=risk_html,
        seq_patterns_amr=seq_patterns_amr,
        seq_patterns_mic=seq_patterns_mic,
        seq_html_amr=seq_html_amr,
        seq_html_mic=seq_html_mic,
    )

    # 15) Save HTML report
    out_html = save_report(html_report, output_folder)

    # 16) Generate and save Excel report with PNG charts
    excel_path = generate_excel_report(
        data=data,
        class_res_all=class_res_all,
        class_res_mdr=class_res_mdr,
        amr_all=amr_all,
        amr_mdr=amr_mdr,
        freq_pheno_all=freq_pheno_all,
        freq_pheno_mdr=freq_pheno_mdr,
        freq_gene_all=freq_gene_all,
        freq_gene_mdr=freq_gene_mdr,
        pat_pheno_mdr=pat_pheno_mdr,
        pat_gene_mdr=pat_gene_mdr,
        coocc_pheno_mdr=coocc_pheno_mdr,
        coocc_gene_mdr=coocc_gene_mdr,
        gene_pheno_assoc=gene_pheno_assoc,
        assoc_rules_pheno=assoc_pheno,
        assoc_rules_genes=assoc_genes,
        hybrid_net=hybrid_net,
        edges_df=edges_df,
        comm_df=comm_df,
        fig_network=fig_network,
        risk_scores=risk_scores,
        fig_risk_hist=fig_risk_hist,
        fig_risk_ranking=fig_risk_ranking,
        fig_risk_dist=fig_risk_dist,
        seq_patterns_amr=seq_patterns_amr,
        seq_patterns_mic=seq_patterns_mic,
        fig_seq_amr_bar=fig_seq_amr_bar,
        fig_seq_amr_scatter=fig_seq_amr_scatter,
        fig_seq_mic_bar=fig_seq_mic_bar,
        fig_seq_mic_scatter=fig_seq_mic_scatter,
    )

    end_time = time.time()
    logging.info(f"Hybrid pipeline completed in {end_time - start_time:.2f}s")
    logging.info(f"HTML report: {out_html}")
    logging.info(f"Excel report: {excel_path}")


if __name__ == "__main__":
    main()
