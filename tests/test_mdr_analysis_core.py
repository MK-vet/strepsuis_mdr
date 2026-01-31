"""
Comprehensive Unit Tests for mdr_analysis_core.py

This module provides extensive test coverage for all core MDR analysis functions,
including:
- Bootstrap resampling algorithms
- Chi-square and Fisher's exact tests
- Phi coefficient calculations
- MDR pattern detection
- Co-occurrence analysis
- Association rule mining
- Network construction
- Community detection

Target: 95%+ coverage for mdr_analysis_core.py
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import fisher_exact


# ============================================================================
# Test safe_contingency function
# ============================================================================
class TestSafeContingency:
    """Test the safe_contingency function for various inputs."""

    def test_non_2x2_table_returns_nan(self):
        """Test that non-2x2 tables return NaN values."""
        from strepsuis_mdr.mdr_analysis_core import safe_contingency
        
        # 3x2 table
        table = pd.DataFrame([[10, 20], [30, 40], [50, 60]])
        chi2, p, phi = safe_contingency(table)
        assert np.isnan(chi2)
        assert np.isnan(p)
        assert np.isnan(phi)
        
        # 2x3 table
        table = pd.DataFrame([[10, 20, 30], [40, 50, 60]])
        chi2, p, phi = safe_contingency(table)
        assert np.isnan(chi2)
        assert np.isnan(p)
        assert np.isnan(phi)

    def test_empty_table_returns_nan(self):
        """Test that empty (all zeros) table returns NaN."""
        from strepsuis_mdr.mdr_analysis_core import safe_contingency
        
        table = pd.DataFrame([[0, 0], [0, 0]])
        chi2, p, phi = safe_contingency(table)
        assert np.isnan(chi2)
        assert np.isnan(p)
        assert np.isnan(phi)

    def test_row_with_zero_total_returns_nan(self):
        """Test table with row/column of zeros."""
        from strepsuis_mdr.mdr_analysis_core import safe_contingency
        
        table = pd.DataFrame([[0, 0], [10, 20]])
        chi2, p, phi = safe_contingency(table)
        assert np.isnan(chi2)
        assert np.isnan(p)
        assert np.isnan(phi)

    def test_column_with_zero_total_returns_nan(self):
        """Test table with column of zeros."""
        from strepsuis_mdr.mdr_analysis_core import safe_contingency
        
        table = pd.DataFrame([[0, 10], [0, 20]])
        chi2, p, phi = safe_contingency(table)
        assert np.isnan(chi2)
        assert np.isnan(p)
        assert np.isnan(phi)

    def test_chi_square_path(self):
        """Test chi-square path for large expected counts."""
        from strepsuis_mdr.mdr_analysis_core import safe_contingency
        
        # Create table with large expected counts (triggers chi-square)
        table = pd.DataFrame([[50, 30], [20, 40]])
        chi2, p, phi = safe_contingency(table)
        
        # Should get valid results
        assert not np.isnan(chi2)
        assert not np.isnan(p)
        assert not np.isnan(phi)
        
        # P-value should be between 0 and 1
        assert 0 <= p <= 1
        
        # Phi should be between -1 and 1
        assert -1 <= phi <= 1

    def test_fisher_exact_path(self):
        """Test Fisher's exact path for small expected counts."""
        from strepsuis_mdr.mdr_analysis_core import safe_contingency
        
        # Create table with small expected counts (triggers Fisher's exact)
        table = pd.DataFrame([[2, 1], [1, 2]])
        chi2, p, phi = safe_contingency(table)
        
        # Should get valid results
        assert not np.isnan(chi2)
        assert not np.isnan(p)
        assert not np.isnan(phi)
        
        # Verify Fisher's exact p-value matches scipy
        _, p_scipy = fisher_exact(table)
        np.testing.assert_almost_equal(p, p_scipy, decimal=10)

    def test_perfect_positive_association(self):
        """Test perfect positive association (phi = 1)."""
        from strepsuis_mdr.mdr_analysis_core import safe_contingency
        
        # Perfect positive association (all on diagonal)
        table = pd.DataFrame([[50, 0], [0, 50]])
        chi2, p, phi = safe_contingency(table)
        
        np.testing.assert_almost_equal(phi, 1.0, decimal=10)
        assert p < 0.001  # Should be highly significant

    def test_perfect_negative_association(self):
        """Test perfect negative association (phi = -1)."""
        from strepsuis_mdr.mdr_analysis_core import safe_contingency
        
        # Perfect negative association (all off diagonal)
        table = pd.DataFrame([[0, 50], [50, 0]])
        chi2, p, phi = safe_contingency(table)
        
        np.testing.assert_almost_equal(phi, -1.0, decimal=10)
        assert p < 0.001  # Should be highly significant

    def test_independence(self):
        """Test statistical independence (phi ≈ 0)."""
        from strepsuis_mdr.mdr_analysis_core import safe_contingency
        
        # Independence (proportional cells)
        table = pd.DataFrame([[25, 25], [25, 25]])
        chi2, p, phi = safe_contingency(table)
        
        np.testing.assert_almost_equal(phi, 0.0, decimal=10)
        np.testing.assert_almost_equal(chi2, 0.0, decimal=10)


# ============================================================================
# Test add_significance_stars function
# ============================================================================
class TestAddSignificanceStars:
    """Test the significance stars formatting function."""

    def test_none_returns_empty(self):
        """Test that None input returns empty string."""
        from strepsuis_mdr.mdr_analysis_core import add_significance_stars
        
        assert add_significance_stars(None) == ""

    def test_nan_returns_empty(self):
        """Test that NaN input returns empty string."""
        from strepsuis_mdr.mdr_analysis_core import add_significance_stars
        
        assert add_significance_stars(np.nan) == ""

    def test_three_stars_for_highly_significant(self):
        """Test *** for p < 0.001."""
        from strepsuis_mdr.mdr_analysis_core import add_significance_stars
        
        result = add_significance_stars(0.0001)
        assert "***" in result
        assert "**" not in result.replace("***", "")

    def test_two_stars_for_significant(self):
        """Test ** for 0.001 <= p < 0.01."""
        from strepsuis_mdr.mdr_analysis_core import add_significance_stars
        
        result = add_significance_stars(0.005)
        assert "**" in result
        assert "***" not in result

    def test_one_star_for_marginally_significant(self):
        """Test * for 0.01 <= p < 0.05."""
        from strepsuis_mdr.mdr_analysis_core import add_significance_stars
        
        result = add_significance_stars(0.03)
        assert "*" in result
        # Should not contain ** or *** (single star only)
        assert result.count("*") == 1

    def test_no_stars_for_non_significant(self):
        """Test no stars for p >= 0.05."""
        from strepsuis_mdr.mdr_analysis_core import add_significance_stars
        
        result = add_significance_stars(0.1)
        assert "*" not in result


# ============================================================================
# Test bootstrap functions
# ============================================================================
class TestBootstrapFunctions:
    """Test bootstrap resampling functions."""

    def test_bootstrap_col_basic(self):
        """Test _bootstrap_col with simple data."""
        from strepsuis_mdr.mdr_analysis_core import _bootstrap_col
        
        np.random.seed(42)
        data = np.array([1, 1, 1, 0, 0])  # 60% ones
        
        mean, ci_low, ci_high = _bootstrap_col(data, 5, 1000, 0.95)
        
        # Mean should be close to 0.6
        assert 0.4 < mean < 0.8
        
        # CI should contain the mean
        assert ci_low <= mean <= ci_high
        
        # CI bounds should be valid
        assert 0 <= ci_low <= 1
        assert 0 <= ci_high <= 1
        assert ci_low <= ci_high

    def test_bootstrap_col_all_ones(self):
        """Test _bootstrap_col with all ones."""
        from strepsuis_mdr.mdr_analysis_core import _bootstrap_col
        
        np.random.seed(42)
        data = np.array([1, 1, 1, 1, 1])
        
        mean, ci_low, ci_high = _bootstrap_col(data, 5, 1000, 0.95)
        
        # Mean should be exactly 1.0
        np.testing.assert_almost_equal(mean, 1.0, decimal=5)
        
        # CI should be [1.0, 1.0]
        np.testing.assert_almost_equal(ci_low, 1.0, decimal=5)
        np.testing.assert_almost_equal(ci_high, 1.0, decimal=5)

    def test_bootstrap_col_all_zeros(self):
        """Test _bootstrap_col with all zeros."""
        from strepsuis_mdr.mdr_analysis_core import _bootstrap_col
        
        np.random.seed(42)
        data = np.array([0, 0, 0, 0, 0])
        
        mean, ci_low, ci_high = _bootstrap_col(data, 5, 1000, 0.95)
        
        # Mean should be exactly 0.0
        np.testing.assert_almost_equal(mean, 0.0, decimal=5)
        
        # CI should be [0.0, 0.0]
        np.testing.assert_almost_equal(ci_low, 0.0, decimal=5)
        np.testing.assert_almost_equal(ci_high, 0.0, decimal=5)

    def test_compute_bootstrap_ci_basic(self):
        """Test compute_bootstrap_ci with simple DataFrame."""
        from strepsuis_mdr.mdr_analysis_core import compute_bootstrap_ci
        
        np.random.seed(42)
        df = pd.DataFrame({
            'A': [1, 1, 1, 0, 0, 0, 1, 0, 1, 0],  # 50%
            'B': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # 50%
        })
        
        result = compute_bootstrap_ci(df, n_iter=500, confidence_level=0.95)
        
        # Should have correct columns
        assert 'ColumnName' in result.columns
        assert 'Mean' in result.columns
        assert 'CI_Lower' in result.columns
        assert 'CI_Upper' in result.columns
        
        # Should have one row per column
        assert len(result) == 2
        
        # Means should be around 50%
        for _, row in result.iterrows():
            assert 30 < row['Mean'] < 70  # Allowing for bootstrap variability

    def test_compute_bootstrap_ci_empty_dataframe(self):
        """Test compute_bootstrap_ci with empty DataFrame."""
        from strepsuis_mdr.mdr_analysis_core import compute_bootstrap_ci
        
        df = pd.DataFrame()
        result = compute_bootstrap_ci(df)
        
        assert len(result) == 0
        assert 'ColumnName' in result.columns


# ============================================================================
# Test MDR analysis functions
# ============================================================================
class TestMDRFunctions:
    """Test MDR-specific analysis functions."""

    def test_build_class_resistance(self):
        """Test building class resistance matrix."""
        from strepsuis_mdr.mdr_analysis_core import build_class_resistance, ANTIBIOTIC_CLASSES
        
        # Create test data with some antibiotics
        data = pd.DataFrame({
            'Oxytetracycline': [1, 0, 1, 0],
            'Doxycycline': [0, 1, 1, 0],
            'Tulathromycin': [1, 1, 0, 0],
            'Penicillin': [0, 0, 1, 1],
        })
        
        pheno_cols = list(data.columns)
        result = build_class_resistance(data, pheno_cols)
        
        # Should have class columns
        assert 'Tetracyclines' in result.columns
        assert 'Macrolides' in result.columns
        assert 'Penicillins' in result.columns
        
        # Tetracyclines: max of Oxytetracycline and Doxycycline
        # Row 0: max(1, 0) = 1
        # Row 1: max(0, 1) = 1
        # Row 2: max(1, 1) = 1
        # Row 3: max(0, 0) = 0
        expected_tetracyclines = [1, 1, 1, 0]
        assert list(result['Tetracyclines']) == expected_tetracyclines

    def test_identify_mdr_isolates_default_threshold(self):
        """Test MDR identification with default threshold (3)."""
        from strepsuis_mdr.mdr_analysis_core import identify_mdr_isolates
        
        class_df = pd.DataFrame({
            'Tetracyclines': [1, 1, 1, 0],
            'Macrolides': [1, 1, 0, 0],
            'Penicillins': [1, 0, 1, 0],
            'Aminoglycosides': [0, 1, 1, 1],
        })
        
        result = identify_mdr_isolates(class_df, threshold=3)
        
        # Row 0: 3 classes resistant (1+1+1+0) = MDR
        # Row 1: 3 classes resistant (1+1+0+1) = MDR
        # Row 2: 3 classes resistant (1+0+1+1) = MDR
        # Row 3: 1 class resistant = not MDR
        expected = pd.Series([True, True, True, False])
        pd.testing.assert_series_equal(result.reset_index(drop=True), expected)

    def test_identify_mdr_isolates_custom_threshold(self):
        """Test MDR identification with custom threshold."""
        from strepsuis_mdr.mdr_analysis_core import identify_mdr_isolates
        
        class_df = pd.DataFrame({
            'Tetracyclines': [1, 1, 0, 0],
            'Macrolides': [1, 0, 0, 0],
        })
        
        result = identify_mdr_isolates(class_df, threshold=2)
        
        # Row 0: 2 classes = MDR with threshold 2
        # Row 1: 1 class = not MDR
        expected = pd.Series([True, False, False, False])
        pd.testing.assert_series_equal(result.reset_index(drop=True), expected)


# ============================================================================
# Test AMR gene functions
# ============================================================================
class TestAMRGeneFunctions:
    """Test AMR gene extraction and pattern functions."""

    def test_extract_amr_genes_numeric(self):
        """Test extracting AMR genes from numeric data."""
        from strepsuis_mdr.mdr_analysis_core import extract_amr_genes
        
        data = pd.DataFrame({
            'gene1': [1, 0, 1, 0],
            'gene2': [0, 1, 0, 1],
            'gene3': [1, 1, 1, 1],
        })
        
        result = extract_amr_genes(data, list(data.columns))
        
        # Should be all 0/1 values
        assert result['gene1'].isin([0, 1]).all()
        assert result['gene2'].isin([0, 1]).all()
        assert result['gene3'].isin([0, 1]).all()
        
        # Values should match input
        pd.testing.assert_frame_equal(result, data.astype(int))

    def test_extract_amr_genes_non_numeric(self):
        """Test extracting AMR genes from non-numeric data."""
        from strepsuis_mdr.mdr_analysis_core import extract_amr_genes
        
        data = pd.DataFrame({
            'gene1': ['present', '', 'present', np.nan],
            'gene2': ['yes', 'no', '', 'yes'],
        })
        
        result = extract_amr_genes(data, list(data.columns))
        
        # Should convert to binary
        assert result['gene1'].isin([0, 1]).all()
        assert result['gene2'].isin([0, 1]).all()

    def test_get_mdr_patterns_pheno(self):
        """Test getting phenotypic MDR patterns."""
        from strepsuis_mdr.mdr_analysis_core import get_mdr_patterns_pheno
        
        class_df = pd.DataFrame({
            'Tetracyclines': [1, 1, 0],
            'Macrolides': [1, 0, 0],
            'Penicillins': [0, 1, 0],
        })
        
        result = get_mdr_patterns_pheno(class_df)
        
        # Row 0: Macrolides, Tetracyclines (sorted)
        # Row 1: Penicillins, Tetracyclines (sorted)
        # Row 2: No_Resistance
        assert result.iloc[0] == ('Macrolides', 'Tetracyclines')
        assert result.iloc[1] == ('Penicillins', 'Tetracyclines')
        assert result.iloc[2] == ('No_Resistance',)

    def test_get_mdr_patterns_geno(self):
        """Test getting genotypic MDR patterns."""
        from strepsuis_mdr.mdr_analysis_core import get_mdr_patterns_geno
        
        gene_df = pd.DataFrame({
            'tetA': [1, 1, 0],
            'ermB': [1, 0, 0],
            'blaTEM': [0, 1, 0],
        })
        
        result = get_mdr_patterns_geno(gene_df)
        
        # Row 0: blaTEM, ermB, tetA sorted -> (blaTEM removed), (ermB, tetA)
        assert 'ermB' in result.iloc[0]
        assert 'tetA' in result.iloc[0]
        assert result.iloc[2] == ('No_Genes',)


# ============================================================================
# Test bootstrap pattern frequency
# ============================================================================
class TestBootstrapPatternFreq:
    """Test bootstrap pattern frequency calculation."""

    def test_basic_pattern_frequency(self):
        """Test basic pattern frequency calculation."""
        from strepsuis_mdr.mdr_analysis_core import bootstrap_pattern_freq
        
        patterns = pd.Series([
            ('A', 'B'), ('A', 'B'), ('A', 'B'),
            ('C',), ('C',),
            ('D', 'E'),
        ])
        
        result = bootstrap_pattern_freq(patterns, n_iter=500, conf_level=0.95)
        
        # Should have Pattern, Count, Frequency(%), CI_Lower, CI_Upper
        assert 'Pattern' in result.columns
        assert 'Count' in result.columns
        assert 'Frequency(%)' in result.columns
        assert 'CI_Lower' in result.columns
        assert 'CI_Upper' in result.columns
        
        # Should have 3 unique patterns
        assert len(result) == 3
        
        # First pattern should be A, B with count 3
        assert result.iloc[0]['Count'] == 3

    def test_empty_patterns(self):
        """Test with empty patterns series."""
        from strepsuis_mdr.mdr_analysis_core import bootstrap_pattern_freq
        
        patterns = pd.Series([], dtype=object)
        result = bootstrap_pattern_freq(patterns)
        
        assert len(result) == 0


# ============================================================================
# Test co-occurrence analysis functions
# ============================================================================
class TestCooccurrence:
    """Test co-occurrence analysis functions."""

    def test_pairwise_cooccurrence_basic(self):
        """Test basic pairwise co-occurrence."""
        from strepsuis_mdr.mdr_analysis_core import pairwise_cooccurrence
        
        # Create data with clear co-occurrence pattern
        df = pd.DataFrame({
            'A': [1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
            'B': [1, 1, 1, 1, 0, 0, 0, 0, 1, 1],  # Highly correlated with A
            'C': [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],  # Anti-correlated with A
        })
        
        result = pairwise_cooccurrence(df, alpha=0.05)
        
        # Should have Item1, Item2, Phi columns
        assert 'Item1' in result.columns
        assert 'Item2' in result.columns
        assert 'Phi' in result.columns

    def test_pairwise_cooccurrence_single_column(self):
        """Test co-occurrence with single column (should return empty)."""
        from strepsuis_mdr.mdr_analysis_core import pairwise_cooccurrence
        
        df = pd.DataFrame({'A': [1, 0, 1, 0]})
        result = pairwise_cooccurrence(df)
        
        assert len(result) == 0

    def test_phenotype_gene_cooccurrence_basic(self):
        """Test phenotype-gene co-occurrence."""
        from strepsuis_mdr.mdr_analysis_core import phenotype_gene_cooccurrence
        
        phen_df = pd.DataFrame({
            'Tetracycline_R': [1, 1, 0, 0, 1, 1, 0, 0],
        })
        gene_df = pd.DataFrame({
            'tetA': [1, 1, 0, 0, 1, 1, 0, 0],  # Perfect correlation
        })
        
        result = phenotype_gene_cooccurrence(phen_df, gene_df, alpha=0.05)
        
        # Should find the association if significant
        assert 'Phenotype' in result.columns
        assert 'Gene' in result.columns

    def test_phenotype_gene_cooccurrence_empty_inputs(self):
        """Test with empty inputs."""
        from strepsuis_mdr.mdr_analysis_core import phenotype_gene_cooccurrence
        
        # Empty phenotype DataFrame
        result = phenotype_gene_cooccurrence(pd.DataFrame(), pd.DataFrame({'A': [1, 0]}))
        assert len(result) == 0
        
        # Empty gene DataFrame
        result = phenotype_gene_cooccurrence(pd.DataFrame({'A': [1, 0]}), pd.DataFrame())
        assert len(result) == 0


# ============================================================================
# Test association rules
# ============================================================================
class TestAssociationRules:
    """Test association rule mining functions."""

    def test_association_rules_phenotypic_basic(self):
        """Test phenotypic association rules."""
        from strepsuis_mdr.mdr_analysis_core import association_rules_phenotypic
        
        # Create data where some classes co-occur frequently
        np.random.seed(42)
        df = pd.DataFrame({
            'Tetracyclines': [1] * 40 + [0] * 10,
            'Macrolides': [1] * 35 + [0] * 15,
            'Penicillins': [1] * 30 + [0] * 20,
        })
        
        result = association_rules_phenotypic(df, min_support=0.1, lift_thresh=1.0)
        
        # Result may be empty if mlxtend is not available or no rules found
        # Just check the function runs without error
        assert isinstance(result, pd.DataFrame)

    def test_association_rules_phenotypic_empty(self):
        """Test with empty DataFrame."""
        from strepsuis_mdr.mdr_analysis_core import association_rules_phenotypic
        
        result = association_rules_phenotypic(pd.DataFrame())
        assert len(result) == 0

    def test_association_rules_genes_basic(self):
        """Test gene association rules."""
        from strepsuis_mdr.mdr_analysis_core import association_rules_genes
        
        np.random.seed(42)
        df = pd.DataFrame({
            'tetA': [1] * 40 + [0] * 10,
            'tetM': [1] * 38 + [0] * 12,
            'ermB': [1] * 35 + [0] * 15,
        })
        
        result = association_rules_genes(df, min_support=0.1, lift_thresh=1.0)
        assert isinstance(result, pd.DataFrame)


# ============================================================================
# Test network functions
# ============================================================================
class TestNetworkFunctions:
    """Test network construction and analysis functions."""

    def test_build_hybrid_network_basic(self):
        """Test basic hybrid network construction."""
        from strepsuis_mdr.mdr_analysis_core import build_hybrid_co_resistance_network
        import networkx as nx
        
        # Create data with some associations
        data = pd.DataFrame({
            'Penicillin': [1, 1, 1, 1, 0, 0, 0, 0],
            'Ampicillin': [1, 1, 1, 1, 0, 0, 0, 0],  # Correlated
            'blaTEM': [1, 1, 1, 1, 0, 0, 0, 0],  # Also correlated
        })
        
        pheno_cols = ['Penicillin', 'Ampicillin']
        gene_cols = ['blaTEM']
        
        G = build_hybrid_co_resistance_network(data, pheno_cols, gene_cols)
        
        assert isinstance(G, nx.Graph)

    def test_build_hybrid_network_empty(self):
        """Test with empty column lists."""
        from strepsuis_mdr.mdr_analysis_core import build_hybrid_co_resistance_network
        import networkx as nx
        
        data = pd.DataFrame({'A': [1, 0]})
        G = build_hybrid_co_resistance_network(data, [], [])
        
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == 0

    def test_compute_louvain_communities_basic(self):
        """Test Louvain community detection."""
        from strepsuis_mdr.mdr_analysis_core import compute_louvain_communities
        import networkx as nx
        
        # Create a simple graph with communities
        G = nx.Graph()
        G.add_edge('A', 'B', phi=0.5)
        G.add_edge('B', 'C', phi=0.4)
        G.add_edge('D', 'E', phi=0.6)
        
        result = compute_louvain_communities(G)
        
        assert 'Node' in result.columns
        assert 'Community' in result.columns

    def test_compute_louvain_communities_empty_graph(self):
        """Test with empty graph."""
        from strepsuis_mdr.mdr_analysis_core import compute_louvain_communities
        import networkx as nx
        
        G = nx.Graph()
        result = compute_louvain_communities(G)
        
        assert len(result) == 0


# ============================================================================
# Test HTML/report generation functions
# ============================================================================
class TestReportFunctions:
    """Test report generation functions."""

    def test_df_to_html_basic(self):
        """Test DataFrame to HTML conversion."""
        from strepsuis_mdr.mdr_analysis_core import df_to_html
        
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4.0, 5.0, 6.0]})
        html = df_to_html(df, "Test Table")
        
        assert "Test Table" in html
        assert "<table" in html

    def test_df_to_html_empty(self):
        """Test with empty DataFrame."""
        from strepsuis_mdr.mdr_analysis_core import df_to_html
        
        df = pd.DataFrame()
        html = df_to_html(df, "Empty Table")
        
        assert "Empty Table" in html
        assert "No data" in html


# ============================================================================
# Test network visualization
# ============================================================================
class TestNetworkVisualization:
    """Test network visualization functions."""

    def test_create_hybrid_network_figure_basic(self):
        """Test basic network figure creation."""
        from strepsuis_mdr.mdr_analysis_core import create_hybrid_network_figure
        import networkx as nx
        
        G = nx.Graph()
        G.add_node('A', node_type='Phenotype')
        G.add_node('B', node_type='Genotype')
        G.add_edge('A', 'B', phi=0.5, pvalue=0.01, edge_type='pheno-gene')
        
        html, fig = create_hybrid_network_figure(G)
        
        assert isinstance(html, str)
        assert "<div" in html or "plotly" in html.lower()

    def test_create_hybrid_network_figure_empty_graph(self):
        """Test with empty graph."""
        from strepsuis_mdr.mdr_analysis_core import create_hybrid_network_figure
        import networkx as nx
        
        G = nx.Graph()
        html, fig = create_hybrid_network_figure(G)
        
        assert "No hybrid network" in html
        assert fig is None


# ============================================================================
# Test ANTIBIOTIC_CLASSES constant
# ============================================================================
class TestAntibioticClasses:
    """Test antibiotic class definitions."""

    def test_antibiotic_classes_structure(self):
        """Test that ANTIBIOTIC_CLASSES has correct structure."""
        from strepsuis_mdr.mdr_analysis_core import ANTIBIOTIC_CLASSES
        
        assert isinstance(ANTIBIOTIC_CLASSES, dict)
        
        # Should have expected classes
        expected_classes = [
            'Tetracyclines', 'Macrolides', 'Aminoglycosides',
            'Pleuromutilins', 'Sulfonamides', 'Fluoroquinolones',
            'Penicillins', 'Cephalosporins', 'Phenicols'
        ]
        
        for cls in expected_classes:
            assert cls in ANTIBIOTIC_CLASSES
            assert isinstance(ANTIBIOTIC_CLASSES[cls], list)
            assert len(ANTIBIOTIC_CLASSES[cls]) > 0


# ============================================================================
# Integration tests
# ============================================================================
class TestIntegration:
    """Integration tests for combined functionality."""

    def test_full_pipeline_simulation(self):
        """Test a simplified full pipeline."""
        from strepsuis_mdr.mdr_analysis_core import (
            build_class_resistance,
            identify_mdr_isolates,
            compute_bootstrap_ci,
            pairwise_cooccurrence,
        )
        
        # Create realistic test data
        np.random.seed(42)
        n = 50
        data = pd.DataFrame({
            'Oxytetracycline': np.random.binomial(1, 0.6, n),
            'Doxycycline': np.random.binomial(1, 0.5, n),
            'Tulathromycin': np.random.binomial(1, 0.4, n),
            'Penicillin': np.random.binomial(1, 0.3, n),
            'Ampicillin': np.random.binomial(1, 0.35, n),
        })
        
        pheno_cols = list(data.columns)
        
        # Step 1: Build class resistance
        class_res = build_class_resistance(data, pheno_cols)
        assert len(class_res) == n
        
        # Step 2: Identify MDR isolates
        mdr_mask = identify_mdr_isolates(class_res, threshold=2)
        assert mdr_mask.dtype == bool
        
        # Step 3: Compute bootstrap CI
        mdr_class_res = class_res[mdr_mask]
        if len(mdr_class_res) > 0:
            freq = compute_bootstrap_ci(mdr_class_res, n_iter=100)
            assert 'Mean' in freq.columns
        
        # Step 4: Compute co-occurrence (if enough MDR isolates)
        if len(mdr_class_res) >= 5:
            coocc = pairwise_cooccurrence(mdr_class_res, alpha=0.2)  # Relaxed alpha
            assert isinstance(coocc, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
