"""
Tests for report generation functions.

Target: Increase coverage for generate_html_report, generate_excel_report, and save_report
from 0% to 80%+
"""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import networkx as nx
import pandas as pd
import pytest

from strepsuis_mdr.mdr_analysis_core import (
    generate_excel_report,
    generate_html_report,
    save_report,
)


class TestReportGeneration:
    """Test report generation functions."""

    @pytest.fixture
    def sample_data(self):
        """Create minimal sample data for testing."""
        data = pd.DataFrame({
            "Strain_ID": [f"S{i}" for i in range(10)],
            "Oxytetracycline": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "Penicillin": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "Gene_A": [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            "Gene_B": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        })
        return data

    @pytest.fixture
    def sample_analysis_results(self, sample_data):
        """Create sample analysis results."""
        class_res_all = pd.DataFrame({
            "Tetracyclines": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "Penicillins": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        })
        class_res_mdr = class_res_all.iloc[:5]
        
        amr_all = pd.DataFrame({
            "Gene_A": [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            "Gene_B": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        })
        amr_mdr = amr_all.iloc[:5]
        
        freq_pheno_all = pd.DataFrame({
            "ColumnName": ["Tetracyclines", "Penicillins"],
            "Mean": [50.0, 50.0],
            "CI_Lower": [40.0, 40.0],
            "CI_Upper": [60.0, 60.0],
        })
        freq_pheno_mdr = freq_pheno_all.copy()
        freq_gene_all = pd.DataFrame({
            "ColumnName": ["Gene_A", "Gene_B"],
            "Mean": [50.0, 50.0],
            "CI_Lower": [40.0, 40.0],
            "CI_Upper": [60.0, 60.0],
        })
        freq_gene_mdr = freq_gene_all.copy()
        
        pat_pheno_mdr = pd.DataFrame({
            "Pattern": ["Tetracyclines", "Penicillins"],
            "Mean": [50.0, 50.0],
            "CI_Lower": [40.0, 40.0],
            "CI_Upper": [60.0, 60.0],
        })
        pat_gene_mdr = pd.DataFrame({
            "Pattern": ["Gene_A", "Gene_B"],
            "Mean": [50.0, 50.0],
            "CI_Lower": [40.0, 40.0],
            "CI_Upper": [60.0, 60.0],
        })
        
        coocc_pheno_mdr = pd.DataFrame({
            "Item1": ["Tetracyclines"],
            "Item2": ["Penicillins"],
            "Phi": [0.5],
            "Corrected_p": [0.01],
        })
        coocc_gene_mdr = pd.DataFrame({
            "Item1": ["Gene_A"],
            "Item2": ["Gene_B"],
            "Phi": [0.3],
            "Corrected_p": [0.05],
        })
        
        gene_pheno_assoc = pd.DataFrame({
            "Phenotype": ["Tetracyclines"],
            "Gene": ["Gene_A"],
            "Phi": [0.4],
            "Corrected_p": [0.02],
        })
        
        assoc_rules_pheno = pd.DataFrame({
            "Antecedents": ["Tetracyclines"],
            "Consequents": ["Penicillins"],
            "Support": [0.3],
            "Confidence": [0.6],
            "Lift": [1.2],
        })
        assoc_rules_genes = pd.DataFrame({
            "Antecedents": ["Gene_A"],
            "Consequents": ["Gene_B"],
            "Support": [0.2],
            "Confidence": [0.5],
            "Lift": [1.1],
        })
        
        hybrid_net = nx.Graph()
        hybrid_net.add_edge("Tetracyclines", "Penicillins", pvalue=0.01, phi=0.5)
        hybrid_net.add_edge("Gene_A", "Gene_B", pvalue=0.05, phi=0.3)
        
        edges_df = pd.DataFrame({
            "Source": ["Tetracyclines", "Gene_A"],
            "Target": ["Penicillins", "Gene_B"],
            "P_Value": [0.01, 0.05],
            "Phi": [0.5, 0.3],
        })
        
        comm_df = pd.DataFrame({
            "Node": ["Tetracyclines", "Penicillins", "Gene_A", "Gene_B"],
            "Community": [0, 0, 1, 1],
        })
        
        net_html = "<div>Network visualization</div>"
        
        return {
            "class_res_all": class_res_all,
            "class_res_mdr": class_res_mdr,
            "amr_all": amr_all,
            "amr_mdr": amr_mdr,
            "freq_pheno_all": freq_pheno_all,
            "freq_pheno_mdr": freq_pheno_mdr,
            "freq_gene_all": freq_gene_all,
            "freq_gene_mdr": freq_gene_mdr,
            "pat_pheno_mdr": pat_pheno_mdr,
            "pat_gene_mdr": pat_gene_mdr,
            "coocc_pheno_mdr": coocc_pheno_mdr,
            "coocc_gene_mdr": coocc_gene_mdr,
            "gene_pheno_assoc": gene_pheno_assoc,
            "assoc_rules_pheno": assoc_rules_pheno,
            "assoc_rules_genes": assoc_rules_genes,
            "hybrid_net": hybrid_net,
            "edges_df": edges_df,
            "comm_df": comm_df,
            "net_html": net_html,
        }

    def test_generate_html_report_basic(self, sample_data, sample_analysis_results, tmp_path):
        """Test basic HTML report generation."""
        os.chdir(tmp_path)
        os.makedirs("output", exist_ok=True)
        
        html = generate_html_report(
            data=sample_data,
            class_res_all=sample_analysis_results["class_res_all"],
            class_res_mdr=sample_analysis_results["class_res_mdr"],
            amr_all=sample_analysis_results["amr_all"],
            amr_mdr=sample_analysis_results["amr_mdr"],
            freq_pheno_all=sample_analysis_results["freq_pheno_all"],
            freq_pheno_mdr=sample_analysis_results["freq_pheno_mdr"],
            freq_gene_all=sample_analysis_results["freq_gene_all"],
            freq_gene_mdr=sample_analysis_results["freq_gene_mdr"],
            pat_pheno_mdr=sample_analysis_results["pat_pheno_mdr"],
            pat_gene_mdr=sample_analysis_results["pat_gene_mdr"],
            coocc_pheno_mdr=sample_analysis_results["coocc_pheno_mdr"],
            coocc_gene_mdr=sample_analysis_results["coocc_gene_mdr"],
            gene_pheno_assoc=sample_analysis_results["gene_pheno_assoc"],
            assoc_rules_pheno=sample_analysis_results["assoc_rules_pheno"],
            assoc_rules_genes=sample_analysis_results["assoc_rules_genes"],
            hybrid_net=sample_analysis_results["hybrid_net"],
            edges_df=sample_analysis_results["edges_df"],
            comm_df=sample_analysis_results["comm_df"],
            net_html=sample_analysis_results["net_html"],
        )
        
        assert html is not None
        assert len(html) > 1000
        assert "<!DOCTYPE html>" in html
        assert "StrepSuis-AMRPat" in html

    def test_generate_html_report_with_innovations(self, sample_data, sample_analysis_results, tmp_path):
        """Test HTML report generation with innovation features."""
        os.chdir(tmp_path)
        os.makedirs("output", exist_ok=True)
        
        risk_scores = pd.DataFrame({
            "Strain_ID": [f"S{i}" for i in range(5)],
            "Network_Risk_Score": [0.5, 0.6, 0.4, 0.7, 0.3],
            "MDR_Predicted": [True, True, False, True, False],
            "Percentile_Rank": [50, 60, 40, 70, 30],
        })
        
        seq_patterns_amr = pd.DataFrame({
            "Pattern": ["Gene_A→Gene_B"],
            "Support": [0.3],
            "Confidence": [0.6],
            "Lift": [1.2],
            "P_Value": [0.01],
        })
        
        html = generate_html_report(
            data=sample_data,
            class_res_all=sample_analysis_results["class_res_all"],
            class_res_mdr=sample_analysis_results["class_res_mdr"],
            amr_all=sample_analysis_results["amr_all"],
            amr_mdr=sample_analysis_results["amr_mdr"],
            freq_pheno_all=sample_analysis_results["freq_pheno_all"],
            freq_pheno_mdr=sample_analysis_results["freq_pheno_mdr"],
            freq_gene_all=sample_analysis_results["freq_gene_all"],
            freq_gene_mdr=sample_analysis_results["freq_gene_mdr"],
            pat_pheno_mdr=sample_analysis_results["pat_pheno_mdr"],
            pat_gene_mdr=sample_analysis_results["pat_gene_mdr"],
            coocc_pheno_mdr=sample_analysis_results["coocc_pheno_mdr"],
            coocc_gene_mdr=sample_analysis_results["coocc_gene_mdr"],
            gene_pheno_assoc=sample_analysis_results["gene_pheno_assoc"],
            assoc_rules_pheno=sample_analysis_results["assoc_rules_pheno"],
            assoc_rules_genes=sample_analysis_results["assoc_rules_genes"],
            hybrid_net=sample_analysis_results["hybrid_net"],
            edges_df=sample_analysis_results["edges_df"],
            comm_df=sample_analysis_results["comm_df"],
            net_html=sample_analysis_results["net_html"],
            risk_scores=risk_scores,
            risk_html="<div>Risk visualization</div>",
            seq_patterns_amr=seq_patterns_amr,
            seq_html_amr="<div>Sequential patterns AMR</div>",
        )
        
        assert html is not None
        assert "Risk" in html or "risk" in html.lower()

    def test_save_report(self, tmp_path):
        """Test save_report function."""
        os.chdir(tmp_path)
        os.makedirs("output", exist_ok=True)
        
        html_content = "<html><body>Test Report</body></html>"
        filepath = save_report(html_content, "output")
        
        assert filepath is not None
        assert os.path.exists(filepath)
        assert filepath.endswith(".html")
        
        # Verify content
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        assert "Test Report" in content

    def test_generate_excel_report_basic(self, sample_data, sample_analysis_results, tmp_path):
        """Test basic Excel report generation."""
        os.chdir(tmp_path)
        os.makedirs("output", exist_ok=True)
        
        excel_path = generate_excel_report(
            data=sample_data,
            class_res_all=sample_analysis_results["class_res_all"],
            class_res_mdr=sample_analysis_results["class_res_mdr"],
            amr_all=sample_analysis_results["amr_all"],
            amr_mdr=sample_analysis_results["amr_mdr"],
            freq_pheno_all=sample_analysis_results["freq_pheno_all"],
            freq_pheno_mdr=sample_analysis_results["freq_pheno_mdr"],
            freq_gene_all=sample_analysis_results["freq_gene_all"],
            freq_gene_mdr=sample_analysis_results["freq_gene_mdr"],
            pat_pheno_mdr=sample_analysis_results["pat_pheno_mdr"],
            pat_gene_mdr=sample_analysis_results["pat_gene_mdr"],
            coocc_pheno_mdr=sample_analysis_results["coocc_pheno_mdr"],
            coocc_gene_mdr=sample_analysis_results["coocc_gene_mdr"],
            gene_pheno_assoc=sample_analysis_results["gene_pheno_assoc"],
            assoc_rules_pheno=sample_analysis_results["assoc_rules_pheno"],
            assoc_rules_genes=sample_analysis_results["assoc_rules_genes"],
            hybrid_net=sample_analysis_results["hybrid_net"],
            edges_df=sample_analysis_results["edges_df"],
            comm_df=sample_analysis_results["comm_df"],
        )
        
        assert excel_path is not None
        assert os.path.exists(excel_path)
        assert excel_path.endswith(".xlsx")
        
        # Verify Excel file can be read
        xls = pd.ExcelFile(excel_path)
        assert len(xls.sheet_names) > 0

    def test_generate_excel_report_with_innovations(self, sample_data, sample_analysis_results, tmp_path):
        """Test Excel report generation with innovation features."""
        os.chdir(tmp_path)
        os.makedirs("output", exist_ok=True)
        
        risk_scores = pd.DataFrame({
            "Strain_ID": [f"S{i}" for i in range(5)],
            "Network_Risk_Score": [0.5, 0.6, 0.4, 0.7, 0.3],
            "MDR_Predicted": [True, True, False, True, False],
            "Percentile_Rank": [50, 60, 40, 70, 30],
        })
        
        seq_patterns_amr = pd.DataFrame({
            "Pattern": ["Gene_A→Gene_B"],
            "Support": [0.3],
            "Confidence": [0.6],
            "Lift": [1.2],
            "P_Value": [0.01],
        })
        
        excel_path = generate_excel_report(
            data=sample_data,
            class_res_all=sample_analysis_results["class_res_all"],
            class_res_mdr=sample_analysis_results["class_res_mdr"],
            amr_all=sample_analysis_results["amr_all"],
            amr_mdr=sample_analysis_results["amr_mdr"],
            freq_pheno_all=sample_analysis_results["freq_pheno_all"],
            freq_pheno_mdr=sample_analysis_results["freq_pheno_mdr"],
            freq_gene_all=sample_analysis_results["freq_gene_all"],
            freq_gene_mdr=sample_analysis_results["freq_gene_mdr"],
            pat_pheno_mdr=sample_analysis_results["pat_pheno_mdr"],
            pat_gene_mdr=sample_analysis_results["pat_gene_mdr"],
            coocc_pheno_mdr=sample_analysis_results["coocc_pheno_mdr"],
            coocc_gene_mdr=sample_analysis_results["coocc_gene_mdr"],
            gene_pheno_assoc=sample_analysis_results["gene_pheno_assoc"],
            assoc_rules_pheno=sample_analysis_results["assoc_rules_pheno"],
            assoc_rules_genes=sample_analysis_results["assoc_rules_genes"],
            hybrid_net=sample_analysis_results["hybrid_net"],
            edges_df=sample_analysis_results["edges_df"],
            comm_df=sample_analysis_results["comm_df"],
            risk_scores=risk_scores,
            seq_patterns_amr=seq_patterns_amr,
        )
        
        assert excel_path is not None
        assert os.path.exists(excel_path)
        
        # Verify Excel file has expected sheets
        xls = pd.ExcelFile(excel_path)
        sheet_names = xls.sheet_names
        assert len(sheet_names) > 0

    def test_generate_excel_report_empty_dataframes(self, sample_data, tmp_path):
        """Test Excel report generation with empty DataFrames."""
        os.chdir(tmp_path)
        os.makedirs("output", exist_ok=True)
        
        empty_df = pd.DataFrame()
        empty_net = nx.Graph()
        
        excel_path = generate_excel_report(
            data=sample_data,
            class_res_all=empty_df,
            class_res_mdr=empty_df,
            amr_all=empty_df,
            amr_mdr=empty_df,
            freq_pheno_all=empty_df,
            freq_pheno_mdr=empty_df,
            freq_gene_all=empty_df,
            freq_gene_mdr=empty_df,
            pat_pheno_mdr=empty_df,
            pat_gene_mdr=empty_df,
            coocc_pheno_mdr=empty_df,
            coocc_gene_mdr=empty_df,
            gene_pheno_assoc=empty_df,
            assoc_rules_pheno=empty_df,
            assoc_rules_genes=empty_df,
            hybrid_net=empty_net,
            edges_df=empty_df,
            comm_df=empty_df,
        )
        
        assert excel_path is not None
        assert os.path.exists(excel_path)

    def test_generate_excel_report_with_network_figure(self, sample_data, sample_analysis_results, tmp_path):
        """Test Excel report generation with network figure."""
        os.chdir(tmp_path)
        os.makedirs("output", exist_ok=True)
        
        # Create a mock plotly figure
        import plotly.graph_objects as go
        fig_network = go.Figure()
        fig_network.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
        
        excel_path = generate_excel_report(
            data=sample_data,
            class_res_all=sample_analysis_results["class_res_all"],
            class_res_mdr=sample_analysis_results["class_res_mdr"],
            amr_all=sample_analysis_results["amr_all"],
            amr_mdr=sample_analysis_results["amr_mdr"],
            freq_pheno_all=sample_analysis_results["freq_pheno_all"],
            freq_pheno_mdr=sample_analysis_results["freq_pheno_mdr"],
            freq_gene_all=sample_analysis_results["freq_gene_all"],
            freq_gene_mdr=sample_analysis_results["freq_gene_mdr"],
            pat_pheno_mdr=sample_analysis_results["pat_pheno_mdr"],
            pat_gene_mdr=sample_analysis_results["pat_gene_mdr"],
            coocc_pheno_mdr=sample_analysis_results["coocc_pheno_mdr"],
            coocc_gene_mdr=sample_analysis_results["coocc_gene_mdr"],
            gene_pheno_assoc=sample_analysis_results["gene_pheno_assoc"],
            assoc_rules_pheno=sample_analysis_results["assoc_rules_pheno"],
            assoc_rules_genes=sample_analysis_results["assoc_rules_genes"],
            hybrid_net=sample_analysis_results["hybrid_net"],
            edges_df=sample_analysis_results["edges_df"],
            comm_df=sample_analysis_results["comm_df"],
            fig_network=fig_network,
        )
        
        assert excel_path is not None
        assert os.path.exists(excel_path)

    def test_generate_excel_report_with_all_risk_figures(self, sample_data, sample_analysis_results, tmp_path):
        """Test Excel report generation with all risk scoring figures."""
        os.chdir(tmp_path)
        os.makedirs("output", exist_ok=True)
        
        import plotly.graph_objects as go
        
        risk_scores = pd.DataFrame({
            "Strain_ID": [f"S{i}" for i in range(5)],
            "Network_Risk_Score": [0.5, 0.6, 0.4, 0.7, 0.3],
            "MDR_Predicted": [True, True, False, True, False],
            "Percentile_Rank": [50, 60, 40, 70, 30],
        })
        
        fig_risk_hist = go.Figure()
        fig_risk_hist.add_trace(go.Histogram(x=[0.3, 0.4, 0.5, 0.6, 0.7]))
        
        fig_risk_ranking = go.Figure()
        fig_risk_ranking.add_trace(go.Bar(x=["S1", "S2"], y=[0.6, 0.7]))
        
        fig_risk_dist = go.Figure()
        fig_risk_dist.add_trace(go.Box(y=[0.3, 0.4, 0.5, 0.6, 0.7]))
        
        excel_path = generate_excel_report(
            data=sample_data,
            class_res_all=sample_analysis_results["class_res_all"],
            class_res_mdr=sample_analysis_results["class_res_mdr"],
            amr_all=sample_analysis_results["amr_all"],
            amr_mdr=sample_analysis_results["amr_mdr"],
            freq_pheno_all=sample_analysis_results["freq_pheno_all"],
            freq_pheno_mdr=sample_analysis_results["freq_pheno_mdr"],
            freq_gene_all=sample_analysis_results["freq_gene_all"],
            freq_gene_mdr=sample_analysis_results["freq_gene_mdr"],
            pat_pheno_mdr=sample_analysis_results["pat_pheno_mdr"],
            pat_gene_mdr=sample_analysis_results["pat_gene_mdr"],
            coocc_pheno_mdr=sample_analysis_results["coocc_pheno_mdr"],
            coocc_gene_mdr=sample_analysis_results["coocc_gene_mdr"],
            gene_pheno_assoc=sample_analysis_results["gene_pheno_assoc"],
            assoc_rules_pheno=sample_analysis_results["assoc_rules_pheno"],
            assoc_rules_genes=sample_analysis_results["assoc_rules_genes"],
            hybrid_net=sample_analysis_results["hybrid_net"],
            edges_df=sample_analysis_results["edges_df"],
            comm_df=sample_analysis_results["comm_df"],
            risk_scores=risk_scores,
            fig_risk_hist=fig_risk_hist,
            fig_risk_ranking=fig_risk_ranking,
            fig_risk_dist=fig_risk_dist,
        )
        
        assert excel_path is not None
        assert os.path.exists(excel_path)

    def test_generate_excel_report_with_all_sequential_figures(self, sample_data, sample_analysis_results, tmp_path):
        """Test Excel report generation with all sequential pattern figures."""
        os.chdir(tmp_path)
        os.makedirs("output", exist_ok=True)
        
        import plotly.graph_objects as go
        
        seq_patterns_amr = pd.DataFrame({
            "Pattern": ["Gene_A→Gene_B"],
            "Support": [0.3],
            "Confidence": [0.6],
            "Lift": [1.2],
            "P_Value": [0.01],
        })
        
        seq_patterns_mic = pd.DataFrame({
            "Pattern": ["Tetracyclines→Penicillins"],
            "Support": [0.4],
            "Confidence": [0.7],
            "Lift": [1.3],
            "P_Value": [0.02],
        })
        
        fig_seq_amr_bar = go.Figure()
        fig_seq_amr_bar.add_trace(go.Bar(x=["Gene_A→Gene_B"], y=[0.6]))
        
        fig_seq_amr_scatter = go.Figure()
        fig_seq_amr_scatter.add_trace(go.Scatter(x=[0.3], y=[0.6]))
        
        fig_seq_mic_bar = go.Figure()
        fig_seq_mic_bar.add_trace(go.Bar(x=["Tetracyclines→Penicillins"], y=[0.7]))
        
        fig_seq_mic_scatter = go.Figure()
        fig_seq_mic_scatter.add_trace(go.Scatter(x=[0.4], y=[0.7]))
        
        excel_path = generate_excel_report(
            data=sample_data,
            class_res_all=sample_analysis_results["class_res_all"],
            class_res_mdr=sample_analysis_results["class_res_mdr"],
            amr_all=sample_analysis_results["amr_all"],
            amr_mdr=sample_analysis_results["amr_mdr"],
            freq_pheno_all=sample_analysis_results["freq_pheno_all"],
            freq_pheno_mdr=sample_analysis_results["freq_pheno_mdr"],
            freq_gene_all=sample_analysis_results["freq_gene_all"],
            freq_gene_mdr=sample_analysis_results["freq_gene_mdr"],
            pat_pheno_mdr=sample_analysis_results["pat_pheno_mdr"],
            pat_gene_mdr=sample_analysis_results["pat_gene_mdr"],
            coocc_pheno_mdr=sample_analysis_results["coocc_pheno_mdr"],
            coocc_gene_mdr=sample_analysis_results["coocc_gene_mdr"],
            gene_pheno_assoc=sample_analysis_results["gene_pheno_assoc"],
            assoc_rules_pheno=sample_analysis_results["assoc_rules_pheno"],
            assoc_rules_genes=sample_analysis_results["assoc_rules_genes"],
            hybrid_net=sample_analysis_results["hybrid_net"],
            edges_df=sample_analysis_results["edges_df"],
            comm_df=sample_analysis_results["comm_df"],
            seq_patterns_amr=seq_patterns_amr,
            seq_patterns_mic=seq_patterns_mic,
            fig_seq_amr_bar=fig_seq_amr_bar,
            fig_seq_amr_scatter=fig_seq_amr_scatter,
            fig_seq_mic_bar=fig_seq_mic_bar,
            fig_seq_mic_scatter=fig_seq_mic_scatter,
        )
        
        assert excel_path is not None
        assert os.path.exists(excel_path)

    def test_generate_excel_report_with_none_frequencies(self, sample_data, sample_analysis_results, tmp_path):
        """Test Excel report generation with None frequency DataFrames."""
        os.chdir(tmp_path)
        os.makedirs("output", exist_ok=True)
        
        excel_path = generate_excel_report(
            data=sample_data,
            class_res_all=sample_analysis_results["class_res_all"],
            class_res_mdr=sample_analysis_results["class_res_mdr"],
            amr_all=sample_analysis_results["amr_all"],
            amr_mdr=sample_analysis_results["amr_mdr"],
            freq_pheno_all=None,  # Test None handling
            freq_pheno_mdr=None,
            freq_gene_all=None,
            freq_gene_mdr=None,
            pat_pheno_mdr=sample_analysis_results["pat_pheno_mdr"],
            pat_gene_mdr=sample_analysis_results["pat_gene_mdr"],
            coocc_pheno_mdr=sample_analysis_results["coocc_pheno_mdr"],
            coocc_gene_mdr=sample_analysis_results["coocc_gene_mdr"],
            gene_pheno_assoc=sample_analysis_results["gene_pheno_assoc"],
            assoc_rules_pheno=sample_analysis_results["assoc_rules_pheno"],
            assoc_rules_genes=sample_analysis_results["assoc_rules_genes"],
            hybrid_net=sample_analysis_results["hybrid_net"],
            edges_df=sample_analysis_results["edges_df"],
            comm_df=sample_analysis_results["comm_df"],
        )
        
        assert excel_path is not None
        assert os.path.exists(excel_path)

    def test_generate_excel_report_error_handling_network_figure(self, sample_data, sample_analysis_results, tmp_path):
        """Test error handling when saving network figure fails."""
        os.chdir(tmp_path)
        os.makedirs("output", exist_ok=True)
        
        import plotly.graph_objects as go
        fig_network = go.Figure()
        fig_network.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
        
        # Mock save_plotly_figure_fallback to raise exception
        with patch('strepsuis_mdr.excel_report_utils.ExcelReportGenerator.save_plotly_figure_fallback') as mock_save:
            mock_save.side_effect = Exception("Kaleido error")
            
            # Should not crash, but handle errors gracefully
            excel_path = generate_excel_report(
                data=sample_data,
                class_res_all=sample_analysis_results["class_res_all"],
                class_res_mdr=sample_analysis_results["class_res_mdr"],
                amr_all=sample_analysis_results["amr_all"],
                amr_mdr=sample_analysis_results["amr_mdr"],
                freq_pheno_all=sample_analysis_results["freq_pheno_all"],
                freq_pheno_mdr=sample_analysis_results["freq_pheno_mdr"],
                freq_gene_all=sample_analysis_results["freq_gene_all"],
                freq_gene_mdr=sample_analysis_results["freq_gene_mdr"],
                pat_pheno_mdr=sample_analysis_results["pat_pheno_mdr"],
                pat_gene_mdr=sample_analysis_results["pat_gene_mdr"],
                coocc_pheno_mdr=sample_analysis_results["coocc_pheno_mdr"],
                coocc_gene_mdr=sample_analysis_results["coocc_gene_mdr"],
                gene_pheno_assoc=sample_analysis_results["gene_pheno_assoc"],
                assoc_rules_pheno=sample_analysis_results["assoc_rules_pheno"],
                assoc_rules_genes=sample_analysis_results["assoc_rules_genes"],
                hybrid_net=sample_analysis_results["hybrid_net"],
                edges_df=sample_analysis_results["edges_df"],
                comm_df=sample_analysis_results["comm_df"],
                fig_network=fig_network,
            )
            
            assert excel_path is not None
            assert os.path.exists(excel_path)

    def test_generate_excel_report_error_handling_risk_figures(self, sample_data, sample_analysis_results, tmp_path):
        """Test error handling when saving risk scoring figures fails."""
        os.chdir(tmp_path)
        os.makedirs("output", exist_ok=True)
        
        import plotly.graph_objects as go
        
        risk_scores = pd.DataFrame({
            "Strain_ID": [f"S{i}" for i in range(5)],
            "Network_Risk_Score": [0.5, 0.6, 0.4, 0.7, 0.3],
            "MDR_Predicted": [True, True, False, True, False],
            "Percentile_Rank": [50, 60, 40, 70, 30],
        })
        
        fig_risk_hist = go.Figure()
        fig_risk_ranking = go.Figure()
        fig_risk_dist = go.Figure()
        
        # Mock save_plotly_figure_fallback to raise exception
        with patch('strepsuis_mdr.excel_report_utils.ExcelReportGenerator.save_plotly_figure_fallback') as mock_save:
            mock_save.side_effect = Exception("Kaleido error")
            
            excel_path = generate_excel_report(
                data=sample_data,
                class_res_all=sample_analysis_results["class_res_all"],
                class_res_mdr=sample_analysis_results["class_res_mdr"],
                amr_all=sample_analysis_results["amr_all"],
                amr_mdr=sample_analysis_results["amr_mdr"],
                freq_pheno_all=sample_analysis_results["freq_pheno_all"],
                freq_pheno_mdr=sample_analysis_results["freq_pheno_mdr"],
                freq_gene_all=sample_analysis_results["freq_gene_all"],
                freq_gene_mdr=sample_analysis_results["freq_gene_mdr"],
                pat_pheno_mdr=sample_analysis_results["pat_pheno_mdr"],
                pat_gene_mdr=sample_analysis_results["pat_gene_mdr"],
                coocc_pheno_mdr=sample_analysis_results["coocc_pheno_mdr"],
                coocc_gene_mdr=sample_analysis_results["coocc_gene_mdr"],
                gene_pheno_assoc=sample_analysis_results["gene_pheno_assoc"],
                assoc_rules_pheno=sample_analysis_results["assoc_rules_pheno"],
                assoc_rules_genes=sample_analysis_results["assoc_rules_genes"],
                hybrid_net=sample_analysis_results["hybrid_net"],
                edges_df=sample_analysis_results["edges_df"],
                comm_df=sample_analysis_results["comm_df"],
                risk_scores=risk_scores,
                fig_risk_hist=fig_risk_hist,
                fig_risk_ranking=fig_risk_ranking,
                fig_risk_dist=fig_risk_dist,
            )
            
            assert excel_path is not None
            assert os.path.exists(excel_path)

    def test_generate_excel_report_error_handling_sequential_figures(self, sample_data, sample_analysis_results, tmp_path):
        """Test error handling when saving sequential pattern figures fails."""
        os.chdir(tmp_path)
        os.makedirs("output", exist_ok=True)
        
        import plotly.graph_objects as go
        
        seq_patterns_amr = pd.DataFrame({
            "Pattern": ["Gene_A→Gene_B"],
            "Support": [0.3],
            "Confidence": [0.6],
            "Lift": [1.2],
            "P_Value": [0.01],
        })
        
        seq_patterns_mic = pd.DataFrame({
            "Pattern": ["Tetracyclines→Penicillins"],
            "Support": [0.4],
            "Confidence": [0.7],
            "Lift": [1.3],
            "P_Value": [0.02],
        })
        
        fig_seq_amr_bar = go.Figure()
        fig_seq_amr_scatter = go.Figure()
        fig_seq_mic_bar = go.Figure()
        fig_seq_mic_scatter = go.Figure()
        
        # Mock save_plotly_figure_fallback to raise exception
        with patch('strepsuis_mdr.excel_report_utils.ExcelReportGenerator.save_plotly_figure_fallback') as mock_save:
            mock_save.side_effect = Exception("Kaleido error")
            
            excel_path = generate_excel_report(
                data=sample_data,
                class_res_all=sample_analysis_results["class_res_all"],
                class_res_mdr=sample_analysis_results["class_res_mdr"],
                amr_all=sample_analysis_results["amr_all"],
                amr_mdr=sample_analysis_results["amr_mdr"],
                freq_pheno_all=sample_analysis_results["freq_pheno_all"],
                freq_pheno_mdr=sample_analysis_results["freq_pheno_mdr"],
                freq_gene_all=sample_analysis_results["freq_gene_all"],
                freq_gene_mdr=sample_analysis_results["freq_gene_mdr"],
                pat_pheno_mdr=sample_analysis_results["pat_pheno_mdr"],
                pat_gene_mdr=sample_analysis_results["pat_gene_mdr"],
                coocc_pheno_mdr=sample_analysis_results["coocc_pheno_mdr"],
                coocc_gene_mdr=sample_analysis_results["coocc_gene_mdr"],
                gene_pheno_assoc=sample_analysis_results["gene_pheno_assoc"],
                assoc_rules_pheno=sample_analysis_results["assoc_rules_pheno"],
                assoc_rules_genes=sample_analysis_results["assoc_rules_genes"],
                hybrid_net=sample_analysis_results["hybrid_net"],
                edges_df=sample_analysis_results["edges_df"],
                comm_df=sample_analysis_results["comm_df"],
                seq_patterns_amr=seq_patterns_amr,
                seq_patterns_mic=seq_patterns_mic,
                fig_seq_amr_bar=fig_seq_amr_bar,
                fig_seq_amr_scatter=fig_seq_amr_scatter,
                fig_seq_mic_bar=fig_seq_mic_bar,
                fig_seq_mic_scatter=fig_seq_mic_scatter,
            )
            
            assert excel_path is not None
            assert os.path.exists(excel_path)
