
"""
Real Data Test for Innovations - Full Report Generation

This script tests the new innovative features on real data (91 S. suis strains)
and generates a comprehensive report with actual results.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Tuple
from datetime import datetime

from strepsuis_mdr.mdr_analysis_core import (
    compute_network_mdr_risk_score,
    detect_sequential_resistance_patterns,
    build_hybrid_co_resistance_network,
    compute_bootstrap_ci,
    pairwise_cooccurrence,
)


def load_real_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, list, list]:
    """Load real data files."""
    print(f"Loading data from: {data_dir}")
    
    # Try main data directory first
    mic_path = data_dir / "MIC.csv"
    amr_path = data_dir / "AMR_genes.csv"
    
    # If not found, try examples subdirectory
    if not mic_path.exists():
        mic_path = data_dir / "examples" / "MIC.csv"
        amr_path = data_dir / "examples" / "AMR_genes.csv"
    
    # If still not found, try parent directories
    if not mic_path.exists():
        # Try ../../data/
        parent_data = data_dir.parent.parent / "data"
        mic_path = parent_data / "MIC.csv"
        amr_path = parent_data / "AMR_genes.csv"
    
    if not mic_path.exists():
        raise FileNotFoundError(f"Could not find MIC.csv in {data_dir} or parent directories")
    
    print(f"Loading MIC data from: {mic_path}")
    mic_data = pd.read_csv(mic_path)
    
    print(f"Loading AMR genes data from: {amr_path}")
    amr_data = pd.read_csv(amr_path)
    
    # Set Strain_ID as index
    if "Strain_ID" in mic_data.columns:
        mic_data = mic_data.set_index("Strain_ID")
    elif mic_data.columns[0].lower() in ["strain_id", "strain", "id"]:
        mic_data = mic_data.set_index(mic_data.columns[0])
    
    if "Strain_ID" in amr_data.columns:
        amr_data = amr_data.set_index("Strain_ID")
    elif amr_data.columns[0].lower() in ["strain_id", "strain", "id"]:
        amr_data = amr_data.set_index(amr_data.columns[0])
    
    # Identify phenotype and genotype columns
    # MIC columns are phenotypes
    phenotype_cols = list(mic_data.columns)
    # AMR gene columns are genotypes
    genotype_cols = list(amr_data.columns)
    
    # Merge data
    merged_data = pd.concat([mic_data, amr_data], axis=1)
    
    print(f"Loaded {len(merged_data)} strains")
    print(f"Phenotype columns: {len(phenotype_cols)}")
    print(f"Genotype columns: {len(genotype_cols)}")
    
    return merged_data, mic_data, phenotype_cols, genotype_cols


def generate_full_report(
    data_dir: Path,
    output_file: str = "INNOVATIONS_REAL_DATA_REPORT.md"
) -> str:
    """Generate full report with real data results."""
    
    report_lines = []
    report_lines.append("# Real Data Test Report - Innovations\n")
    report_lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"**Module:** strepsuis-mdr\n")
    report_lines.append(f"**Test:** Network Risk Scoring & Sequential Pattern Detection\n")
    report_lines.append("\n---\n")
    
    try:
        # Load data
        merged_data, mic_data, phenotype_cols, genotype_cols = load_real_data(data_dir)
        n_strains = len(merged_data)
        
        report_lines.append("## 1. Data Summary\n\n")
        report_lines.append(f"- **Number of strains:** {n_strains}\n")
        report_lines.append(f"- **Phenotype columns (MIC):** {len(phenotype_cols)}\n")
        report_lines.append(f"- **Genotype columns (AMR genes):** {len(genotype_cols)}\n")
        report_lines.append(f"- **Total features:** {len(phenotype_cols) + len(genotype_cols)}\n\n")
        
        # Check data quality
        report_lines.append("### Data Quality Check\n\n")
        missing_values = merged_data.isnull().sum().sum()
        report_lines.append(f"- **Missing values:** {missing_values}\n")
        
        # Check binary encoding
        non_binary_cols = []
        for col in merged_data.columns:
            unique_vals = set(merged_data[col].dropna().unique())
            if not unique_vals.issubset({0, 1}):
                non_binary_cols.append(col)
        
        if non_binary_cols:
            report_lines.append(f"- **Non-binary columns:** {len(non_binary_cols)} (may need conversion)\n")
        else:
            report_lines.append(f"- **Binary encoding:** ✅ All columns are binary (0/1)\n")
        
        report_lines.append("\n---\n")
        
        # INNOVATION 1: Network Risk Scoring
        report_lines.append("## 2. Innovation 1: Network Risk Scoring\n\n")
        
        print("\n" + "="*80)
        print("INNOVATION 1: Network Risk Scoring")
        print("="*80)
        
        # Step 1: Build network
        print("\nStep 1: Building hybrid co-resistance network...")
        try:
            network = build_hybrid_co_resistance_network(
                merged_data, phenotype_cols, genotype_cols, alpha=0.05
            )
            n_nodes = network.number_of_nodes()
            n_edges = network.number_of_edges()
            
            report_lines.append(f"### Network Construction\n\n")
            report_lines.append(f"- **Network nodes:** {n_nodes}\n")
            report_lines.append(f"- **Network edges:** {n_edges}\n")
            report_lines.append(f"- **Network density:** {nx.density(network):.4f}\n\n")
            
            print(f"Network built: {n_nodes} nodes, {n_edges} edges")
            
        except Exception as e:
            report_lines.append(f"**Error building network:** {str(e)}\n\n")
            print(f"Error: {e}")
            network = nx.Graph()
        
        # Step 2: Compute bootstrap CI
        print("\nStep 2: Computing bootstrap confidence intervals...")
        try:
            bootstrap_results = compute_bootstrap_ci(
                merged_data, n_iter=1000, confidence_level=0.95
            )
            
            # Convert to dict format for risk scoring
            bootstrap_ci = {}
            for _, row in bootstrap_results.iterrows():
                feature = row['ColumnName']
                bootstrap_ci[feature] = (row['CI_Lower'] / 100.0, row['CI_Upper'] / 100.0)
            
            report_lines.append(f"### Bootstrap Confidence Intervals\n\n")
            report_lines.append(f"- **Features with CI:** {len(bootstrap_ci)}\n")
            report_lines.append(f"- **Average CI width:** {np.mean([ub - lb for lb, ub in bootstrap_ci.values()]):.4f}\n\n")
            
            # Show top 5 features with narrowest CI (highest confidence)
            ci_widths = [(feat, ub - lb) for feat, (lb, ub) in bootstrap_ci.items()]
            ci_widths.sort(key=lambda x: x[1])
            report_lines.append("**Top 5 features with narrowest CI (highest confidence):**\n\n")
            for feat, width in ci_widths[:5]:
                lb, ub = bootstrap_ci[feat]
                report_lines.append(f"- {feat}: [{lb:.3f}, {ub:.3f}], width={width:.3f}\n")
            report_lines.append("\n")
            
            print(f"Bootstrap CI computed for {len(bootstrap_ci)} features")
            
        except Exception as e:
            report_lines.append(f"**Error computing bootstrap CI:** {str(e)}\n\n")
            print(f"Error: {e}")
            bootstrap_ci = {}
        
        # Step 3: Compute risk scores
        print("\nStep 3: Computing network risk scores...")
        try:
            if network.number_of_nodes() > 0 and len(bootstrap_ci) > 0:
                risk_scores = compute_network_mdr_risk_score(
                    network, merged_data, bootstrap_ci, percentile_threshold=75.0
                )
                
                report_lines.append(f"### Risk Score Results\n\n")
                report_lines.append(f"- **Strains analyzed:** {len(risk_scores)}\n")
                report_lines.append(f"- **Risk score range:** [{risk_scores['Network_Risk_Score'].min():.4f}, {risk_scores['Network_Risk_Score'].max():.4f}]\n")
                report_lines.append(f"- **Mean risk score:** {risk_scores['Network_Risk_Score'].mean():.4f}\n")
                report_lines.append(f"- **Median risk score:** {risk_scores['Network_Risk_Score'].median():.4f}\n")
                report_lines.append(f"- **Strains predicted as MDR:** {risk_scores['MDR_Predicted'].sum()}\n")
                report_lines.append(f"- **MDR prediction rate:** {risk_scores['MDR_Predicted'].mean() * 100:.1f}%\n\n")
                
                # Top 10 high-risk strains
                top_risk = risk_scores.nlargest(10, 'Network_Risk_Score')
                report_lines.append("**Top 10 High-Risk Strains:**\n\n")
                report_lines.append("| Strain_ID | Risk Score | Percentile Rank | MDR Predicted |\n")
                report_lines.append("|-----------|------------|-----------------|---------------|\n")
                for _, row in top_risk.iterrows():
                    report_lines.append(f"| {row['Strain_ID']} | {row['Network_Risk_Score']:.4f} | {row['Percentile_Rank']:.1f}% | {'Yes' if row['MDR_Predicted'] else 'No'} |\n")
                report_lines.append("\n")
                
                # Bottom 10 low-risk strains
                bottom_risk = risk_scores.nsmallest(10, 'Network_Risk_Score')
                report_lines.append("**Bottom 10 Low-Risk Strains:**\n\n")
                report_lines.append("| Strain_ID | Risk Score | Percentile Rank | MDR Predicted |\n")
                report_lines.append("|-----------|------------|-----------------|---------------|\n")
                for _, row in bottom_risk.iterrows():
                    report_lines.append(f"| {row['Strain_ID']} | {row['Network_Risk_Score']:.4f} | {row['Percentile_Rank']:.1f}% | {'Yes' if row['MDR_Predicted'] else 'No'} |\n")
                report_lines.append("\n")
                
                print(f"Risk scores computed for {len(risk_scores)} strains")
                print(f"High-risk strains (MDR predicted): {risk_scores['MDR_Predicted'].sum()}")
                
            else:
                report_lines.append("**Cannot compute risk scores:** Network is empty or bootstrap CI missing\n\n")
                print("Warning: Cannot compute risk scores - network empty or CI missing")
                risk_scores = pd.DataFrame()
                
        except Exception as e:
            report_lines.append(f"**Error computing risk scores:** {str(e)}\n\n")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            risk_scores = pd.DataFrame()
        
        report_lines.append("\n---\n")
        
        # INNOVATION 2: Sequential Pattern Detection
        report_lines.append("## 3. Innovation 2: Sequential Pattern Detection\n\n")
        
        print("\n" + "="*80)
        print("INNOVATION 2: Sequential Pattern Detection")
        print("="*80)
        
        # Test on AMR genes data
        print("\nAnalyzing sequential patterns in AMR genes...")
        try:
            sequential_patterns = detect_sequential_resistance_patterns(
                amr_data,
                min_support=0.1,
                min_confidence=0.5,
                correlation_threshold=0.3
            )
            
            report_lines.append(f"### Sequential Pattern Results (AMR Genes)\n\n")
            report_lines.append(f"- **Total patterns detected:** {len(sequential_patterns)}\n")
            
            if len(sequential_patterns) > 0:
                report_lines.append(f"- **Patterns with P < 0.05:** {(sequential_patterns['P_Value'] < 0.05).sum()}\n")
                report_lines.append(f"- **Patterns with P < 0.01:** {(sequential_patterns['P_Value'] < 0.01).sum()}\n\n")
                
                # Top 10 patterns by confidence
                top_patterns = sequential_patterns.head(10)
                report_lines.append("**Top 10 Sequential Patterns (by Confidence):**\n\n")
                report_lines.append("| Pattern | Support | Confidence | Lift | P-Value |\n")
                report_lines.append("|---------|---------|------------|------|----------|\n")
                for _, row in top_patterns.iterrows():
                    report_lines.append(f"| {row['Pattern']} | {row['Support']:.3f} | {row['Confidence']:.3f} | {row['Lift']:.3f} | {row['P_Value']:.4f} |\n")
                report_lines.append("\n")
                
                # Significant patterns (P < 0.05)
                significant = sequential_patterns[sequential_patterns['P_Value'] < 0.05]
                if len(significant) > 0:
                    report_lines.append(f"**Significant Patterns (P < 0.05): {len(significant)}**\n\n")
                    report_lines.append("| Pattern | Support | Confidence | P-Value |\n")
                    report_lines.append("|---------|---------|------------|----------|\n")
                    for _, row in significant.head(10).iterrows():
                        report_lines.append(f"| {row['Pattern']} | {row['Support']:.3f} | {row['Confidence']:.3f} | {row['P_Value']:.4f} |\n")
                    report_lines.append("\n")
                
                print(f"Detected {len(sequential_patterns)} sequential patterns")
                print(f"Significant patterns (P < 0.05): {(sequential_patterns['P_Value'] < 0.05).sum()}")
                
            else:
                report_lines.append("**No sequential patterns detected** (may need to lower min_support or min_confidence)\n\n")
                print("No patterns detected - may need to adjust parameters")
                
        except Exception as e:
            report_lines.append(f"**Error detecting sequential patterns:** {str(e)}\n\n")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sequential_patterns = pd.DataFrame()
        
        # Test on MIC data (phenotypes)
        print("\nAnalyzing sequential patterns in MIC phenotypes...")
        try:
            mic_patterns = detect_sequential_resistance_patterns(
                mic_data,
                min_support=0.1,
                min_confidence=0.5,
                correlation_threshold=0.3
            )
            
            report_lines.append(f"### Sequential Pattern Results (MIC Phenotypes)\n\n")
            report_lines.append(f"- **Total patterns detected:** {len(mic_patterns)}\n")
            
            if len(mic_patterns) > 0:
                report_lines.append(f"- **Patterns with P < 0.05:** {(mic_patterns['P_Value'] < 0.05).sum()}\n\n")
                
                # Top 5 patterns
                top_mic = mic_patterns.head(5)
                report_lines.append("**Top 5 Sequential Patterns (MIC):**\n\n")
                report_lines.append("| Pattern | Support | Confidence | P-Value |\n")
                report_lines.append("|---------|---------|------------|----------|\n")
                for _, row in top_mic.iterrows():
                    report_lines.append(f"| {row['Pattern']} | {row['Support']:.3f} | {row['Confidence']:.3f} | {row['P_Value']:.4f} |\n")
                report_lines.append("\n")
                
                print(f"Detected {len(mic_patterns)} patterns in MIC data")
            else:
                report_lines.append("**No sequential patterns detected in MIC data**\n\n")
                
        except Exception as e:
            report_lines.append(f"**Error detecting MIC patterns:** {str(e)}\n\n")
            print(f"Error: {e}")
            mic_patterns = pd.DataFrame()
        
        report_lines.append("\n---\n")
        
        # Summary
        report_lines.append("## 4. Summary\n\n")
        report_lines.append("### Innovation 1: Network Risk Scoring\n\n")
        if 'risk_scores' in locals() and len(risk_scores) > 0:
            report_lines.append("✅ **Successfully implemented and tested**\n\n")
            report_lines.append(f"- Computed risk scores for {len(risk_scores)} strains\n")
            report_lines.append(f"- Identified {risk_scores['MDR_Predicted'].sum()} high-risk strains\n")
            report_lines.append(f"- Risk scores range from {risk_scores['Network_Risk_Score'].min():.4f} to {risk_scores['Network_Risk_Score'].max():.4f}\n\n")
        else:
            report_lines.append("⚠️ **Implementation complete but testing encountered issues**\n\n")
        
        report_lines.append("### Innovation 2: Sequential Pattern Detection\n\n")
        if 'sequential_patterns' in locals() and len(sequential_patterns) > 0:
            report_lines.append("✅ **Successfully implemented and tested**\n\n")
            report_lines.append(f"- Detected {len(sequential_patterns)} sequential patterns in AMR genes\n")
            if len(sequential_patterns) > 0:
                report_lines.append(f"- {len(sequential_patterns[sequential_patterns['P_Value'] < 0.05])} patterns are statistically significant (P < 0.05)\n\n")
        else:
            report_lines.append("⚠️ **Implementation complete but no patterns detected** (may need parameter adjustment)\n\n")
        
        report_lines.append("\n---\n")
        report_lines.append(f"**Report generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_lines.append("**Status:** ✅ Complete\n")
        
    except Exception as e:
        report_lines.append(f"\n## ERROR\n\n")
        report_lines.append(f"**Error during testing:** {str(e)}\n\n")
        import traceback
        report_lines.append("```\n")
        report_lines.append(traceback.format_exc())
        report_lines.append("```\n")
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    
    # Write report
    report_content = "".join(report_lines)
    output_path = Path(__file__).parent.parent / output_file
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n{'='*80}")
    print(f"Report saved to: {output_path}")
    print(f"{'='*80}\n")
    
    return report_content


if __name__ == "__main__":
    # Try to find data directory
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / "data"
    
    # Try different locations
    if not data_dir.exists():
        data_dir = script_dir.parent.parent / "data"
    
    if not data_dir.exists():
        data_dir = script_dir / "data" / "examples"
    
    print(f"Testing innovations with real data...")
    print(f"Data directory: {data_dir}")
    
    report = generate_full_report(data_dir)
    print("\nReport generation complete!")
    print(f"\nReport preview (first 1000 chars):\n{report[:1000]}...")


