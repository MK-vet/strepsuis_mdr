"""
Output handler for StrepSuisMDR with StandardOutput integration.

This module wraps all output operations to ensure standardized formatting
with statistical interpretation and QA checklists.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# Import StandardOutput from shared module
try:
    from shared import StandardOutput, create_qa_checklist
except ImportError:
    # Fallback if shared module not in path
    import sys
    shared_path = Path(__file__).parent.parent.parent / "shared" / "src"
    sys.path.insert(0, str(shared_path))
    from shared import StandardOutput, create_qa_checklist


class MDROutputHandler:
    """
    Output handler for MDR analysis results.

    Wraps all output operations with StandardOutput to ensure consistent
    formatting and inclusion of statistical interpretations and QA checks.
    """

    def __init__(self, output_dir: str):
        """
        Initialize output handler.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_mdr_patterns(
        self,
        mdr_df: pd.DataFrame,
        n_samples: int
    ) -> None:
        """
        Save MDR patterns with interpretation.

        Args:
            mdr_df: DataFrame with MDR pattern results
            n_samples: Total number of samples analyzed
        """
        output = StandardOutput(data=mdr_df)

        # Calculate MDR statistics
        if 'MDR_Class' in mdr_df.columns:
            mdr_counts = mdr_df['MDR_Class'].value_counts()
            n_mdr = (mdr_df['MDR_Class'] != 'Non-MDR').sum()
            pct_mdr = 100 * n_mdr / n_samples

            interp = f"MDR analysis identified {n_mdr}/{n_samples} ({pct_mdr:.1f}%) isolates "
            interp += f"with multidrug resistance phenotypes. "

            if 'XDR' in mdr_counts.index:
                n_xdr = mdr_counts['XDR']
                interp += f"Extensive drug resistance (XDR) detected in {n_xdr} isolates "
                interp += f"({100*n_xdr/n_samples:.1f}%). "

            if 'PDR' in mdr_counts.index:
                n_pdr = mdr_counts['PDR']
                interp += f"Pan-drug resistance (PDR) detected in {n_pdr} isolates "
                interp += f"({100*n_pdr/n_samples:.1f}%). "
        else:
            interp = f"MDR pattern analysis completed for {n_samples} samples."

        output.add_statistical_interpretation(interp)

        # QA checklist
        qa_items = [
            f"✓ {n_samples} samples analyzed for MDR patterns",
            "✓ MDR classification completed (MDR/XDR/PDR)",
        ]

        if 'MDR_Class' in mdr_df.columns:
            if n_mdr > 0:
                qa_items.append(f"✓ {n_mdr} MDR isolates identified")
            else:
                qa_items.append("✓ No MDR isolates detected")

        output.add_quick_qa(qa_items)
        output.add_metadata("n_samples", n_samples)
        output.add_metadata("analysis_type", "mdr_patterns")

        base_path = self.output_dir / "mdr_patterns"
        output.to_json(f"{base_path}.json")
        output.to_csv(f"{base_path}.csv")
        output.to_markdown(f"{base_path}.md")

    def save_resistance_cooccurrence(
        self,
        cooccur_df: pd.DataFrame,
        significance_threshold: float = 0.05
    ) -> None:
        """
        Save resistance co-occurrence analysis with interpretation.

        Args:
            cooccur_df: DataFrame with co-occurrence results
            significance_threshold: P-value threshold for significance
        """
        output = StandardOutput(data=cooccur_df)

        # Count significant co-occurrences
        if 'P_Value' in cooccur_df.columns:
            n_significant = (cooccur_df['P_Value'] < significance_threshold).sum()
            n_total = len(cooccur_df)

            interp = f"Resistance co-occurrence analysis identified {n_significant}/{n_total} "
            interp += f"significant paired resistance patterns (p < {significance_threshold}). "

            if 'Phi_Coefficient' in cooccur_df.columns:
                strong_assoc = (abs(cooccur_df['Phi_Coefficient']) > 0.5).sum()
                interp += f"{strong_assoc} pairs show strong correlation (|φ| > 0.5), "
                interp += f"indicating linked resistance mechanisms or co-selection."
        else:
            interp = f"Co-occurrence analysis completed for {len(cooccur_df)} resistance pairs."

        output.add_statistical_interpretation(interp)

        # QA checklist
        qa_items = [
            f"✓ Co-occurrence analysis performed on {len(cooccur_df)} resistance pairs",
        ]

        if 'P_Value' in cooccur_df.columns:
            if n_significant > 0:
                qa_items.append(f"✓ {n_significant} significant associations (p < {significance_threshold})")
            else:
                qa_items.append("⚠ No significant co-occurrences found")

        if 'Phi_Coefficient' in cooccur_df.columns and strong_assoc > 0:
            qa_items.append(f"✓ {strong_assoc} strong correlations detected (|φ| > 0.5)")

        output.add_quick_qa(qa_items)
        output.add_metadata("significance_threshold", significance_threshold)

        base_path = self.output_dir / "resistance_cooccurrence"
        output.to_json(f"{base_path}.json")
        output.to_csv(f"{base_path}.csv")
        output.to_markdown(f"{base_path}.md")

    def save_genotype_phenotype_concordance(
        self,
        concordance_df: pd.DataFrame
    ) -> None:
        """
        Save genotype-phenotype concordance results.

        Args:
            concordance_df: DataFrame with concordance metrics
        """
        output = StandardOutput(data=concordance_df)

        # Calculate concordance statistics
        if 'Concordance' in concordance_df.columns:
            avg_concordance = concordance_df['Concordance'].mean()
            high_concordance = (concordance_df['Concordance'] > 0.9).sum()

            interp = f"Genotype-phenotype concordance analysis shows average agreement of {avg_concordance:.1%}. "
            interp += f"{high_concordance} resistance markers demonstrate high concordance (>90%), "
            interp += f"indicating reliable genotype-based prediction. "

            if avg_concordance < 0.7:
                interp += "Lower concordance suggests potential discrepancies requiring further investigation."
        else:
            interp = "Genotype-phenotype concordance analysis completed."

        output.add_statistical_interpretation(interp)

        # QA checklist
        qa_items = [
            f"✓ Concordance analysis performed on {len(concordance_df)} markers",
        ]

        if 'Concordance' in concordance_df.columns:
            if avg_concordance > 0.8:
                qa_items.append(f"✓ High average concordance ({avg_concordance:.1%})")
            elif avg_concordance > 0.7:
                qa_items.append(f"✓ Moderate concordance ({avg_concordance:.1%})")
            else:
                qa_items.append(f"⚠ Low concordance ({avg_concordance:.1%}) - review markers")

        output.add_quick_qa(qa_items)

        base_path = self.output_dir / "genotype_phenotype_concordance"
        output.to_json(f"{base_path}.json")
        output.to_csv(f"{base_path}.csv")
        output.to_markdown(f"{base_path}.md")

    def save_generic_results(
        self,
        df: pd.DataFrame,
        filename: str,
        interpretation: str,
        qa_items: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save generic results with custom interpretation and QA.

        Args:
            df: DataFrame with results
            filename: Base filename (without extension)
            interpretation: Statistical interpretation text
            qa_items: QA checklist items
            metadata: Optional additional metadata
        """
        output = StandardOutput(data=df)
        output.add_statistical_interpretation(interpretation)
        output.add_quick_qa(qa_items)

        if metadata:
            for key, value in metadata.items():
                output.add_metadata(key, value)

        base_path = self.output_dir / filename
        output.to_json(f"{base_path}.json")
        output.to_csv(f"{base_path}.csv")
        output.to_markdown(f"{base_path}.md")
