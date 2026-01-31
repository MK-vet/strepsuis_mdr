#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Intervention Target Prioritization Module

This module implements multi-criteria ranking for drug/vaccine targets based on:
- MDR association strength
- Network centrality
- Prevalence
- Mobility prediction (co-selection modules)

Innovation: Translational value - bridges genomics to therapeutics by identifying
high-priority intervention targets.
"""

import logging
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterventionTargetRanker:
    """
    Multi-criteria ranking system for drug/vaccine targets.
    
    Integrates multiple criteria using existing analysis results:
    - Gene-MDR associations (from gene_pheno_assoc)
    - Network structure (from hybrid network)
    - Community assignments (from Louvain communities)
    - Prevalence data (from original data)
    
    Adds:
    - Mobility prediction (based on co-selection modules)
    - Integrated priority score
    """
    
    def __init__(
        self,
        network: nx.Graph,
        gene_associations: pd.DataFrame,
        communities: Optional[pd.DataFrame] = None,
        data: Optional[pd.DataFrame] = None,
        coselection_modules: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize InterventionTargetRanker.
        
        Args:
            network: Hybrid co-resistance network
            gene_associations: DataFrame with gene-phenotype associations
                Expected columns: Gene (or Feature1), Phenotype (or Feature2), Phi, Corrected_p
            communities: DataFrame with community assignments (optional)
            data: Original binary data matrix (optional, for prevalence)
            coselection_modules: DataFrame with co-selection modules (optional)
                Expected columns: module_id, genes, predicted_mge
        """
        self.network = network
        self.gene_associations = gene_associations
        self.communities = communities
        self.data = data
        self.coselection_modules = coselection_modules
        
        # Build gene-to-module mapping
        self._gene_to_module = {}
        if coselection_modules is not None and not coselection_modules.empty:
            for _, row in coselection_modules.iterrows():
                genes = row['genes'] if isinstance(row['genes'], list) else eval(row['genes'])
                is_mge = row.get('predicted_mge', False)
                for gene in genes:
                    self._gene_to_module[gene] = is_mge
    
    def rank_intervention_targets(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Rank genes by intervention priority using multi-criteria analysis.
        
        Integrates multiple criteria:
        1. MDR association strength (from existing gene_associations)
        2. Network centrality (degree, betweenness)
        3. Prevalence (from data)
        4. Mobility prediction (from co-selection modules)
        
        Args:
            weights: Dictionary of weights for each criterion
                Default: {
                    'mdr_association': 0.3,
                    'degree': 0.2,
                    'betweenness': 0.2,
                    'prevalence': 0.2,
                    'mobility': 0.1
                }
        
        Returns:
            DataFrame with ranked targets and all criteria scores
        """
        if weights is None:
            weights = {
                'mdr_association': 0.3,
                'degree': 0.2,
                'betweenness': 0.2,
                'prevalence': 0.2,
                'mobility': 0.1,
            }
        
        rankings = []
        
        # Compute network centralities (once for all genes)
        try:
            degree_cent = nx.degree_centrality(self.network)
            betweenness_cent = nx.betweenness_centrality(self.network)
        except Exception as e:
            logger.warning(f"Error computing centralities: {e}")
            degree_cent = {node: 0.0 for node in self.network.nodes()}
            betweenness_cent = {node: 0.0 for node in self.network.nodes()}
        
        # Get max values for normalization
        max_degree = max(degree_cent.values()) if degree_cent else 1.0
        max_betweenness = max(betweenness_cent.values()) if betweenness_cent else 1.0
        
        # Process each gene in network
        for gene in self.network.nodes():
            # 1. MDR association (uses existing gene_associations!)
            # Try different column name formats
            if 'Gene' in self.gene_associations.columns:
                assoc_data = self.gene_associations[self.gene_associations['Gene'] == gene]
            elif 'Feature1' in self.gene_associations.columns:
                assoc_data = self.gene_associations[
                    (self.gene_associations['Feature1'] == gene) |
                    (self.gene_associations.get('Feature2', pd.Series()) == gene)
                ]
            else:
                # Try first column as gene name
                first_col = self.gene_associations.columns[0]
                assoc_data = self.gene_associations[self.gene_associations[first_col] == gene]
            
            if len(assoc_data) > 0:
                # Use strongest association
                assoc_data = assoc_data.loc[assoc_data['Phi'].abs().idxmax()]
                mdr_association = abs(assoc_data['Phi'])
                p_value = assoc_data.get('Corrected_p', assoc_data.get('P_Value', 1.0))
            else:
                mdr_association = 0.0
                p_value = 1.0
            
            # 2. Network centrality (from existing network!)
            degree = self.network.degree(gene)
            degree_norm = degree_cent.get(gene, 0.0)  # Already normalized
            betweenness_norm = betweenness_cent.get(gene, 0.0)  # Already normalized
            
            # 3. Prevalence (from data if available)
            if self.data is not None and gene in self.data.columns:
                prevalence = self.data[gene].mean()
            else:
                # Estimate from network degree (more connections = more prevalent)
                prevalence = degree / max(len(self.network.nodes()) - 1, 1)
            
            # 4. Mobility prediction (from co-selection modules)
            is_mobile = self._gene_to_module.get(gene, False)
            mobility_score = 1.0 if is_mobile else 0.0
            
            # Alternative: community size heuristic
            if not is_mobile and self.communities is not None:
                community = self._get_community(gene)
                if community is not None:
                    community_size = sum(1 for _, row in self.communities.iterrows() 
                                        if row['Community'] == community)
                    # Large communities (>=3 genes) likely on MGE
                    if community_size >= 3:
                        mobility_score = 0.5  # Moderate mobility
            
            # 5. Integrated priority score
            priority = (
                weights['mdr_association'] * mdr_association +  # Strong MDR effect
                weights['degree'] * degree_norm +                # Network importance
                weights['betweenness'] * betweenness_norm +    # Bottleneck position
                weights['prevalence'] * prevalence +            # Common gene
                weights['mobility'] * mobility_score            # Transmissible
            )
            
            rankings.append({
                'gene': gene,
                'mdr_association': round(mdr_association, 4),
                'p_value': round(p_value, 4),
                'degree': degree,
                'degree_centrality': round(degree_norm, 4),
                'betweenness_centrality': round(betweenness_norm, 4),
                'prevalence': round(prevalence, 4),
                'predicted_mobile': mobility_score > 0,
                'mobility_score': round(mobility_score, 4),
                'priority_score': round(priority, 4),
                'priority_rank': None,  # Fill after sorting
            })
        
        if not rankings:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=[
                'gene', 'mdr_association', 'p_value', 'degree', 'degree_centrality',
                'betweenness_centrality', 'prevalence', 'predicted_mobile',
                'mobility_score', 'priority_score', 'priority_rank'
            ])
        
        df = pd.DataFrame(rankings)
        df = df.sort_values('priority_score', ascending=False)
        df['priority_rank'] = range(1, len(df) + 1)
        
        # Reset index
        df.reset_index(drop=True, inplace=True)
        
        return df
    
    def _get_community(self, node: str) -> Optional[int]:
        """Get community ID for a node."""
        if self.communities is None or self.communities.empty:
            return None
        
        comm_data = self.communities[self.communities['Node'] == node]
        if len(comm_data) > 0:
            return comm_data.iloc[0]['Community']
        return None
    
    def get_top_targets(
        self,
        top_n: int = 20,
        min_mdr_association: float = 0.3,
        min_prevalence: float = 0.1,
    ) -> pd.DataFrame:
        """
        Get top intervention targets with filtering criteria.
        
        Args:
            top_n: Number of top targets to return
            min_mdr_association: Minimum MDR association strength
            min_prevalence: Minimum prevalence
        
        Returns:
            Filtered and ranked DataFrame
        """
        rankings = self.rank_intervention_targets()
        
        # Filter
        filtered = rankings[
            (rankings['mdr_association'] >= min_mdr_association) &
            (rankings['prevalence'] >= min_prevalence)
        ]
        
        return filtered.head(top_n)
    
    def generate_target_report(
        self,
        top_n: int = 20,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate a formatted report of top intervention targets.
        
        Args:
            top_n: Number of targets to include
            output_path: Optional path to save report
        
        Returns:
            Formatted report string
        """
        top_targets = self.get_top_targets(top_n=top_n)
        
        report_lines = [
            "=" * 80,
            "INTERVENTION TARGET PRIORITIZATION REPORT",
            "=" * 80,
            "",
            f"Top {len(top_targets)} Intervention Targets",
            "",
            "Ranking Criteria:",
            "  - MDR Association: Strength of association with multidrug resistance",
            "  - Network Centrality: Importance in resistance network",
            "  - Prevalence: Frequency in population",
            "  - Mobility: Predicted to be on mobile genetic element",
            "",
            "-" * 80,
        ]
        
        for idx, row in top_targets.iterrows():
            report_lines.extend([
                f"\nRank {row['priority_rank']}: {row['gene']}",
                f"  Priority Score: {row['priority_score']:.4f}",
                f"  MDR Association: {row['mdr_association']:.4f} (p={row['p_value']:.4f})",
                f"  Network Centrality: Degree={row['degree']}, Betweenness={row['betweenness_centrality']:.4f}",
                f"  Prevalence: {row['prevalence']:.2%}",
                f"  Predicted Mobile: {'Yes' if row['predicted_mobile'] else 'No'}",
            ])
        
        report_lines.extend([
            "",
            "=" * 80,
            f"Report generated: {pd.Timestamp.now()}",
            "=" * 80,
        ])
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Target report saved to: {output_path}")
        
        return report
