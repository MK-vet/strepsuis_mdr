#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Co-Selection Analysis Module

This module implements co-selection score calculation and mobile genetic element
(MGE) prediction based on network topology, co-occurrence patterns, and community
membership.

Innovation: Novel metric combining multiple data sources to identify genes likely
co-selected on mobile genetic elements.
"""

import logging
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoSelectionAnalyzer:
    """
    Analyzes co-selection patterns between resistance genes using network topology,
    co-occurrence statistics, and community detection.
    
    Uses existing analysis results:
    - Network structure (from hybrid network)
    - Co-occurrence statistics (from pairwise_cooccurrence)
    - Community assignments (from Louvain communities)
    
    Adds:
    - Co-selection score calculation
    - Mobile genetic element prediction
    """
    
    def __init__(
        self,
        network: nx.Graph,
        cooccurrence_results: pd.DataFrame,
        communities: Optional[pd.DataFrame] = None,
        data: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize CoSelectionAnalyzer.
        
        Args:
            network: Hybrid co-resistance network (NetworkX Graph)
            cooccurrence_results: DataFrame with co-occurrence statistics
                Expected columns: Item1, Item2, Phi, Corrected_p
            communities: DataFrame with community assignments (optional)
                Expected columns: Node, Community
            data: Original binary data matrix (optional, for prevalence calculation)
        """
        self.network = network
        self.cooccurrence = cooccurrence_results
        self.communities = communities
        self.data = data
        
        # Build community lookup dictionary
        self._community_lookup = {}
        if communities is not None and not communities.empty:
            for _, row in communities.iterrows():
                node = row['Node']
                comm = row['Community']
                self._community_lookup[node] = comm
    
    def calculate_coselection_score(
        self,
        gene1: str,
        gene2: str,
        weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
    ) -> float:
        """
        Calculate co-selection score between two genes.
        
        Novel metric combining:
        1. Co-occurrence strength (Phi coefficient)
        2. Network proximity (shortest path length)
        3. Community membership (same community = higher score)
        
        Formula:
        cs_score = w1 * |cooccur_freq| + w2 * network_proximity + w3 * same_community
        
        Args:
            gene1: First gene name
            gene2: Second gene name
            weights: Tuple of (cooccurrence_weight, network_weight, community_weight)
                Default: (0.4, 0.3, 0.3)
        
        Returns:
            Co-selection score (0.0 to 1.0, higher = more likely co-selected)
        """
        w_cooccur, w_network, w_community = weights
        
        # 1. Co-occurrence strength (uses existing cooccurrence data!)
        cooccur_data = self.cooccurrence[
            ((self.cooccurrence['Item1'] == gene1) & (self.cooccurrence['Item2'] == gene2)) |
            ((self.cooccurrence['Item1'] == gene2) & (self.cooccurrence['Item2'] == gene1))
        ]
        
        if len(cooccur_data) == 0:
            cooccur_freq = 0.0
        else:
            # Use absolute Phi coefficient (strength of association)
            cooccur_freq = abs(cooccur_data.iloc[0]['Phi'])
        
        # Normalize to [0, 1] (Phi ranges from -1 to 1)
        cooccur_norm = min(cooccur_freq, 1.0)
        
        # 2. Network proximity (uses existing network!)
        try:
            if gene1 in self.network.nodes() and gene2 in self.network.nodes():
                path_length = nx.shortest_path_length(self.network, gene1, gene2)
                # Inverse distance: closer = higher score
                network_proximity = 1.0 / (path_length + 1)
            else:
                network_proximity = 0.0
        except nx.NetworkXNoPath:
            network_proximity = 0.0
        except Exception as e:
            logger.warning(f"Error computing network path between {gene1} and {gene2}: {e}")
            network_proximity = 0.0
        
        # 3. Community membership (uses existing communities!)
        gene1_community = self._get_community(gene1)
        gene2_community = self._get_community(gene2)
        same_community = 1.0 if gene1_community == gene2_community else 0.0
        
        # 4. Co-selection score (NEW FORMULA)
        cs_score = (
            w_cooccur * cooccur_norm +      # Co-occurrence strength
            w_network * network_proximity +  # Network closeness
            w_community * same_community     # Community membership
        )
        
        return min(cs_score, 1.0)  # Cap at 1.0
    
    def _get_community(self, node: str) -> Optional[int]:
        """Get community ID for a node."""
        return self._community_lookup.get(node, None)
    
    def identify_coselection_modules(
        self,
        threshold: float = 0.7,
        min_module_size: int = 2,
    ) -> pd.DataFrame:
        """
        Identify gene modules likely co-selected on mobile genetic elements.
        
        Uses calculate_coselection_score() for all pairs within communities.
        Genes with high co-selection scores are predicted to be on the same MGE.
        
        Args:
            threshold: Minimum co-selection score to consider (default: 0.7)
            min_module_size: Minimum number of genes in a module (default: 2)
        
        Returns:
            DataFrame with columns:
                - module_id: Community ID
                - genes: List of gene names in module
                - n_genes: Number of genes
                - avg_cs_score: Average co-selection score
                - predicted_mge: Boolean (True if likely on MGE)
        """
        if self.communities is None or self.communities.empty:
            logger.warning("No communities provided, cannot identify modules")
            return pd.DataFrame(columns=[
                'module_id', 'genes', 'n_genes', 'avg_cs_score', 'predicted_mge'
            ])
        
        modules = []
        
        # Group nodes by community
        community_groups = {}
        for _, row in self.communities.iterrows():
            comm_id = row['Community']
            node = row['Node']
            if comm_id not in community_groups:
                community_groups[comm_id] = []
            community_groups[comm_id].append(node)
        
        # Iterate over communities
        for community_id, genes in community_groups.items():
            if len(genes) < min_module_size:
                continue
            
            module_scores = []
            
            # Calculate co-selection scores for all pairs in community
            for gene1, gene2 in combinations(genes, 2):
                score = self.calculate_coselection_score(gene1, gene2)
                if score >= threshold:
                    module_scores.append({
                        'gene1': gene1,
                        'gene2': gene2,
                        'cs_score': score
                    })
            
            # If enough high-scoring pairs, consider it a module
            if len(module_scores) >= min_module_size - 1:  # At least n-1 pairs for n genes
                avg_score = np.mean([s['cs_score'] for s in module_scores]) if module_scores else 0.0
                
                modules.append({
                    'module_id': community_id,
                    'genes': genes,
                    'n_genes': len(genes),
                    'avg_cs_score': round(avg_score, 4),
                    'predicted_mge': True,  # Candidate mobile genetic element
                    'n_high_scoring_pairs': len(module_scores),
                })
        
        if not modules:
            logger.info("No co-selection modules identified above threshold")
            return pd.DataFrame(columns=[
                'module_id', 'genes', 'n_genes', 'avg_cs_score', 'predicted_mge', 'n_high_scoring_pairs'
            ])
        
        return pd.DataFrame(modules)
    
    def get_gene_coselection_network(
        self,
        threshold: float = 0.5,
    ) -> nx.Graph:
        """
        Build network of co-selection relationships.
        
        Creates a network where edges represent co-selection relationships
        (cs_score >= threshold).
        
        Args:
            threshold: Minimum co-selection score for edge (default: 0.5)
        
        Returns:
            NetworkX Graph with co-selection relationships
        """
        cs_network = nx.Graph()
        
        # Get all unique gene pairs from cooccurrence data
        gene_pairs = set()
        for _, row in self.cooccurrence.iterrows():
            gene1 = row['Item1']
            gene2 = row['Item2']
            if gene1 < gene2:
                gene_pairs.add((gene1, gene2))
            else:
                gene_pairs.add((gene2, gene1))
        
        # Calculate co-selection scores and add edges
        for gene1, gene2 in gene_pairs:
            cs_score = self.calculate_coselection_score(gene1, gene2)
            if cs_score >= threshold:
                cs_network.add_edge(gene1, gene2, weight=cs_score, cs_score=cs_score)
        
        return cs_network
    
    def rank_genes_by_coselection_potential(
        self,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Rank genes by their potential to be part of co-selection modules.
        
        Uses:
        - Number of high co-selection relationships
        - Average co-selection score
        - Community membership (genes in large communities score higher)
        
        Args:
            top_n: Number of top genes to return (default: 20)
        
        Returns:
            DataFrame with gene rankings
        """
        if self.communities is None or self.communities.empty:
            logger.warning("No communities provided, cannot rank genes")
            return pd.DataFrame()
        
        gene_stats = {}
        
        # Get all genes from network
        all_genes = list(self.network.nodes())
        
        for gene in all_genes:
            # Count high co-selection relationships
            high_cs_count = 0
            cs_scores = []
            
            for other_gene in all_genes:
                if gene == other_gene:
                    continue
                
                cs_score = self.calculate_coselection_score(gene, other_gene)
                cs_scores.append(cs_score)
                if cs_score >= 0.7:  # High threshold
                    high_cs_count += 1
            
            # Community size
            community = self._get_community(gene)
            community_size = 0
            if community is not None:
                community_size = sum(1 for _, row in self.communities.iterrows() 
                                    if row['Community'] == community)
            
            # Average co-selection score
            avg_cs = np.mean(cs_scores) if cs_scores else 0.0
            
            # Combined score
            potential_score = (
                0.4 * (high_cs_count / max(len(all_genes) - 1, 1)) +  # Normalized count
                0.4 * avg_cs +                                        # Average score
                0.2 * min(community_size / 10.0, 1.0)                 # Community size (normalized)
            )
            
            gene_stats[gene] = {
                'gene': gene,
                'high_cs_relationships': high_cs_count,
                'avg_cs_score': round(avg_cs, 4),
                'community_size': community_size,
                'potential_score': round(potential_score, 4),
            }
        
        df = pd.DataFrame(list(gene_stats.values()))
        df = df.sort_values('potential_score', ascending=False)
        df['rank'] = range(1, len(df) + 1)
        
        return df.head(top_n)
