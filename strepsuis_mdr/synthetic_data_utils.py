"""
Synthetic Data Generator for StrepSuis Suite

Generates synthetic datasets with known properties for validation and testing.
This module allows end-to-end testing without requiring real data files.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Import validation utilities at module level for better testing and import handling
from .validation_utils import validate_input_data, get_data_summary


def generate_synthetic_amr_data(
    n_strains: int = 50,
    n_antibiotics: int = 9,
    n_genes: int = 15,
    noise_level: float = 0.1,
    n_clusters: int = 2,
    random_state: Optional[int] = 42,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Generate synthetic AMR/MIC/Virulence data with known cluster structure.

    Creates a dataset with configurable number of strains, features, and clusters.
    The data includes phenotypic resistance (antibiotics) and genotypic data (genes)
    with known associations for validation purposes.

    Parameters:
        n_strains: Number of bacterial strains to generate (default: 50)
        n_antibiotics: Number of antibiotic resistance columns (default: 9)
        n_genes: Number of AMR gene columns (default: 15)
        noise_level: Proportion of random flips to add as noise (default: 0.1)
        n_clusters: Number of distinct resistance profiles/clusters (default: 2)
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (DataFrame, metadata_dict) where:
            - DataFrame contains the synthetic data with Strain_ID, antibiotic, and gene columns
            - metadata_dict contains ground truth information for validation:
                - 'true_clusters': array of true cluster assignments
                - 'antibiotic_cols': list of antibiotic column names
                - 'gene_cols': list of gene column names
                - 'cluster_profiles': dict of cluster -> typical profile
                - 'known_associations': list of (gene, antibiotic) pairs that are associated

    Example:
        >>> data, metadata = generate_synthetic_amr_data(n_strains=100, random_state=42)
        >>> print(data.shape)
        (100, 25)
        >>> print(metadata['true_clusters'][:5])
        [0 0 1 0 1]
    """
    rng = np.random.default_rng(random_state)
    
    # Define antibiotic columns (matching real dataset naming)
    default_antibiotics = [
        "Oxytetracycline", "Doxycycline", "Tulathromycin", "Spectinomycin",
        "Gentamicin", "Tiamulin", "Trimethoprim_Sulphamethoxazole", 
        "Enrofloxacin", "Penicillin", "Ampicillin", "Ceftiofur", "Florfenicol"
    ]
    antibiotic_cols = default_antibiotics[:n_antibiotics]
    
    # Define gene columns
    gene_cols = [f"Gene_{i:02d}" for i in range(1, n_genes + 1)]
    
    # Create cluster profiles (each cluster has a characteristic resistance pattern)
    cluster_profiles = {}
    for cluster_id in range(n_clusters):
        # Each cluster has different base resistance probabilities
        antibiotic_probs = rng.beta(2, 5, size=n_antibiotics)
        # Make some antibiotics more resistant in certain clusters
        high_res_indices = rng.choice(n_antibiotics, size=max(2, n_antibiotics // 3), replace=False)
        antibiotic_probs[high_res_indices] = rng.beta(5, 2, size=len(high_res_indices))
        
        gene_probs = rng.beta(2, 5, size=n_genes)
        high_gene_indices = rng.choice(n_genes, size=max(2, n_genes // 3), replace=False)
        gene_probs[high_gene_indices] = rng.beta(5, 2, size=len(high_gene_indices))
        
        cluster_profiles[cluster_id] = {
            'antibiotic_probs': antibiotic_probs,
            'gene_probs': gene_probs,
        }
    
    # Assign strains to clusters
    cluster_sizes = [n_strains // n_clusters] * n_clusters
    cluster_sizes[-1] += n_strains - sum(cluster_sizes)  # Handle remainder
    
    true_clusters = np.concatenate([
        np.full(size, cluster_id) for cluster_id, size in enumerate(cluster_sizes)
    ])
    rng.shuffle(true_clusters)
    
    # Generate data based on cluster assignments
    data = {"Strain_ID": [f"Strain_{i:03d}" for i in range(1, n_strains + 1)]}
    
    # Generate antibiotic resistance data
    for ab_idx, ab_name in enumerate(antibiotic_cols):
        values = []
        for strain_idx in range(n_strains):
            cluster = true_clusters[strain_idx]
            prob = cluster_profiles[cluster]['antibiotic_probs'][ab_idx]
            values.append(int(rng.random() < prob))
        data[ab_name] = values
    
    # Generate gene data with some associations to antibiotics
    known_associations = []
    for gene_idx, gene_name in enumerate(gene_cols):
        values = []
        # Create association with an antibiotic (every 2nd gene is associated)
        associated_ab_idx = gene_idx % n_antibiotics if gene_idx % 2 == 0 else None
        
        if associated_ab_idx is not None:
            known_associations.append((gene_name, antibiotic_cols[associated_ab_idx]))
        
        for strain_idx in range(n_strains):
            cluster = true_clusters[strain_idx]
            base_prob = cluster_profiles[cluster]['gene_probs'][gene_idx]
            
            # If associated with an antibiotic, increase probability when antibiotic is resistant
            if associated_ab_idx is not None:
                ab_value = data[antibiotic_cols[associated_ab_idx]][strain_idx]
                if ab_value == 1:
                    base_prob = min(1.0, base_prob + 0.3)  # Increase probability
            
            values.append(int(rng.random() < base_prob))
        data[gene_name] = values
    
    # Add noise
    df = pd.DataFrame(data)
    feature_cols = antibiotic_cols + gene_cols
    
    for col in feature_cols:
        n_flips = int(n_strains * noise_level)
        flip_indices = rng.choice(n_strains, size=n_flips, replace=False)
        df.loc[flip_indices, col] = 1 - df.loc[flip_indices, col]
    
    metadata = {
        'true_clusters': true_clusters,
        'antibiotic_cols': antibiotic_cols,
        'gene_cols': gene_cols,
        'cluster_profiles': cluster_profiles,
        'known_associations': known_associations,
        'n_strains': n_strains,
        'noise_level': noise_level,
        'random_state': random_state,
    }
    
    return df, metadata


def generate_cooccurrence_data(
    n_samples: int = 100,
    n_features: int = 6,
    association_strength: float = 0.7,
    random_state: Optional[int] = 42,
) -> Tuple[pd.DataFrame, List[Tuple[str, str, float]]]:
    """
    Generate synthetic data with known co-occurrence patterns.

    Parameters:
        n_samples: Number of samples
        n_features: Number of binary features
        association_strength: Strength of induced associations (0-1)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (DataFrame, known_associations) where:
            - DataFrame contains binary feature data
            - known_associations is list of (feat1, feat2, expected_phi) tuples
    """
    rng = np.random.default_rng(random_state)
    
    feature_names = [f"Feature_{chr(65 + i)}" for i in range(n_features)]
    
    data = {}
    
    # Generate first feature randomly
    data[feature_names[0]] = rng.binomial(1, 0.5, n_samples)
    
    # Generate pairs with known associations
    known_associations = []
    
    for i in range(1, n_features):
        if i % 2 == 1 and i > 0:
            # Create association with previous feature
            base_values = data[feature_names[i - 1]]
            # When base is 1, higher chance of being 1
            probs = np.where(base_values == 1, association_strength, 1 - association_strength)
            probs = probs * 0.5 + 0.25  # Normalize to reasonable range
            values = (rng.random(n_samples) < probs).astype(int)
            known_associations.append((feature_names[i - 1], feature_names[i], association_strength))
        else:
            # Independent feature
            values = rng.binomial(1, rng.uniform(0.3, 0.7), n_samples)
        
        data[feature_names[i]] = values
    
    return pd.DataFrame(data), known_associations


def generate_phylogenetic_tree(
    n_tips: int = 30,
    random_state: Optional[int] = 42,
) -> str:
    """
    Generate a random Newick tree string for testing.

    Parameters:
        n_tips: Number of tip labels (strain names)
        random_state: Random seed

    Returns:
        Newick format tree string
    """
    rng = np.random.default_rng(random_state)
    
    tip_names = [f"Strain_{i:03d}" for i in range(1, n_tips + 1)]
    
    def random_branch() -> str:
        return f":{rng.uniform(0.01, 0.5):.4f}"
    
    def build_tree(names: List[str]) -> str:
        if len(names) == 1:
            return names[0] + random_branch()
        if len(names) == 2:
            return f"({names[0]}{random_branch()},{names[1]}{random_branch()}){random_branch()}"
        
        # Split randomly
        split_point = rng.integers(1, len(names))
        left = names[:split_point]
        right = names[split_point:]
        
        return f"({build_tree(left)},{build_tree(right)}){random_branch()}"
    
    rng.shuffle(tip_names)
    return build_tree(tip_names) + ";"


def run_synthetic_smoke_test(
    n_strains: int = 50,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, any]:
    """
    Run a complete smoke test using synthetic data.

    Generates synthetic data and runs the analysis pipeline to verify
    that all components work correctly together.

    Parameters:
        n_strains: Number of strains in synthetic dataset
        random_state: Random seed for reproducibility
        verbose: Print progress messages

    Returns:
        Dictionary with test results:
            - 'success': bool indicating if all tests passed
            - 'data_generated': True if data generation worked
            - 'validation_passed': True if validation passed
            - 'analysis_ran': True if analysis completed
            - 'errors': list of error messages
    """
    # Use module-level imports (already imported at top of file)
    
    results = {
        'success': True,
        'data_generated': False,
        'validation_passed': False,
        'analysis_ran': False,
        'errors': [],
    }
    
    if verbose:
        print(f"Generating synthetic data with {n_strains} strains...")
    
    try:
        # Generate synthetic data
        data, metadata = generate_synthetic_amr_data(
            n_strains=n_strains,
            random_state=random_state,
        )
        results['data_generated'] = True
        
        if verbose:
            print(f"  Generated {data.shape[0]} strains x {data.shape[1]} columns")
            print(f"  Antibiotics: {len(metadata['antibiotic_cols'])}")
            print(f"  Genes: {len(metadata['gene_cols'])}")
        
    except Exception as e:
        results['errors'].append(f"Data generation failed: {e}")
        results['success'] = False
        return results
    
    # Validate the synthetic data
    if verbose:
        print("Validating synthetic data...")
    
    try:
        is_valid, msg = validate_input_data(data, id_col="Strain_ID")
        results['validation_passed'] = is_valid
        
        if not is_valid:
            results['errors'].append(f"Validation failed: {msg}")
        elif verbose:
            print(f"  Validation: {msg}")
        
        # Get data summary
        summary = get_data_summary(data, id_col="Strain_ID")
        if verbose:
            print(f"  Summary: {summary['n_samples']} samples, {summary['n_features']} features")
        
    except Exception as e:
        results['errors'].append(f"Validation failed with exception: {e}")
        results['validation_passed'] = False
    
    # Try to run a minimal analysis (just import and basic function calls)
    if verbose:
        print("Running minimal analysis tests...")
    
    try:
        from .mdr_analysis_core import (
            safe_contingency,
            compute_bootstrap_ci,
            build_class_resistance,
            identify_mdr_isolates,
        )
        
        # Test safe_contingency (pd already imported at module level)
        test_table = pd.DataFrame([[10, 5], [5, 10]])
        chi2, p, phi = safe_contingency(test_table)
        assert not pd.isna(chi2), "Chi2 should not be NaN"
        assert not pd.isna(p), "P-value should not be NaN"
        
        # Test bootstrap CI on a subset
        test_df = data[metadata['antibiotic_cols'][:3]].head(20).copy()
        ci_result = compute_bootstrap_ci(test_df, n_iter=100, confidence_level=0.95)
        assert len(ci_result) == 3, "Should have CI for 3 columns"
        
        results['analysis_ran'] = True
        
        if verbose:
            print("  All analysis functions executed successfully")
            
    except Exception as e:
        results['errors'].append(f"Analysis test failed: {e}")
        results['analysis_ran'] = False
    
    results['success'] = (
        results['data_generated'] and 
        results['validation_passed'] and 
        results['analysis_ran']
    )
    
    if verbose:
        status = "PASSED" if results['success'] else "FAILED"
        print(f"\nSmoke test {status}")
        if results['errors']:
            for err in results['errors']:
                print(f"  ERROR: {err}")
    
    return results
