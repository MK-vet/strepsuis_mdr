"""
Large Data Integration for MDR Module

This module integrates server-side large data pipeline components for
MDR pattern analysis and heatmap visualization of large datasets.

Features:
- DuckDB-based MDR pattern queries
- HoloViews tiled heatmaps for large correlation matrices
- Datashader for MDR pattern visualization
- Server-side processing (no raw data to browser)

Author: MK-vet
License: MIT
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

# Import shared large data pipeline components
try:
    from shared.duckdb_handler import DuckDBHandler
    from shared.datashader_plots import DatashaderPlots
    from shared.holoviews_heatmaps import HoloViewsHeatmaps, create_distance_heatmap
    LARGE_DATA_AVAILABLE = True
except ImportError:
    LARGE_DATA_AVAILABLE = False

logger = logging.getLogger(__name__)


class LargeDataMDR:
    """
    Large-scale data processing for MDR pattern analysis.

    Handles large datasets (100k+ gene combinations) using server-side processing.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize large data integration.

        Parameters
        ----------
        output_dir : Path, optional
            Directory for output files
        """
        if not LARGE_DATA_AVAILABLE:
            raise ImportError(
                "Large data pipeline not available. "
                "Install with: pip install duckdb datashader holoviews"
            )

        self.output_dir = output_dir or Path("mdr_large_data_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.db_handler = DuckDBHandler()
        self.plotter = DatashaderPlots(width=1200, height=900)
        self.heatmap = HoloViewsHeatmaps(tile_size=512)

        logger.info(f"Large data MDR initialized: {self.output_dir}")

    def load_mdr_patterns(
        self,
        data_path: Union[str, Path],
        table_name: str = "mdr_patterns"
    ) -> Dict[str, Any]:
        """
        Load large MDR pattern dataset into DuckDB.

        Parameters
        ----------
        data_path : str or Path
            Path to CSV or Parquet file
        table_name : str
            Table name in DuckDB

        Returns
        -------
        dict
            Dataset metadata
        """
        data_path = Path(data_path)

        if data_path.suffix == '.csv':
            metadata = self.db_handler.load_csv(data_path, table_name)
        elif data_path.suffix == '.parquet':
            metadata = self.db_handler.load_parquet(data_path, table_name)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

        logger.info(f"Loaded {metadata['row_count']:,} MDR patterns")
        return metadata

    def query_mdr_patterns(
        self,
        table_name: str = "mdr_patterns",
        min_support: Optional[float] = None,
        min_confidence: Optional[float] = None,
        gene_list: Optional[List[str]] = None,
        page: int = 1,
        page_size: int = 50
    ) -> Dict[str, Any]:
        """
        Query MDR patterns with filtering and pagination.

        Parameters
        ----------
        table_name : str
            Table name
        min_support : float, optional
            Minimum support threshold
        min_confidence : float, optional
            Minimum confidence threshold
        gene_list : list of str, optional
            Filter by specific genes
        page : int
            Page number
        page_size : int
            Rows per page

        Returns
        -------
        dict
            Paginated query results
        """
        where_clauses = []

        if min_support is not None:
            where_clauses.append(f"support >= {min_support}")

        if min_confidence is not None:
            where_clauses.append(f"confidence >= {min_confidence}")

        if gene_list:
            gene_conditions = [f"genes LIKE '%{gene}%'" for gene in gene_list]
            where_clauses.append(f"({' OR '.join(gene_conditions)})")

        where = " AND ".join(where_clauses) if where_clauses else None

        return self.db_handler.query_table(
            table_name,
            where=where,
            order_by="support DESC, confidence DESC",
            page=page,
            page_size=page_size
        )

    def create_mdr_heatmap(
        self,
        df: pd.DataFrame,
        gene_columns: Optional[List[str]] = None,
        output_dir: Optional[Path] = None,
        method: str = 'correlation'
    ) -> Dict[str, Any]:
        """
        Create tiled heatmap for MDR gene patterns.

        Parameters
        ----------
        df : pd.DataFrame
            MDR gene data (presence/absence)
        gene_columns : list of str, optional
            Gene columns to include
        output_dir : Path, optional
            Output directory for tiles
        method : str
            'correlation' or 'distance'

        Returns
        -------
        dict
            Heatmap metadata
        """
        if output_dir is None:
            output_dir = self.output_dir / "mdr_heatmap_tiles"

        if gene_columns:
            df_genes = df[gene_columns]
        else:
            df_genes = df.select_dtypes(include=[np.number])

        logger.info(f"Creating {method} heatmap: {df_genes.shape[1]} genes")

        if method == 'correlation':
            # Correlation heatmap
            from shared.holoviews_heatmaps import create_correlation_heatmap
            return create_correlation_heatmap(
                df_genes,
                method='phi',  # Phi coefficient for binary data
                output_dir=output_dir
            )
        elif method == 'distance':
            # Distance matrix heatmap
            from scipy.spatial.distance import pdist, squareform
            distances = pdist(df_genes.T, metric='jaccard')  # Jaccard for binary
            distance_matrix = squareform(distances)

            return create_distance_heatmap(
                distance_matrix,
                labels=df_genes.columns.tolist(),
                output_dir=output_dir
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def visualize_mdr_network(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        pattern_col: str,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Visualize MDR pattern network/clusters.

        Parameters
        ----------
        df : pd.DataFrame
            MDR pattern data with coordinates
        x_col : str
            X-axis column
        y_col : str
            Y-axis column
        pattern_col : str
            Pattern type column
        output_path : Path, optional
            Output image path

        Returns
        -------
        Path
            Path to saved image
        """
        if output_path is None:
            output_path = self.output_dir / "mdr_network_visualization.png"

        logger.info(f"Visualizing {len(df):,} MDR patterns")

        return self.plotter.scatter_plot(
            df, x_col, y_col,
            color_by=pattern_col,
            aggregation='count',
            output_path=output_path
        )

    def export_top_patterns(
        self,
        table_name: str = "mdr_patterns",
        n_patterns: int = 1000,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Export top MDR patterns by support/confidence.

        Parameters
        ----------
        table_name : str
            Table name
        n_patterns : int
            Number of top patterns to export
        output_path : Path, optional
            Output CSV path

        Returns
        -------
        Path
            Path to saved CSV
        """
        if output_path is None:
            output_path = self.output_dir / "top_mdr_patterns.csv"

        query = f"""
        SELECT *
        FROM {table_name}
        ORDER BY support DESC, confidence DESC
        LIMIT {n_patterns}
        """

        return self.db_handler.export_filtered(query, output_path, format='csv')

    def get_pattern_statistics(
        self,
        table_name: str = "mdr_patterns"
    ) -> Dict[str, Any]:
        """
        Get summary statistics for MDR patterns.

        Parameters
        ----------
        table_name : str
            Table name

        Returns
        -------
        dict
            Pattern statistics
        """
        stats = self.db_handler.get_summary_statistics(
            table_name,
            columns=['support', 'confidence', 'lift']
        )

        # Additional pattern-specific stats
        query = f"""
        SELECT
            COUNT(*) as total_patterns,
            COUNT(DISTINCT pattern_type) as unique_pattern_types,
            AVG(pattern_length) as avg_pattern_length,
            MAX(pattern_length) as max_pattern_length
        FROM {table_name}
        """

        result = self.db_handler.connection.execute(query).fetchone()

        stats['pattern_summary'] = {
            'total_patterns': result[0],
            'unique_pattern_types': result[1],
            'avg_pattern_length': result[2],
            'max_pattern_length': result[3]
        }

        return stats

    def close(self):
        """Close database connection."""
        self.db_handler.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def process_large_mdr_dataset(
    data_path: Union[str, Path],
    output_dir: Union[str, Path],
    gene_data: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Convenience function to process large MDR dataset.

    Parameters
    ----------
    data_path : str or Path
        Path to input data
    output_dir : str or Path
        Output directory
    gene_data : pd.DataFrame, optional
        Gene presence/absence data for heatmaps

    Returns
    -------
    dict
        Processing results and output paths
    """
    output_dir = Path(output_dir)

    with LargeDataMDR(output_dir) as processor:
        # Load data
        metadata = processor.load_mdr_patterns(data_path)

        results = {
            'metadata': metadata,
            'outputs': {}
        }

        # Get statistics
        stats = processor.get_pattern_statistics()
        results['statistics'] = stats

        # Export top patterns
        top_patterns = processor.export_top_patterns(
            n_patterns=1000,
            output_path=output_dir / "top_patterns.csv"
        )
        results['outputs']['top_patterns'] = str(top_patterns)

        # Create visualizations if gene data provided
        if gene_data is not None:
            # Correlation heatmap
            heatmap_corr = processor.create_mdr_heatmap(
                gene_data,
                output_dir=output_dir / "correlation_tiles",
                method='correlation'
            )
            results['outputs']['correlation_heatmap'] = heatmap_corr

            # Distance heatmap
            heatmap_dist = processor.create_mdr_heatmap(
                gene_data,
                output_dir=output_dir / "distance_tiles",
                method='distance'
            )
            results['outputs']['distance_heatmap'] = heatmap_dist

        return results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Large Data Integration - MDR Module")
    print("=" * 60)

    if not LARGE_DATA_AVAILABLE:
        print("ERROR: Large data pipeline not available")
        print("Install with: pip install duckdb datashader holoviews pillow matplotlib")
        exit(1)

    # Create sample large MDR dataset
    print("\nCreating sample dataset (100,000 patterns)...")
    np.random.seed(42)

    n_patterns = 100_000
    n_genes = 50

    data = {
        'pattern_id': [f"PAT_{i:06d}" for i in range(n_patterns)],
        'pattern_type': np.random.choice(['2-way', '3-way', '4-way'], n_patterns),
        'pattern_length': np.random.randint(2, 6, n_patterns),
        'support': np.random.uniform(0.01, 0.5, n_patterns),
        'confidence': np.random.uniform(0.5, 1.0, n_patterns),
        'lift': np.random.uniform(1.0, 10.0, n_patterns),
        'genes': [
            ','.join(np.random.choice([f"gene_{i}" for i in range(n_genes)],
                                    np.random.randint(2, 6), replace=False))
            for _ in range(n_patterns)
        ]
    }

    df_patterns = pd.DataFrame(data)
    test_file = Path("test_mdr_large.csv")
    df_patterns.to_csv(test_file, index=False)
    print(f"Created: {test_file}")

    # Create sample gene data
    n_isolates = 1000
    gene_data = pd.DataFrame(
        np.random.randint(0, 2, (n_isolates, n_genes)),
        columns=[f"gene_{i}" for i in range(n_genes)]
    )

    # Process dataset
    print("\nProcessing large MDR dataset...")
    results = process_large_mdr_dataset(
        test_file,
        "test_mdr_output",
        gene_data=gene_data
    )

    print(f"\n✓ Processing complete")
    print(f"Loaded: {results['metadata']['row_count']:,} patterns")
    print(f"Statistics: {results['statistics']['pattern_summary']}")
    print(f"Outputs: {list(results['outputs'].keys())}")
