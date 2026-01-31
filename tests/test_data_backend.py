"""
Tests for high-performance data backend module.

Tests Parquet caching, DuckDB queries, and performance improvements.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from strepsuis_mdr.data_backend import (
    DataBackend,
    load_data_efficient,
    get_backend_status,
    PYARROW_AVAILABLE,
    DUCKDB_AVAILABLE
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp)


@pytest.fixture
def sample_csv(temp_dir):
    """Create sample CSV file."""
    csv_path = temp_dir / "test_data.csv"
    df = pd.DataFrame({
        'Strain_ID': [f"Strain_{i}" for i in range(100)],
        'feature_1': np.random.randint(0, 2, 100),
        'feature_2': np.random.randint(0, 2, 100),
        'feature_3': np.random.randint(0, 2, 100),
        'cluster': np.random.randint(1, 4, 100)
    })
    df.to_csv(csv_path, index=False)
    return csv_path


class TestDataBackend:
    """Tests for DataBackend class."""

    def test_initialization(self, temp_dir):
        """Test backend initialization."""
        backend = DataBackend(cache_dir=temp_dir)
        assert backend.cache_dir == temp_dir
        assert backend.cache_dir.exists()

    def test_csv_loading(self, temp_dir, sample_csv):
        """Test basic CSV loading."""
        backend = DataBackend(cache_dir=temp_dir)
        df = backend.load(sample_csv)

        assert len(df) == 100
        assert 'Strain_ID' in df.columns
        assert 'feature_1' in df.columns

    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not available")
    def test_parquet_caching(self, temp_dir, sample_csv):
        """Test automatic Parquet caching."""
        backend = DataBackend(cache_dir=temp_dir, use_parquet=True)

        # First load - creates cache
        df1 = backend.load(sample_csv)
        cache_info_1 = backend.get_cache_info()
        assert len(cache_info_1) == 1
        assert cache_info_1.iloc[0]['rows'] == 100

        # Second load - uses cache
        df2 = backend.load(sample_csv)
        pd.testing.assert_frame_equal(df1, df2)

        # Verify cache file exists
        cache_files = list(temp_dir.glob("*.parquet"))
        assert len(cache_files) == 1

    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not available")
    def test_cache_invalidation(self, temp_dir, sample_csv):
        """Test cache invalidation on file modification."""
        backend = DataBackend(cache_dir=temp_dir)

        # First load
        df1 = backend.load(sample_csv)
        cache_info_1 = backend.get_cache_info()
        old_cache_hash = cache_info_1.iloc[0]['source_file']

        # Modify CSV file
        import time
        time.sleep(0.1)  # Ensure mtime changes
        df_modified = pd.read_csv(sample_csv)
        df_modified['feature_4'] = np.random.randint(0, 2, 100)
        df_modified.to_csv(sample_csv, index=False)

        # Second load - should detect change and reload
        df2 = backend.load(sample_csv)
        assert 'feature_4' in df2.columns
        assert len(df2.columns) == len(df1.columns) + 1

    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not available")
    def test_force_reload(self, temp_dir, sample_csv):
        """Test force reload bypasses cache."""
        backend = DataBackend(cache_dir=temp_dir)

        # First load
        df1 = backend.load(sample_csv)

        # Force reload
        df2 = backend.load(sample_csv, force_reload=True)
        pd.testing.assert_frame_equal(df1, df2)

    @pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not available")
    def test_duckdb_query(self, temp_dir, sample_csv):
        """Test DuckDB query execution."""
        backend = DataBackend(cache_dir=temp_dir, use_duckdb=True)
        df = backend.load(sample_csv)

        # Simple aggregation query
        result = backend.query(
            "SELECT cluster, COUNT(*) as count FROM data GROUP BY cluster ORDER BY cluster",
            data=df
        )

        assert len(result) <= 3  # Max 3 clusters
        assert 'cluster' in result.columns
        assert 'count' in result.columns
        assert result['count'].sum() == 100

    @pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not available")
    def test_duckdb_filtering(self, temp_dir, sample_csv):
        """Test DuckDB filtering."""
        backend = DataBackend(cache_dir=temp_dir)
        df = backend.load(sample_csv)

        # Filter query
        result = backend.query(
            "SELECT * FROM data WHERE feature_1 = 1 AND feature_2 = 1",
            data=df
        )

        assert len(result) <= len(df)
        assert all(result['feature_1'] == 1)
        assert all(result['feature_2'] == 1)

    @pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not available")
    def test_register_table(self, temp_dir, sample_csv):
        """Test table registration."""
        backend = DataBackend(cache_dir=temp_dir)
        df = backend.load(sample_csv)

        backend.register_table("mydata", df)
        result = backend.query("SELECT COUNT(*) as n FROM mydata")

        assert result.iloc[0]['n'] == 100

    def test_clear_cache(self, temp_dir, sample_csv):
        """Test cache clearing."""
        backend = DataBackend(cache_dir=temp_dir, use_parquet=PYARROW_AVAILABLE)

        if PYARROW_AVAILABLE:
            # Create cache
            backend.load(sample_csv)
            assert len(backend.get_cache_info()) == 1

            # Clear cache
            backend.clear_cache()
            assert len(backend.get_cache_info()) == 0

    def test_context_manager(self, temp_dir):
        """Test context manager usage."""
        with DataBackend(cache_dir=temp_dir) as backend:
            assert backend.cache_dir == temp_dir

        # Connection should be closed
        if backend.conn is not None:
            # DuckDB connection closed, should raise error
            with pytest.raises(Exception):
                backend.conn.execute("SELECT 1")

    def test_file_not_found(self, temp_dir):
        """Test error handling for missing file."""
        backend = DataBackend(cache_dir=temp_dir)

        with pytest.raises(FileNotFoundError):
            backend.load(temp_dir / "nonexistent.csv")


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_load_data_efficient(self, temp_dir, sample_csv):
        """Test convenience loading function."""
        df = load_data_efficient(sample_csv)
        assert len(df) == 100

    def test_get_backend_status(self):
        """Test backend status function."""
        status = get_backend_status()
        assert 'parquet' in status
        assert 'duckdb' in status
        assert isinstance(status['parquet'], bool)
        assert isinstance(status['duckdb'], bool)


@pytest.mark.skipif(
    not (PYARROW_AVAILABLE and DUCKDB_AVAILABLE),
    reason="Full backend not available"
)
class TestPerformance:
    """Performance benchmarks."""

    def test_parquet_faster_than_csv(self, temp_dir):
        """Test that Parquet loading is faster than CSV."""
        import time

        # Create larger dataset
        large_df = pd.DataFrame({
            f'feature_{i}': np.random.randint(0, 2, 10000)
            for i in range(50)
        })
        csv_path = temp_dir / "large.csv"
        large_df.to_csv(csv_path, index=False)

        backend = DataBackend(cache_dir=temp_dir)

        # First load (CSV → Parquet)
        t0 = time.perf_counter()
        df1 = backend.load(csv_path)
        t_first = time.perf_counter() - t0

        # Second load (Parquet from cache)
        t0 = time.perf_counter()
        df2 = backend.load(csv_path)
        t_cache = time.perf_counter() - t0

        # Cache should be faster (at least 2x)
        assert t_cache < t_first / 2
        pd.testing.assert_frame_equal(df1, df2)

    def test_duckdb_aggregation_performance(self, temp_dir):
        """Test DuckDB aggregation performance."""
        import time

        # Create dataset with many groups
        df = pd.DataFrame({
            'group_id': np.random.randint(0, 100, 50000),
            'value': np.random.randn(50000)
        })

        backend = DataBackend(cache_dir=temp_dir)

        # DuckDB aggregation
        t0 = time.perf_counter()
        result_duckdb = backend.query(
            "SELECT group_id, AVG(value) as mean, COUNT(*) as n FROM data GROUP BY group_id",
            data=df
        )
        t_duckdb = time.perf_counter() - t0

        # Pandas aggregation
        t0 = time.perf_counter()
        result_pandas = df.groupby('group_id').agg({'value': ['mean', 'count']}).reset_index()
        t_pandas = time.perf_counter() - t0

        assert len(result_duckdb) == len(result_pandas)
        # DuckDB should be competitive or faster
        # (Not asserting specific speedup as it depends on system)
