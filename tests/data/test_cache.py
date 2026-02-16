"""Tests for SQLite price cache."""

import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portopt.data.cache import CacheDB


@pytest.fixture
def tmp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_cache.db"
        db = CacheDB(db_path)
        yield db
        db.close()


@pytest.fixture
def sample_prices():
    """Sample OHLCV DataFrame."""
    dates = pd.bdate_range(start="2024-01-02", periods=10)
    return pd.DataFrame({
        "Open": np.random.uniform(100, 110, 10),
        "High": np.random.uniform(110, 120, 10),
        "Low": np.random.uniform(90, 100, 10),
        "Close": np.random.uniform(100, 110, 10),
        "Volume": np.random.randint(1000, 10000, 10),
    }, index=dates)


class TestCacheDB:
    def test_store_and_retrieve_prices(self, tmp_db, sample_prices):
        tmp_db.store_prices("AAPL", sample_prices)
        retrieved = tmp_db.get_prices("AAPL")
        assert len(retrieved) == 10
        # Cache stores columns as lowercase
        assert "close" in retrieved.columns or "Close" in retrieved.columns

    def test_empty_for_unknown_symbol(self, tmp_db):
        result = tmp_db.get_prices("UNKNOWN")
        assert len(result) == 0

    def test_get_latest_date(self, tmp_db, sample_prices):
        tmp_db.store_prices("AAPL", sample_prices)
        latest = tmp_db.get_latest_date("AAPL")
        assert latest is not None

    def test_store_and_get_asset(self, tmp_db):
        tmp_db.store_asset("AAPL", name="Apple Inc", asset_type="STOCK",
                           sector="Technology")
        asset = tmp_db.get_asset("AAPL")
        assert asset is not None
        assert asset["name"] == "Apple Inc"

    def test_cache_size(self, tmp_db, sample_prices):
        tmp_db.store_prices("AAPL", sample_prices)
        size = tmp_db.get_cache_size_mb()
        assert size >= 0

    def test_clear_symbol(self, tmp_db, sample_prices):
        tmp_db.store_prices("AAPL", sample_prices)
        tmp_db.clear_symbol("AAPL")
        result = tmp_db.get_prices("AAPL")
        assert len(result) == 0

    def test_date_range_filter(self, tmp_db, sample_prices):
        tmp_db.store_prices("AAPL", sample_prices)
        start = sample_prices.index[3].date()
        end = sample_prices.index[7].date()
        result = tmp_db.get_prices("AAPL", start=start, end=end)
        assert len(result) <= 5

    def test_portfolio_snapshot(self, tmp_db):
        data = {"symbols": ["AAPL", "MSFT"], "weights": [0.6, 0.4]}
        tmp_db.save_portfolio_snapshot("test_portfolio", data)
        snapshot = tmp_db.get_latest_snapshot("test_portfolio")
        assert snapshot is not None
