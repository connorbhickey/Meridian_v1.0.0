"""SQLite cache for price data, asset info, and portfolio snapshots."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from portopt.config import get_cache_db_path


class CacheDB:
    """SQLite-backed local cache for market data."""

    def __init__(self, db_path: Path | None = None):
        self._path = db_path or get_cache_db_path()
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._ensure_tables()

    @property
    def conn(self) -> sqlite3.Connection:
        with self._lock:
            if self._conn is None:
                self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
                self._conn.row_factory = sqlite3.Row
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA foreign_keys=ON")
            return self._conn

    def _ensure_tables(self):
        c = self.conn
        c.executescript("""
            CREATE TABLE IF NOT EXISTS assets (
                symbol TEXT PRIMARY KEY,
                name TEXT,
                asset_type TEXT,
                sector TEXT,
                exchange TEXT,
                currency TEXT DEFAULT 'USD',
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS prices (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                adj_close REAL,
                PRIMARY KEY (symbol, date)
            );

            CREATE INDEX IF NOT EXISTS idx_prices_symbol ON prices(symbol);
            CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);

            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                data TEXT,  -- JSON serialized
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
        c.commit()

    # ── Prices ───────────────────────────────────────────────────────
    def get_prices(
        self, symbol: str, start: date | None = None, end: date | None = None
    ) -> pd.DataFrame:
        """Fetch cached price data as a DataFrame."""
        query = "SELECT date, open, high, low, close, volume, adj_close FROM prices WHERE symbol = ?"
        params: list = [symbol]
        if start:
            query += " AND date >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND date <= ?"
            params.append(end.isoformat())
        query += " ORDER BY date"
        df = pd.read_sql_query(query, self.conn, params=params)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        return df

    def store_prices(self, symbol: str, df: pd.DataFrame):
        """Store price data from a DataFrame (index=date, cols=open,high,low,close,volume,adj_close)."""
        if df.empty:
            return
        records = []
        for idx, row in df.iterrows():
            dt = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
            records.append((
                symbol, dt,
                row.get("Open", row.get("open")),
                row.get("High", row.get("high")),
                row.get("Low", row.get("low")),
                row.get("Close", row.get("close")),
                row.get("Volume", row.get("volume", 0)),
                row.get("Adj Close", row.get("adj_close")),
            ))
        self.conn.executemany(
            "INSERT OR REPLACE INTO prices (symbol, date, open, high, low, close, volume, adj_close) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            records,
        )
        self.conn.commit()

    def get_latest_date(self, symbol: str) -> date | None:
        """Return the most recent cached date for a symbol."""
        row = self.conn.execute(
            "SELECT MAX(date) as max_date FROM prices WHERE symbol = ?", (symbol,)
        ).fetchone()
        if row and row["max_date"]:
            return date.fromisoformat(row["max_date"])
        return None

    # ── Assets ───────────────────────────────────────────────────────
    def get_asset(self, symbol: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM assets WHERE symbol = ?", (symbol,)).fetchone()
        return dict(row) if row else None

    def store_asset(self, symbol: str, name: str = "", asset_type: str = "STOCK",
                    sector: str = "", exchange: str = "", currency: str = "USD"):
        self.conn.execute(
            "INSERT OR REPLACE INTO assets (symbol, name, asset_type, sector, exchange, currency, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (symbol, name, asset_type, sector, exchange, currency, datetime.now().isoformat()),
        )
        self.conn.commit()

    # ── Portfolio Snapshots ──────────────────────────────────────────
    def save_portfolio_snapshot(self, name: str, data: dict):
        self.conn.execute(
            "INSERT INTO portfolio_snapshots (name, data) VALUES (?, ?)",
            (name, json.dumps(data)),
        )
        self.conn.commit()

    def get_latest_snapshot(self, name: str) -> dict | None:
        row = self.conn.execute(
            "SELECT data FROM portfolio_snapshots WHERE name = ? ORDER BY created_at DESC LIMIT 1",
            (name,),
        ).fetchone()
        return json.loads(row["data"]) if row else None

    # ── Maintenance ──────────────────────────────────────────────────
    def get_cache_size_mb(self) -> float:
        """Return the cache database file size in MB."""
        if self._path.exists():
            return self._path.stat().st_size / (1024 * 1024)
        return 0.0

    def clear_symbol(self, symbol: str):
        self.conn.execute("DELETE FROM prices WHERE symbol = ?", (symbol,))
        self.conn.execute("DELETE FROM assets WHERE symbol = ?", (symbol,))
        self.conn.commit()

    def clear_all(self):
        self.conn.execute("DELETE FROM prices")
        self.conn.execute("DELETE FROM assets")
        self.conn.execute("DELETE FROM portfolio_snapshots")
        self.conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
