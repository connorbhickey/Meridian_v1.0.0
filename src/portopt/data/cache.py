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

            CREATE TABLE IF NOT EXISTS plaid_items (
                item_id TEXT PRIMARY KEY,
                institution_id TEXT,
                institution_name TEXT,
                sync_cursor TEXT DEFAULT '',
                last_synced TEXT,
                error TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS plaid_accounts (
                account_id TEXT PRIMARY KEY,
                item_id TEXT,
                institution_name TEXT,
                name TEXT,
                official_name TEXT,
                account_type TEXT,
                subtype TEXT,
                mask TEXT,
                current_balance REAL DEFAULT 0,
                available_balance REAL,
                credit_limit REAL,
                last_synced TEXT,
                FOREIGN KEY (item_id) REFERENCES plaid_items(item_id)
            );

            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                account_id TEXT,
                account_name TEXT,
                date TEXT,
                authorized_date TEXT,
                amount REAL,
                merchant_name TEXT,
                name TEXT,
                category TEXT,
                status TEXT,
                pending INTEGER DEFAULT 0,
                institution_name TEXT,
                source TEXT,
                iso_currency_code TEXT DEFAULT 'USD',
                metadata TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_txn_account ON transactions(account_id);
            CREATE INDEX IF NOT EXISTS idx_txn_date ON transactions(date);
            CREATE INDEX IF NOT EXISTS idx_txn_status ON transactions(status);
            CREATE INDEX IF NOT EXISTS idx_txn_source ON transactions(source);
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

    # ── Plaid Items ──────────────────────────────────────────────────
    def upsert_plaid_item(self, item_id: str, institution_id: str = "",
                          institution_name: str = "", sync_cursor: str = "",
                          error: str = ""):
        self.conn.execute(
            "INSERT OR REPLACE INTO plaid_items "
            "(item_id, institution_id, institution_name, sync_cursor, last_synced, error) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (item_id, institution_id, institution_name, sync_cursor,
             datetime.now().isoformat(), error),
        )
        self.conn.commit()

    def get_plaid_items(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM plaid_items").fetchall()
        return [dict(r) for r in rows]

    def delete_plaid_item(self, item_id: str):
        # Delete transactions first (references account_ids), then accounts, then item
        self.conn.execute(
            "DELETE FROM transactions WHERE account_id IN "
            "(SELECT account_id FROM plaid_accounts WHERE item_id = ?)", (item_id,),
        )
        self.conn.execute("DELETE FROM plaid_accounts WHERE item_id = ?", (item_id,))
        self.conn.execute("DELETE FROM plaid_items WHERE item_id = ?", (item_id,))
        self.conn.commit()

    def update_plaid_sync_cursor(self, item_id: str, cursor: str):
        self.conn.execute(
            "UPDATE plaid_items SET sync_cursor = ?, last_synced = ? WHERE item_id = ?",
            (cursor, datetime.now().isoformat(), item_id),
        )
        self.conn.commit()

    # ── Plaid Accounts ────────────────────────────────────────────────
    def upsert_plaid_accounts(self, accounts: list[dict]):
        for acct in accounts:
            self.conn.execute(
                "INSERT OR REPLACE INTO plaid_accounts "
                "(account_id, item_id, institution_name, name, official_name, "
                "account_type, subtype, mask, current_balance, available_balance, "
                "credit_limit, last_synced) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    acct["account_id"], acct.get("item_id", ""),
                    acct.get("institution_name", ""), acct.get("name", ""),
                    acct.get("official_name", ""), acct.get("account_type", ""),
                    acct.get("subtype", ""), acct.get("mask", ""),
                    acct.get("current_balance", 0), acct.get("available_balance"),
                    acct.get("credit_limit"), datetime.now().isoformat(),
                ),
            )
        self.conn.commit()

    def get_plaid_accounts(self, item_id: str | None = None) -> list[dict]:
        if item_id:
            rows = self.conn.execute(
                "SELECT * FROM plaid_accounts WHERE item_id = ?", (item_id,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM plaid_accounts").fetchall()
        return [dict(r) for r in rows]

    def update_account_balances(self, account_id: str, current_balance: float,
                                available_balance: float | None = None,
                                credit_limit: float | None = None):
        self.conn.execute(
            "UPDATE plaid_accounts SET current_balance = ?, available_balance = ?, "
            "credit_limit = ?, last_synced = ? WHERE account_id = ?",
            (current_balance, available_balance, credit_limit,
             datetime.now().isoformat(), account_id),
        )
        self.conn.commit()

    # ── Transactions (unified) ────────────────────────────────────────
    def upsert_transactions(self, txns: list[dict]):
        for txn in txns:
            self.conn.execute(
                "INSERT OR REPLACE INTO transactions "
                "(transaction_id, account_id, account_name, date, authorized_date, "
                "amount, merchant_name, name, category, status, pending, "
                "institution_name, source, iso_currency_code, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    txn["transaction_id"], txn.get("account_id", ""),
                    txn.get("account_name", ""), txn.get("date", ""),
                    txn.get("authorized_date", ""), txn.get("amount", 0),
                    txn.get("merchant_name", ""), txn.get("name", ""),
                    txn.get("category", ""), txn.get("status", "POSTED"),
                    1 if txn.get("pending", False) else 0,
                    txn.get("institution_name", ""), txn.get("source", "PLAID"),
                    txn.get("iso_currency_code", "USD"),
                    json.dumps(txn.get("metadata", {})),
                ),
            )
        self.conn.commit()

    def remove_transactions(self, transaction_ids: list[str]):
        if not transaction_ids:
            return
        placeholders = ",".join("?" for _ in transaction_ids)
        self.conn.execute(
            f"DELETE FROM transactions WHERE transaction_id IN ({placeholders})",
            transaction_ids,
        )
        self.conn.commit()

    def get_transactions(
        self, account_id: str | None = None, start: date | None = None,
        end: date | None = None, status: str | None = None,
        source: str | None = None, limit: int = 100, offset: int = 0,
    ) -> list[dict]:
        query = "SELECT * FROM transactions WHERE 1=1"
        params: list = []
        if account_id:
            query += " AND account_id = ?"
            params.append(account_id)
        if start:
            query += " AND date >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND date <= ?"
            params.append(end.isoformat())
        if status:
            query += " AND status = ?"
            params.append(status)
        if source:
            query += " AND source = ?"
            params.append(source)
        query += " ORDER BY date DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_transaction_count(self, source: str | None = None) -> int:
        if source:
            row = self.conn.execute(
                "SELECT COUNT(*) as cnt FROM transactions WHERE source = ?", (source,)
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) as cnt FROM transactions").fetchone()
        return row["cnt"] if row else 0

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
