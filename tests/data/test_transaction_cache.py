"""Tests for CacheDB transaction, Plaid item, and Plaid account CRUD."""

from __future__ import annotations

import tempfile
from datetime import date, datetime
from pathlib import Path

import pytest

from portopt.data.cache import CacheDB


@pytest.fixture
def db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_txn_cache.db"
        cache = CacheDB(db_path)
        yield cache
        cache.close()


# ── Plaid Items ──────────────────────────────────────────────────────

class TestPlaidItems:
    def test_upsert_and_get(self, db):
        db.upsert_plaid_item("item_1", "ins_1", "Chase")
        items = db.get_plaid_items()
        assert len(items) == 1
        assert items[0]["item_id"] == "item_1"
        assert items[0]["institution_name"] == "Chase"

    def test_upsert_updates_existing(self, db):
        db.upsert_plaid_item("item_1", "ins_1", "Chase")
        db.upsert_plaid_item("item_1", "ins_1", "JPMorgan Chase")
        items = db.get_plaid_items()
        assert len(items) == 1
        assert items[0]["institution_name"] == "JPMorgan Chase"

    def test_multiple_items(self, db):
        db.upsert_plaid_item("item_1", "ins_1", "Chase")
        db.upsert_plaid_item("item_2", "ins_2", "Bank of America")
        items = db.get_plaid_items()
        assert len(items) == 2

    def test_delete_item(self, db):
        db.upsert_plaid_item("item_1", "ins_1", "Chase")
        # Add related accounts and transactions
        db.upsert_plaid_accounts([{
            "account_id": "acc_1", "item_id": "item_1",
            "institution_name": "Chase", "name": "Checking",
            "account_type": "CHECKING",
        }])
        db.upsert_transactions([{
            "transaction_id": "txn_1", "account_id": "acc_1",
            "date": "2025-01-15", "amount": 50.0, "source": "PLAID",
        }])
        db.delete_plaid_item("item_1")
        assert len(db.get_plaid_items()) == 0
        assert len(db.get_plaid_accounts("item_1")) == 0

    def test_update_sync_cursor(self, db):
        db.upsert_plaid_item("item_1", "ins_1", "Chase")
        db.update_plaid_sync_cursor("item_1", "cursor_abc123")
        items = db.get_plaid_items()
        assert items[0]["sync_cursor"] == "cursor_abc123"


# ── Plaid Accounts ───────────────────────────────────────────────────

class TestPlaidAccounts:
    def _ensure_item(self, db, item_id="item_1", inst="Chase"):
        """Create parent item to satisfy FK constraint."""
        db.upsert_plaid_item(item_id, f"ins_{item_id}", inst)

    def test_upsert_and_get(self, db):
        self._ensure_item(db)
        db.upsert_plaid_accounts([{
            "account_id": "acc_1",
            "item_id": "item_1",
            "institution_name": "Chase",
            "name": "Total Checking",
            "official_name": "Chase Total Checking",
            "account_type": "CHECKING",
            "subtype": "checking",
            "mask": "1234",
            "current_balance": 5000.00,
            "available_balance": 4800.00,
        }])
        accounts = db.get_plaid_accounts()
        assert len(accounts) == 1
        assert accounts[0]["account_id"] == "acc_1"
        assert accounts[0]["name"] == "Total Checking"
        assert float(accounts[0]["current_balance"]) == 5000.00

    def test_get_by_item_id(self, db):
        self._ensure_item(db, "item_1")
        self._ensure_item(db, "item_2", "BoA")
        db.upsert_plaid_accounts([
            {"account_id": "acc_1", "item_id": "item_1", "name": "Checking",
             "account_type": "CHECKING"},
            {"account_id": "acc_2", "item_id": "item_1", "name": "Savings",
             "account_type": "SAVINGS"},
            {"account_id": "acc_3", "item_id": "item_2", "name": "Credit",
             "account_type": "CREDIT_CARD"},
        ])
        item1_accounts = db.get_plaid_accounts("item_1")
        assert len(item1_accounts) == 2
        item2_accounts = db.get_plaid_accounts("item_2")
        assert len(item2_accounts) == 1

    def test_update_balances(self, db):
        self._ensure_item(db)
        db.upsert_plaid_accounts([{
            "account_id": "acc_1", "item_id": "item_1",
            "account_type": "CHECKING", "current_balance": 1000.00,
        }])
        db.update_account_balances("acc_1", 2000.00, 1900.00)
        accounts = db.get_plaid_accounts()
        assert float(accounts[0]["current_balance"]) == 2000.00
        assert float(accounts[0]["available_balance"]) == 1900.00

    def test_upsert_replaces_existing(self, db):
        self._ensure_item(db)
        db.upsert_plaid_accounts([{
            "account_id": "acc_1", "item_id": "item_1",
            "name": "Old Name", "account_type": "CHECKING",
        }])
        db.upsert_plaid_accounts([{
            "account_id": "acc_1", "item_id": "item_1",
            "name": "New Name", "account_type": "CHECKING",
        }])
        accounts = db.get_plaid_accounts()
        assert len(accounts) == 1
        assert accounts[0]["name"] == "New Name"


# ── Transactions (unified) ──────────────────────────────────────────

def _make_txn(txn_id: str, account_id: str = "acc_1",
              txn_date: str = "2025-03-15", amount: float = 25.00,
              source: str = "PLAID", status: str = "POSTED",
              merchant_name: str = "Amazon", name: str = "Amazon.com",
              category: str = "Shopping") -> dict:
    """Helper to build a transaction dict."""
    return {
        "transaction_id": txn_id,
        "account_id": account_id,
        "account_name": "Test Account",
        "date": txn_date,
        "authorized_date": txn_date,
        "amount": amount,
        "merchant_name": merchant_name,
        "name": name,
        "category": category,
        "status": status,
        "pending": False,
        "institution_name": "Chase",
        "source": source,
        "iso_currency_code": "USD",
        "metadata": "{}",
    }


class TestTransactions:
    def test_upsert_and_get(self, db):
        db.upsert_transactions([_make_txn("txn_1")])
        txns = db.get_transactions()
        assert len(txns) == 1
        assert txns[0]["transaction_id"] == "txn_1"

    def test_upsert_replaces_duplicate(self, db):
        db.upsert_transactions([_make_txn("txn_1", amount=10.00)])
        db.upsert_transactions([_make_txn("txn_1", amount=20.00)])
        txns = db.get_transactions()
        assert len(txns) == 1
        assert float(txns[0]["amount"]) == 20.00

    def test_multiple_transactions(self, db):
        db.upsert_transactions([
            _make_txn("txn_1", txn_date="2025-03-15"),
            _make_txn("txn_2", txn_date="2025-03-14"),
            _make_txn("txn_3", txn_date="2025-03-13"),
        ])
        txns = db.get_transactions()
        assert len(txns) == 3

    def test_filter_by_account_id(self, db):
        db.upsert_transactions([
            _make_txn("txn_1", account_id="acc_1"),
            _make_txn("txn_2", account_id="acc_2"),
            _make_txn("txn_3", account_id="acc_1"),
        ])
        txns = db.get_transactions(account_id="acc_1")
        assert len(txns) == 2
        assert all(t["account_id"] == "acc_1" for t in txns)

    def test_filter_by_date_range(self, db):
        db.upsert_transactions([
            _make_txn("txn_1", txn_date="2025-01-10"),
            _make_txn("txn_2", txn_date="2025-02-15"),
            _make_txn("txn_3", txn_date="2025-03-20"),
        ])
        txns = db.get_transactions(start=date(2025, 2, 1), end=date(2025, 2, 28))
        assert len(txns) == 1
        assert txns[0]["transaction_id"] == "txn_2"

    def test_filter_by_status(self, db):
        db.upsert_transactions([
            _make_txn("txn_1", status="POSTED"),
            _make_txn("txn_2", status="PENDING"),
        ])
        txns = db.get_transactions(status="PENDING")
        assert len(txns) == 1
        assert txns[0]["transaction_id"] == "txn_2"

    def test_filter_by_source(self, db):
        db.upsert_transactions([
            _make_txn("txn_1", source="PLAID"),
            _make_txn("txn_2", source="FIDELITY"),
            _make_txn("txn_3", source="PLAID"),
        ])
        txns = db.get_transactions(source="FIDELITY")
        assert len(txns) == 1
        txns = db.get_transactions(source="PLAID")
        assert len(txns) == 2

    def test_limit_and_offset(self, db):
        for i in range(10):
            db.upsert_transactions([_make_txn(f"txn_{i}", txn_date=f"2025-03-{10+i:02d}")])
        txns = db.get_transactions(limit=3, offset=0)
        assert len(txns) == 3
        txns2 = db.get_transactions(limit=3, offset=3)
        assert len(txns2) == 3
        # Should not overlap
        ids1 = {t["transaction_id"] for t in txns}
        ids2 = {t["transaction_id"] for t in txns2}
        assert ids1.isdisjoint(ids2)

    def test_remove_transactions(self, db):
        db.upsert_transactions([
            _make_txn("txn_1"),
            _make_txn("txn_2"),
            _make_txn("txn_3"),
        ])
        db.remove_transactions(["txn_1", "txn_3"])
        txns = db.get_transactions()
        assert len(txns) == 1
        assert txns[0]["transaction_id"] == "txn_2"

    def test_get_transaction_count_all(self, db):
        db.upsert_transactions([
            _make_txn("txn_1", source="PLAID"),
            _make_txn("txn_2", source="FIDELITY"),
            _make_txn("txn_3", source="PLAID"),
        ])
        assert db.get_transaction_count() == 3

    def test_get_transaction_count_by_source(self, db):
        db.upsert_transactions([
            _make_txn("txn_1", source="PLAID"),
            _make_txn("txn_2", source="FIDELITY"),
            _make_txn("txn_3", source="PLAID"),
        ])
        assert db.get_transaction_count(source="PLAID") == 2
        assert db.get_transaction_count(source="FIDELITY") == 1

    def test_empty_transactions(self, db):
        txns = db.get_transactions()
        assert txns == []
        assert db.get_transaction_count() == 0
