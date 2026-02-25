"""Tests for PlaidController with mocked PlaidClient and CacheDB."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from portopt.data.models import (
    PlaidAccount, PlaidAccountType,
    Transaction, TransactionSource,
)

pytest.importorskip("plaid", reason="plaid-python not installed")

from portopt.gui.controllers.plaid_controller import PlaidController


@pytest.fixture
def controller(tmp_path):
    """Create a PlaidController with mocked dependencies."""
    with patch("portopt.gui.controllers.plaid_controller.CacheDB") as MockCache, \
         patch("portopt.gui.controllers.plaid_controller.PlaidClient") as MockClient:
        mock_cache = MagicMock()
        MockCache.return_value = mock_cache
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        ctrl = PlaidController()
        ctrl._client = mock_client
        ctrl._cache = mock_cache
        yield ctrl, mock_client, mock_cache


# ── Configuration ────────────────────────────────────────────────────

class TestConfigure:
    @patch("portopt.gui.controllers.plaid_controller.store_credential")
    def test_configure_stores_credentials(self, mock_store, controller):
        ctrl, mock_client, _ = controller
        ctrl.configure("test_client_id", "test_secret", "sandbox")
        assert mock_store.call_count == 3  # client_id, secret, environment

    @patch("portopt.gui.controllers.plaid_controller.get_credential")
    def test_is_configured_checks_keyring(self, mock_get_cred, controller):
        ctrl, _, _ = controller
        mock_get_cred.side_effect = lambda key: {
            "plaid_client_id": "test_id",
            "plaid_secret": "test_secret",
        }.get(key)
        assert ctrl.is_configured


# ── Link Flow ────────────────────────────────────────────────────────

class TestLinkFlow:
    def test_complete_link_stores_item(self, controller):
        ctrl, mock_client, mock_cache = controller
        mock_client.exchange_public_token.return_value = ("access-token-abc", "item_abc")
        mock_client.get_accounts.return_value = [
            PlaidAccount(account_id="acc_1", item_id="item_abc",
                         institution_name="Chase", name="Checking",
                         account_type=PlaidAccountType.CHECKING),
        ]

        with patch("portopt.gui.controllers.plaid_controller.store_credential") as mock_store:
            ctrl._do_complete_link("public-token-xyz", "Chase")
            # Should store access token
            mock_store.assert_called_once()
            mock_cache.upsert_plaid_item.assert_called_once()
            mock_cache.upsert_plaid_accounts.assert_called_once()


# ── Transaction Sync ─────────────────────────────────────────────────

class TestTransactionSync:
    def test_sync_all_iterates_items(self, controller):
        ctrl, mock_client, mock_cache = controller

        items = [
            {"item_id": "item_1", "institution_name": "Chase", "sync_cursor": ""},
            {"item_id": "item_2", "institution_name": "BoA", "sync_cursor": "cursor_old"},
        ]
        mock_client.sync_transactions.return_value = ([], [], [], "new_cursor")

        with patch("portopt.gui.controllers.plaid_controller.get_credential", return_value="access-token"):
            count = ctrl._do_sync_all(items)

        # Should have been called once per item
        assert mock_client.sync_transactions.call_count == 2


# ── Item Removal ─────────────────────────────────────────────────────

class TestRemoveItem:
    @patch("portopt.gui.controllers.plaid_controller.delete_credential")
    def test_remove_cleans_cache_and_keyring(self, mock_del, controller):
        ctrl, _, mock_cache = controller
        ctrl.remove_item("item_xyz")
        mock_cache.delete_plaid_item.assert_called_once_with("item_xyz")
        mock_del.assert_called_once()


# ── Auto-sync Timer ──────────────────────────────────────────────────

class TestAutoSync:
    def test_set_sync_interval(self, controller):
        ctrl, _, _ = controller
        ctrl.set_sync_interval(30)
        assert ctrl._sync_interval_min == 30

    @patch("portopt.gui.controllers.plaid_controller.get_credential")
    def test_start_auto_sync_skips_if_not_configured(self, mock_cred, controller):
        """start_auto_sync does nothing if no credentials."""
        ctrl, _, mock_cache = controller
        mock_cred.return_value = None  # not configured
        ctrl.start_auto_sync()
        assert not ctrl._sync_timer.isActive()

    def test_close_stops_timer(self, controller):
        ctrl, _, _ = controller
        ctrl.close()
        assert not ctrl._sync_timer.isActive()


# ── Static Helpers ───────────────────────────────────────────────────

class TestStaticHelpers:
    def test_account_to_dict(self):
        acct = PlaidAccount(
            account_id="acc_1",
            item_id="item_1",
            institution_name="Chase",
            name="Checking",
            account_type=PlaidAccountType.CHECKING,
            mask="1234",
            current_balance=5000.0,
        )
        result = PlaidController._account_to_dict(acct)
        assert result["account_id"] == "acc_1"
        assert result["account_type"] == "CHECKING"
        assert result["current_balance"] == 5000.0

    def test_transaction_to_dict(self):
        txn = Transaction(
            transaction_id="txn_1",
            account_id="acc_1",
            account_name="Checking",
            date=date(2025, 3, 15),
            amount=42.50,
            source=TransactionSource.PLAID,
        )
        result = PlaidController._transaction_to_dict(txn)
        assert result["transaction_id"] == "txn_1"
        assert result["source"] == "PLAID"
        assert result["amount"] == 42.50

    def test_dict_to_transaction(self):
        row = {
            "transaction_id": "txn_1",
            "account_id": "acc_1",
            "account_name": "Checking",
            "date": "2025-03-15",
            "authorized_date": "",
            "amount": 42.50,
            "merchant_name": "Amazon",
            "name": "Amazon.com",
            "category": "Shopping",
            "status": "POSTED",
            "pending": False,
            "institution_name": "Chase",
            "source": "PLAID",
            "iso_currency_code": "USD",
            "metadata": "{}",
        }
        txn = PlaidController._dict_to_transaction(row)
        assert isinstance(txn, Transaction)
        assert txn.transaction_id == "txn_1"
        assert txn.source == TransactionSource.PLAID
        assert txn.date == date(2025, 3, 15)
