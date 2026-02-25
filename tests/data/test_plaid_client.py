"""Tests for PlaidClient with mocked plaid-python SDK."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from portopt.data.models import PlaidAccount, PlaidAccountType, Transaction, TransactionSource


# ── Skip if plaid SDK not installed ──────────────────────────────────

plaid = pytest.importorskip("plaid", reason="plaid-python not installed")

from portopt.data.providers.plaid_client import PlaidClient  # noqa: E402


@pytest.fixture
def mock_api_client():
    """Create a PlaidClient with fully mocked internal SDK client."""
    with patch("portopt.data.providers.plaid_client.plaid") as mock_plaid_module:
        mock_api = MagicMock()
        mock_plaid_module.ApiClient.return_value = mock_api
        mock_plaid_api = MagicMock()
        mock_plaid_module.api.PlaidApi.return_value = mock_plaid_api
        client = PlaidClient("test_client_id", "test_secret", "sandbox")
        client._client = mock_plaid_api
        yield client, mock_plaid_api


# ── create_link_token ────────────────────────────────────────────────

class TestCreateLinkToken:
    def test_returns_token_string(self, mock_api_client):
        client, mock_api = mock_api_client
        mock_api.link_token_create.return_value = {"link_token": "link-sandbox-abc123"}
        token = client.create_link_token()
        assert token == "link-sandbox-abc123"
        mock_api.link_token_create.assert_called_once()

    def test_custom_user_id(self, mock_api_client):
        client, mock_api = mock_api_client
        mock_api.link_token_create.return_value = {"link_token": "link-sandbox-xyz"}
        token = client.create_link_token(user_id="custom-user-42")
        assert token == "link-sandbox-xyz"


# ── exchange_public_token ────────────────────────────────────────────

class TestExchangePublicToken:
    def test_returns_access_token_and_item_id(self, mock_api_client):
        client, mock_api = mock_api_client
        mock_api.item_public_token_exchange.return_value = {
            "access_token": "access-sandbox-abc",
            "item_id": "item_abc",
        }
        access_token, item_id = client.exchange_public_token("public-sandbox-xyz")
        assert access_token == "access-sandbox-abc"
        assert item_id == "item_abc"
        mock_api.item_public_token_exchange.assert_called_once()


# ── get_accounts ─────────────────────────────────────────────────────

class TestGetAccounts:
    def test_parses_checking_account(self, mock_api_client):
        client, mock_api = mock_api_client
        mock_api.accounts_get.return_value = {
            "accounts": [
                {
                    "account_id": "acc_001",
                    "name": "Total Checking",
                    "official_name": "Chase Total Checking",
                    "type": "depository",
                    "subtype": "checking",
                    "mask": "1234",
                    "balances": {
                        "current": 5000.00,
                        "available": 4800.00,
                        "limit": None,
                    },
                },
            ],
        }
        accounts = client.get_accounts("access-token", "Chase", "item_1")
        assert len(accounts) == 1
        acct = accounts[0]
        assert isinstance(acct, PlaidAccount)
        assert acct.account_id == "acc_001"
        assert acct.name == "Total Checking"
        assert acct.official_name == "Chase Total Checking"
        assert acct.account_type == PlaidAccountType.CHECKING
        assert acct.mask == "1234"
        assert acct.current_balance == 5000.00
        assert acct.available_balance == 4800.00
        assert acct.limit is None
        assert acct.item_id == "item_1"
        assert acct.institution_name == "Chase"

    def test_parses_credit_card_account(self, mock_api_client):
        client, mock_api = mock_api_client
        mock_api.accounts_get.return_value = {
            "accounts": [
                {
                    "account_id": "acc_002",
                    "name": "Freedom Flex",
                    "official_name": "",
                    "type": "credit",
                    "subtype": "credit card",
                    "mask": "5678",
                    "balances": {
                        "current": 1500.00,
                        "available": 3500.00,
                        "limit": 5000.00,
                    },
                },
            ],
        }
        accounts = client.get_accounts("access-token", "Chase", "item_1")
        acct = accounts[0]
        assert acct.account_type == PlaidAccountType.CREDIT_CARD
        assert acct.limit == 5000.00

    def test_multiple_accounts(self, mock_api_client):
        client, mock_api = mock_api_client
        mock_api.accounts_get.return_value = {
            "accounts": [
                {
                    "account_id": "acc_1", "name": "Checking",
                    "type": "depository", "subtype": "checking",
                    "mask": "1111", "balances": {"current": 100, "available": 90, "limit": None},
                },
                {
                    "account_id": "acc_2", "name": "Savings",
                    "type": "depository", "subtype": "savings",
                    "mask": "2222", "balances": {"current": 200, "available": 200, "limit": None},
                },
            ],
        }
        accounts = client.get_accounts("access-token")
        assert len(accounts) == 2


# ── sync_transactions ────────────────────────────────────────────────

class TestSyncTransactions:
    def test_handles_added_modified_removed(self, mock_api_client):
        client, mock_api = mock_api_client
        mock_api.transactions_sync.return_value = {
            "added": [
                {
                    "transaction_id": "txn_001",
                    "account_id": "acc_1",
                    "date": date(2025, 3, 15),
                    "authorized_date": date(2025, 3, 14),
                    "amount": 42.50,
                    "merchant_name": "Starbucks",
                    "name": "STARBUCKS #1234",
                    "category": ["Food and Drink", "Restaurants", "Coffee"],
                    "pending": False,
                    "iso_currency_code": "USD",
                },
            ],
            "modified": [
                {
                    "transaction_id": "txn_002",
                    "account_id": "acc_1",
                    "date": date(2025, 3, 10),
                    "authorized_date": None,
                    "amount": 100.00,
                    "merchant_name": "Amazon",
                    "name": "Amazon.com",
                    "category": ["Shopping"],
                    "pending": False,
                    "iso_currency_code": "USD",
                },
            ],
            "removed": [
                {"transaction_id": "txn_003"},
            ],
            "has_more": False,
            "next_cursor": "cursor_xyz",
        }
        added, modified, removed_ids, cursor = client.sync_transactions(
            "access-token", "", "Chase", {"acc_1": "Checking"}
        )
        assert len(added) == 1
        assert len(modified) == 1
        assert removed_ids == ["txn_003"]
        assert cursor == "cursor_xyz"

        # Verify added transaction fields
        txn = added[0]
        assert isinstance(txn, Transaction)
        assert txn.transaction_id == "txn_001"
        assert txn.source == TransactionSource.PLAID
        assert txn.merchant_name == "Starbucks"

    def test_returns_cursor(self, mock_api_client):
        client, mock_api = mock_api_client
        mock_api.transactions_sync.return_value = {
            "added": [], "modified": [], "removed": [],
            "has_more": False, "next_cursor": "new_cursor_abc",
        }
        _, _, _, cursor = client.sync_transactions("access-token", "old_cursor")
        assert cursor == "new_cursor_abc"

    def test_handles_pagination(self, mock_api_client):
        """When has_more=True, should make multiple API calls."""
        client, mock_api = mock_api_client
        # First call: has_more=True
        mock_api.transactions_sync.side_effect = [
            {
                "added": [
                    {"transaction_id": "txn_1", "account_id": "acc_1",
                     "date": date(2025, 3, 15), "amount": 10.0,
                     "merchant_name": "A", "name": "A", "category": [],
                     "pending": False, "iso_currency_code": "USD"},
                ],
                "modified": [], "removed": [],
                "has_more": True, "next_cursor": "cursor_2",
            },
            {
                "added": [
                    {"transaction_id": "txn_2", "account_id": "acc_1",
                     "date": date(2025, 3, 14), "amount": 20.0,
                     "merchant_name": "B", "name": "B", "category": [],
                     "pending": False, "iso_currency_code": "USD"},
                ],
                "modified": [], "removed": [],
                "has_more": False, "next_cursor": "cursor_final",
            },
        ]
        added, _, _, cursor = client.sync_transactions("access-token")
        assert len(added) == 2
        assert cursor == "cursor_final"
        assert mock_api.transactions_sync.call_count == 2


# ── refresh_balances ─────────────────────────────────────────────────

class TestRefreshBalances:
    def test_delegates_to_get_accounts(self, mock_api_client):
        client, mock_api = mock_api_client
        mock_api.accounts_get.return_value = {
            "accounts": [
                {
                    "account_id": "acc_1", "name": "Checking",
                    "type": "depository", "subtype": "checking",
                    "mask": "1234", "balances": {"current": 999, "available": 900, "limit": None},
                },
            ],
        }
        accounts = client.refresh_balances("access-token", "item_1", "Chase")
        assert len(accounts) == 1
        assert accounts[0].current_balance == 999
