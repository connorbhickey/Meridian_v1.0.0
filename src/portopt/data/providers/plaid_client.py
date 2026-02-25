"""Plaid API client wrapper — zero GUI knowledge.

Wraps the plaid-python SDK to provide structured access to
transactions, accounts, and balances from linked financial institutions.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

import plaid
from plaid.api import plaid_api
from plaid.model.accounts_get_request import AccountsGetRequest
from plaid.model.country_code import CountryCode
from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
from plaid.model.link_token_create_request import LinkTokenCreateRequest
from plaid.model.link_token_create_request_user import LinkTokenCreateRequestUser
from plaid.model.products import Products
from plaid.model.transactions_sync_request import TransactionsSyncRequest

from portopt.data.models import (
    PlaidAccount,
    PlaidAccountType,
    PlaidItem,
    Transaction,
    TransactionSource,
    TransactionStatus,
)

logger = logging.getLogger(__name__)

# Map Plaid account type strings to our enum
_ACCOUNT_TYPE_MAP = {
    "depository": {
        "checking": PlaidAccountType.CHECKING,
        "savings": PlaidAccountType.SAVINGS,
        "money market": PlaidAccountType.SAVINGS,
        "cash management": PlaidAccountType.CASH_MANAGEMENT,
    },
    "credit": {
        "credit card": PlaidAccountType.CREDIT_CARD,
    },
    "investment": {
        "ira": PlaidAccountType.IRA,
        "401k": PlaidAccountType.IRA,
        "brokerage": PlaidAccountType.BROKERAGE,
    },
}


def _resolve_account_type(acct_type: str, subtype: str) -> PlaidAccountType:
    """Map Plaid type/subtype to PlaidAccountType."""
    type_map = _ACCOUNT_TYPE_MAP.get(acct_type, {})
    if subtype in type_map:
        return type_map[subtype]
    # Fallback by top-level type
    if acct_type == "depository":
        return PlaidAccountType.CHECKING
    if acct_type == "credit":
        return PlaidAccountType.CREDIT_CARD
    if acct_type == "investment":
        return PlaidAccountType.BROKERAGE
    return PlaidAccountType.OTHER


class PlaidClient:
    """Wraps the Plaid Python SDK for transaction/account operations."""

    def __init__(self, client_id: str, secret: str, environment: str = "sandbox"):
        env_map = {
            "sandbox": plaid.Environment.Sandbox,
            "development": plaid.Environment.Development,
            "production": plaid.Environment.Production,
        }
        host = env_map.get(environment.lower(), plaid.Environment.Sandbox)

        configuration = plaid.Configuration(
            host=host,
            api_key={
                "clientId": client_id,
                "secret": secret,
                "plaidVersion": "2020-09-14",
            },
        )
        api_client = plaid.ApiClient(configuration)
        self._client = plaid_api.PlaidApi(api_client)
        logger.info("PlaidClient initialized (env=%s)", environment)

    def create_link_token(self, user_id: str = "meridian-user") -> str:
        """Create a link token for initializing Plaid Link UI.

        Returns the link_token string.
        """
        request = LinkTokenCreateRequest(
            user=LinkTokenCreateRequestUser(client_user_id=user_id),
            client_name="Meridian",
            products=[Products("transactions")],
            country_codes=[CountryCode("US")],
            language="en",
        )
        response = self._client.link_token_create(request)
        token = response["link_token"]
        logger.info("Created Plaid link token")
        return token

    def exchange_public_token(self, public_token: str) -> tuple[str, str]:
        """Exchange a public token from Plaid Link for an access token.

        Returns (access_token, item_id).
        """
        request = ItemPublicTokenExchangeRequest(public_token=public_token)
        response = self._client.item_public_token_exchange(request)
        access_token = response["access_token"]
        item_id = response["item_id"]
        logger.info("Exchanged public token -> item_id=%s", item_id)
        return access_token, item_id

    def get_accounts(self, access_token: str, institution_name: str = "",
                     item_id: str = "") -> list[PlaidAccount]:
        """Fetch all accounts for an access token."""
        request = AccountsGetRequest(access_token=access_token)
        response = self._client.accounts_get(request)
        accounts = []
        for acct in response["accounts"]:
            acct_type = str(acct.get("type", ""))
            subtype = str(acct.get("subtype", ""))
            balances = acct.get("balances", {})
            accounts.append(PlaidAccount(
                account_id=acct["account_id"],
                item_id=item_id,
                institution_name=institution_name,
                name=acct.get("name", ""),
                official_name=acct.get("official_name", "") or "",
                account_type=_resolve_account_type(acct_type, subtype),
                subtype=subtype,
                mask=acct.get("mask", "") or "",
                current_balance=float(balances.get("current", 0) or 0),
                available_balance=float(balances["available"]) if balances.get("available") is not None else None,
                limit=float(balances["limit"]) if balances.get("limit") is not None else None,
                last_synced=datetime.now(),
            ))
        logger.info("Fetched %d accounts for item %s", len(accounts), item_id)
        return accounts

    def sync_transactions(
        self, access_token: str, cursor: str = "",
        institution_name: str = "", account_map: dict[str, str] | None = None,
    ) -> tuple[list[Transaction], list[Transaction], list[str], str]:
        """Incrementally sync transactions using /transactions/sync.

        Args:
            access_token: Plaid access token for the Item
            cursor: Previous sync cursor (empty string for initial sync)
            institution_name: Institution name for labeling
            account_map: account_id -> account_name mapping

        Returns:
            (added, modified, removed_ids, next_cursor)
        """
        account_map = account_map or {}
        added = []
        modified = []
        removed_ids = []

        has_more = True
        while has_more:
            request = TransactionsSyncRequest(
                access_token=access_token,
                cursor=cursor if cursor else None,
            )
            response = self._client.transactions_sync(request)

            for txn in response["added"]:
                added.append(self._map_transaction(txn, institution_name, account_map))
            for txn in response["modified"]:
                modified.append(self._map_transaction(txn, institution_name, account_map))
            for txn in response["removed"]:
                tid = txn.get("transaction_id", "")
                if tid:
                    removed_ids.append(tid)

            has_more = response["has_more"]
            cursor = response["next_cursor"]

        logger.info(
            "Transaction sync: +%d modified=%d removed=%d",
            len(added), len(modified), len(removed_ids),
        )
        return added, modified, removed_ids, cursor

    def refresh_balances(self, access_token: str, item_id: str = "",
                         institution_name: str = "") -> list[PlaidAccount]:
        """Refresh account balances (calls accounts/get)."""
        return self.get_accounts(access_token, institution_name, item_id)

    def _map_transaction(
        self, txn: dict, institution_name: str, account_map: dict[str, str],
    ) -> Transaction:
        """Convert a Plaid transaction dict to our Transaction dataclass."""
        acct_id = txn.get("account_id", "")
        category_list = txn.get("category") or []
        category_str = " > ".join(category_list) if isinstance(category_list, list) else str(category_list)

        txn_date = txn.get("date")
        auth_date = txn.get("authorized_date")

        return Transaction(
            transaction_id=txn.get("transaction_id", ""),
            account_id=acct_id,
            account_name=account_map.get(acct_id, ""),
            date=txn_date,
            authorized_date=auth_date,
            amount=float(txn.get("amount", 0)),
            merchant_name=txn.get("merchant_name", "") or "",
            name=txn.get("name", "") or "",
            category=category_str,
            status=TransactionStatus.PENDING if txn.get("pending") else TransactionStatus.POSTED,
            pending=bool(txn.get("pending", False)),
            institution_name=institution_name,
            source=TransactionSource.PLAID,
            iso_currency_code=txn.get("iso_currency_code", "USD") or "USD",
            metadata={},
        )
