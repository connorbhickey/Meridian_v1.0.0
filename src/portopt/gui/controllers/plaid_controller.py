"""Controller managing Plaid account linking and transaction sync."""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime

from PySide6.QtCore import QObject, QTimer, Signal

from portopt.data.cache import CacheDB
from portopt.data.models import PlaidAccount, PlaidItem, Transaction
from portopt.data.providers.plaid_client import PlaidClient
from portopt.utils.credentials import (
    PLAID_CLIENT_ID, PLAID_ENVIRONMENT, PLAID_SECRET,
    delete_credential, get_credential, plaid_access_token_key,
    store_credential,
)
from portopt.utils.threading import run_in_thread

logger = logging.getLogger(__name__)


class PlaidController(QObject):
    """Manages Plaid connection lifecycle: link, sync, balances."""

    # Signals
    link_token_ready = Signal(str)        # link_token for Plaid Link UI
    account_linked = Signal(str)          # institution_name
    accounts_updated = Signal(list)       # list[PlaidAccount]
    balances_updated = Signal(list)       # list[PlaidAccount] with fresh balances
    transactions_synced = Signal(int)     # count of new/updated transactions
    transactions_loaded = Signal(list)    # list[Transaction] for display
    status_changed = Signal(str)
    error = Signal(str)
    item_removed = Signal(str)            # item_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._client: PlaidClient | None = None
        self._cache = CacheDB()
        self._worker = None  # Prevent GC

        # Auto-sync timer
        self._sync_timer = QTimer(self)
        self._sync_timer.timeout.connect(self.sync_all_transactions)
        self._sync_interval_min = 15  # Default 15 minutes

    @property
    def is_configured(self) -> bool:
        """True if Plaid credentials are stored."""
        return (get_credential(PLAID_CLIENT_ID) is not None
                and get_credential(PLAID_SECRET) is not None)

    @property
    def has_linked_items(self) -> bool:
        """True if at least one Plaid Item exists in the cache."""
        return len(self._cache.get_plaid_items()) > 0

    def configure(self, client_id: str, secret: str, environment: str = "development"):
        """Store Plaid credentials and initialize the client."""
        store_credential(PLAID_CLIENT_ID, client_id)
        store_credential(PLAID_SECRET, secret)
        store_credential(PLAID_ENVIRONMENT, environment)
        self._init_client()
        self.status_changed.emit("Plaid configured")

    def _init_client(self) -> bool:
        """Initialize PlaidClient from stored credentials. Returns True on success."""
        client_id = get_credential(PLAID_CLIENT_ID)
        secret = get_credential(PLAID_SECRET)
        if not client_id or not secret:
            return False
        environment = get_credential(PLAID_ENVIRONMENT) or "development"
        self._client = PlaidClient(client_id, secret, environment)
        return True

    def _ensure_client(self) -> bool:
        """Ensure client is initialized."""
        if self._client is not None:
            return True
        return self._init_client()

    # ── Link flow ─────────────────────────────────────────────────────

    def request_link_token(self):
        """Request a link token for Plaid Link UI (runs in background)."""
        if not self._ensure_client():
            self.error.emit("Plaid not configured. Add credentials in API Key dialog.")
            return
        self.status_changed.emit("Creating link token...")
        self._worker = run_in_thread(
            self._client.create_link_token,
            on_result=self._on_link_token,
            on_error=self._on_error,
        )

    def _on_link_token(self, token: str):
        self.status_changed.emit("Link token ready")
        self.link_token_ready.emit(token)

    def complete_link(self, public_token: str, institution_name: str = ""):
        """Exchange public token after user completes Plaid Link."""
        if not self._ensure_client():
            self.error.emit("Plaid client not initialized")
            return
        self.status_changed.emit(f"Linking {institution_name or 'institution'}...")
        self._worker = run_in_thread(
            self._do_complete_link, public_token, institution_name,
            on_result=self._on_link_complete,
            on_error=self._on_error,
        )

    def _do_complete_link(self, public_token: str, institution_name: str):
        """Background: exchange token, fetch accounts, store everything."""
        access_token, item_id = self._client.exchange_public_token(public_token)

        # Store access token securely in keyring
        store_credential(plaid_access_token_key(item_id), access_token)

        # Store the Item in cache
        self._cache.upsert_plaid_item(
            item_id=item_id,
            institution_name=institution_name,
        )

        # Fetch and store accounts
        accounts = self._client.get_accounts(access_token, institution_name, item_id)
        self._cache.upsert_plaid_accounts([
            self._account_to_dict(a) for a in accounts
        ])

        return item_id, institution_name, accounts

    def _on_link_complete(self, result):
        item_id, institution_name, accounts = result
        self.status_changed.emit(f"Linked {institution_name} ({len(accounts)} accounts)")
        self.account_linked.emit(institution_name)
        self.accounts_updated.emit(accounts)

        # Auto-start sync for the new item
        self._sync_item_transactions(item_id)

    # ── Transaction sync ──────────────────────────────────────────────

    def sync_all_transactions(self):
        """Sync transactions for all linked items (runs in background)."""
        if not self._ensure_client():
            self.error.emit("Plaid not configured")
            return
        items = self._cache.get_plaid_items()
        if not items:
            self.status_changed.emit("No linked accounts")
            return
        self.status_changed.emit("Syncing transactions...")
        self._worker = run_in_thread(
            self._do_sync_all, items,
            on_result=self._on_sync_complete,
            on_error=self._on_error,
        )

    def _do_sync_all(self, items: list[dict]) -> int:
        """Background: sync transactions for all items."""
        total_new = 0
        for item in items:
            item_id = item["item_id"]
            try:
                total_new += self._sync_single_item(item_id, item.get("sync_cursor", ""))
            except Exception as exc:
                logger.error("Failed to sync item %s: %s", item_id, exc)
                self._cache.upsert_plaid_item(
                    item_id=item_id,
                    institution_name=item.get("institution_name", ""),
                    sync_cursor=item.get("sync_cursor", ""),
                    error=str(exc),
                )
        return total_new

    def _sync_item_transactions(self, item_id: str):
        """Sync transactions for a single item (runs in background)."""
        if not self._ensure_client():
            return
        items = self._cache.get_plaid_items()
        item = next((i for i in items if i["item_id"] == item_id), None)
        cursor = item.get("sync_cursor", "") if item else ""
        self.status_changed.emit("Syncing transactions...")
        self._worker = run_in_thread(
            self._sync_single_item, item_id, cursor,
            on_result=self._on_sync_complete,
            on_error=self._on_error,
        )

    def _sync_single_item(self, item_id: str, cursor: str) -> int:
        """Background: sync one item. Returns count of added+modified."""
        access_token = get_credential(plaid_access_token_key(item_id))
        if not access_token:
            logger.warning("No access token for item %s", item_id)
            return 0

        # Build account_id -> name map for labeling
        cached_accounts = self._cache.get_plaid_accounts(item_id)
        account_map = {a["account_id"]: a.get("name", "") for a in cached_accounts}
        institution_name = cached_accounts[0].get("institution_name", "") if cached_accounts else ""

        added, modified, removed_ids, next_cursor = self._client.sync_transactions(
            access_token, cursor, institution_name, account_map,
        )

        # Persist to cache
        if added or modified:
            all_txns = added + modified
            self._cache.upsert_transactions([
                self._transaction_to_dict(t) for t in all_txns
            ])
        if removed_ids:
            self._cache.remove_transactions(removed_ids)

        # Update cursor
        self._cache.update_plaid_sync_cursor(item_id, next_cursor)

        return len(added) + len(modified)

    def _on_sync_complete(self, count: int):
        self.status_changed.emit(f"Synced {count} transactions")
        self.transactions_synced.emit(count)

    # ── Balance refresh ───────────────────────────────────────────────

    def refresh_balances(self):
        """Refresh balances for all linked accounts."""
        if not self._ensure_client():
            self.error.emit("Plaid not configured")
            return
        items = self._cache.get_plaid_items()
        if not items:
            self.status_changed.emit("No linked accounts")
            return
        self.status_changed.emit("Refreshing balances...")
        self._worker = run_in_thread(
            self._do_refresh_balances, items,
            on_result=self._on_balances_refreshed,
            on_error=self._on_error,
        )

    def _do_refresh_balances(self, items: list[dict]) -> list[PlaidAccount]:
        """Background: refresh balances for all items."""
        all_accounts = []
        for item in items:
            item_id = item["item_id"]
            access_token = get_credential(plaid_access_token_key(item_id))
            if not access_token:
                continue
            try:
                accounts = self._client.refresh_balances(
                    access_token, item_id, item.get("institution_name", ""),
                )
                # Update cache
                self._cache.upsert_plaid_accounts([
                    self._account_to_dict(a) for a in accounts
                ])
                all_accounts.extend(accounts)
            except Exception as exc:
                logger.error("Failed to refresh balances for item %s: %s", item_id, exc)
        return all_accounts

    def _on_balances_refreshed(self, accounts: list[PlaidAccount]):
        self.status_changed.emit(f"Refreshed {len(accounts)} accounts")
        self.balances_updated.emit(accounts)

    # ── Load transactions from cache ──────────────────────────────────

    def load_transactions(self, account_id: str | None = None,
                          start=None, end=None, status: str | None = None,
                          limit: int = 500, offset: int = 0):
        """Load Plaid transactions from cache for display."""
        self._worker = run_in_thread(
            self._cache.get_transactions,
            account_id=account_id, start=start, end=end,
            status=status, source="PLAID", limit=limit, offset=offset,
            on_result=self._on_transactions_loaded,
            on_error=self._on_error,
        )

    def _on_transactions_loaded(self, rows: list[dict]):
        txns = [self._dict_to_transaction(r) for r in rows]
        self.transactions_loaded.emit(txns)

    # ── Load accounts from cache ──────────────────────────────────────

    def load_accounts(self):
        """Load all Plaid accounts from cache."""
        rows = self._cache.get_plaid_accounts()
        accounts = [self._dict_to_plaid_account(r) for r in rows]
        self.accounts_updated.emit(accounts)

    # ── Remove item ───────────────────────────────────────────────────

    def remove_item(self, item_id: str):
        """Remove a linked institution (delete from cache + keyring)."""
        self.status_changed.emit("Removing linked account...")
        # Delete access token from keyring
        delete_credential(plaid_access_token_key(item_id))
        # Delete from cache (cascades to accounts + transactions)
        self._cache.delete_plaid_item(item_id)
        self.status_changed.emit("Account removed")
        self.item_removed.emit(item_id)

    # ── Auto-sync ─────────────────────────────────────────────────────

    def set_sync_interval(self, minutes: int):
        """Set the auto-sync interval. 0 to disable."""
        self._sync_interval_min = minutes
        self._sync_timer.stop()
        if minutes > 0 and self.has_linked_items:
            self._sync_timer.start(minutes * 60 * 1000)
            logger.info("Plaid auto-sync set to %d min", minutes)
        else:
            logger.info("Plaid auto-sync disabled")

    def start_auto_sync(self):
        """Start auto-sync if configured and has linked items."""
        if not self.is_configured or not self.has_linked_items:
            return
        if not self._ensure_client():
            return
        # Initial sync
        self.sync_all_transactions()
        self.refresh_balances()
        # Start timer
        if self._sync_interval_min > 0:
            self._sync_timer.start(self._sync_interval_min * 60 * 1000)

    # ── Error handling ────────────────────────────────────────────────

    def _on_error(self, error_msg: str):
        logger.error("Plaid error: %s", error_msg)
        self.status_changed.emit("Error")
        self.error.emit(error_msg)

    # ── Cleanup ───────────────────────────────────────────────────────

    def close(self):
        """Clean up resources."""
        self._sync_timer.stop()

    # ── Serialization helpers ─────────────────────────────────────────

    @staticmethod
    def _account_to_dict(acct: PlaidAccount) -> dict:
        """Convert PlaidAccount dataclass to dict for cache storage."""
        return {
            "account_id": acct.account_id,
            "item_id": acct.item_id,
            "institution_name": acct.institution_name,
            "name": acct.name,
            "official_name": acct.official_name,
            "account_type": acct.account_type.name,
            "subtype": acct.subtype,
            "mask": acct.mask,
            "current_balance": acct.current_balance,
            "available_balance": acct.available_balance,
            "credit_limit": acct.limit,
            "last_synced": acct.last_synced.isoformat() if acct.last_synced else None,
        }

    @staticmethod
    def _transaction_to_dict(txn: Transaction) -> dict:
        """Convert Transaction dataclass to dict for cache storage."""
        return {
            "transaction_id": txn.transaction_id,
            "account_id": txn.account_id,
            "account_name": txn.account_name,
            "date": txn.date.isoformat() if txn.date else "",
            "authorized_date": txn.authorized_date.isoformat() if txn.authorized_date else "",
            "amount": txn.amount,
            "merchant_name": txn.merchant_name,
            "name": txn.name,
            "category": txn.category,
            "status": txn.status.name,
            "pending": txn.pending,
            "institution_name": txn.institution_name,
            "source": txn.source.name,
            "iso_currency_code": txn.iso_currency_code,
            "metadata": txn.metadata,
        }

    @staticmethod
    def _dict_to_transaction(row: dict) -> Transaction:
        """Convert a cache dict row to a Transaction dataclass."""
        from portopt.data.models import TransactionSource, TransactionStatus

        txn_date = None
        if row.get("date"):
            try:
                txn_date = datetime.fromisoformat(row["date"]).date()
            except (ValueError, TypeError):
                pass

        auth_date = None
        if row.get("authorized_date"):
            try:
                auth_date = datetime.fromisoformat(row["authorized_date"]).date()
            except (ValueError, TypeError):
                pass

        status = TransactionStatus.POSTED
        status_str = row.get("status", "POSTED")
        try:
            status = TransactionStatus[status_str]
        except (KeyError, TypeError):
            pass

        source = TransactionSource.PLAID
        source_str = row.get("source", "PLAID")
        try:
            source = TransactionSource[source_str]
        except (KeyError, TypeError):
            pass

        metadata = row.get("metadata", {})
        if isinstance(metadata, str):
            import json
            try:
                metadata = json.loads(metadata)
            except (json.JSONDecodeError, TypeError):
                metadata = {}

        return Transaction(
            transaction_id=row.get("transaction_id", ""),
            account_id=row.get("account_id", ""),
            account_name=row.get("account_name", ""),
            date=txn_date,
            authorized_date=auth_date,
            amount=float(row.get("amount", 0)),
            merchant_name=row.get("merchant_name", ""),
            name=row.get("name", ""),
            category=row.get("category", ""),
            status=status,
            pending=bool(row.get("pending", False)),
            institution_name=row.get("institution_name", ""),
            source=source,
            iso_currency_code=row.get("iso_currency_code", "USD"),
            metadata=metadata,
        )

    @staticmethod
    def _dict_to_plaid_account(row: dict) -> PlaidAccount:
        """Convert a cache dict row to a PlaidAccount dataclass."""
        from portopt.data.models import PlaidAccountType

        acct_type = PlaidAccountType.OTHER
        type_str = row.get("account_type", "OTHER")
        try:
            acct_type = PlaidAccountType[type_str]
        except (KeyError, TypeError):
            pass

        last_synced = None
        if row.get("last_synced"):
            try:
                last_synced = datetime.fromisoformat(row["last_synced"])
            except (ValueError, TypeError):
                pass

        return PlaidAccount(
            account_id=row.get("account_id", ""),
            item_id=row.get("item_id", ""),
            institution_name=row.get("institution_name", ""),
            name=row.get("name", ""),
            official_name=row.get("official_name", ""),
            account_type=acct_type,
            subtype=row.get("subtype", ""),
            mask=row.get("mask", ""),
            current_balance=float(row.get("current_balance", 0)),
            available_balance=float(row["available_balance"]) if row.get("available_balance") is not None else None,
            limit=float(row["credit_limit"]) if row.get("credit_limit") is not None else None,
            last_synced=last_synced,
        )
