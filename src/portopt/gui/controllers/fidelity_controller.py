"""Controller managing the Fidelity connection lifecycle."""

from __future__ import annotations

import json
import logging
from datetime import datetime

from PySide6.QtCore import QObject, QTimer, Signal

from portopt.data.cache import CacheDB
from portopt.data.importers.fidelity_auto import (
    FidelityAutoImporter, PlaywrightNotInstalledError,
)
from portopt.data.models import (
    AccountSummary, Asset, AssetType, Holding, Portfolio,
)
from portopt.utils.credentials import (
    FIDELITY_USERNAME, FIDELITY_PASSWORD, FIDELITY_TOTP_SECRET,
    get_credential, store_credential, delete_credential,
)
from portopt.utils.threading import run_in_thread

logger = logging.getLogger(__name__)

_SNAPSHOT_NAME = "fidelity"


class FidelityController(QObject):
    """Manages Fidelity connection: login, 2FA, session persistence, position refresh."""

    connected = Signal(Portfolio)     # Emitted when positions are loaded
    disconnected = Signal()
    connection_error = Signal(str)
    needs_2fa = Signal()             # Emitted when 2FA code is required
    status_changed = Signal(str)     # Status text updates
    playwright_missing = Signal()    # Emitted when Playwright Firefox is not installed

    def __init__(self, parent=None):
        super().__init__(parent)
        self._importer = FidelityAutoImporter()
        self._cache = CacheDB()
        self._worker = None  # Keep reference to prevent GC

        # Auto-refresh timer
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self.refresh_positions)
        self._refresh_interval_min = 0  # 0 = disabled

    @property
    def is_connected(self) -> bool:
        return self._importer.is_connected

    @property
    def has_saved_session(self) -> bool:
        return self._importer.has_saved_session

    @property
    def has_saved_credentials(self) -> bool:
        return get_credential(FIDELITY_USERNAME) is not None

    def get_saved_credentials(self) -> tuple[str, str, str]:
        """Return (username, password, totp_secret) from keyring for dialog pre-fill."""
        return (
            get_credential(FIDELITY_USERNAME) or "",
            get_credential(FIDELITY_PASSWORD) or "",
            get_credential(FIDELITY_TOTP_SECRET) or "",
        )

    def set_refresh_interval(self, minutes: int):
        """Set the auto-refresh interval. 0 to disable."""
        self._refresh_interval_min = minutes
        self._refresh_timer.stop()
        if minutes > 0 and self._importer.is_connected:
            self._refresh_timer.start(minutes * 60 * 1000)
            logger.info("Fidelity auto-refresh set to %d min", minutes)
        else:
            logger.info("Fidelity auto-refresh disabled")

    def try_auto_connect(self):
        """Attempt to connect using saved session (called on app startup)."""
        if not self._importer.has_saved_session:
            self.status_changed.emit("No saved session")
            return

        self.status_changed.emit("Restoring session...")
        self._worker = run_in_thread(
            self._importer.connect_with_saved_session,
            on_result=self._on_auto_connect_result,
            on_error=self._on_connect_error,
        )

    def _on_auto_connect_result(self, success):
        if success:
            self.status_changed.emit("Session restored, fetching positions...")
            self._fetch_positions()
        else:
            self.status_changed.emit("Session expired")
            self.disconnected.emit()

    def login(self, username: str, password: str, totp_secret: str = "", save_credentials: bool = True):
        """Start the login process."""
        self.status_changed.emit("Logging in...")

        if save_credentials:
            store_credential(FIDELITY_USERNAME, username)
            store_credential(FIDELITY_PASSWORD, password)
            if totp_secret:
                store_credential(FIDELITY_TOTP_SECRET, totp_secret)

        self._worker = run_in_thread(
            self._importer.login,
            username, password, totp_secret or None,
            on_result=self._on_login_result,
            on_error=self._on_connect_error,
        )

    def _on_login_result(self, result):
        success, needs_2fa = result
        if success and not needs_2fa:
            self.status_changed.emit("Logged in, fetching positions...")
            self._fetch_positions()
        elif success and needs_2fa:
            self.status_changed.emit("2FA required")
            self.needs_2fa.emit()
        else:
            self.status_changed.emit("Login failed")
            self.connection_error.emit("Login failed. Check your credentials.")

    def submit_2fa(self, code: str):
        """Submit 2FA verification code."""
        self.status_changed.emit("Verifying 2FA...")
        self._worker = run_in_thread(
            self._importer.complete_2fa,
            code, True,
            on_result=self._on_2fa_result,
            on_error=self._on_connect_error,
        )

    def _on_2fa_result(self, success):
        if success:
            self.status_changed.emit("2FA verified, fetching positions...")
            self._fetch_positions()
        else:
            self.status_changed.emit("2FA failed")
            self.connection_error.emit("Invalid verification code. Try again.")

    def refresh_positions(self):
        """Refresh positions from Fidelity."""
        if not self._importer.is_connected:
            self.connection_error.emit("Not connected to Fidelity")
            return
        self.status_changed.emit("Refreshing positions...")
        self._fetch_positions()

    def _fetch_positions(self):
        """Fetch positions in a background thread."""
        self._worker = run_in_thread(
            self._importer.get_positions,
            on_result=self._on_positions_loaded,
            on_error=self._on_connect_error,
        )

    def _on_positions_loaded(self, portfolio: Portfolio):
        self.status_changed.emit(f"Connected ({len(portfolio.holdings)} positions)")

        # Save snapshot for offline startup
        self._save_portfolio_snapshot(portfolio)

        # Start auto-refresh if configured
        if self._refresh_interval_min > 0 and not self._refresh_timer.isActive():
            self._refresh_timer.start(self._refresh_interval_min * 60 * 1000)

        self.connected.emit(portfolio)

    def _save_portfolio_snapshot(self, portfolio: Portfolio):
        """Serialize portfolio to CacheDB for offline startup."""
        try:
            data = {
                "name": portfolio.name,
                "last_updated": portfolio.last_updated.isoformat() if portfolio.last_updated else None,
                "holdings": [
                    {
                        "symbol": h.asset.symbol,
                        "name": h.asset.name,
                        "asset_type": h.asset.asset_type.name,
                        "quantity": h.quantity,
                        "cost_basis": h.cost_basis,
                        "current_price": h.current_price,
                        "account": h.account,
                    }
                    for h in portfolio.holdings
                ],
                "accounts": [
                    {
                        "account_id": a.account_id,
                        "account_name": a.account_name,
                        "total_value": a.total_value,
                        "holdings_count": a.holdings_count,
                    }
                    for a in portfolio.accounts
                ],
            }
            self._cache.save_portfolio_snapshot(_SNAPSHOT_NAME, data)
            logger.info("Saved Fidelity portfolio snapshot (%d holdings)", len(portfolio.holdings))
        except Exception as e:
            logger.warning("Failed to save portfolio snapshot: %s", e)

    def load_cached_portfolio(self) -> Portfolio | None:
        """Reconstruct Portfolio from cached snapshot for offline startup."""
        try:
            data = self._cache.get_latest_snapshot(_SNAPSHOT_NAME)
            if not data:
                return None

            holdings = []
            for h in data.get("holdings", []):
                asset = Asset(
                    symbol=h["symbol"],
                    name=h.get("name", ""),
                    asset_type=AssetType[h.get("asset_type", "STOCK")],
                )
                holdings.append(Holding(
                    asset=asset,
                    quantity=h["quantity"],
                    cost_basis=h.get("cost_basis", 0.0),
                    current_price=h.get("current_price", 0.0),
                    account=h.get("account", ""),
                ))

            accounts = []
            for a in data.get("accounts", []):
                accounts.append(AccountSummary(
                    account_id=a["account_id"],
                    account_name=a.get("account_name", ""),
                    total_value=a.get("total_value", 0.0),
                    holdings_count=a.get("holdings_count", 0),
                ))

            last_updated = None
            if data.get("last_updated"):
                try:
                    last_updated = datetime.fromisoformat(data["last_updated"])
                except (ValueError, TypeError):
                    pass

            portfolio = Portfolio(
                name=data.get("name", "Fidelity Portfolio (Cached)"),
                holdings=holdings,
                accounts=accounts,
                last_updated=last_updated,
            )
            logger.info("Loaded cached Fidelity portfolio (%d holdings)", len(holdings))
            return portfolio
        except Exception as e:
            logger.warning("Failed to load cached portfolio: %s", e)
            return None

    def _on_connect_error(self, error_msg: str):
        # Check for Playwright not installed
        if "PlaywrightNotInstalledError" in error_msg or "playwright" in error_msg.lower():
            logger.warning("Playwright Firefox not installed")
            self.status_changed.emit("Playwright Firefox not installed")
            self.playwright_missing.emit()
            return

        logger.error("Fidelity connection error: %s", error_msg)
        self.status_changed.emit("Connection error")
        self.connection_error.emit(error_msg)

    def disconnect(self):
        """Disconnect and close browser."""
        self._refresh_timer.stop()
        try:
            self._importer.close()
        except Exception:
            pass
        self.status_changed.emit("Disconnected")
        self.disconnected.emit()

    def clear_credentials(self):
        """Remove saved credentials from keyring."""
        delete_credential(FIDELITY_USERNAME)
        delete_credential(FIDELITY_PASSWORD)
        delete_credential(FIDELITY_TOTP_SECRET)

    def clear_session(self):
        """Remove saved browser session."""
        self._importer.clear_saved_session()

    def close(self):
        """Clean up resources."""
        self._refresh_timer.stop()
        try:
            self._importer.close()
        except Exception:
            pass
