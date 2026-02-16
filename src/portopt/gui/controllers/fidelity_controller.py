"""Controller managing the Fidelity connection lifecycle."""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal

from portopt.data.importers.fidelity_auto import FidelityAutoImporter
from portopt.data.models import Portfolio
from portopt.utils.credentials import (
    FIDELITY_USERNAME, FIDELITY_PASSWORD, FIDELITY_TOTP_SECRET,
    get_credential, store_credential, delete_credential,
)
from portopt.utils.threading import run_in_thread

logger = logging.getLogger(__name__)


class FidelityController(QObject):
    """Manages Fidelity connection: login, 2FA, session persistence, position refresh."""

    connected = Signal(Portfolio)     # Emitted when positions are loaded
    disconnected = Signal()
    connection_error = Signal(str)
    needs_2fa = Signal()             # Emitted when 2FA code is required
    status_changed = Signal(str)     # Status text updates

    def __init__(self, parent=None):
        super().__init__(parent)
        self._importer = FidelityAutoImporter()
        self._worker = None  # Keep reference to prevent GC

    @property
    def is_connected(self) -> bool:
        return self._importer.is_connected

    @property
    def has_saved_session(self) -> bool:
        return self._importer.has_saved_session

    @property
    def has_saved_credentials(self) -> bool:
        return get_credential(FIDELITY_USERNAME) is not None

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
        self.connected.emit(portfolio)

    def _on_connect_error(self, error_msg: str):
        logger.error("Fidelity connection error: %s", error_msg)
        self.status_changed.emit("Connection error")
        self.connection_error.emit(error_msg)

    def disconnect(self):
        """Disconnect and close browser."""
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
        try:
            self._importer.close()
        except Exception:
            pass
