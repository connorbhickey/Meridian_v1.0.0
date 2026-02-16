"""Tests for Fidelity controller and login dialog."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QTimer

from portopt.data.models import (
    AccountSummary, Asset, AssetType, Holding, Portfolio,
)


# ── Login Dialog Tests ────────────────────────────────────────────────

class TestFidelityLoginDialog:
    """Test the login dialog UI behavior."""

    def test_dialog_creates(self, qtbot):
        from portopt.gui.dialogs.fidelity_login_dialog import FidelityLoginDialog
        dialog = FidelityLoginDialog()
        qtbot.addWidget(dialog)
        # Default page is credentials (page 1)
        assert dialog._stack.currentIndex() == 1

    def test_dialog_playwright_setup_page(self, qtbot):
        from portopt.gui.dialogs.fidelity_login_dialog import FidelityLoginDialog
        dialog = FidelityLoginDialog(show_playwright_setup=True)
        qtbot.addWidget(dialog)
        # Should show playwright setup page
        assert dialog._stack.currentIndex() == 0

    def test_prefill_credentials(self, qtbot):
        from portopt.gui.dialogs.fidelity_login_dialog import FidelityLoginDialog
        dialog = FidelityLoginDialog()
        qtbot.addWidget(dialog)

        dialog.prefill_credentials("testuser", "testpass", "totpsecret")
        assert dialog._username_input.text() == "testuser"
        assert dialog._password_input.text() == "testpass"
        assert dialog._totp_input.text() == "totpsecret"

    def test_show_2fa(self, qtbot):
        from portopt.gui.dialogs.fidelity_login_dialog import FidelityLoginDialog, _Page
        dialog = FidelityLoginDialog()
        qtbot.addWidget(dialog)

        dialog.show_2fa()
        assert dialog._stack.currentIndex() == _Page.TWOFA

    def test_show_progress(self, qtbot):
        from portopt.gui.dialogs.fidelity_login_dialog import FidelityLoginDialog, _Page
        dialog = FidelityLoginDialog()
        qtbot.addWidget(dialog)

        dialog.show_progress("Testing...")
        assert dialog._stack.currentIndex() == _Page.PROGRESS
        assert dialog._progress_label.text() == "Testing..."

    def test_show_success(self, qtbot):
        from portopt.gui.dialogs.fidelity_login_dialog import FidelityLoginDialog, _Page
        dialog = FidelityLoginDialog()
        qtbot.addWidget(dialog)

        dialog.show_success("5 positions loaded")
        assert dialog._stack.currentIndex() == _Page.RESULT
        assert "Connected" in dialog._result_label.text()

    def test_show_error(self, qtbot):
        from portopt.gui.dialogs.fidelity_login_dialog import FidelityLoginDialog, _Page
        dialog = FidelityLoginDialog()
        qtbot.addWidget(dialog)

        dialog.show_error("Bad credentials")
        assert dialog._stack.currentIndex() == _Page.RESULT
        assert "Failed" in dialog._result_label.text()

    def test_login_validation(self, qtbot):
        from portopt.gui.dialogs.fidelity_login_dialog import FidelityLoginDialog
        dialog = FidelityLoginDialog()
        qtbot.addWidget(dialog)

        # Empty fields should show error, not emit signal
        signals = []
        dialog.login_requested.connect(lambda *a: signals.append(a))
        dialog._on_login()
        assert len(signals) == 0
        # Label.show() was called but parent dialog isn't visible, so check isHidden()
        assert not dialog._login_error.isHidden()

    def test_login_emits_signal(self, qtbot):
        from portopt.gui.dialogs.fidelity_login_dialog import FidelityLoginDialog
        dialog = FidelityLoginDialog()
        qtbot.addWidget(dialog)

        dialog._username_input.setText("user")
        dialog._password_input.setText("pass")
        dialog._totp_input.setText("totp")

        signals = []
        dialog.login_requested.connect(lambda *a: signals.append(a))
        dialog._on_login()
        assert len(signals) == 1
        assert signals[0] == ("user", "pass", "totp")


# ── Controller Tests ──────────────────────────────────────────────────

def _make_test_portfolio() -> Portfolio:
    """Create a minimal test portfolio."""
    return Portfolio(
        name="Test",
        holdings=[
            Holding(
                asset=Asset(symbol="AAPL", name="Apple"),
                quantity=10,
                cost_basis=1000,
                current_price=150,
                account="Individual",
            ),
            Holding(
                asset=Asset(symbol="MSFT", name="Microsoft"),
                quantity=5,
                cost_basis=1500,
                current_price=300,
                account="Roth IRA",
            ),
        ],
        accounts=[
            AccountSummary(account_id="1234", account_name="Individual", total_value=1500),
            AccountSummary(account_id="5678", account_name="Roth IRA", total_value=1500),
        ],
        last_updated=datetime.now(),
    )


class TestFidelityController:
    """Test the Fidelity controller (QObject, not QWidget)."""

    @patch("portopt.gui.controllers.fidelity_controller.CacheDB")
    @patch("portopt.gui.controllers.fidelity_controller.FidelityAutoImporter")
    def test_controller_creates(self, mock_importer_cls, mock_cache_cls):
        from portopt.gui.controllers.fidelity_controller import FidelityController
        ctrl = FidelityController()
        assert ctrl._refresh_timer is not None
        assert ctrl._refresh_interval_min == 0

    @patch("portopt.gui.controllers.fidelity_controller.CacheDB")
    @patch("portopt.gui.controllers.fidelity_controller.FidelityAutoImporter")
    def test_get_saved_credentials(self, mock_importer_cls, mock_cache_cls):
        from portopt.gui.controllers.fidelity_controller import FidelityController
        ctrl = FidelityController()

        with patch("portopt.gui.controllers.fidelity_controller.get_credential") as mock_get:
            mock_get.side_effect = lambda key: {
                "fidelity_username": "user1",
                "fidelity_password": "pass1",
                "fidelity_totp_secret": "totp1",
            }.get(key)

            u, p, t = ctrl.get_saved_credentials()
            assert u == "user1"
            assert p == "pass1"
            assert t == "totp1"

    @patch("portopt.gui.controllers.fidelity_controller.CacheDB")
    @patch("portopt.gui.controllers.fidelity_controller.FidelityAutoImporter")
    def test_set_refresh_interval(self, mock_importer_cls, mock_cache_cls):
        from portopt.gui.controllers.fidelity_controller import FidelityController
        ctrl = FidelityController()

        # Not connected, timer should not start
        ctrl._importer.is_connected = False
        ctrl.set_refresh_interval(5)
        assert ctrl._refresh_interval_min == 5
        assert not ctrl._refresh_timer.isActive()

    @patch("portopt.gui.controllers.fidelity_controller.CacheDB")
    @patch("portopt.gui.controllers.fidelity_controller.FidelityAutoImporter")
    def test_save_and_load_portfolio_snapshot(self, mock_importer_cls, mock_cache_cls):
        from portopt.gui.controllers.fidelity_controller import FidelityController
        ctrl = FidelityController()

        portfolio = _make_test_portfolio()

        # Mock the cache to capture and return data
        saved_data = {}

        def mock_save(name, data):
            saved_data[name] = data

        def mock_get(name):
            return saved_data.get(name)

        ctrl._cache.save_portfolio_snapshot = mock_save
        ctrl._cache.get_latest_snapshot = mock_get

        # Save
        ctrl._save_portfolio_snapshot(portfolio)
        assert "fidelity" in saved_data
        assert len(saved_data["fidelity"]["holdings"]) == 2

        # Load
        loaded = ctrl.load_cached_portfolio()
        assert loaded is not None
        assert len(loaded.holdings) == 2
        assert loaded.holdings[0].asset.symbol == "AAPL"
        assert loaded.holdings[1].asset.symbol == "MSFT"
        assert loaded.holdings[0].quantity == 10
        assert loaded.holdings[0].current_price == 150

    @patch("portopt.gui.controllers.fidelity_controller.CacheDB")
    @patch("portopt.gui.controllers.fidelity_controller.FidelityAutoImporter")
    def test_load_cached_portfolio_returns_none_when_empty(self, mock_importer_cls, mock_cache_cls):
        from portopt.gui.controllers.fidelity_controller import FidelityController
        ctrl = FidelityController()

        ctrl._cache.get_latest_snapshot = MagicMock(return_value=None)
        result = ctrl.load_cached_portfolio()
        assert result is None

    @patch("portopt.gui.controllers.fidelity_controller.CacheDB")
    @patch("portopt.gui.controllers.fidelity_controller.FidelityAutoImporter")
    def test_playwright_error_emits_signal(self, mock_importer_cls, mock_cache_cls):
        from portopt.gui.controllers.fidelity_controller import FidelityController
        ctrl = FidelityController()

        signals = []
        ctrl.playwright_missing.connect(lambda: signals.append(True))
        ctrl._on_connect_error("PlaywrightNotInstalledError: Firefox not installed")
        assert len(signals) == 1

    @patch("portopt.gui.controllers.fidelity_controller.CacheDB")
    @patch("portopt.gui.controllers.fidelity_controller.FidelityAutoImporter")
    def test_disconnect_stops_timer(self, mock_importer_cls, mock_cache_cls):
        from portopt.gui.controllers.fidelity_controller import FidelityController
        ctrl = FidelityController()

        signals = []
        ctrl.disconnected.connect(lambda: signals.append(True))
        ctrl.disconnect()
        assert len(signals) == 1
        assert not ctrl._refresh_timer.isActive()


# ── Portfolio Panel Tests ─────────────────────────────────────────────

class TestPortfolioPanel:
    """Test the portfolio panel with account filtering."""

    def test_set_portfolio_populates_table(self, qtbot):
        from portopt.gui.panels.portfolio_panel import PortfolioPanel
        panel = PortfolioPanel()
        qtbot.addWidget(panel)

        portfolio = _make_test_portfolio()
        panel.set_portfolio(portfolio)

        assert panel._table.rowCount() == 2
        assert panel._account_filter.count() == 3  # "All Accounts" + 2 accounts

    def test_account_filter(self, qtbot):
        from portopt.gui.panels.portfolio_panel import PortfolioPanel
        panel = PortfolioPanel()
        qtbot.addWidget(panel)

        portfolio = _make_test_portfolio()
        panel.set_portfolio(portfolio)

        # Filter to one account
        panel._account_filter.setCurrentText("Individual")
        assert panel._table.rowCount() == 1
        assert panel._table.item(0, 0).text() == "AAPL"

        # Filter to other account
        panel._account_filter.setCurrentText("Roth IRA")
        assert panel._table.rowCount() == 1
        assert panel._table.item(0, 0).text() == "MSFT"

        # Reset to all
        panel._account_filter.setCurrentText("All Accounts")
        assert panel._table.rowCount() == 2

    def test_sector_summary(self, qtbot):
        from portopt.gui.panels.portfolio_panel import PortfolioPanel
        panel = PortfolioPanel()
        qtbot.addWidget(panel)

        portfolio = _make_test_portfolio()
        # Set sectors on assets
        portfolio.holdings[0].asset.sector = "Technology"
        portfolio.holdings[1].asset.sector = "Technology"

        panel.set_portfolio(portfolio)
        assert "Technology" in panel._sector_label.text()
