"""Unit tests for FidelityAutoImporter with mocked fidelity-api library."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from portopt.data.importers.fidelity_auto import (
    FidelityAutoImporter,
    PlaywrightNotInstalledError,
    _LIBRARY_STATE_FILE,
)
from portopt.data.models import Portfolio


@pytest.fixture
def state_dir(tmp_path):
    """Use a temporary directory for Fidelity state."""
    d = tmp_path / "fidelity_state"
    d.mkdir()
    return d


@pytest.fixture
def importer(state_dir):
    """Create an importer with a temp state dir."""
    return FidelityAutoImporter(state_dir=state_dir)


# ── Bug Fix 1: profile_path is a directory, not a file ──────────────

@patch("portopt.utils.playwright_check.is_playwright_firefox_installed", return_value=True)
@patch("portopt.data.importers.fidelity_auto.FidelityAutoImporter._check_playwright")
def test_create_automation_passes_directory(mock_pw_check, mock_pw_installed, importer, state_dir):
    """profile_path should be the state DIRECTORY, not a file path."""
    mock_automation_cls = MagicMock()
    with patch("fidelity.fidelity.FidelityAutomation", mock_automation_cls):
        importer._create_automation()
        mock_automation_cls.assert_called_once_with(
            headless=True,
            save_state=True,
            profile_path=str(state_dir),
        )


# ── Bug Fix 2: login() return value mapping ──────────────────────────

@patch("portopt.data.importers.fidelity_auto.FidelityAutoImporter._check_playwright")
def test_login_fully_logged_in(mock_pw_check, importer):
    """Library (True, True) means fully logged in → Meridian (True, False)."""
    mock_fa = MagicMock()
    mock_fa.login.return_value = (True, True)  # Library: success + fully logged in

    with patch("fidelity.fidelity.FidelityAutomation", return_value=mock_fa):
        success, needs_2fa = importer.login("user", "pass")

    assert success is True
    assert needs_2fa is False
    assert importer._logged_in is True


@patch("portopt.data.importers.fidelity_auto.FidelityAutoImporter._check_playwright")
def test_login_needs_2fa(mock_pw_check, importer):
    """Library (True, False) means needs 2FA → Meridian (True, True)."""
    mock_fa = MagicMock()
    mock_fa.login.return_value = (True, False)  # Library: success + needs 2FA

    with patch("fidelity.fidelity.FidelityAutomation", return_value=mock_fa):
        success, needs_2fa = importer.login("user", "pass")

    assert success is True
    assert needs_2fa is True
    assert importer._logged_in is False


@patch("portopt.data.importers.fidelity_auto.FidelityAutoImporter._check_playwright")
def test_login_failed(mock_pw_check, importer):
    """Library (False, False) means login failed → Meridian (False, False)."""
    mock_fa = MagicMock()
    mock_fa.login.return_value = (False, False)

    with patch("fidelity.fidelity.FidelityAutomation", return_value=mock_fa):
        success, needs_2fa = importer.login("user", "wrong")

    assert success is False
    assert needs_2fa is False
    assert importer._logged_in is False


# ── Bug Fix 3: connect_with_saved_session checks return value ────────

@patch("portopt.data.importers.fidelity_auto.FidelityAutoImporter._check_playwright")
def test_saved_session_expired_returns_none(mock_pw_check, importer, state_dir):
    """getAccountInfo() returning None means session expired → False."""
    # Create the state file so has_saved_session is True
    (state_dir / _LIBRARY_STATE_FILE).write_text("{}")

    mock_fa = MagicMock()
    mock_fa.getAccountInfo.return_value = None

    with patch("fidelity.fidelity.FidelityAutomation", return_value=mock_fa):
        result = importer.connect_with_saved_session()

    assert result is False
    assert importer._logged_in is False


@patch("portopt.data.importers.fidelity_auto.FidelityAutoImporter._check_playwright")
def test_saved_session_valid(mock_pw_check, importer, state_dir):
    """getAccountInfo() returning non-None means session is valid → True."""
    (state_dir / _LIBRARY_STATE_FILE).write_text("{}")

    mock_fa = MagicMock()
    mock_fa.getAccountInfo.return_value = {"some": "data"}

    with patch("fidelity.fidelity.FidelityAutomation", return_value=mock_fa):
        result = importer.connect_with_saved_session()

    assert result is True
    assert importer._logged_in is True


# ── Bug Fix 4: has_saved_session / clear_saved_session use correct file ──

def test_has_saved_session_checks_fidelity_json(importer, state_dir):
    """has_saved_session should check for Fidelity.json, not fidelity_state.json."""
    assert importer.has_saved_session is False

    # Create the file the library actually creates
    (state_dir / _LIBRARY_STATE_FILE).write_text("{}")
    assert importer.has_saved_session is True


def test_clear_saved_session_removes_fidelity_json(importer, state_dir):
    """clear_saved_session should delete Fidelity.json."""
    fidelity_file = state_dir / _LIBRARY_STATE_FILE
    fidelity_file.write_text("{}")
    assert fidelity_file.exists()

    importer.clear_saved_session()
    assert not fidelity_file.exists()


# ── Playwright check ─────────────────────────────────────────────────

def test_playwright_not_installed_raises(importer):
    """Should raise PlaywrightNotInstalledError when Firefox is missing."""
    with patch("portopt.utils.playwright_check.is_playwright_firefox_installed", return_value=False):
        with pytest.raises(PlaywrightNotInstalledError):
            importer._check_playwright()


def test_playwright_installed_no_error(importer):
    """Should not raise when Firefox is installed."""
    with patch("portopt.utils.playwright_check.is_playwright_firefox_installed", return_value=True):
        importer._check_playwright()  # should not raise


# ── get_positions parsing ─────────────────────────────────────────────

@patch("portopt.data.importers.fidelity_auto.FidelityAutoImporter._check_playwright")
def test_get_positions_parses_accounts(mock_pw_check, importer):
    """get_positions should parse account_dict into Portfolio."""
    mock_fa = MagicMock()
    mock_fa.account_dict = {
        "1234": {
            "balance": "50000.00",
            "nickname": "Individual",
            "stocks": [
                {"ticker": "AAPL", "name": "Apple Inc", "quantity": "10", "last_price": "150.00", "value": "1500.00", "cost_basis": "1200.00"},
                {"ticker": "MSFT", "name": "Microsoft", "quantity": "5", "last_price": "300.00", "value": "1500.00"},
                {"ticker": "SPAXX**", "name": "Cash", "quantity": "100", "last_price": "1.00", "value": "100.00"},  # Should be skipped
            ],
        },
    }
    mock_fa.getAccountInfo.return_value = True

    importer._fidelity = mock_fa
    importer._logged_in = True

    portfolio = importer.get_positions()
    assert isinstance(portfolio, Portfolio)
    assert len(portfolio.holdings) == 2  # SPAXX** is filtered out
    assert portfolio.holdings[0].asset.symbol == "AAPL"
    assert portfolio.holdings[0].quantity == 10.0
    assert portfolio.holdings[0].cost_basis == 1200.0
    assert portfolio.holdings[1].asset.symbol == "MSFT"
    assert portfolio.holdings[1].cost_basis == 0.0  # No cost_basis key
    assert len(portfolio.accounts) == 1
    assert portfolio.accounts[0].account_id == "1234"


def test_get_positions_not_logged_in_raises(importer):
    """get_positions should raise RuntimeError if not logged in."""
    with pytest.raises(RuntimeError, match="Not logged in"):
        importer.get_positions()
