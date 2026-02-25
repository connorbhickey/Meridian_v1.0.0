"""Tests for Fidelity transaction scraping (get_activity) with mocked Playwright."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from portopt.data.importers.fidelity_auto import FidelityAutoImporter
from portopt.data.models import Transaction, TransactionSource


@pytest.fixture
def state_dir(tmp_path):
    d = tmp_path / "fidelity_state"
    d.mkdir()
    return d


@pytest.fixture
def importer(state_dir):
    return FidelityAutoImporter(state_dir=state_dir)


def _setup_connected(importer):
    """Configure importer as if logged in with a mock page."""
    mock_fidelity = MagicMock()
    mock_page = MagicMock()
    mock_fidelity.page = mock_page
    mock_page.goto.return_value = None
    mock_page.wait_for_load_state.return_value = None
    mock_page.wait_for_timeout.return_value = None
    mock_page.on.return_value = None
    mock_page.remove_listener.return_value = None
    importer._fidelity = mock_fidelity
    importer._logged_in = True
    return mock_page


# ── get_activity tests ───────────────────────────────────────────────

class TestGetActivity:
    def test_raises_when_not_connected(self, importer):
        """get_activity should raise RuntimeError if not logged in."""
        with pytest.raises(RuntimeError, match="Not logged in"):
            importer.get_activity(days=30)

    def test_returns_list_from_dom_scraping(self, importer):
        """When connected and DOM has data, should return Transaction objects."""
        mock_page = _setup_connected(importer)

        # _parse_captured_activity returns [] (no XHR data captured)
        # _scrape_activity_dom returns transactions from DOM
        dom_txns = [
            Transaction(
                transaction_id="fid_1",
                account_id="Z12345678",
                account_name="Z12345678",
                date=date(2025, 3, 15),
                amount=1234.56,
                name="YOU BOUGHT APPLE INC (AAPL)",
                source=TransactionSource.FIDELITY,
            ),
            Transaction(
                transaction_id="fid_2",
                account_id="Z12345678",
                account_name="Z12345678",
                date=date(2025, 3, 14),
                amount=-3.45,
                name="DIVIDEND RECEIVED APPLE INC (AAPL)",
                source=TransactionSource.FIDELITY,
            ),
        ]

        with patch.object(importer, "_parse_captured_activity", return_value=[]), \
             patch.object(importer, "_scrape_activity_dom", return_value=dom_txns):
            result = importer.get_activity(days=30)

        assert len(result) == 2
        for txn in result:
            assert isinstance(txn, Transaction)
            assert txn.source == TransactionSource.FIDELITY

    def test_prefers_xhr_over_dom(self, importer):
        """If XHR data is captured, DOM scraping should not be used."""
        mock_page = _setup_connected(importer)

        xhr_txns = [
            Transaction(
                transaction_id="xhr_1",
                account_id="Z12345678",
                date=date(2025, 3, 15),
                amount=500.00,
                source=TransactionSource.FIDELITY,
            ),
        ]

        with patch.object(importer, "_parse_captured_activity", return_value=xhr_txns) as mock_parse, \
             patch.object(importer, "_scrape_activity_dom") as mock_dom:
            result = importer.get_activity(days=30)

        assert len(result) == 1
        assert result[0].transaction_id == "xhr_1"
        mock_dom.assert_not_called()

    def test_all_transactions_have_fidelity_source(self, importer):
        """Every transaction from get_activity must have source=FIDELITY."""
        mock_page = _setup_connected(importer)

        txns = [
            Transaction(
                transaction_id=f"t{i}",
                account_id="Z99999999",
                source=TransactionSource.FIDELITY,
            )
            for i in range(5)
        ]

        with patch.object(importer, "_parse_captured_activity", return_value=txns):
            result = importer.get_activity(days=30)

        assert all(t.source == TransactionSource.FIDELITY for t in result)

    def test_graceful_failure_on_error(self, importer):
        """get_activity should never crash — return [] on any error."""
        mock_page = _setup_connected(importer)
        mock_page.goto.side_effect = Exception("Network error")

        result = importer.get_activity(days=30)
        assert result == []

    def test_empty_activity_page(self, importer):
        """get_activity should return [] when no data is captured or scraped."""
        mock_page = _setup_connected(importer)

        with patch.object(importer, "_parse_captured_activity", return_value=[]), \
             patch.object(importer, "_scrape_activity_dom", return_value=[]):
            result = importer.get_activity(days=30)

        assert result == []

    def test_registers_and_removes_response_listener(self, importer):
        """get_activity should register page.on('response') then remove it."""
        mock_page = _setup_connected(importer)

        with patch.object(importer, "_parse_captured_activity", return_value=[]), \
             patch.object(importer, "_scrape_activity_dom", return_value=[]):
            importer.get_activity(days=30)

        mock_page.on.assert_called_once()
        assert mock_page.on.call_args[0][0] == "response"
        mock_page.remove_listener.assert_called_once()
        assert mock_page.remove_listener.call_args[0][0] == "response"


# ── _parse_captured_activity tests ──────────────────────────────────

class TestParseCapturedActivity:
    def test_empty_data(self, importer):
        result = importer._parse_captured_activity([])
        assert result == []

    def test_parses_json_list(self, importer):
        """Should parse a JSON list of activity items."""
        captured = [
            ("json", "https://fidelity.com/api/activity", [
                {
                    "activityId": "act_1",
                    "date": "2025-03-15",
                    "description": "Buy AAPL",
                    "amount": "$1,234.56",
                    "type": "Buy",
                    "accountNumber": "Z12345",
                },
            ]),
        ]
        result = importer._parse_captured_activity(captured)
        assert len(result) >= 1
        assert all(isinstance(t, Transaction) for t in result)

    def test_parses_nested_activities_key(self, importer):
        """Should find transactions under body['activities'] key."""
        captured = [
            ("json", "https://fidelity.com/api/data", {
                "activities": [
                    {
                        "activityId": "act_2",
                        "date": "2025-03-10",
                        "description": "Sell MSFT",
                        "amount": "-$500.00",
                    },
                ],
            }),
        ]
        result = importer._parse_captured_activity(captured)
        assert len(result) >= 1

    def test_skips_non_json(self, importer):
        """Should skip non-JSON captured data."""
        captured = [("html", "https://fidelity.com/page", "<html></html>")]
        result = importer._parse_captured_activity(captured)
        assert result == []


# ── _scrape_activity_dom tests ──────────────────────────────────────

class TestScrapeActivityDom:
    def test_returns_transactions_from_table(self, importer):
        """Should parse DOM rows into Transaction objects."""
        mock_page = MagicMock()
        mock_page.evaluate.return_value = [
            {"index": 0, "cells": ["03/15/2025", "Buy", "AAPL", "$1,234.56", "Settled"], "html": ""},
            {"index": 1, "cells": ["03/14/2025", "Dividend", "AAPL", "$3.45", "Settled"], "html": ""},
        ]
        result = importer._scrape_activity_dom(mock_page)
        assert isinstance(result, list)
        for txn in result:
            assert isinstance(txn, Transaction)

    def test_empty_dom(self, importer):
        """Should return [] when no rows found."""
        mock_page = MagicMock()
        mock_page.evaluate.return_value = []
        result = importer._scrape_activity_dom(mock_page)
        assert result == []

    def test_dom_evaluation_error(self, importer):
        """Should return [] when page.evaluate raises."""
        mock_page = MagicMock()
        mock_page.evaluate.side_effect = Exception("Page not loaded")
        result = importer._scrape_activity_dom(mock_page)
        assert result == []
