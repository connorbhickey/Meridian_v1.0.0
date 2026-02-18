"""Tests for the SankeyPanel — rebalance flow visualization.

Requires: pytest-qt (provides qtbot fixture)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")


# ── Fixture ──────────────────────────────────────────────────────────


@pytest.fixture
def panel(qtbot):
    from portopt.gui.panels.sankey_panel import SankeyPanel

    p = SankeyPanel()
    qtbot.addWidget(p)
    return p


# ── Helpers ──────────────────────────────────────────────────────────


def _table_column_values(table, col: int) -> list[str]:
    """Read all values from a table column as strings."""
    return [table.item(row, col).text() for row in range(table.rowCount())]


def _table_row_dict(table, row: int) -> dict[str, str]:
    """Read a single table row as a dict keyed by column header."""
    headers = [
        table.horizontalHeaderItem(c).text() for c in range(table.columnCount())
    ]
    return {
        headers[c]: table.item(row, c).text() for c in range(table.columnCount())
    }


def _find_row_by_symbol(table, symbol: str) -> int | None:
    """Return the row index for a given symbol, or None."""
    for row in range(table.rowCount()):
        if table.item(row, 0).text() == symbol:
            return row
    return None


# ── Panel Initialization ─────────────────────────────────────────────


class TestPanelInitialization:
    """Verify panel identity and initial widget state."""

    def test_panel_id(self, panel):
        assert panel.panel_id == "sankey"

    def test_panel_title(self, panel):
        assert panel.panel_title == "REBALANCE FLOW"

    def test_table_has_five_columns(self, panel):
        assert panel._table.columnCount() == 5

    def test_table_column_headers(self, panel):
        headers = [
            panel._table.horizontalHeaderItem(c).text()
            for c in range(panel._table.columnCount())
        ]
        assert headers == ["Symbol", "Current %", "Target %", "Change %", "Action"]

    def test_table_starts_empty(self, panel):
        assert panel._table.rowCount() == 0

    def test_info_label_default_text(self, panel):
        assert "Load current and target weights" in panel._info_label.text()


# ── set_weights() ────────────────────────────────────────────────────


class TestSetWeights:
    """Test the set_weights() public API."""

    def test_row_count_matches_unique_symbols(self, panel):
        current = {"AAPL": 0.40, "MSFT": 0.30, "GOOG": 0.30}
        target = {"AAPL": 0.25, "MSFT": 0.50, "GOOG": 0.25}
        panel.set_weights(current, target)
        assert panel._table.rowCount() == 3

    def test_row_count_with_new_symbol_in_target(self, panel):
        """A symbol appearing only in target should still get a row."""
        current = {"AAPL": 0.60, "MSFT": 0.40}
        target = {"AAPL": 0.40, "MSFT": 0.30, "GOOG": 0.30}
        panel.set_weights(current, target)
        assert panel._table.rowCount() == 3

    def test_row_count_with_symbol_only_in_current(self, panel):
        """A symbol only in current (sold entirely) should still get a row."""
        current = {"AAPL": 0.50, "MSFT": 0.30, "GOOG": 0.20}
        target = {"AAPL": 0.60, "MSFT": 0.40}
        panel.set_weights(current, target)
        assert panel._table.rowCount() == 3

    def test_current_and_target_values(self, panel):
        current = {"AAPL": 0.60, "MSFT": 0.40}
        target = {"AAPL": 0.50, "MSFT": 0.50}
        panel.set_weights(current, target)

        row_aapl = _find_row_by_symbol(panel._table, "AAPL")
        assert row_aapl is not None
        data = _table_row_dict(panel._table, row_aapl)
        assert data["Current %"] == "60.00%"
        assert data["Target %"] == "50.00%"
        assert data["Change %"] == "-10.00%"

        row_msft = _find_row_by_symbol(panel._table, "MSFT")
        assert row_msft is not None
        data = _table_row_dict(panel._table, row_msft)
        assert data["Current %"] == "40.00%"
        assert data["Target %"] == "50.00%"
        assert data["Change %"] == "+10.00%"

    def test_action_buy(self, panel):
        """Symbol increasing in weight should show BUY action."""
        current = {"AAPL": 0.30}
        target = {"AAPL": 0.70}
        panel.set_weights(current, target)

        row = _find_row_by_symbol(panel._table, "AAPL")
        data = _table_row_dict(panel._table, row)
        assert data["Action"] == "BUY"

    def test_action_sell(self, panel):
        """Symbol decreasing in weight should show SELL action."""
        current = {"AAPL": 0.70}
        target = {"AAPL": 0.30}
        panel.set_weights(current, target)

        row = _find_row_by_symbol(panel._table, "AAPL")
        data = _table_row_dict(panel._table, row)
        assert data["Action"] == "SELL"

    def test_action_hold(self, panel):
        """Symbol with same weight should show HOLD action."""
        current = {"AAPL": 0.50, "MSFT": 0.50}
        target = {"AAPL": 0.50, "MSFT": 0.50}
        panel.set_weights(current, target)

        row = _find_row_by_symbol(panel._table, "AAPL")
        data = _table_row_dict(panel._table, row)
        assert data["Action"] == "HOLD"

    def test_action_new_position(self, panel):
        """Symbol only in target (new position) should show BUY."""
        current = {"AAPL": 1.0}
        target = {"AAPL": 0.70, "GOOG": 0.30}
        panel.set_weights(current, target)

        row = _find_row_by_symbol(panel._table, "GOOG")
        data = _table_row_dict(panel._table, row)
        assert data["Action"] == "BUY"
        assert data["Current %"] == "0.00%"
        assert data["Target %"] == "30.00%"

    def test_action_full_exit(self, panel):
        """Symbol only in current (fully sold) should show SELL."""
        current = {"AAPL": 0.60, "GOOG": 0.40}
        target = {"AAPL": 1.0}
        panel.set_weights(current, target)

        row = _find_row_by_symbol(panel._table, "GOOG")
        data = _table_row_dict(panel._table, row)
        assert data["Action"] == "SELL"
        assert data["Target %"] == "0.00%"

    def test_empty_weights_clears_table(self, panel):
        """Passing empty dicts should result in an empty table."""
        # First populate
        panel.set_weights({"AAPL": 0.5, "MSFT": 0.5}, {"AAPL": 0.6, "MSFT": 0.4})
        assert panel._table.rowCount() == 2

        # Now clear
        panel.set_weights({}, {})
        assert panel._table.rowCount() == 0

    def test_none_weights_treated_as_empty(self, panel):
        """Passing None should be handled gracefully as empty."""
        panel.set_weights(None, None)
        assert panel._table.rowCount() == 0

    def test_table_sorted_by_absolute_change(self, panel):
        """Rows should be sorted by absolute change descending."""
        current = {"AAPL": 0.30, "MSFT": 0.40, "GOOG": 0.30}
        target = {"AAPL": 0.10, "MSFT": 0.45, "GOOG": 0.45}
        panel.set_weights(current, target)

        # AAPL: -20%, GOOG: +15%, MSFT: +5%
        symbols = _table_column_values(panel._table, 0)
        assert symbols[0] == "AAPL"  # |change| = 20%
        assert symbols[1] == "GOOG"  # |change| = 15%
        assert symbols[2] == "MSFT"  # |change| = 5%


# ── set_rebalance_event() ────────────────────────────────────────────


class TestSetRebalanceEvent:
    """Test the set_rebalance_event() convenience method."""

    def test_rebalance_event_populates_table(self, panel):
        event = SimpleNamespace(
            weights_before={"AAPL": 0.50, "MSFT": 0.50},
            weights_after={"AAPL": 0.30, "MSFT": 0.70},
        )
        panel.set_rebalance_event(event)
        assert panel._table.rowCount() == 2

    def test_rebalance_event_values(self, panel):
        event = SimpleNamespace(
            weights_before={"AAPL": 0.40, "MSFT": 0.60},
            weights_after={"AAPL": 0.55, "MSFT": 0.45},
        )
        panel.set_rebalance_event(event)

        row_aapl = _find_row_by_symbol(panel._table, "AAPL")
        data = _table_row_dict(panel._table, row_aapl)
        assert data["Action"] == "BUY"
        assert data["Current %"] == "40.00%"
        assert data["Target %"] == "55.00%"

        row_msft = _find_row_by_symbol(panel._table, "MSFT")
        data = _table_row_dict(panel._table, row_msft)
        assert data["Action"] == "SELL"

    def test_rebalance_event_with_three_symbols(self, panel):
        event = SimpleNamespace(
            weights_before={"AAPL": 0.33, "MSFT": 0.34, "GOOG": 0.33},
            weights_after={"AAPL": 0.50, "MSFT": 0.25, "GOOG": 0.25},
        )
        panel.set_rebalance_event(event)
        assert panel._table.rowCount() == 3


# ── Info Label ───────────────────────────────────────────────────────


class TestInfoLabel:
    """Test that the info label updates with turnover information."""

    def test_info_label_shows_symbol_count(self, panel):
        current = {"AAPL": 0.50, "MSFT": 0.50}
        target = {"AAPL": 0.60, "MSFT": 0.40}
        panel.set_weights(current, target)
        assert "2 symbols" in panel._info_label.text()

    def test_info_label_shows_turnover(self, panel):
        current = {"AAPL": 0.50, "MSFT": 0.50}
        target = {"AAPL": 0.60, "MSFT": 0.40}
        panel.set_weights(current, target)
        # Turnover = sum(|delta|) / 2 = (0.10 + 0.10) / 2 = 0.10 = 10%
        assert "Turnover: 10.0%" in panel._info_label.text()

    def test_info_label_shows_buy_sell_counts(self, panel):
        current = {"AAPL": 0.40, "MSFT": 0.30, "GOOG": 0.30}
        target = {"AAPL": 0.50, "MSFT": 0.20, "GOOG": 0.30}
        panel.set_weights(current, target)
        # AAPL: +10% (buy), MSFT: -10% (sell), GOOG: 0% (hold)
        assert "1 buys" in panel._info_label.text()
        assert "1 sells" in panel._info_label.text()

    def test_info_label_zero_turnover(self, panel):
        current = {"AAPL": 0.50, "MSFT": 0.50}
        target = {"AAPL": 0.50, "MSFT": 0.50}
        panel.set_weights(current, target)
        assert "Turnover: 0.0%" in panel._info_label.text()
        assert "0 buys" in panel._info_label.text()
        assert "0 sells" in panel._info_label.text()

    def test_info_label_large_rebalance(self, panel):
        current = {"AAPL": 1.0}
        target = {"AAPL": 0.25, "MSFT": 0.25, "GOOG": 0.25, "AMZN": 0.25}
        panel.set_weights(current, target)
        assert "4 symbols" in panel._info_label.text()
        # 3 buys (MSFT, GOOG, AMZN), 1 sell (AAPL: 1.0 -> 0.25)
        assert "3 buys" in panel._info_label.text()
        assert "1 sells" in panel._info_label.text()
