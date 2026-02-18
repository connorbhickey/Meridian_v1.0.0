"""Tests for the OrderPanel -- trade order generation and review.

Requires: pytest-qt (provides qtbot fixture)
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from portopt.constants import Colors
from portopt.gui.panels.order_panel import OrderPanel


# -- Fixture ----------------------------------------------------------------


@pytest.fixture
def panel(qtbot):
    p = OrderPanel()
    qtbot.addWidget(p)
    return p


# -- Helpers ----------------------------------------------------------------


def _make_orders() -> list[dict]:
    """Build a small set of sample orders for testing."""
    return [
        {
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 50,
            "price": 175.00,
            "limit_price": 174.50,
            "value": 8750,
            "estimated_cost": 1.25,
            "status": "PENDING",
        },
        {
            "symbol": "MSFT",
            "side": "SELL",
            "quantity": 30,
            "price": 420.00,
            "limit_price": None,
            "value": 12600,
            "estimated_cost": 0.85,
            "status": "PENDING",
        },
        {
            "symbol": "GOOG",
            "side": "BUY",
            "quantity": 10,
            "price": 178.50,
            "limit_price": 178.00,
            "value": 1785,
            "estimated_cost": 0.50,
            "status": "READY",
        },
    ]


def _make_batch_stats() -> dict:
    """Build sample batch statistics."""
    return {
        "total_buy_value": 10535,
        "total_sell_value": 12600,
        "net_cash_flow": 2065,
        "total_estimated_cost": 2.60,
        "turnover": 0.12,
    }


def _table_cell_text(table, row: int, col: int) -> str:
    """Read the text of a single table cell."""
    item = table.item(row, col)
    return item.text() if item else ""


def _table_column_values(table, col: int) -> list[str]:
    """Read all values from a table column as strings."""
    return [_table_cell_text(table, row, col) for row in range(table.rowCount())]


def _find_row_by_symbol(table, symbol: str) -> int | None:
    """Return the row index for a given symbol, or None."""
    for row in range(table.rowCount()):
        if _table_cell_text(table, row, 0) == symbol:
            return row
    return None


# -- TestPanelInitialization ------------------------------------------------


class TestPanelInitialization:
    """Verify panel identity and initial widget state."""

    def test_panel_id(self, panel):
        assert panel.panel_id == "orders"

    def test_panel_title(self, panel):
        assert panel.panel_title == "ORDERS"

    def test_table_has_eight_columns(self, panel):
        assert panel._table.columnCount() == 8

    def test_table_column_headers(self, panel):
        headers = [
            panel._table.horizontalHeaderItem(c).text()
            for c in range(panel._table.columnCount())
        ]
        assert headers == [
            "Symbol", "Side", "Shares", "Price",
            "Limit", "Value", "Est. Cost", "Status",
        ]

    def test_table_starts_empty(self, panel):
        assert panel._table.rowCount() == 0

    def test_generate_button_exists(self, panel):
        assert panel._generate_btn is not None
        assert panel._generate_btn.text() == "Generate Orders"

    def test_order_type_combo_has_market_and_limit(self, panel):
        combo = panel._order_type_combo
        items = [combo.itemText(i) for i in range(combo.count())]
        assert "MARKET" in items
        assert "LIMIT" in items

    def test_info_label_default_text(self, panel):
        assert panel._info_label.text() == "No orders generated"

    def test_blotter_button_disabled_initially(self, panel):
        assert not panel._blotter_btn.isEnabled()


# -- TestSetOrders ----------------------------------------------------------


class TestSetOrders:
    """Test the set_orders() public API."""

    def test_row_count_matches_orders(self, panel):
        orders = _make_orders()
        panel.set_orders(orders)
        assert panel._table.rowCount() == 3

    def test_buy_side_has_profit_color(self, panel):
        orders = _make_orders()
        panel.set_orders(orders)
        row = _find_row_by_symbol(panel._table, "AAPL")
        assert row is not None
        side_item = panel._table.item(row, 1)
        assert side_item.foreground().color() == QColor(Colors.PROFIT)

    def test_sell_side_has_loss_color(self, panel):
        orders = _make_orders()
        panel.set_orders(orders)
        row = _find_row_by_symbol(panel._table, "MSFT")
        assert row is not None
        side_item = panel._table.item(row, 1)
        assert side_item.foreground().color() == QColor(Colors.LOSS)

    def test_symbol_column_populated(self, panel):
        orders = _make_orders()
        panel.set_orders(orders)
        symbols = set(_table_column_values(panel._table, 0))
        assert symbols == {"AAPL", "MSFT", "GOOG"}

    def test_quantity_column_populated(self, panel):
        orders = _make_orders()
        panel.set_orders(orders)
        row = _find_row_by_symbol(panel._table, "AAPL")
        assert _table_cell_text(panel._table, row, 2) == "50"

    def test_price_column_populated(self, panel):
        orders = _make_orders()
        panel.set_orders(orders)
        row = _find_row_by_symbol(panel._table, "AAPL")
        assert _table_cell_text(panel._table, row, 3) == "$175.00"

    def test_limit_price_shown_when_present(self, panel):
        orders = _make_orders()
        panel.set_orders(orders)
        row = _find_row_by_symbol(panel._table, "AAPL")
        assert _table_cell_text(panel._table, row, 4) == "$174.50"

    def test_limit_price_dash_when_absent(self, panel):
        orders = _make_orders()
        panel.set_orders(orders)
        row = _find_row_by_symbol(panel._table, "MSFT")
        # limit_price is None, so should show em dash
        assert _table_cell_text(panel._table, row, 4) == "\u2014"

    def test_status_column_shows_status(self, panel):
        orders = _make_orders()
        panel.set_orders(orders)
        row = _find_row_by_symbol(panel._table, "GOOG")
        assert _table_cell_text(panel._table, row, 7) == "READY"

    def test_info_label_updates_with_order_count(self, panel):
        orders = _make_orders()
        panel.set_orders(orders)
        label_text = panel._info_label.text()
        assert "3 orders" in label_text
        assert "2 buys" in label_text
        assert "1 sells" in label_text

    def test_batch_stats_populate_summary_labels(self, panel):
        orders = _make_orders()
        stats = _make_batch_stats()
        panel.set_orders(orders, batch_stats=stats)

        assert panel._summary_labels["total_buy"].text() == "$10,535"
        assert panel._summary_labels["total_sell"].text() == "$12,600"
        assert panel._summary_labels["net_cash"].text() == "$+2,065"
        assert panel._summary_labels["est_cost"].text() == "$2.60"
        assert panel._summary_labels["turnover"].text() == "12.0%"

    def test_blotter_button_enabled_after_set_orders(self, panel):
        panel.set_orders(_make_orders())
        assert panel._blotter_btn.isEnabled()

    def test_set_orders_with_empty_list(self, panel):
        panel.set_orders([])
        assert panel._table.rowCount() == 0
        assert not panel._blotter_btn.isEnabled()

    def test_set_orders_replaces_previous(self, panel):
        """Calling set_orders again should replace, not append."""
        panel.set_orders(_make_orders())
        assert panel._table.rowCount() == 3

        new_orders = [_make_orders()[0]]
        panel.set_orders(new_orders)
        assert panel._table.rowCount() == 1


# -- TestClearOrders --------------------------------------------------------


class TestClearOrders:
    """Test the clear_orders() method."""

    def test_clear_empties_table(self, panel):
        panel.set_orders(_make_orders())
        assert panel._table.rowCount() == 3
        panel.clear_orders()
        assert panel._table.rowCount() == 0

    def test_info_label_resets(self, panel):
        panel.set_orders(_make_orders())
        panel.clear_orders()
        assert panel._info_label.text() == "No orders generated"

    def test_blotter_button_disabled_after_clear(self, panel):
        panel.set_orders(_make_orders())
        assert panel._blotter_btn.isEnabled()
        panel.clear_orders()
        assert not panel._blotter_btn.isEnabled()

    def test_summary_labels_reset_to_dash(self, panel):
        panel.set_orders(_make_orders(), batch_stats=_make_batch_stats())
        panel.clear_orders()
        for label in panel._summary_labels.values():
            assert label.text() == "\u2014"


# -- TestOrderType ----------------------------------------------------------


class TestOrderType:
    """Test the order_type property and combo box."""

    def test_default_order_type_is_market(self, panel):
        assert panel.order_type == "market"

    def test_change_to_limit(self, panel):
        panel._order_type_combo.setCurrentText("LIMIT")
        assert panel.order_type == "limit"

    def test_change_back_to_market(self, panel):
        panel._order_type_combo.setCurrentText("LIMIT")
        panel._order_type_combo.setCurrentText("MARKET")
        assert panel.order_type == "market"


# -- TestSignals ------------------------------------------------------------


class TestSignals:
    """Test signal emissions from user interactions."""

    def test_generate_requested_fires_on_button_click(self, panel, qtbot):
        with qtbot.waitSignal(panel.generate_requested, timeout=1000):
            panel._generate_btn.click()

    def test_send_to_blotter_fires_with_trade_dicts(self, panel, qtbot):
        orders = _make_orders()
        panel.set_orders(orders)

        with qtbot.waitSignal(panel.send_to_blotter, timeout=1000) as blocker:
            panel._blotter_btn.click()

        trades = blocker.args[0]
        assert isinstance(trades, list)
        assert len(trades) == 3

        # Verify blotter trade dict structure
        first_trade = trades[0]
        assert "date" in first_trade
        assert "symbol" in first_trade
        assert "side" in first_trade
        assert "quantity" in first_trade
        assert "price" in first_trade
        assert "cost" in first_trade

    def test_blotter_trades_have_correct_symbol_data(self, panel, qtbot):
        orders = _make_orders()
        panel.set_orders(orders)

        with qtbot.waitSignal(panel.send_to_blotter, timeout=1000) as blocker:
            panel._blotter_btn.click()

        trades = blocker.args[0]
        symbols = {t["symbol"] for t in trades}
        assert symbols == {"AAPL", "MSFT", "GOOG"}
