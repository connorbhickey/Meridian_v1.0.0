"""Dockable TRADE BLOTTER panel — backtest trade log with sort/filter."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QLineEdit, QLabel, QComboBox,
)

from portopt.gui.panels.base_panel import BasePanel
from portopt.constants import Colors, Fonts


class TradeBlotterPanel(BasePanel):
    panel_id = "trade_blotter"
    panel_title = "TRADE BLOTTER"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._trades = []   # list of trade dicts
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        # Filter by symbol
        lbl = QLabel("Filter:")
        lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 10px;")
        toolbar.addWidget(lbl)

        self._filter_input = QLineEdit()
        self._filter_input.setPlaceholderText("Symbol...")
        self._filter_input.setFixedWidth(80)
        self._filter_input.textChanged.connect(self._apply_filter)
        toolbar.addWidget(self._filter_input)

        # Side filter
        self._side_combo = QComboBox()
        self._side_combo.addItems(["All", "BUY", "SELL"])
        self._side_combo.setFixedWidth(70)
        self._side_combo.currentTextChanged.connect(self._apply_filter)
        toolbar.addWidget(self._side_combo)

        toolbar.addStretch()

        self._count_label = QLabel("")
        self._count_label.setStyleSheet(
            f"color: {Colors.TEXT_MUTED}; font-family: {Fonts.MONO}; font-size: 9px;"
        )
        toolbar.addWidget(self._count_label)

        layout.addLayout(toolbar)

        # Trade table
        self._table = QTableWidget()
        self._table.setColumnCount(7)
        self._table.setHorizontalHeaderLabels([
            "Date", "Symbol", "Side", "Qty", "Price", "Cost", "Weight After",
        ])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        self._table.setStyleSheet(f"""
            QTableWidget {{
                background: {Colors.BG_SECONDARY};
                gridline-color: {Colors.BORDER};
                color: {Colors.TEXT_PRIMARY};
                font-family: {Fonts.MONO};
                font-size: 10px;
                border: 1px solid {Colors.BORDER};
            }}
            QTableWidget::item {{
                padding: 2px 6px;
            }}
            QHeaderView::section {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_SECONDARY};
                border: 1px solid {Colors.BORDER};
                padding: 3px;
                font-size: 9px;
                font-weight: bold;
            }}
        """)
        layout.addWidget(self._table)
        self.content_layout.addLayout(layout)

    # ── Public API ───────────────────────────────────────────────────

    def set_trades(self, trades: list[dict]):
        """Set trade data. Each dict should have keys:
        date, symbol, side, quantity, price, cost, weight_after
        """
        self._trades = list(trades)
        self._apply_filter()

    def clear_trades(self):
        self._trades.clear()
        self._table.setRowCount(0)
        self._count_label.setText("")

    # ── Internal ─────────────────────────────────────────────────────

    def _apply_filter(self, _=None):
        symbol_filter = self._filter_input.text().upper().strip()
        side_filter = self._side_combo.currentText()

        filtered = self._trades
        if symbol_filter:
            filtered = [t for t in filtered if symbol_filter in t.get("symbol", "").upper()]
        if side_filter != "All":
            filtered = [t for t in filtered if t.get("side", "").upper() == side_filter]

        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(filtered))

        for i, trade in enumerate(filtered):
            date_str = str(trade.get("date", ""))[:10]
            self._table.setItem(i, 0, QTableWidgetItem(date_str))

            sym_item = QTableWidgetItem(trade.get("symbol", ""))
            self._table.setItem(i, 1, sym_item)

            side = trade.get("side", "")
            side_item = QTableWidgetItem(side)
            color = Colors.PROFIT if side.upper() == "BUY" else Colors.LOSS
            side_item.setForeground(pg_color(color))
            self._table.setItem(i, 2, side_item)

            qty = trade.get("quantity", 0)
            qty_item = QTableWidgetItem(f"{qty:,.2f}")
            qty_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(i, 3, qty_item)

            price = trade.get("price", 0)
            price_item = QTableWidgetItem(f"${price:,.2f}")
            price_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(i, 4, price_item)

            cost = trade.get("cost", 0)
            cost_item = QTableWidgetItem(f"${cost:,.4f}")
            cost_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(i, 5, cost_item)

            w_after = trade.get("weight_after", 0)
            w_item = QTableWidgetItem(f"{w_after:.2%}")
            w_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(i, 6, w_item)

        self._table.setSortingEnabled(True)
        self._count_label.setText(f"{len(filtered)} / {len(self._trades)} trades")


def pg_color(hex_str: str):
    """Create a QColor from hex string."""
    from PySide6.QtGui import QColor
    return QColor(hex_str)
