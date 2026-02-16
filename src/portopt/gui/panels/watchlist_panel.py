"""Dockable WATCHLIST panel — ticker list with prices, change %, sparklines."""

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QLineEdit, QLabel, QAbstractItemView,
)

from portopt.constants import Colors, Fonts
from portopt.gui.panels.base_panel import BasePanel


class WatchlistPanel(BasePanel):
    panel_id = "watchlist"
    panel_title = "WATCHLIST"

    ticker_selected = Signal(str)
    add_ticker_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Header with add ticker input
        header = QHBoxLayout()
        self._input = QLineEdit()
        self._input.setPlaceholderText("Add ticker...")
        self._input.setFont(QFont(Fonts.MONO, Fonts.SIZE_SMALL))
        self._input.setMaximumWidth(120)
        self._input.returnPressed.connect(self._on_add_ticker)

        add_btn = QPushButton("+")
        add_btn.setFixedWidth(28)
        add_btn.clicked.connect(self._on_add_ticker)

        header.addWidget(self._input)
        header.addWidget(add_btn)
        header.addStretch()
        self._layout.addLayout(header)

        # Table
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(["Symbol", "Price", "Chg", "Chg %", "Volume"])
        self._table.horizontalHeader().setFont(QFont(Fonts.MONO, Fonts.SIZE_SMALL))
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setFont(QFont(Fonts.MONO, Fonts.SIZE_SMALL))
        self._table.cellDoubleClicked.connect(self._on_row_selected)
        self._layout.addWidget(self._table)

        self._watchlist: list[dict] = []

    def _on_add_ticker(self):
        ticker = self._input.text().strip().upper()
        if ticker:
            self.add_ticker_requested.emit(ticker)
            self._input.clear()

    def _on_row_selected(self, row, _col):
        item = self._table.item(row, 0)
        if item:
            self.ticker_selected.emit(item.text())

    def set_watchlist(self, items: list[dict]):
        """Set watchlist data. Each item: {symbol, price, change, change_pct, volume}."""
        self._watchlist = items
        self._table.setRowCount(len(items))

        for i, item in enumerate(items):
            sym = item.get("symbol", "")
            price = item.get("price", 0.0)
            change = item.get("change", 0.0)
            change_pct = item.get("change_pct", 0.0)
            volume = item.get("volume", 0)

            color = Colors.PROFIT if change >= 0 else Colors.LOSS

            self._set_cell(i, 0, sym, Colors.TEXT_PRIMARY)
            self._set_cell(i, 1, f"${price:,.2f}", Colors.TEXT_PRIMARY)
            self._set_cell(i, 2, f"{change:+,.2f}", color)
            self._set_cell(i, 3, f"{change_pct:+.2f}%", color)
            self._set_cell(i, 4, self._fmt_volume(volume), Colors.TEXT_SECONDARY)

        self._table.setRowCount(len(items))

    def _set_cell(self, row, col, text, color):
        item = QTableWidgetItem(text)
        item.setForeground(QColor(color))
        item.setFont(QFont(Fonts.MONO, Fonts.SIZE_SMALL))
        if col > 0:
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._table.setItem(row, col, item)

    def get_symbols(self) -> list[str]:
        """Return list of ticker symbols currently in the watchlist."""
        return [item["symbol"] for item in self._watchlist if item.get("symbol")]

    def _fmt_volume(self, vol):
        if vol >= 1_000_000:
            return f"{vol / 1_000_000:.1f}M"
        if vol >= 1_000:
            return f"{vol / 1_000:.1f}K"
        return str(vol)
