"""Dockable PORTFOLIO panel — Fidelity positions blotter with P&L coloring."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QHBoxLayout, QHeaderView, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from portopt.constants import Colors, Fonts
from portopt.gui.widgets.table_context_menu import setup_table_context_menu
from portopt.data.models import Portfolio, Holding
from portopt.gui.panels.base_panel import BasePanel


class PortfolioPanel(BasePanel):
    panel_id = "portfolio"
    panel_title = "PORTFOLIO"

    refresh_requested = Signal()
    connect_requested = Signal()

    _COLUMNS = ["Symbol", "Qty", "Price", "Value", "Cost", "P&L", "P&L %", "Weight", "Account"]

    def __init__(self, parent=None):
        super().__init__(parent)

        # Header bar
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)

        title = QLabel("POSITIONS")
        title.setProperty("header", True)
        header_layout.addWidget(title)

        header_layout.addStretch()

        self._total_label = QLabel("$0.00")
        self._total_label.setFont(QFont(Fonts.MONO, Fonts.SIZE_LARGE, QFont.Weight.Bold))
        self._total_label.setStyleSheet(f"color: {Colors.ACCENT};")
        header_layout.addWidget(self._total_label)

        self._pnl_label = QLabel("")
        self._pnl_label.setFont(QFont(Fonts.MONO, Fonts.SIZE_NORMAL))
        header_layout.addWidget(self._pnl_label)

        self._layout.addWidget(header)

        # Positions table
        self._table = QTableWidget()
        self._table.setColumnCount(len(self._COLUMNS))
        self._table.setHorizontalHeaderLabels(self._COLUMNS)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(22)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setFont(QFont(Fonts.MONO, Fonts.SIZE_SMALL))
        setup_table_context_menu(self._table)
        self._layout.addWidget(self._table)

        # Bottom bar
        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(4)

        self._status_label = QLabel("No positions loaded")
        self._status_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: {Fonts.SIZE_SMALL}pt;")
        bottom_layout.addWidget(self._status_label)
        bottom_layout.addStretch()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedHeight(22)
        refresh_btn.clicked.connect(self.refresh_requested.emit)
        bottom_layout.addWidget(refresh_btn)

        connect_btn = QPushButton("Connect Fidelity")
        connect_btn.setFixedHeight(22)
        connect_btn.setProperty("primary", True)
        connect_btn.clicked.connect(self.connect_requested.emit)
        bottom_layout.addWidget(connect_btn)

        self._layout.addWidget(bottom)

    def set_portfolio(self, portfolio: Portfolio):
        """Populate the table with portfolio holdings."""
        self._table.setRowCount(len(portfolio.holdings))
        total_value = portfolio.total_value

        for row, holding in enumerate(portfolio.holdings):
            self._set_cell(row, 0, holding.asset.symbol, align=Qt.AlignmentFlag.AlignLeft)
            self._set_cell(row, 1, f"{holding.quantity:,.2f}")
            self._set_cell(row, 2, f"${holding.current_price:,.2f}")
            self._set_cell(row, 3, f"${holding.market_value:,.2f}")
            self._set_cell(row, 4, f"${holding.cost_basis:,.2f}" if holding.cost_basis else "---")

            # P&L with coloring
            pnl = holding.unrealized_pnl
            pnl_pct = holding.unrealized_pnl_pct
            pnl_color = Colors.PROFIT if pnl >= 0 else Colors.LOSS
            sign = "+" if pnl >= 0 else ""
            self._set_cell(row, 5, f"{sign}${pnl:,.2f}", color=pnl_color)
            self._set_cell(row, 6, f"{sign}{pnl_pct:.2f}%", color=pnl_color)

            # Weight
            weight = (holding.market_value / total_value * 100) if total_value else 0
            self._set_cell(row, 7, f"{weight:.1f}%")

            # Account
            self._set_cell(row, 8, holding.account, align=Qt.AlignmentFlag.AlignLeft)

        # Update summary
        self._total_label.setText(f"${total_value:,.2f}")
        total_pnl = portfolio.total_pnl
        sign = "+" if total_pnl >= 0 else ""
        color = Colors.PROFIT if total_pnl >= 0 else Colors.LOSS
        self._pnl_label.setText(f"{sign}${total_pnl:,.2f} ({sign}{portfolio.total_pnl_pct:.2f}%)")
        self._pnl_label.setStyleSheet(f"color: {color};")

        n_accts = len(portfolio.accounts)
        self._status_label.setText(
            f"{len(portfolio.holdings)} positions across {n_accts} account{'s' if n_accts != 1 else ''}"
        )

        # Auto-resize columns
        self._table.resizeColumnsToContents()

    def _set_cell(self, row: int, col: int, text: str,
                  color: str = None, align: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignRight):
        item = QTableWidgetItem(text)
        item.setTextAlignment(align | Qt.AlignmentFlag.AlignVCenter)
        if color:
            item.setForeground(QColor(color))
        self._table.setItem(row, col, item)
