"""Dockable PORTFOLIO panel — Fidelity positions blotter with P&L coloring."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QHeaderView, QLabel, QPushButton, QTableWidget,
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
        self._portfolio = None  # stored for filtering

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

        # Account filter dropdown
        filter_row = QWidget()
        filter_layout = QHBoxLayout(filter_row)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.setSpacing(6)

        filter_label = QLabel("Account:")
        filter_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: {Fonts.SIZE_SMALL}pt;")
        filter_layout.addWidget(filter_label)

        self._account_filter = QComboBox()
        self._account_filter.addItem("All Accounts")
        self._account_filter.setFixedHeight(22)
        self._account_filter.currentTextChanged.connect(self._on_account_filter_changed)
        filter_layout.addWidget(self._account_filter)

        filter_layout.addStretch()
        self._layout.addWidget(filter_row)

        # Positions table
        self._table = QTableWidget()
        self._table.setColumnCount(len(self._COLUMNS))
        self._table.setHorizontalHeaderLabels(self._COLUMNS)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(22)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setFont(QFont(Fonts.MONO, Fonts.SIZE_SMALL))
        setup_table_context_menu(self._table)
        self._layout.addWidget(self._table)

        # Sector summary label
        self._sector_label = QLabel("")
        self._sector_label.setStyleSheet(
            f"color: {Colors.TEXT_MUTED}; font-size: {Fonts.SIZE_SMALL}pt; padding: 2px 0;"
        )
        self._sector_label.setWordWrap(True)
        self._layout.addWidget(self._sector_label)

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
        """Store the portfolio and populate the table."""
        self._portfolio = portfolio

        # Update account filter dropdown
        self._account_filter.blockSignals(True)
        current_filter = self._account_filter.currentText()
        self._account_filter.clear()
        self._account_filter.addItem("All Accounts")
        account_names = sorted({h.account for h in portfolio.holdings if h.account})
        for name in account_names:
            self._account_filter.addItem(name)
        # Restore previous selection if still valid
        idx = self._account_filter.findText(current_filter)
        self._account_filter.setCurrentIndex(idx if idx >= 0 else 0)
        self._account_filter.blockSignals(False)

        self._display_holdings()

    def _on_account_filter_changed(self, text: str):
        """Re-display holdings filtered by account."""
        if self._portfolio:
            self._display_holdings()

    def _display_holdings(self):
        """Populate the table with holdings, applying the current account filter."""
        if not self._portfolio:
            return

        account_filter = self._account_filter.currentText()
        if account_filter == "All Accounts":
            holdings = self._portfolio.holdings
        else:
            holdings = [h for h in self._portfolio.holdings if h.account == account_filter]

        self._table.setRowCount(len(holdings))
        total_value = self._portfolio.total_value  # always use full portfolio total for weights

        for row, holding in enumerate(holdings):
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

        # Update summary (always show full portfolio summary)
        self._total_label.setText(f"${total_value:,.2f}")
        total_pnl = self._portfolio.total_pnl
        sign = "+" if total_pnl >= 0 else ""
        color = Colors.PROFIT if total_pnl >= 0 else Colors.LOSS
        self._pnl_label.setText(f"{sign}${total_pnl:,.2f} ({sign}{self._portfolio.total_pnl_pct:.2f}%)")
        self._pnl_label.setStyleSheet(f"color: {color};")

        n_accts = len(self._portfolio.accounts)
        filtered = f" ({len(holdings)} shown)" if account_filter != "All Accounts" else ""
        self._status_label.setText(
            f"{len(self._portfolio.holdings)} positions across {n_accts} account{'s' if n_accts != 1 else ''}{filtered}"
        )

        # Sector summary
        self._update_sector_summary(holdings)

        # Auto-resize columns
        self._table.resizeColumnsToContents()

    def _update_sector_summary(self, holdings: list[Holding]):
        """Show top sectors by weight below the table."""
        sector_weight: dict[str, float] = {}
        total_val = sum(h.market_value for h in holdings)
        if total_val <= 0:
            self._sector_label.setText("")
            return

        for h in holdings:
            sector = h.asset.sector or "Unknown"
            sector_weight[sector] = sector_weight.get(sector, 0) + h.market_value

        # Sort by weight descending, show top 5
        top = sorted(sector_weight.items(), key=lambda x: x[1], reverse=True)[:5]
        parts = [f"{name}: {val / total_val * 100:.1f}%" for name, val in top]
        self._sector_label.setText("Sectors: " + " | ".join(parts))

    def _set_cell(self, row: int, col: int, text: str,
                  color: str = None, align: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignRight):
        item = QTableWidgetItem(text)
        item.setTextAlignment(align | Qt.AlignmentFlag.AlignVCenter)
        if color:
            item.setForeground(QColor(color))
        self._table.setItem(row, col, item)
