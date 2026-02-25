"""Dockable ACCOUNT BALANCES panel — unified view of all linked financial accounts."""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from portopt.constants import Colors, Fonts
from portopt.data.models import PlaidAccount, PlaidAccountType
from portopt.gui.panels.base_panel import BasePanel
from portopt.gui.widgets.table_context_menu import setup_table_context_menu


class BalancePanel(BasePanel):
    """Unified account balances — Plaid + Fidelity accounts in one table."""

    panel_id = "account_balances"
    panel_title = "ACCOUNT BALANCES"

    refresh_requested = Signal()
    link_account_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._plaid_accounts: list[PlaidAccount] = []
        self._fidelity_accounts: list[dict] = []  # AccountSummary-style dicts
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Summary header ────────────────────────────────────────────
        summary_row = QHBoxLayout()
        summary_row.setSpacing(12)

        self._net_worth_label = QLabel("Net Worth: —")
        self._net_worth_label.setStyleSheet(
            f"color: {Colors.ACCENT}; font-family: {Fonts.MONO}; font-size: 11px; font-weight: bold;"
        )
        summary_row.addWidget(self._net_worth_label)

        self._assets_label = QLabel("Assets: —")
        self._assets_label.setStyleSheet(
            f"color: {Colors.PROFIT}; font-family: {Fonts.MONO}; font-size: 9px;"
        )
        summary_row.addWidget(self._assets_label)

        self._liabilities_label = QLabel("Liabilities: —")
        self._liabilities_label.setStyleSheet(
            f"color: {Colors.LOSS}; font-family: {Fonts.MONO}; font-size: 9px;"
        )
        summary_row.addWidget(self._liabilities_label)

        summary_row.addStretch()

        # Buttons
        self._link_btn = QPushButton("Link Account")
        self._link_btn.setFixedHeight(24)
        self._link_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT_DIM};
                color: {Colors.ACCENT};
                border: 1px solid {Colors.ACCENT};
                border-radius: 3px;
                padding: 0 12px;
                font-family: {Fonts.SANS};
                font-size: 9px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background: {Colors.ACCENT}; color: {Colors.BG_PRIMARY}; }}
        """)
        self._link_btn.clicked.connect(self.link_account_requested.emit)
        summary_row.addWidget(self._link_btn)

        self._refresh_btn = QPushButton("Refresh All")
        self._refresh_btn.setFixedHeight(24)
        self._refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_SECONDARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 3px;
                padding: 0 12px;
                font-family: {Fonts.SANS};
                font-size: 9px;
            }}
            QPushButton:hover {{ background: {Colors.BG_ELEVATED}; color: {Colors.TEXT_PRIMARY}; }}
        """)
        self._refresh_btn.clicked.connect(self.refresh_requested.emit)
        summary_row.addWidget(self._refresh_btn)

        layout.addLayout(summary_row)

        # ── Account table ─────────────────────────────────────────────
        self._table = QTableWidget()
        self._table.setColumnCount(8)
        self._table.setHorizontalHeaderLabels([
            "Source", "Institution", "Account", "Type",
            "Balance", "Available", "Limit", "Last Synced",
        ])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
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
        setup_table_context_menu(self._table)
        layout.addWidget(self._table)

        self.content_layout.addLayout(layout)

    # ── Public API ────────────────────────────────────────────────────

    def set_plaid_accounts(self, accounts: list[PlaidAccount]):
        """Update Plaid accounts and re-render."""
        self._plaid_accounts = list(accounts)
        self._render()

    def set_fidelity_accounts(self, accounts: list[dict]):
        """Update Fidelity accounts (from AccountSummary or similar dicts) and re-render.

        Expected dict keys: account_id, account_name, total_value
        """
        self._fidelity_accounts = list(accounts)
        self._render()

    def clear(self):
        self._plaid_accounts.clear()
        self._fidelity_accounts.clear()
        self._table.setRowCount(0)
        self._net_worth_label.setText("Net Worth: —")
        self._assets_label.setText("Assets: —")
        self._liabilities_label.setText("Liabilities: —")

    # ── Rendering ─────────────────────────────────────────────────────

    def _render(self):
        rows = []

        # Plaid accounts
        for acct in self._plaid_accounts:
            is_liability = acct.account_type == PlaidAccountType.CREDIT_CARD
            rows.append({
                "source": "Plaid",
                "institution": acct.institution_name,
                "account": acct.display_name,
                "type": acct.account_type.name.replace("_", " ").title(),
                "balance": acct.current_balance,
                "available": acct.available_balance,
                "limit": acct.limit,
                "last_synced": acct.last_synced,
                "is_liability": is_liability,
            })

        # Fidelity accounts
        for acct in self._fidelity_accounts:
            rows.append({
                "source": "Fidelity",
                "institution": "Fidelity",
                "account": acct.get("account_name", acct.get("account_id", "")),
                "type": "Investment",
                "balance": acct.get("total_value", 0),
                "available": None,
                "limit": None,
                "last_synced": None,
                "is_liability": False,
            })

        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(rows))

        total_assets = 0.0
        total_liabilities = 0.0

        for row_idx, data in enumerate(rows):
            # Source
            item = QTableWidgetItem(data["source"])
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row_idx, 0, item)

            # Institution
            item = QTableWidgetItem(data["institution"])
            self._table.setItem(row_idx, 1, item)

            # Account
            item = QTableWidgetItem(data["account"])
            self._table.setItem(row_idx, 2, item)

            # Type
            item = QTableWidgetItem(data["type"])
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row_idx, 3, item)

            # Balance
            balance = data["balance"] or 0
            item = QTableWidgetItem(f"${balance:,.2f}")
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            item.setFont(QFont(Fonts.MONO, Fonts.SIZE_NORMAL))
            if data["is_liability"]:
                item.setForeground(QColor(Colors.LOSS))
                total_liabilities += abs(balance)
            else:
                item.setForeground(QColor(Colors.PROFIT))
                total_assets += balance
            self._table.setItem(row_idx, 4, item)

            # Available
            avail = data["available"]
            item = QTableWidgetItem(f"${avail:,.2f}" if avail is not None else "—")
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._table.setItem(row_idx, 5, item)

            # Limit
            limit = data["limit"]
            item = QTableWidgetItem(f"${limit:,.2f}" if limit is not None else "—")
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._table.setItem(row_idx, 6, item)

            # Last Synced
            synced = data["last_synced"]
            if isinstance(synced, datetime):
                synced_str = synced.strftime("%Y-%m-%d %H:%M")
            elif synced:
                synced_str = str(synced)
            else:
                synced_str = "—"
            item = QTableWidgetItem(synced_str)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setForeground(QColor(Colors.TEXT_MUTED))
            self._table.setItem(row_idx, 7, item)

        self._table.setSortingEnabled(True)

        # Update summary
        net_worth = total_assets - total_liabilities
        self._net_worth_label.setText(f"Net Worth: ${net_worth:,.2f}")
        self._assets_label.setText(f"Assets: ${total_assets:,.2f}")
        self._liabilities_label.setText(f"Liabilities: ${total_liabilities:,.2f}")
