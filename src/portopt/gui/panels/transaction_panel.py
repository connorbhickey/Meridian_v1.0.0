"""Dockable TRANSACTIONS panel — unified view of transactions from all sources."""

from __future__ import annotations

from datetime import date

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QComboBox, QDateEdit, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from portopt.constants import Colors, Fonts
from portopt.data.models import Transaction, TransactionSource, TransactionStatus
from portopt.gui.panels.base_panel import BasePanel
from portopt.gui.widgets.table_context_menu import setup_table_context_menu


class TransactionPanel(BasePanel):
    """Unified transaction view — Plaid + Fidelity transactions merged by date."""

    panel_id = "transactions"
    panel_title = "TRANSACTIONS"

    sync_requested = Signal()
    filter_changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._transactions: list[Transaction] = []
        self._filtered: list[Transaction] = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Summary header ────────────────────────────────────────────
        summary_row = QHBoxLayout()
        summary_row.setSpacing(12)

        self._count_label = QLabel("0 transactions")
        self._count_label.setStyleSheet(
            f"color: {Colors.TEXT_MUTED}; font-family: {Fonts.MONO}; font-size: 9px;"
        )
        summary_row.addWidget(self._count_label)

        self._debit_label = QLabel("Debit: $0")
        self._debit_label.setStyleSheet(
            f"color: {Colors.LOSS}; font-family: {Fonts.MONO}; font-size: 9px;"
        )
        summary_row.addWidget(self._debit_label)

        self._credit_label = QLabel("Credit: $0")
        self._credit_label.setStyleSheet(
            f"color: {Colors.PROFIT}; font-family: {Fonts.MONO}; font-size: 9px;"
        )
        summary_row.addWidget(self._credit_label)

        summary_row.addStretch()

        self._sync_btn = QPushButton("Sync")
        self._sync_btn.setFixedHeight(24)
        self._sync_btn.setStyleSheet(f"""
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
        self._sync_btn.clicked.connect(self.sync_requested.emit)
        summary_row.addWidget(self._sync_btn)

        layout.addLayout(summary_row)

        # ── Filter bar ────────────────────────────────────────────────
        filter_row = QHBoxLayout()
        filter_row.setSpacing(6)

        filter_style = f"""
            QComboBox, QLineEdit, QDateEdit {{
                background: {Colors.BG_INPUT};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 3px;
                padding: 2px 6px;
                font-family: {Fonts.MONO};
                font-size: 9px;
            }}
        """

        # Source filter
        lbl = QLabel("Source:")
        lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 9px;")
        filter_row.addWidget(lbl)
        self._source_combo = QComboBox()
        self._source_combo.addItems(["All", "Plaid", "Fidelity"])
        self._source_combo.setFixedWidth(80)
        self._source_combo.setStyleSheet(filter_style)
        self._source_combo.currentTextChanged.connect(self._apply_filters)
        filter_row.addWidget(self._source_combo)

        # Account filter
        lbl = QLabel("Account:")
        lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 9px;")
        filter_row.addWidget(lbl)
        self._account_combo = QComboBox()
        self._account_combo.addItems(["All"])
        self._account_combo.setFixedWidth(140)
        self._account_combo.setStyleSheet(filter_style)
        self._account_combo.currentTextChanged.connect(self._apply_filters)
        filter_row.addWidget(self._account_combo)

        # Status filter
        lbl = QLabel("Status:")
        lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 9px;")
        filter_row.addWidget(lbl)
        self._status_combo = QComboBox()
        self._status_combo.addItems(["All", "Pending", "Posted"])
        self._status_combo.setFixedWidth(80)
        self._status_combo.setStyleSheet(filter_style)
        self._status_combo.currentTextChanged.connect(self._apply_filters)
        filter_row.addWidget(self._status_combo)

        # Search box
        self._search_box = QLineEdit()
        self._search_box.setPlaceholderText("Search merchant/description...")
        self._search_box.setFixedWidth(180)
        self._search_box.setStyleSheet(filter_style)
        self._search_box.textChanged.connect(self._apply_filters)
        filter_row.addWidget(self._search_box)

        filter_row.addStretch()
        layout.addLayout(filter_row)

        # ── Transaction table ─────────────────────────────────────────
        self._table = QTableWidget()
        self._table.setColumnCount(9)
        self._table.setHorizontalHeaderLabels([
            "Date", "Status", "Source", "Account", "Merchant",
            "Description", "Amount", "Type", "Category",
        ])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeMode.ResizeToContents)
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

    def set_transactions(self, transactions: list[Transaction]):
        """Set the full list of transactions (from all sources merged)."""
        self._transactions = sorted(
            transactions,
            key=lambda t: t.date or date.min,
            reverse=True,
        )
        self._update_account_filter()
        self._apply_filters()

    def add_transactions(self, transactions: list[Transaction]):
        """Append new transactions and re-sort."""
        seen = {t.transaction_id for t in self._transactions}
        for txn in transactions:
            if txn.transaction_id not in seen:
                self._transactions.append(txn)
                seen.add(txn.transaction_id)
        self._transactions.sort(
            key=lambda t: t.date or date.min, reverse=True,
        )
        self._update_account_filter()
        self._apply_filters()

    def clear_transactions(self):
        self._transactions.clear()
        self._filtered.clear()
        self._table.setRowCount(0)
        self._count_label.setText("0 transactions")
        self._debit_label.setText("Debit: $0")
        self._credit_label.setText("Credit: $0")

    # ── Filtering ─────────────────────────────────────────────────────

    def _update_account_filter(self):
        """Rebuild account dropdown from current transactions."""
        accounts = sorted({
            t.account_name or t.account_id
            for t in self._transactions if t.account_name or t.account_id
        })
        current = self._account_combo.currentText()
        self._account_combo.blockSignals(True)
        self._account_combo.clear()
        self._account_combo.addItem("All")
        self._account_combo.addItems(accounts)
        idx = self._account_combo.findText(current)
        if idx >= 0:
            self._account_combo.setCurrentIndex(idx)
        self._account_combo.blockSignals(False)

    def _apply_filters(self):
        """Filter transactions based on current control values."""
        source_text = self._source_combo.currentText()
        account_text = self._account_combo.currentText()
        status_text = self._status_combo.currentText()
        search_text = self._search_box.text().lower().strip()

        filtered = self._transactions

        if source_text != "All":
            source_enum = TransactionSource.PLAID if source_text == "Plaid" else TransactionSource.FIDELITY
            filtered = [t for t in filtered if t.source == source_enum]

        if account_text != "All":
            filtered = [
                t for t in filtered
                if (t.account_name or t.account_id) == account_text
            ]

        if status_text != "All":
            status_enum = TransactionStatus.PENDING if status_text == "Pending" else TransactionStatus.POSTED
            filtered = [t for t in filtered if t.status == status_enum]

        if search_text:
            filtered = [
                t for t in filtered
                if search_text in (t.merchant_name or "").lower()
                or search_text in (t.name or "").lower()
            ]

        self._filtered = filtered
        self._render_table()

    # ── Rendering ─────────────────────────────────────────────────────

    def _render_table(self):
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(self._filtered))

        total_debit = 0.0
        total_credit = 0.0

        for row, txn in enumerate(self._filtered):
            # Date
            date_str = txn.date.isoformat() if txn.date else ""
            item = QTableWidgetItem(date_str)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, 0, item)

            # Status
            status_str = txn.status.name if txn.status else "POSTED"
            item = QTableWidgetItem(status_str)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if txn.status == TransactionStatus.PENDING:
                item.setForeground(QColor(Colors.WARNING))
            self._table.setItem(row, 1, item)

            # Source
            source_str = txn.source.name if txn.source else ""
            item = QTableWidgetItem(source_str)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, 2, item)

            # Account
            item = QTableWidgetItem(txn.account_name or txn.account_id)
            self._table.setItem(row, 3, item)

            # Merchant
            item = QTableWidgetItem(txn.merchant_name or "")
            self._table.setItem(row, 4, item)

            # Description
            item = QTableWidgetItem(txn.name or "")
            self._table.setItem(row, 5, item)

            # Amount
            amount_str = txn.display_amount
            item = QTableWidgetItem(amount_str)
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            item.setFont(QFont(Fonts.MONO, Fonts.SIZE_NORMAL))
            if txn.is_credit:
                item.setForeground(QColor(Colors.PROFIT))
                total_credit += abs(txn.amount)
            else:
                item.setForeground(QColor(Colors.LOSS))
                total_debit += txn.amount
            self._table.setItem(row, 6, item)

            # Type
            type_str = "Credit" if txn.is_credit else "Debit"
            item = QTableWidgetItem(type_str)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, 7, item)

            # Category
            item = QTableWidgetItem(txn.category or "")
            self._table.setItem(row, 8, item)

        self._table.setSortingEnabled(True)

        # Update summary
        self._count_label.setText(f"{len(self._filtered)} transactions")
        self._debit_label.setText(f"Debit: ${total_debit:,.2f}")
        self._credit_label.setText(f"Credit: ${total_credit:,.2f}")
