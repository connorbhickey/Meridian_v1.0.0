"""Tests for TransactionPanel."""

from __future__ import annotations

from datetime import date

import pytest

from portopt.data.models import Transaction, TransactionSource, TransactionStatus

pytest.importorskip("PySide6", reason="PySide6 not installed")

from portopt.gui.panels.transaction_panel import TransactionPanel


@pytest.fixture
def sample_transactions():
    """Mixed transactions from Plaid and Fidelity."""
    return [
        Transaction(
            transaction_id="txn_1",
            account_id="acc_1",
            account_name="Chase Checking",
            date=date(2025, 3, 15),
            amount=42.50,
            merchant_name="Starbucks",
            name="STARBUCKS #1234",
            category="Food > Coffee",
            status=TransactionStatus.POSTED,
            institution_name="Chase",
            source=TransactionSource.PLAID,
        ),
        Transaction(
            transaction_id="txn_2",
            account_id="acc_2",
            account_name="BoA Credit",
            date=date(2025, 3, 14),
            amount=-150.00,
            merchant_name="",
            name="Payment received",
            status=TransactionStatus.POSTED,
            institution_name="Bank of America",
            source=TransactionSource.PLAID,
        ),
        Transaction(
            transaction_id="txn_3",
            account_id="fid_1",
            account_name="IRA Z12345678",
            date=date(2025, 3, 13),
            amount=1234.56,
            merchant_name="",
            name="YOU BOUGHT AAPL",
            category="Investment",
            status=TransactionStatus.POSTED,
            institution_name="Fidelity",
            source=TransactionSource.FIDELITY,
        ),
        Transaction(
            transaction_id="txn_4",
            account_id="acc_1",
            account_name="Chase Checking",
            date=date(2025, 3, 16),
            amount=25.00,
            merchant_name="Amazon",
            name="AMAZON.COM",
            category="Shopping",
            status=TransactionStatus.PENDING,
            pending=True,
            institution_name="Chase",
            source=TransactionSource.PLAID,
        ),
    ]


class TestTransactionPanel:
    @pytest.fixture
    def panel(self, qtbot):
        p = TransactionPanel()
        qtbot.addWidget(p)
        return p

    def test_panel_id(self, panel):
        assert panel.panel_id == "transactions"

    def test_panel_title(self, panel):
        assert panel.panel_title == "TRANSACTIONS"

    def test_set_transactions(self, panel, sample_transactions):
        panel.set_transactions(sample_transactions)
        # All 4 should be in the internal list
        assert len(panel._transactions) == 4

    def test_add_transactions_merges(self, panel, sample_transactions):
        panel.set_transactions(sample_transactions[:2])
        panel.add_transactions(sample_transactions[2:])
        assert len(panel._transactions) == 4

    def test_clear_transactions(self, panel, sample_transactions):
        panel.set_transactions(sample_transactions)
        panel.clear_transactions()
        assert len(panel._transactions) == 0

    def test_table_has_correct_columns(self, panel):
        col_count = panel._table.columnCount()
        assert col_count == 9  # Date, Status, Source, Account, Merchant, Description, Amount, Type, Category

    def test_set_transactions_populates_table(self, panel, sample_transactions):
        panel.set_transactions(sample_transactions)
        # Table should show all transactions (no filter active)
        assert panel._table.rowCount() >= 1


class TestTransactionPanelFilters:
    @pytest.fixture
    def panel(self, qtbot, sample_transactions):
        p = TransactionPanel()
        qtbot.addWidget(p)
        p.set_transactions(sample_transactions)
        return p

    def test_source_filter_all(self, panel):
        panel._source_combo.setCurrentText("All")
        panel._apply_filters()
        assert panel._table.rowCount() == 4

    def test_source_filter_plaid(self, panel):
        panel._source_combo.setCurrentText("Plaid")
        panel._apply_filters()
        assert panel._table.rowCount() == 3  # 3 Plaid transactions

    def test_source_filter_fidelity(self, panel):
        panel._source_combo.setCurrentText("Fidelity")
        panel._apply_filters()
        assert panel._table.rowCount() == 1  # 1 Fidelity transaction

    def test_status_filter_pending(self, panel):
        panel._status_combo.setCurrentText("Pending")
        panel._apply_filters()
        assert panel._table.rowCount() == 1  # 1 pending

    def test_status_filter_posted(self, panel):
        panel._status_combo.setCurrentText("Posted")
        panel._apply_filters()
        assert panel._table.rowCount() == 3  # 3 posted

    def test_search_filter(self, panel):
        panel._search_box.setText("starbucks")
        panel._apply_filters()
        # Should match merchant_name or name containing "starbucks" (case-insensitive)
        assert panel._table.rowCount() == 1
