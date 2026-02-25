"""Tests for transaction-related models: Transaction, PlaidAccount, PlaidItem, enums."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from portopt.data.models import (
    PlaidAccount,
    PlaidAccountType,
    PlaidItem,
    Transaction,
    TransactionSource,
    TransactionStatus,
    TransactionType,
)


# ── TransactionStatus enum ──────────────────────────────────────────

class TestTransactionStatus:
    def test_has_pending(self):
        assert TransactionStatus.PENDING is not None

    def test_has_posted(self):
        assert TransactionStatus.POSTED is not None

    def test_has_removed(self):
        assert TransactionStatus.REMOVED is not None

    def test_all_members(self):
        assert len(TransactionStatus) == 3


# ── TransactionType enum ────────────────────────────────────────────

class TestTransactionType:
    def test_has_debit(self):
        assert TransactionType.DEBIT is not None

    def test_has_credit(self):
        assert TransactionType.CREDIT is not None


# ── PlaidAccountType enum ───────────────────────────────────────────

class TestPlaidAccountType:
    def test_all_types(self):
        expected = {"CHECKING", "SAVINGS", "CREDIT_CARD", "BROKERAGE", "IRA", "CASH_MANAGEMENT", "OTHER"}
        actual = {m.name for m in PlaidAccountType}
        assert actual == expected


# ── TransactionSource enum ──────────────────────────────────────────

class TestTransactionSource:
    def test_plaid(self):
        assert TransactionSource.PLAID is not None

    def test_fidelity(self):
        assert TransactionSource.FIDELITY is not None

    def test_name_access(self):
        assert TransactionSource.PLAID.name == "PLAID"
        assert TransactionSource.FIDELITY.name == "FIDELITY"


# ── Transaction dataclass ───────────────────────────────────────────

class TestTransaction:
    @pytest.fixture
    def debit_txn(self):
        return Transaction(
            transaction_id="txn_001",
            account_id="acct_001",
            account_name="Chase Checking",
            date=date(2025, 3, 15),
            amount=42.50,
            merchant_name="Amazon",
            name="Amazon.com",
            source=TransactionSource.PLAID,
        )

    @pytest.fixture
    def credit_txn(self):
        return Transaction(
            transaction_id="txn_002",
            account_id="acct_002",
            account_name="Chase Credit",
            date=date(2025, 3, 14),
            amount=-150.00,
            merchant_name="",
            name="Payment received",
            source=TransactionSource.FIDELITY,
        )

    def test_display_amount_debit(self, debit_txn):
        assert debit_txn.display_amount == "-$42.50"

    def test_display_amount_credit(self, credit_txn):
        assert credit_txn.display_amount == "+$150.00"

    def test_display_amount_zero(self):
        txn = Transaction(transaction_id="txn_0", account_id="acct_0", amount=0.0)
        assert txn.display_amount == "-$0.00"

    def test_display_amount_large(self):
        txn = Transaction(transaction_id="txn_0", account_id="acct_0", amount=1234567.89)
        assert txn.display_amount == "-$1,234,567.89"

    def test_is_credit_positive(self, debit_txn):
        assert debit_txn.is_credit is False

    def test_is_credit_negative(self, credit_txn):
        assert credit_txn.is_credit is True

    def test_is_credit_zero(self):
        txn = Transaction(transaction_id="txn_0", account_id="acct_0", amount=0.0)
        assert txn.is_credit is False

    def test_default_values(self):
        txn = Transaction(transaction_id="txn_0", account_id="acct_0")
        assert txn.account_name == ""
        assert txn.date is None
        assert txn.authorized_date is None
        assert txn.amount == 0.0
        assert txn.merchant_name == ""
        assert txn.name == ""
        assert txn.category == ""
        assert txn.status == TransactionStatus.POSTED
        assert txn.pending is False
        assert txn.institution_name == ""
        assert txn.source == TransactionSource.PLAID
        assert txn.iso_currency_code == "USD"
        assert txn.metadata == {}

    def test_metadata_mutable_default(self):
        """Ensure metadata default doesn't share between instances."""
        t1 = Transaction(transaction_id="a", account_id="b")
        t2 = Transaction(transaction_id="c", account_id="d")
        t1.metadata["key"] = "val"
        assert "key" not in t2.metadata

    def test_source_plaid(self, debit_txn):
        assert debit_txn.source == TransactionSource.PLAID

    def test_source_fidelity(self, credit_txn):
        assert credit_txn.source == TransactionSource.FIDELITY


# ── PlaidAccount dataclass ──────────────────────────────────────────

class TestPlaidAccount:
    def test_display_name_official(self):
        acct = PlaidAccount(
            account_id="acc1",
            official_name="Platinum Checking",
            mask="1234",
        )
        assert acct.display_name == "Platinum Checking (***1234)"

    def test_display_name_official_no_mask(self):
        acct = PlaidAccount(account_id="acc1", official_name="Platinum Checking")
        assert acct.display_name == "Platinum Checking"

    def test_display_name_name_only(self):
        acct = PlaidAccount(account_id="acc1", name="Checking", mask="5678")
        assert acct.display_name == "Checking (***5678)"

    def test_display_name_name_no_mask(self):
        acct = PlaidAccount(account_id="acc1", name="Checking")
        assert acct.display_name == "Checking"

    def test_display_name_mask_only(self):
        acct = PlaidAccount(account_id="acc1", mask="9999")
        assert acct.display_name == "Account ***9999"

    def test_display_name_fallback_id(self):
        acct = PlaidAccount(account_id="acc1")
        assert acct.display_name == "acc1"

    def test_default_type(self):
        acct = PlaidAccount(account_id="acc1")
        assert acct.account_type == PlaidAccountType.OTHER

    def test_balance_fields(self):
        acct = PlaidAccount(
            account_id="acc1",
            current_balance=1500.50,
            available_balance=1400.00,
            limit=5000.00,
        )
        assert acct.current_balance == 1500.50
        assert acct.available_balance == 1400.00
        assert acct.limit == 5000.00

    def test_nullable_balances(self):
        acct = PlaidAccount(account_id="acc1")
        assert acct.available_balance is None
        assert acct.limit is None


# ── PlaidItem dataclass ─────────────────────────────────────────────

class TestPlaidItem:
    def test_basic_fields(self):
        item = PlaidItem(
            item_id="item_001",
            institution_id="ins_001",
            institution_name="Chase",
        )
        assert item.item_id == "item_001"
        assert item.institution_name == "Chase"

    def test_default_accounts(self):
        item = PlaidItem(item_id="item_001")
        assert item.accounts == []

    def test_accounts_mutable_default(self):
        """Ensure accounts default list doesn't share between instances."""
        i1 = PlaidItem(item_id="a")
        i2 = PlaidItem(item_id="b")
        i1.accounts.append(PlaidAccount(account_id="acc1"))
        assert len(i2.accounts) == 0

    def test_sync_cursor(self):
        item = PlaidItem(item_id="item_001", sync_cursor="cursor_abc")
        assert item.sync_cursor == "cursor_abc"
