"""Fidelity CSV position export parser.

Handles the standard CSV format downloaded from Fidelity's Positions page.
Fidelity CSV typically has columns like:
  Account Number, Account Name, Symbol, Description, Quantity, Last Price,
  Last Price Change, Current Value, Today's Gain/Loss Dollar, ...
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path

from portopt.data.models import (
    AccountSummary, Asset, AssetType, Holding, Portfolio,
)

logger = logging.getLogger(__name__)


def _clean_numeric(value: str) -> float:
    """Parse a Fidelity numeric value (handles $, commas, --, n/a)."""
    if not value or value.strip() in ("--", "n/a", "N/A", ""):
        return 0.0
    cleaned = value.replace("$", "").replace(",", "").replace("+", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


_MONEY_MARKET_SYMBOLS = {"SPAXX**", "FCASH**", "FDRXX**", "FZFXX**", "CORE**"}


def _detect_asset_type(symbol: str, description: str) -> AssetType:
    if symbol in _MONEY_MARKET_SYMBOLS:
        return AssetType.MONEY_MARKET
    desc_lower = description.lower()
    if "etf" in desc_lower:
        return AssetType.ETF
    if "fund" in desc_lower or "index" in desc_lower:
        return AssetType.MUTUAL_FUND
    if "bond" in desc_lower or "treasury" in desc_lower:
        return AssetType.BOND
    return AssetType.STOCK


def parse_fidelity_csv(file_path: str | Path) -> Portfolio:
    """Parse a Fidelity positions CSV export into a Portfolio.

    Args:
        file_path: Path to the Fidelity CSV file

    Returns:
        Portfolio with holdings and account summaries
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    holdings = []
    accounts: dict[str, AccountSummary] = {}

    with open(file_path, "r", encoding="utf-8-sig") as f:
        # Fidelity CSVs sometimes have a header line before the data
        content = f.read()

    # Find the actual CSV header row
    lines = content.strip().split("\n")
    header_idx = 0
    for i, line in enumerate(lines):
        lower = line.lower()
        if "account" in lower and "symbol" in lower:
            header_idx = i
            break

    # Parse from header row onwards
    csv_text = "\n".join(lines[header_idx:])
    reader = csv.DictReader(csv_text.splitlines())

    for row in reader:
        # Normalize column names (Fidelity uses various formats)
        norm_row = {k.strip().lower().replace(" ", "_"): v.strip() if v else "" for k, v in row.items() if k}

        symbol = norm_row.get("symbol", "").strip().upper()
        if not symbol or symbol in ("CASH", "PENDING"):
            continue

        # Account info
        acct_num = norm_row.get("account_number", norm_row.get("account_name/number", "")).strip()
        acct_name = norm_row.get("account_name", "").strip()
        if not acct_num and "/" in norm_row.get("account_name/number", ""):
            parts = norm_row["account_name/number"].split("/")
            acct_name = parts[0].strip()
            acct_num = parts[-1].strip()

        description = norm_row.get("description", norm_row.get("security_description", ""))
        quantity = _clean_numeric(norm_row.get("quantity", "0"))
        last_price = _clean_numeric(norm_row.get("last_price", norm_row.get("current_price", "0")))
        current_value = _clean_numeric(norm_row.get("current_value", norm_row.get("value", "0")))
        cost_basis = _clean_numeric(norm_row.get("cost_basis_total", norm_row.get("cost_basis", "0")))

        asset = Asset(
            symbol=symbol,
            name=description,
            asset_type=_detect_asset_type(symbol, description),
        )

        holdings.append(Holding(
            asset=asset,
            quantity=quantity,
            cost_basis=cost_basis,
            current_price=last_price,
            account=acct_name or acct_num,
        ))

        # Accumulate account summaries
        key = acct_num or acct_name
        if key and key not in accounts:
            accounts[key] = AccountSummary(
                account_id=acct_num,
                account_name=acct_name,
                total_value=0.0,
                holdings_count=0,
                last_updated=datetime.now(),
            )
        if key:
            accounts[key].total_value += current_value
            accounts[key].holdings_count += 1

    return Portfolio(
        name="Fidelity (CSV Import)",
        holdings=holdings,
        accounts=list(accounts.values()),
        last_updated=datetime.now(),
    )
