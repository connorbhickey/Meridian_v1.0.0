"""Charles Schwab CSV position export parser.

Handles the standard CSV format downloaded from Schwab's Positions page
("All-Accounts-Positions-*.csv").  Schwab CSVs may contain multiple account
sections, each preceded by a header line like:

    "Positions for account XXXX-1234 as of 02/15/2026"

Columns:
  Symbol, Description, Quantity, Price, Price Change %, Price Change $,
  Market Value, Day Change %, Day Change $, Cost Basis, Gain/Loss %,
  Gain/Loss $, Ratings, Reinvest Dividends?, Capital Gains?,
  % Of Account, Security Type
"""

from __future__ import annotations

import csv
import io
import logging
import re
from datetime import datetime
from pathlib import Path

from portopt.data.models import (
    AccountSummary, Asset, AssetType, Holding, Portfolio,
)

logger = logging.getLogger(__name__)

_ACCOUNT_HEADER_RE = re.compile(
    r"Positions\s+for\s+account\s+(.+?)\s+as\s+of\s+(\d{2}/\d{2}/\d{4})",
    re.IGNORECASE,
)

_SECURITY_TYPE_MAP: dict[str, AssetType] = {
    "equity":                    AssetType.STOCK,
    "etf":                       AssetType.ETF,
    "mutual fund":               AssetType.MUTUAL_FUND,
    "fixed income":              AssetType.BOND,
    "option":                    AssetType.OPTION,
    "cash & cash equivalents":   AssetType.MONEY_MARKET,
}

_MONEY_MARKET_KEYWORDS = frozenset({
    "schwab value advantage money",
    "schwab money",
    "money market",
    "cash",
})

_SKIP_SYMBOLS = frozenset({"", "CASH", "PENDING", "TOTAL"})


def _clean_numeric(value: str) -> float:
    """Parse a Schwab numeric value (handles $, commas, %, --, N/A)."""
    if not value or value.strip() in ("--", "n/a", "N/A", ""):
        return 0.0
    cleaned = value.replace("$", "").replace(",", "").replace("%", "").replace("+", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def _detect_asset_type(symbol: str, description: str, security_type: str) -> AssetType:
    """Map Schwab Security Type column to AssetType, with fallback heuristics."""
    mapped = _SECURITY_TYPE_MAP.get(security_type.lower().strip())
    if mapped is not None:
        return mapped

    desc_lower = description.lower()
    for keyword in _MONEY_MARKET_KEYWORDS:
        if keyword in desc_lower:
            return AssetType.MONEY_MARKET

    if "etf" in desc_lower:
        return AssetType.ETF
    if "fund" in desc_lower or "index" in desc_lower:
        return AssetType.MUTUAL_FUND
    if "bond" in desc_lower or "treasury" in desc_lower:
        return AssetType.BOND

    return AssetType.STOCK


def _is_summary_row(symbol: str, description: str) -> bool:
    """Return True for total/summary rows that should be skipped.

    Only checks the *symbol* field for summary keywords so that fund names
    containing 'total' (e.g. 'VANGUARD TOTAL STOCK MKT ETF') are not
    mistakenly excluded.
    """
    sym_lower = symbol.lower().strip()
    return sym_lower in ("account total", "grand total", "total")


def _split_account_sections(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Split raw lines into (account_id, csv_lines) sections.

    Each section starts with a "Positions for account ..." header line.
    If no such header is found the entire file is treated as a single
    unnamed section.
    """
    sections: list[tuple[str, list[str]]] = []
    current_account = ""
    current_lines: list[str] = []

    for line in lines:
        match = _ACCOUNT_HEADER_RE.search(line)
        if match:
            if current_lines:
                sections.append((current_account, current_lines))
            current_account = match.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_account, current_lines))

    return sections


def _find_csv_header(lines: list[str]) -> int:
    """Return the index of the CSV header row containing 'Symbol'."""
    for i, line in enumerate(lines):
        if "symbol" in line.lower() and "description" in line.lower():
            return i
    return -1


def _parse_section(
    account_id: str,
    csv_lines: list[str],
    holdings: list[Holding],
    accounts: dict[str, AccountSummary],
) -> None:
    """Parse a single account section and append results to holdings/accounts."""
    header_idx = _find_csv_header(csv_lines)
    if header_idx < 0:
        logger.warning("No CSV header found for account %r — skipping section", account_id)
        return

    csv_text = "\n".join(csv_lines[header_idx:])
    reader = csv.DictReader(io.StringIO(csv_text))

    for row_num, row in enumerate(reader, start=1):
        norm = {
            k.strip().lower().replace(" ", "_"): v.strip() if v else ""
            for k, v in row.items()
            if k
        }

        symbol = norm.get("symbol", "").strip().upper()
        if not symbol or symbol.upper() in _SKIP_SYMBOLS:
            continue

        description = norm.get("description", "")
        security_type = norm.get("security_type", "")

        if _is_summary_row(symbol, description):
            continue

        quantity = _clean_numeric(norm.get("quantity", "0"))
        price = _clean_numeric(norm.get("price", "0"))
        market_value = _clean_numeric(norm.get("market_value", "0"))
        cost_basis = _clean_numeric(norm.get("cost_basis", "0"))

        if quantity == 0.0 and market_value == 0.0:
            logger.debug("Skipping zero-quantity, zero-value row: %s", symbol)
            continue

        asset_type = _detect_asset_type(symbol, description, security_type)
        asset = Asset(
            symbol=symbol,
            name=description,
            asset_type=asset_type,
        )

        holdings.append(Holding(
            asset=asset,
            quantity=quantity,
            cost_basis=cost_basis,
            current_price=price,
            account=account_id,
        ))

        # Accumulate account summary
        if account_id and account_id not in accounts:
            accounts[account_id] = AccountSummary(
                account_id=account_id,
                account_name=account_id,
                total_value=0.0,
                holdings_count=0,
                last_updated=datetime.now(),
            )
        if account_id:
            accounts[account_id].total_value += market_value or (quantity * price)
            accounts[account_id].holdings_count += 1


def parse_schwab_csv(file_path: str | Path) -> Portfolio:
    """Parse a Charles Schwab positions CSV export into a Portfolio.

    Args:
        file_path: Path to the Schwab CSV file

    Returns:
        Portfolio with holdings and account summaries
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    lines = content.strip().split("\n")
    sections = _split_account_sections(lines)

    if not sections:
        raise ValueError(f"No parseable data found in {file_path}")

    holdings: list[Holding] = []
    accounts: dict[str, AccountSummary] = {}

    for account_id, csv_lines in sections:
        _parse_section(account_id, csv_lines, holdings, accounts)

    if not holdings:
        logger.warning("No holdings parsed from %s", file_path)

    return Portfolio(
        name="Schwab (CSV Import)",
        holdings=holdings,
        accounts=list(accounts.values()),
        last_updated=datetime.now(),
    )
