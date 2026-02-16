"""Generic CSV portfolio importer.

Supports any CSV with at minimum a 'symbol' column and optionally
'quantity', 'price', 'cost_basis', 'account' columns.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path

from portopt.data.models import Asset, AssetType, Holding, Portfolio

logger = logging.getLogger(__name__)

# Column name aliases (all lowercase)
SYMBOL_ALIASES = {"symbol", "ticker", "sym", "stock", "security"}
QTY_ALIASES = {"quantity", "qty", "shares", "amount", "units"}
PRICE_ALIASES = {"price", "last_price", "current_price", "close", "last"}
COST_ALIASES = {"cost_basis", "cost", "avg_cost", "purchase_price"}
NAME_ALIASES = {"name", "description", "security_name", "company"}
ACCOUNT_ALIASES = {"account", "account_name", "portfolio"}


def _find_column(headers: list[str], aliases: set[str]) -> str | None:
    """Find the first matching column header from a set of aliases."""
    for h in headers:
        if h.lower().strip().replace(" ", "_") in aliases:
            return h
    return None


def parse_generic_csv(file_path: str | Path) -> Portfolio:
    """Parse a generic portfolio CSV into a Portfolio object."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        sym_col = _find_column(headers, SYMBOL_ALIASES)
        if not sym_col:
            raise ValueError(f"CSV must have a symbol column. Found headers: {headers}")

        qty_col = _find_column(headers, QTY_ALIASES)
        price_col = _find_column(headers, PRICE_ALIASES)
        cost_col = _find_column(headers, COST_ALIASES)
        name_col = _find_column(headers, NAME_ALIASES)
        acct_col = _find_column(headers, ACCOUNT_ALIASES)

        holdings = []
        for row in reader:
            symbol = row.get(sym_col, "").strip().upper()
            if not symbol:
                continue

            quantity = float(row.get(qty_col, "0").replace(",", "") or "0") if qty_col else 1.0
            price = float(row.get(price_col, "0").replace(",", "").replace("$", "") or "0") if price_col else 0.0
            cost = float(row.get(cost_col, "0").replace(",", "").replace("$", "") or "0") if cost_col else 0.0
            name = row.get(name_col, symbol) if name_col else symbol
            account = row.get(acct_col, "") if acct_col else ""

            asset = Asset(symbol=symbol, name=name, asset_type=AssetType.STOCK)
            holdings.append(Holding(
                asset=asset,
                quantity=quantity,
                cost_basis=cost,
                current_price=price,
                account=account,
            ))

    return Portfolio(
        name=f"Import: {file_path.stem}",
        holdings=holdings,
        last_updated=datetime.now(),
    )
