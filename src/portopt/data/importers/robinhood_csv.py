"""Robinhood CSV position export parser.

Handles the standard CSV format downloaded from Robinhood's portfolio page.
Robinhood CSV typically has columns like:
  Symbol, Name, Type, Quantity, Average Cost, Current Price,
  Total Return, Equity, Percent Change
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path

from portopt.data.models import Asset, AssetType, Holding, Portfolio

logger = logging.getLogger(__name__)


def _clean_numeric(value: str) -> float:
    """Parse a Robinhood numeric value (handles $, commas, %, --, n/a)."""
    if not value or value.strip() in ("--", "n/a", "N/A", ""):
        return 0.0
    cleaned = (
        value.replace("$", "")
        .replace(",", "")
        .replace("%", "")
        .replace("+", "")
        .strip()
    )
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


_TYPE_MAP: dict[str, AssetType] = {
    "stock": AssetType.STOCK,
    "etf": AssetType.ETF,
    "adr": AssetType.STOCK,
    "crypto": AssetType.CRYPTO,
    "option": AssetType.OPTION,
}


def _detect_asset_type(type_value: str, name: str) -> AssetType:
    """Map Robinhood's Type column to AssetType.

    Falls back to name-based heuristics if the type column is empty
    or contains an unrecognized value.
    """
    asset_type = _TYPE_MAP.get(type_value.strip().lower())
    if asset_type is not None:
        return asset_type

    name_lower = name.lower()
    if "etf" in name_lower:
        return AssetType.ETF
    if "fund" in name_lower or "index" in name_lower:
        return AssetType.MUTUAL_FUND
    if "bond" in name_lower or "treasury" in name_lower:
        return AssetType.BOND

    if type_value.strip():
        logger.warning("Unrecognized Robinhood asset type: %r — defaulting to STOCK", type_value)
    return AssetType.STOCK


def parse_robinhood_csv(file_path: str | Path) -> Portfolio:
    """Parse a Robinhood positions CSV export into a Portfolio.

    Args:
        file_path: Path to the Robinhood CSV file.

    Returns:
        Portfolio with holdings from the Robinhood export.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    # Find the actual CSV header row (skip any preamble lines)
    lines = content.strip().split("\n")
    header_idx = 0
    for i, line in enumerate(lines):
        lower = line.lower()
        if "symbol" in lower and "quantity" in lower:
            header_idx = i
            break

    csv_text = "\n".join(lines[header_idx:])
    reader = csv.DictReader(csv_text.splitlines())

    holdings: list[Holding] = []
    skipped = 0

    for row_num, row in enumerate(reader, start=header_idx + 2):
        # Normalize column names: strip whitespace, lowercase, underscores
        norm_row = {
            k.strip().lower().replace(" ", "_"): v.strip() if v else ""
            for k, v in row.items()
            if k
        }

        symbol = norm_row.get("symbol", "").strip().upper()
        if not symbol:
            skipped += 1
            logger.debug("Row %d: skipping — no symbol", row_num)
            continue

        name = norm_row.get("name", "")
        type_value = norm_row.get("type", "")
        quantity = _clean_numeric(norm_row.get("quantity", "0"))

        if quantity == 0.0:
            skipped += 1
            logger.debug("Row %d: skipping %s — zero quantity", row_num, symbol)
            continue

        avg_cost = _clean_numeric(norm_row.get("average_cost", "0"))
        current_price = _clean_numeric(norm_row.get("current_price", "0"))

        # Robinhood provides "Equity" (market value) directly, but we can also
        # compute it from quantity * current_price. Use Equity as a fallback
        # if current_price is missing.
        if current_price == 0.0:
            equity = _clean_numeric(norm_row.get("equity", "0"))
            if equity > 0.0 and quantity > 0.0:
                current_price = equity / quantity

        # Total cost basis = average cost per share * quantity
        cost_basis = avg_cost * quantity

        asset = Asset(
            symbol=symbol,
            name=name,
            asset_type=_detect_asset_type(type_value, name),
        )

        holdings.append(Holding(
            asset=asset,
            quantity=quantity,
            cost_basis=cost_basis,
            current_price=current_price,
            account="Robinhood",
        ))

    if skipped:
        logger.info("Robinhood CSV: skipped %d rows (empty symbol or zero quantity)", skipped)

    logger.info("Robinhood CSV: imported %d holdings from %s", len(holdings), file_path.name)

    return Portfolio(
        name="Robinhood Import",
        holdings=holdings,
        last_updated=datetime.now(),
    )
