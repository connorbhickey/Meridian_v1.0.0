"""OFX/QFX investment position file parser.

Handles the OFX (Open Financial Exchange) and QFX (Quicken) file formats
exported by brokerages such as TD Ameritrade, Vanguard, E*Trade, Schwab, etc.

OFX is SGML-based (not strict XML). Tags may be unclosed, attributes are
unquoted, and nesting is implied by document structure. This module uses
regex-based extraction rather than an XML parser.

Relevant OFX sections:
  INVPOSLIST  — investment positions (POSSTOCK, POSMF, POSDEBT, POSOPT, POSOTHER)
  SECLIST     — security definitions that map CUSIPs to tickers and names
  INVACCTFROM — brokerage account identification (broker ID + account ID)
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

from portopt.data.models import (
    AccountSummary, Asset, AssetType, Holding, Portfolio,
)

logger = logging.getLogger(__name__)

# Position tag -> AssetType mapping
_POSITION_TYPES: dict[str, AssetType] = {
    "POSSTOCK": AssetType.STOCK,
    "POSMF": AssetType.MUTUAL_FUND,
    "POSDEBT": AssetType.BOND,
    "POSOPT": AssetType.OPTION,
    "POSOTHER": AssetType.OTHER,
}

# Security info tag -> AssetType mapping
_SECURITY_TYPES: dict[str, AssetType] = {
    "STOCKINFO": AssetType.STOCK,
    "MFINFO": AssetType.MUTUAL_FUND,
    "DEBTINFO": AssetType.BOND,
    "OPTINFO": AssetType.OPTION,
    "OTHERINFO": AssetType.OTHER,
}


def _read_ofx_file(file_path: Path) -> str:
    """Read an OFX file, trying UTF-8 first then Latin-1 as fallback."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return file_path.read_text(encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Cannot decode OFX file with UTF-8 or Latin-1: {file_path}")


def _extract_tag_value(block: str, tag: str) -> str:
    """Extract the value following an OFX tag within a block.

    OFX tag values appear as ``<TAG>value`` where value extends to the next
    ``<`` character or end of line. Handles both closed (``<TAG>value</TAG>``)
    and unclosed (``<TAG>value\\n``) forms.
    """
    pattern = rf"<{tag}>([^<\r\n]+)"
    match = re.search(pattern, block, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _split_blocks(content: str, tag: str) -> list[str]:
    """Split content into blocks delimited by the given OFX open tag.

    Returns the text from each ``<TAG>`` to the next occurrence of the same
    tag or to the end of the content.
    """
    pattern = rf"<{tag}>"
    parts = re.split(pattern, content, flags=re.IGNORECASE)
    # First element is text before the first occurrence; discard it
    return parts[1:] if len(parts) > 1 else []


def _parse_ofx_datetime(value: str) -> datetime | None:
    """Parse an OFX datetime string like ``20260215120000`` or ``20260215``."""
    digits = re.sub(r"[^\d]", "", value)
    if len(digits) >= 14:
        try:
            return datetime.strptime(digits[:14], "%Y%m%d%H%M%S")
        except ValueError:
            pass
    if len(digits) >= 8:
        try:
            return datetime.strptime(digits[:8], "%Y%m%d")
        except ValueError:
            pass
    return None


def _parse_security_list(content: str) -> dict[str, tuple[str, str, AssetType]]:
    """Build a CUSIP -> (ticker, name, asset_type) mapping from the SECLIST.

    Searches for STOCKINFO, MFINFO, DEBTINFO, OPTINFO, and OTHERINFO blocks,
    each of which contains a SECINFO sub-block with UNIQUEID (CUSIP), TICKER,
    and SECNAME tags.
    """
    sec_map: dict[str, tuple[str, str, AssetType]] = {}

    for sec_tag, asset_type in _SECURITY_TYPES.items():
        for block in _split_blocks(content, sec_tag):
            cusip = _extract_tag_value(block, "UNIQUEID")
            if not cusip:
                continue

            ticker = _extract_tag_value(block, "TICKER")
            name = _extract_tag_value(block, "SECNAME")

            # Refine type: mutual fund tickers ending in X that contain "ETF"
            # in the name are more likely ETFs
            resolved_type = asset_type
            if asset_type == AssetType.MUTUAL_FUND and name:
                if "etf" in name.lower():
                    resolved_type = AssetType.ETF

            sec_map[cusip] = (ticker, name, resolved_type)
            logger.debug("Security: CUSIP=%s ticker=%s name=%s type=%s",
                         cusip, ticker, name, resolved_type)

    return sec_map


def _parse_positions(
    content: str,
    sec_map: dict[str, tuple[str, str, AssetType]],
) -> list[Holding]:
    """Extract holdings from the INVPOSLIST section.

    Parses POSSTOCK, POSMF, POSDEBT, POSOPT, and POSOTHER blocks. Each
    contains an INVPOS sub-block with SECID/UNIQUEID, UNITS, UNITPRICE,
    and MKTVAL tags.
    """
    holdings: list[Holding] = []

    for pos_tag, default_type in _POSITION_TYPES.items():
        for block in _split_blocks(content, pos_tag):
            cusip = _extract_tag_value(block, "UNIQUEID")
            if not cusip:
                logger.warning("Position block <%s> has no UNIQUEID, skipping", pos_tag)
                continue

            units_str = _extract_tag_value(block, "UNITS")
            price_str = _extract_tag_value(block, "UNITPRICE")
            mktval_str = _extract_tag_value(block, "MKTVAL")

            units = _safe_float(units_str)
            price = _safe_float(price_str)
            mktval = _safe_float(mktval_str)

            # Use security list for ticker/name/type; fall back to CUSIP
            if cusip in sec_map:
                ticker, name, asset_type = sec_map[cusip]
            else:
                logger.warning(
                    "No security info for CUSIP %s — using CUSIP as symbol", cusip,
                )
                ticker = cusip
                name = ""
                asset_type = default_type

            if not ticker:
                ticker = cusip

            # If market value is present but units/price are missing, skip
            if units == 0.0 and mktval == 0.0:
                logger.warning("Position %s has zero units and zero market value, skipping", ticker)
                continue

            # Derive price from market value if price tag is absent
            if price == 0.0 and units != 0.0 and mktval != 0.0:
                price = mktval / units

            asset = Asset(
                symbol=ticker.upper(),
                name=name,
                asset_type=asset_type,
            )

            holdings.append(Holding(
                asset=asset,
                quantity=units,
                cost_basis=0.0,
                current_price=price,
            ))

    return holdings


def _parse_accounts(content: str) -> list[AccountSummary]:
    """Extract account summaries from INVACCTFROM sections."""
    accounts: list[AccountSummary] = []

    for block in _split_blocks(content, "INVACCTFROM"):
        broker_id = _extract_tag_value(block, "BROKERID")
        acct_id = _extract_tag_value(block, "ACCTID")
        acct_type = _extract_tag_value(block, "ACCTTYPE")

        if not acct_id:
            continue

        display_name = f"{broker_id} - {acct_id}" if broker_id else acct_id

        accounts.append(AccountSummary(
            account_id=acct_id,
            account_name=display_name,
            account_type=acct_type,
            last_updated=datetime.now(),
        ))

    return accounts


def _safe_float(value: str, default: float = 0.0) -> float:
    """Parse a numeric string, returning a default on failure."""
    if not value:
        return default
    cleaned = value.strip().replace(",", "")
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        logger.warning("Could not parse numeric value: %r", value)
        return default


def _extract_price_date(content: str) -> datetime | None:
    """Find the most recent DTPRICEASOF in the file for the portfolio timestamp."""
    dates: list[datetime] = []
    for match in re.finditer(r"<DTPRICEASOF>([^<\r\n]+)", content, re.IGNORECASE):
        parsed = _parse_ofx_datetime(match.group(1).strip())
        if parsed:
            dates.append(parsed)
    return max(dates) if dates else None


def parse_ofx_file(file_path: str | Path) -> Portfolio:
    """Parse an OFX/QFX investment position file into a Portfolio.

    Extracts security definitions (SECLIST), position data (INVPOSLIST),
    and account info (INVACCTFROM) from the OFX SGML structure.

    Args:
        file_path: Path to the .ofx or .qfx file.

    Returns:
        Portfolio with holdings, account summaries, and computed weights.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be decoded.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"OFX file not found: {file_path}")

    content = _read_ofx_file(file_path)
    logger.info("Parsing OFX file: %s (%d bytes)", file_path.name, len(content))

    # Build CUSIP -> (ticker, name, type) lookup from the security list
    sec_map = _parse_security_list(content)
    logger.info("Found %d securities in SECLIST", len(sec_map))

    # Extract positions
    holdings = _parse_positions(content, sec_map)
    logger.info("Parsed %d holdings from INVPOSLIST", len(holdings))

    if not holdings:
        logger.warning("No holdings found in OFX file: %s", file_path.name)

    # Extract account info
    accounts = _parse_accounts(content)

    # Update account summaries with computed totals
    total_value = sum(h.market_value for h in holdings)
    for acct in accounts:
        acct.total_value = total_value
        acct.holdings_count = len(holdings)

    # Use the most recent price date as portfolio timestamp, or now
    price_date = _extract_price_date(content)
    last_updated = price_date or datetime.now()

    suffix = file_path.suffix.upper().lstrip(".")
    portfolio_name = f"OFX Import ({suffix})" if suffix in ("OFX", "QFX") else "OFX Import"

    return Portfolio(
        name=portfolio_name,
        holdings=holdings,
        accounts=accounts,
        last_updated=last_updated,
    )
