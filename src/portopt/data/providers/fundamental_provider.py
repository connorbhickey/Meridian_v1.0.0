"""Fundamental data provider — P/E, P/B, dividend yield, sector, market cap."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class FundamentalData:
    """Fundamental metrics for a single asset."""
    symbol: str
    name: str = ""
    sector: str = ""
    industry: str = ""
    market_cap: float = 0.0
    pe_ratio: float | None = None       # Trailing P/E
    forward_pe: float | None = None     # Forward P/E
    pb_ratio: float | None = None       # Price/Book
    ps_ratio: float | None = None       # Price/Sales
    dividend_yield: float | None = None  # Annual dividend yield (decimal)
    beta: float | None = None
    eps: float | None = None             # Trailing EPS
    revenue: float | None = None
    profit_margin: float | None = None
    roe: float | None = None             # Return on equity
    debt_to_equity: float | None = None
    free_cash_flow: float | None = None
    fifty_two_week_high: float | None = None
    fifty_two_week_low: float | None = None
    avg_volume: float | None = None
    description: str = ""


class FundamentalProvider:
    """Fetches fundamental data for equities via yfinance."""

    def get_fundamentals(self, symbol: str) -> FundamentalData:
        """Fetch fundamental metrics for a single symbol."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}
        except Exception as e:
            logger.warning("Failed to fetch fundamentals for %s: %s", symbol, e)
            return FundamentalData(symbol=symbol)

        return FundamentalData(
            symbol=symbol,
            name=info.get("longName") or info.get("shortName", symbol),
            sector=info.get("sector", ""),
            industry=info.get("industry", ""),
            market_cap=info.get("marketCap", 0.0) or 0.0,
            pe_ratio=_safe_float(info.get("trailingPE")),
            forward_pe=_safe_float(info.get("forwardPE")),
            pb_ratio=_safe_float(info.get("priceToBook")),
            ps_ratio=_safe_float(info.get("priceToSalesTrailing12Months")),
            dividend_yield=_safe_float(info.get("dividendYield")),
            beta=_safe_float(info.get("beta")),
            eps=_safe_float(info.get("trailingEps")),
            revenue=_safe_float(info.get("totalRevenue")),
            profit_margin=_safe_float(info.get("profitMargins")),
            roe=_safe_float(info.get("returnOnEquity")),
            debt_to_equity=_safe_float(info.get("debtToEquity")),
            free_cash_flow=_safe_float(info.get("freeCashflow")),
            fifty_two_week_high=_safe_float(info.get("fiftyTwoWeekHigh")),
            fifty_two_week_low=_safe_float(info.get("fiftyTwoWeekLow")),
            avg_volume=_safe_float(info.get("averageVolume")),
            description=info.get("longBusinessSummary", ""),
        )

    def get_multiple_fundamentals(
        self, symbols: list[str]
    ) -> dict[str, FundamentalData]:
        """Fetch fundamentals for multiple symbols."""
        results = {}
        for sym in symbols:
            results[sym] = self.get_fundamentals(sym)
        return results


def _safe_float(val) -> float | None:
    """Convert to float if possible, else None."""
    if val is None:
        return None
    try:
        f = float(val)
        if f != f:  # NaN check
            return None
        return f
    except (ValueError, TypeError):
        return None
