"""Prediction data provider — replaces Anthropic API with native providers.

Assembles the ~50-field data dict required by run_prediction() using:
  - YFinance: price history, fundamentals, technicals, analyst targets
  - FRED: fed funds rate, VIX, yield curve, credit spread
  - Computed: RSI-14, SMA-50/200, annualized vol, performance metrics

Latency: ~2-3s (vs 10-15s via Anthropic API).
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_prediction_data(
    symbol: str,
    horizon_days: int = 252,
    fred_provider=None,
) -> dict:
    """Build the complete data dict for run_prediction().

    Args:
        symbol: Ticker symbol (e.g. 'AAPL', 'SPY')
        horizon_days: Prediction horizon in trading days (default: 1 year)
        fred_provider: Optional FredProvider instance for macro data.
            If None, uses sensible defaults.

    Returns:
        Dict with all fields needed by run_prediction().
    """
    ticker = yf.Ticker(symbol)

    # ── YFinance info ──
    try:
        info = ticker.info or {}
    except Exception as e:
        logger.warning("Failed to fetch info for %s: %s", symbol, e)
        info = {}

    # ── Price history (1 year) ──
    try:
        hist = ticker.history(period="1y")
        if hist.empty:
            hist = ticker.history(period="6mo")
    except Exception:
        hist = pd.DataFrame()

    current_price = _safe(info.get("currentPrice")) or _safe(info.get("regularMarketPrice"))
    if current_price is None and not hist.empty:
        current_price = float(hist["Close"].iloc[-1])
    if current_price is None:
        current_price = 100.0  # fallback

    # ── Technicals from history ──
    close = hist["Close"] if not hist.empty else pd.Series(dtype=float)
    sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
    rsi14 = _compute_rsi(close, 14) if len(close) >= 15 else 50.0

    # Annualized vol from daily returns
    if len(close) >= 20:
        returns = close.pct_change().dropna()
        ann_vol = float(returns.std() * math.sqrt(252))
    else:
        ann_vol = 0.30  # default

    # 52-week range
    high52w = _safe(info.get("fiftyTwoWeekHigh")) or (float(close.max()) if not close.empty else current_price * 1.2)
    low52w = _safe(info.get("fiftyTwoWeekLow")) or (float(close.min()) if not close.empty else current_price * 0.8)

    # 3-month performance
    stock_perf_3m = 0.0
    if len(close) >= 63:
        stock_perf_3m = (float(close.iloc[-1]) / float(close.iloc[-63]) - 1) * 100

    # ── Fundamentals ──
    pe_ratio = _safe(info.get("trailingPE")) or 20.0
    forward_pe = _safe(info.get("forwardPE"))
    pb_ratio = _safe(info.get("priceToBook")) or 3.0
    beta = _safe(info.get("beta")) or 1.0
    div_yield = (_safe(info.get("dividendYield")) or 0.0) * 100  # to percent
    eps = _safe(info.get("trailingEps"))
    forward_eps = _safe(info.get("forwardEps"))
    profit_margin = (_safe(info.get("profitMargins")) or 0.10) * 100  # to percent
    roe = (_safe(info.get("returnOnEquity")) or 0.15) * 100  # to percent
    de_ratio = _safe(info.get("debtToEquity"))
    if de_ratio is not None:
        de_ratio = de_ratio / 100  # yfinance returns as percentage
    else:
        de_ratio = 0.8
    fcf = _safe(info.get("freeCashflow")) or 0.0
    market_cap = _safe(info.get("marketCap")) or 0.0
    market_cap_b = market_cap / 1e9 if market_cap > 0 else 50.0

    # FCF yield
    fcf_yield = 0.0
    if market_cap > 0 and fcf != 0:
        fcf_yield = fcf / market_cap * 100

    # ── Analyst targets ──
    analyst_avg = _safe(info.get("targetMeanPrice"))
    analyst_high = _safe(info.get("targetHighPrice"))
    analyst_low = _safe(info.get("targetLowPrice"))

    # ── Earnings dates ──
    next_earnings = None
    try:
        cal = ticker.calendar
        if cal is not None:
            if isinstance(cal, dict) and "Earnings Date" in cal:
                ed = cal["Earnings Date"]
                if isinstance(ed, list) and len(ed) > 0:
                    next_earnings = str(ed[0])
                elif isinstance(ed, (str, datetime)):
                    next_earnings = str(ed)
            elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
                next_earnings = str(cal.loc["Earnings Date"].iloc[0])
    except Exception:
        pass

    # Earnings surprise (from earnings history)
    last_surprise = 0.0
    days_since = 45
    try:
        earnings_hist = ticker.earnings_dates
        if earnings_hist is not None and not earnings_hist.empty:
            # Look for most recent past earnings
            now = pd.Timestamp.now(tz="UTC")
            past = earnings_hist[earnings_hist.index <= now]
            if not past.empty:
                latest = past.iloc[0]
                surprise_col = "Surprise(%)"
                if surprise_col in past.columns:
                    val = _safe(latest.get(surprise_col))
                    if val is not None:
                        last_surprise = val
                days_since = (now - past.index[0]).days
    except Exception:
        pass

    # ── ETF detection ──
    quote_type = info.get("quoteType", "").upper()
    is_etf = quote_type == "ETF"
    expense_ratio = (_safe(info.get("annualReportExpenseRatio")) or 0.0) * 100

    # ── FRED macro data ──
    fed_funds = 4.5
    vix_val = 18.0
    yield_curve = 0.2
    credit_spread = 3.5

    if fred_provider and fred_provider.available:
        try:
            start = date.today() - timedelta(days=30)
            for series_id, default, setter in [
                ("DFF", 4.5, "fed_funds"),
                ("VIXCLS", 18.0, "vix"),
                ("T10Y2Y", 0.2, "yield_curve"),
                ("BAMLH0A0HYM2", 3.5, "credit_spread"),
            ]:
                try:
                    val = fred_provider.get_current_price(series_id)
                    if val and val > 0:
                        if setter == "fed_funds":
                            fed_funds = val
                        elif setter == "vix":
                            vix_val = val
                        elif setter == "yield_curve":
                            yield_curve = val
                        elif setter == "credit_spread":
                            credit_spread = val
                except Exception:
                    pass
        except Exception as e:
            logger.debug("FRED data fetch skipped: %s", e)

    # ── Sector data (defaults for unavailable fields) ──
    sector = info.get("sector", "Technology")
    sector_avg_pe = _sector_avg_pe(sector)

    # ── Assemble output dict ──
    return {
        "symbol": symbol.upper(),
        "currentPrice": current_price,
        "annualizedVol": ann_vol,
        "beta": beta,
        "tradingDaysRemaining": horizon_days,

        # Technicals
        "sma50": sma50,
        "sma200": sma200,
        "rsi14": rsi14,
        "high52w": high52w,
        "low52w": low52w,
        "stockPerformance3m": stock_perf_3m,

        # Fundamentals
        "peRatio": pe_ratio,
        "forwardPE": forward_pe,
        "priceToBook": pb_ratio,
        "forwardEps": forward_eps,
        "forwardEps2": forward_eps * 1.08 if forward_eps else None,  # estimated FY2
        "roe": roe,
        "profitMargin": profit_margin,
        "debtToEquity": de_ratio,
        "fcfYield": fcf_yield,
        "marketCapNum": market_cap_b,
        "dividendYield": div_yield,
        "expenseRatio": expense_ratio,

        # Analyst
        "analystAvgPt": analyst_avg,
        "analystHigh": analyst_high,
        "analystLow": analyst_low,
        "recentAnalystAvg": analyst_avg,  # best available proxy

        # Earnings
        "nextEarningsDate": next_earnings,
        "lastEarningsSurprise": last_surprise,
        "daysSinceEarnings": days_since,
        "historicalEarningsMoveAvg": 6.0,  # default

        # Macro (FRED or defaults)
        "fedFundsRate": fed_funds,
        "vix": vix_val,
        "yieldCurve2s10s": yield_curve,
        "creditSpreadHY": credit_spread,

        # Sector
        "sectorAvgPE": sector_avg_pe,
        "sectorAvgPB": 3.0,  # broad default
        "sectorPerformance3m": 0.0,  # neutral default

        # Defaults for rarely-available fields
        "shortInterest": _safe(info.get("shortPercentOfFloat", 3)) or 3.0,
        "putCallRatio": 0.7,
        "institutionalOwnership": (_safe(info.get("heldPercentInstitutions")) or 0.65) * 100,
        "earningsRevision3m": 0.0,
        "capexToRevenue": 8.0,
        "ivRank": 50.0,
        "ivSkew": 0.0,
        "insiderNetBuying": 0.0,
        "revenueGrowthPct": (_safe(info.get("revenueGrowth")) or 0.08) * 100,
        "priorRevenueGrowthPct": 8.0,
        "interestCoverage": 10.0,
        "shareCountChange": 0.0,

        # Metadata
        "isETF": is_etf,
        "sector": sector,
        "name": info.get("longName") or info.get("shortName", symbol),
    }


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _safe(val) -> float | None:
    """Convert to float, returning None for NaN / missing."""
    if val is None:
        return None
    try:
        f = float(val)
        if f != f:  # NaN
            return None
        return f
    except (ValueError, TypeError):
        return None


def _compute_rsi(close: pd.Series, period: int = 14) -> float:
    """Compute RSI-14 from close prices."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()

    rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 100.0
    return round(100 - 100 / (1 + rs), 1)


def _sector_avg_pe(sector: str) -> float:
    """Approximate sector-average P/E ratios."""
    averages = {
        "Technology": 28.0,
        "Communication Services": 22.0,
        "Consumer Cyclical": 22.0,
        "Consumer Defensive": 23.0,
        "Healthcare": 25.0,
        "Financial Services": 14.0,
        "Industrials": 20.0,
        "Energy": 12.0,
        "Basic Materials": 16.0,
        "Real Estate": 35.0,
        "Utilities": 18.0,
    }
    return averages.get(sector, 20.0)
