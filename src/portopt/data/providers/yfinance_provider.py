"""Yahoo Finance data provider using yfinance."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from portopt.data.models import Asset, AssetType
from portopt.data.providers.base import BaseDataProvider

logger = logging.getLogger(__name__)


_TYPE_MAP = {
    "EQUITY": AssetType.STOCK,
    "ETF": AssetType.ETF,
    "MUTUALFUND": AssetType.MUTUAL_FUND,
    "CRYPTOCURRENCY": AssetType.CRYPTO,
    "BOND": AssetType.BOND,
    "INDEX": AssetType.OTHER,
}


class YFinanceProvider(BaseDataProvider):
    """Data provider backed by Yahoo Finance (yfinance)."""

    def get_prices(
        self, symbol: str, start: date, end: date | None = None
    ) -> pd.DataFrame:
        end = end or date.today()
        # yfinance end date is exclusive, so add one day
        end_adj = end + timedelta(days=1)
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start.isoformat(), end=end_adj.isoformat(), auto_adjust=False)
        if df.empty:
            logger.warning("No data returned from yfinance for %s", symbol)
            return pd.DataFrame()
        # Standardize column names
        df = df.rename(columns={
            "Adj Close": "Adj Close",
        })
        # Keep only the columns we need
        cols = ["Open", "High", "Low", "Close", "Volume"]
        if "Adj Close" in df.columns:
            cols.append("Adj Close")
        return df[cols]

    def get_asset_info(self, symbol: str) -> Asset:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        quote_type = info.get("quoteType", "EQUITY")
        asset_type = _TYPE_MAP.get(quote_type, AssetType.OTHER)
        return Asset(
            symbol=symbol,
            name=info.get("longName") or info.get("shortName", symbol),
            asset_type=asset_type,
            sector=info.get("sector", ""),
            exchange=info.get("exchange", ""),
            currency=info.get("currency", "USD"),
        )

    def get_current_price(self, symbol: str) -> float:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        price = info.get("regularMarketPrice") or info.get("previousClose", 0.0)
        return float(price)

    def get_multiple_prices(
        self, symbols: list[str], start: date, end: date | None = None
    ) -> dict[str, pd.DataFrame]:
        """Batch download for efficiency."""
        end = end or date.today()
        end_adj = end + timedelta(days=1)
        tickers_str = " ".join(symbols)
        try:
            data = yf.download(
                tickers_str,
                start=start.isoformat(),
                end=end_adj.isoformat(),
                auto_adjust=False,
                group_by="ticker",
                threads=True,
            )
        except Exception as e:
            logger.error("Batch download failed: %s", e)
            return super().get_multiple_prices(symbols, start, end)

        results = {}
        if len(symbols) == 1:
            # yf.download returns flat columns for single ticker
            results[symbols[0]] = data
        else:
            for sym in symbols:
                try:
                    df = data[sym].dropna(how="all")
                    results[sym] = df
                except (KeyError, AttributeError):
                    results[sym] = pd.DataFrame()
        return results
