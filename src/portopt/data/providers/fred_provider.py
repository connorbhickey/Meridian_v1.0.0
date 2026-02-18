"""FRED (Federal Reserve Economic Data) provider for macro indicators."""

from __future__ import annotations

import logging
from datetime import date
from io import StringIO

import pandas as pd
import requests

from portopt.data.models import Asset, AssetType
from portopt.data.providers.base import BaseDataProvider

logger = logging.getLogger(__name__)

FRED_BASE_URL = "https://api.stlouisfed.org/fred"

# Common FRED series for portfolio analysis
FRED_SERIES = {
    "DGS10": "10-Year Treasury Rate",
    "DGS2": "2-Year Treasury Rate",
    "DGS3MO": "3-Month Treasury Rate",
    "DFF": "Federal Funds Rate",
    "T10Y2Y": "10Y-2Y Treasury Spread",
    "CPIAUCSL": "CPI (All Urban Consumers)",
    "CPILFESL": "Core CPI (Ex Food & Energy)",
    "UNRATE": "Unemployment Rate",
    "VIXCLS": "CBOE Volatility Index (VIX)",
    "DTWEXBGS": "Trade-Weighted USD Index",
    "BAMLH0A0HYM2": "ICE BofA US High Yield Spread",
    "SP500": "S&P 500 Index",
    "DCOILWTICO": "WTI Crude Oil Price",
    "GOLDAMGBD228NLBM": "Gold Price (London PM Fix)",
}


class FredProvider(BaseDataProvider):
    """Data provider backed by the FRED API.

    Requires a FRED API key. Get one free at:
    https://fred.stlouisfed.org/docs/api/api_key.html

    Set via environment variable FRED_API_KEY or pass to constructor.
    """

    def __init__(self, api_key: str | None = None):
        import os
        if api_key is not None:
            self._api_key = api_key
        else:
            from portopt.utils.credentials import FRED_API_KEY, get_credential
            self._api_key = get_credential(FRED_API_KEY) or os.environ.get("FRED_API_KEY", "")

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def get_prices(
        self, symbol: str, start: date, end: date | None = None
    ) -> pd.DataFrame:
        """Fetch a FRED series as OHLCV-compatible DataFrame.

        For FRED data, Close = the series value. Open/High/Low are set
        equal to Close. Volume is 0.
        """
        end = end or date.today()
        params = {
            "series_id": symbol.upper(),
            "api_key": self._api_key,
            "file_type": "json",
            "observation_start": start.isoformat(),
            "observation_end": end.isoformat(),
        }

        try:
            resp = requests.get(
                f"{FRED_BASE_URL}/series/observations",
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("FRED fetch failed for %s: %s", symbol, e)
            return pd.DataFrame()

        if "observations" not in data:
            logger.warning("FRED response missing observations for %s", symbol)
            return pd.DataFrame()

        rows = []
        for obs in data["observations"]:
            val = obs.get("value", ".")
            if val == ".":
                continue
            try:
                v = float(val)
            except ValueError:
                continue
            rows.append({"date": obs["date"], "value": v})

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.index.name = None

        # Map to OHLCV format
        result = pd.DataFrame({
            "Open": df["value"],
            "High": df["value"],
            "Low": df["value"],
            "Close": df["value"],
            "Volume": 0,
            "Adj Close": df["value"],
        })
        return result

    def get_asset_info(self, symbol: str) -> Asset:
        """Fetch FRED series metadata."""
        params = {
            "series_id": symbol.upper(),
            "api_key": self._api_key,
            "file_type": "json",
        }
        try:
            resp = requests.get(
                f"{FRED_BASE_URL}/series",
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            series = data.get("seriess", [{}])[0]
            return Asset(
                symbol=symbol.upper(),
                name=series.get("title", FRED_SERIES.get(symbol.upper(), symbol)),
                asset_type=AssetType.OTHER,
                sector="Macro",
                exchange="FRED",
                currency="USD",
            )
        except Exception:
            return Asset(
                symbol=symbol.upper(),
                name=FRED_SERIES.get(symbol.upper(), symbol),
                asset_type=AssetType.OTHER,
                sector="Macro",
                exchange="FRED",
            )

    def get_current_price(self, symbol: str) -> float:
        """Get the most recent FRED observation value."""
        params = {
            "series_id": symbol.upper(),
            "api_key": self._api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 1,
        }
        try:
            resp = requests.get(
                f"{FRED_BASE_URL}/series/observations",
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            obs = data.get("observations", [])
            if obs and obs[0].get("value", ".") != ".":
                return float(obs[0]["value"])
        except Exception as e:
            logger.warning("FRED current value failed for %s: %s", symbol, e)
        return 0.0

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Search FRED for series matching a query."""
        params = {
            "search_text": query,
            "api_key": self._api_key,
            "file_type": "json",
            "limit": limit,
        }
        try:
            resp = requests.get(
                f"{FRED_BASE_URL}/series/search",
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                {
                    "id": s["id"],
                    "title": s.get("title", ""),
                    "frequency": s.get("frequency", ""),
                    "units": s.get("units", ""),
                }
                for s in data.get("seriess", [])
            ]
        except Exception as e:
            logger.warning("FRED search failed: %s", e)
            return []
