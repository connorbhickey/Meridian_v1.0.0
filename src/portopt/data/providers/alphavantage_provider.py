"""Alpha Vantage data provider (fallback)."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import requests

from portopt.config import get_alpha_vantage_key
from portopt.utils.credentials import ALPHA_VANTAGE_API_KEY, get_credential
from portopt.data.models import Asset, AssetType
from portopt.data.providers.base import BaseDataProvider

logger = logging.getLogger(__name__)

BASE_URL = "https://www.alphavantage.co/query"


class AlphaVantageProvider(BaseDataProvider):
    """Data provider backed by Alpha Vantage REST API."""

    def __init__(self, api_key: str | None = None):
        if api_key is not None:
            self._api_key = api_key
        else:
            self._api_key = get_credential(ALPHA_VANTAGE_API_KEY) or get_alpha_vantage_key()

    @property
    def available(self) -> bool:
        return self._api_key is not None

    def _request(self, params: dict) -> dict:
        if not self._api_key:
            raise RuntimeError("Alpha Vantage API key not configured")
        params["apikey"] = self._api_key
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "Error Message" in data:
            raise ValueError(data["Error Message"])
        if "Note" in data:
            raise RuntimeError(f"Alpha Vantage rate limit: {data['Note']}")
        return data

    def get_prices(
        self, symbol: str, start: date, end: date | None = None
    ) -> pd.DataFrame:
        data = self._request({
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full",
        })
        ts = data.get("Time Series (Daily)", {})
        if not ts:
            return pd.DataFrame()

        records = []
        end = end or date.today()
        for dt_str, vals in ts.items():
            dt = date.fromisoformat(dt_str)
            if start <= dt <= end:
                records.append({
                    "date": pd.Timestamp(dt),
                    "Open": float(vals["1. open"]),
                    "High": float(vals["2. high"]),
                    "Low": float(vals["3. low"]),
                    "Close": float(vals["4. close"]),
                    "Volume": float(vals["6. volume"]),
                    "Adj Close": float(vals["5. adjusted close"]),
                })

        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)
        return df

    def get_asset_info(self, symbol: str) -> Asset:
        try:
            data = self._request({
                "function": "OVERVIEW",
                "symbol": symbol,
            })
            return Asset(
                symbol=symbol,
                name=data.get("Name", symbol),
                asset_type=AssetType.STOCK if data.get("AssetType") == "Common Stock" else AssetType.OTHER,
                sector=data.get("Sector", ""),
                exchange=data.get("Exchange", ""),
                currency=data.get("Currency", "USD"),
            )
        except Exception:
            return Asset(symbol=symbol)

    def get_current_price(self, symbol: str) -> float:
        data = self._request({
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
        })
        quote = data.get("Global Quote", {})
        return float(quote.get("05. price", 0.0))
