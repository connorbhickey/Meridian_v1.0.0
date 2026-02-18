"""Tiingo data provider — reliable equity, ETF, and crypto prices."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import requests

from portopt.data.models import Asset, AssetType
from portopt.data.providers.base import BaseDataProvider

logger = logging.getLogger(__name__)

TIINGO_BASE_URL = "https://api.tiingo.com"


class TiingoProvider(BaseDataProvider):
    """Data provider backed by the Tiingo API.

    Requires a Tiingo API token. Get one free at:
    https://www.tiingo.com/account/api/token

    Set via environment variable TIINGO_API_KEY or pass to constructor.
    """

    def __init__(self, api_key: str | None = None):
        import os
        if api_key is not None:
            self._api_key = api_key
        else:
            from portopt.utils.credentials import TIINGO_API_KEY, get_credential
            self._api_key = get_credential(TIINGO_API_KEY) or os.environ.get("TIINGO_API_KEY", "")
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {self._api_key}",
        }

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def get_prices(
        self, symbol: str, start: date, end: date | None = None
    ) -> pd.DataFrame:
        end = end or date.today()

        try:
            resp = requests.get(
                f"{TIINGO_BASE_URL}/tiingo/daily/{symbol}/prices",
                headers=self._headers,
                params={
                    "startDate": start.isoformat(),
                    "endDate": end.isoformat(),
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("Tiingo fetch failed for %s: %s", symbol, e)
            return pd.DataFrame()

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df.set_index("date", inplace=True)
        df.index.name = None

        # Map Tiingo columns to standard OHLCV
        result = pd.DataFrame({
            "Open": df.get("adjOpen", df.get("open", pd.Series(dtype=float))),
            "High": df.get("adjHigh", df.get("high", pd.Series(dtype=float))),
            "Low": df.get("adjLow", df.get("low", pd.Series(dtype=float))),
            "Close": df.get("close", pd.Series(dtype=float)),
            "Volume": df.get("adjVolume", df.get("volume", 0)),
            "Adj Close": df.get("adjClose", df.get("close", pd.Series(dtype=float))),
        })
        return result

    def get_asset_info(self, symbol: str) -> Asset:
        try:
            resp = requests.get(
                f"{TIINGO_BASE_URL}/tiingo/daily/{symbol}",
                headers=self._headers,
                timeout=10,
            )
            resp.raise_for_status()
            meta = resp.json()
        except Exception as e:
            logger.warning("Tiingo meta failed for %s: %s", symbol, e)
            return Asset(symbol=symbol)

        return Asset(
            symbol=symbol.upper(),
            name=meta.get("name", symbol),
            asset_type=AssetType.STOCK,
            sector="",
            exchange=meta.get("exchangeCode", ""),
            currency="USD",
        )

    def get_current_price(self, symbol: str) -> float:
        try:
            resp = requests.get(
                f"{TIINGO_BASE_URL}/iex/{symbol}",
                headers=self._headers,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data and isinstance(data, list):
                return float(data[0].get("last") or data[0].get("tngoLast", 0))
        except Exception as e:
            logger.warning("Tiingo current price failed for %s: %s", symbol, e)
        return 0.0

    def get_multiple_prices(
        self, symbols: list[str], start: date, end: date | None = None
    ) -> dict[str, pd.DataFrame]:
        """Fetch prices sequentially (Tiingo has no batch endpoint for daily)."""
        results = {}
        for sym in symbols:
            try:
                results[sym] = self.get_prices(sym, start, end)
            except Exception:
                results[sym] = pd.DataFrame()
        return results
