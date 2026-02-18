"""Tests for Tiingo data provider."""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from portopt.data.models import Asset, AssetType
from portopt.data.providers.tiingo_provider import TiingoProvider


# ---------------------------------------------------------------------------
# Initialization & availability
# ---------------------------------------------------------------------------

def test_init_with_api_key():
    """TiingoProvider stores the key passed to constructor."""
    provider = TiingoProvider(api_key="tiingo-test-key")
    assert provider._api_key == "tiingo-test-key"
    assert "Token tiingo-test-key" in provider._headers["Authorization"]


@patch("portopt.utils.credentials.get_credential", return_value=None)
@patch.dict("os.environ", {"TIINGO_API_KEY": "env-tiingo-key"})
def test_init_from_env(_mock_cred):
    """TiingoProvider reads TIINGO_API_KEY from environment."""
    provider = TiingoProvider()
    assert provider._api_key == "env-tiingo-key"


def test_init_no_key():
    """Without an API key, the provider is not available."""
    provider = TiingoProvider(api_key="")
    assert provider._api_key == ""
    assert provider.available is False


def test_available_with_key():
    """With an API key, the provider is available."""
    provider = TiingoProvider(api_key="my-key")
    assert provider.available is True


def test_available_without_key():
    """Without an API key, available is False."""
    provider = TiingoProvider(api_key="")
    assert provider.available is False


def test_headers_set_correctly():
    """Constructor sets authorization and content-type headers."""
    provider = TiingoProvider(api_key="abc123")
    assert provider._headers["Content-Type"] == "application/json"
    assert provider._headers["Authorization"] == "Token abc123"


# ---------------------------------------------------------------------------
# get_prices
# ---------------------------------------------------------------------------

@patch("portopt.data.providers.tiingo_provider.requests.get")
def test_get_prices_success(mock_get):
    """get_prices returns OHLCV DataFrame from Tiingo daily data."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = [
        {
            "date": "2024-01-02T00:00:00+00:00",
            "close": 185.64,
            "high": 186.50,
            "low": 184.20,
            "open": 185.00,
            "volume": 50123456,
            "adjOpen": 184.50,
            "adjHigh": 186.00,
            "adjLow": 183.70,
            "adjClose": 185.14,
            "adjVolume": 50123456,
        },
        {
            "date": "2024-01-03T00:00:00+00:00",
            "close": 184.25,
            "high": 186.00,
            "low": 183.50,
            "open": 185.50,
            "volume": 48000000,
            "adjOpen": 185.00,
            "adjHigh": 185.50,
            "adjLow": 183.00,
            "adjClose": 183.75,
            "adjVolume": 48000000,
        },
    ]
    mock_get.return_value = mock_resp

    provider = TiingoProvider(api_key="test-key")
    df = provider.get_prices("AAPL", start=date(2024, 1, 1), end=date(2024, 1, 5))

    assert not df.empty
    assert len(df) == 2
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
    # Uses adjusted values (adjOpen, adjHigh, etc.)
    assert df["Open"].iloc[0] == pytest.approx(184.50)
    assert df["High"].iloc[0] == pytest.approx(186.00)
    assert df["Low"].iloc[0] == pytest.approx(183.70)
    assert df["Close"].iloc[0] == pytest.approx(185.64)
    assert df["Adj Close"].iloc[0] == pytest.approx(185.14)
    assert df["Volume"].iloc[0] == pytest.approx(50123456)


@patch("portopt.data.providers.tiingo_provider.requests.get")
def test_get_prices_datetime_index(mock_get):
    """Returned DataFrame has DatetimeIndex with timezone stripped."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = [
        {
            "date": "2024-06-15T00:00:00+00:00",
            "close": 200.0,
            "high": 202.0,
            "low": 198.0,
            "open": 199.0,
            "volume": 1000000,
            "adjClose": 200.0,
            "adjOpen": 199.0,
            "adjHigh": 202.0,
            "adjLow": 198.0,
            "adjVolume": 1000000,
        }
    ]
    mock_get.return_value = mock_resp

    provider = TiingoProvider(api_key="test-key")
    df = provider.get_prices("MSFT", start=date(2024, 6, 1), end=date(2024, 6, 30))

    assert isinstance(df.index, pd.DatetimeIndex)
    # Timezone should be localized to None (naive)
    assert df.index.tz is None
    assert df.index[0] == pd.Timestamp("2024-06-15")


@patch("portopt.data.providers.tiingo_provider.requests.get")
def test_get_prices_empty_response(mock_get):
    """Empty JSON list returns empty DataFrame."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = []
    mock_get.return_value = mock_resp

    provider = TiingoProvider(api_key="test-key")
    df = provider.get_prices("UNKNOWN", start=date(2024, 1, 1))

    assert df.empty


@patch("portopt.data.providers.tiingo_provider.requests.get")
def test_get_prices_network_failure(mock_get):
    """Network error returns empty DataFrame."""
    mock_get.side_effect = ConnectionError("Network unreachable")

    provider = TiingoProvider(api_key="test-key")
    df = provider.get_prices("AAPL", start=date(2024, 1, 1))

    assert isinstance(df, pd.DataFrame)
    assert df.empty


@patch("portopt.data.providers.tiingo_provider.requests.get")
def test_get_prices_http_error(mock_get):
    """HTTP error (e.g. 403 Forbidden) returns empty DataFrame."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = Exception("403 Forbidden")
    mock_get.return_value = mock_resp

    provider = TiingoProvider(api_key="test-key")
    df = provider.get_prices("AAPL", start=date(2024, 1, 1))

    assert df.empty


@patch("portopt.data.providers.tiingo_provider.requests.get")
def test_get_prices_fallback_to_non_adj_columns(mock_get):
    """When adjusted columns are absent, falls back to raw OHLC."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = [
        {
            "date": "2024-01-02T00:00:00+00:00",
            "close": 150.0,
            "high": 152.0,
            "low": 148.0,
            "open": 149.0,
            "volume": 30000000,
            # No adj* columns
        },
    ]
    mock_get.return_value = mock_resp

    provider = TiingoProvider(api_key="test-key")
    df = provider.get_prices("XYZ", start=date(2024, 1, 1))

    assert not df.empty
    assert df["Open"].iloc[0] == pytest.approx(149.0)
    assert df["High"].iloc[0] == pytest.approx(152.0)
    assert df["Low"].iloc[0] == pytest.approx(148.0)
    assert df["Close"].iloc[0] == pytest.approx(150.0)


@patch("portopt.data.providers.tiingo_provider.requests.get")
def test_get_prices_default_end_date(mock_get):
    """When end is None, the request should succeed (defaults to today)."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = [
        {
            "date": "2024-01-02T00:00:00+00:00",
            "close": 100.0, "high": 101.0, "low": 99.0, "open": 100.0,
            "volume": 1000, "adjClose": 100.0, "adjOpen": 100.0,
            "adjHigh": 101.0, "adjLow": 99.0, "adjVolume": 1000,
        }
    ]
    mock_get.return_value = mock_resp

    provider = TiingoProvider(api_key="test-key")
    df = provider.get_prices("AAPL", start=date(2024, 1, 1), end=None)

    assert len(df) == 1


@patch("portopt.data.providers.tiingo_provider.requests.get")
def test_get_prices_sends_auth_headers(mock_get):
    """Verify auth headers are passed in the request."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = []
    mock_get.return_value = mock_resp

    provider = TiingoProvider(api_key="secret-token")
    provider.get_prices("AAPL", start=date(2024, 1, 1))

    call_kwargs = mock_get.call_args
    headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
    assert headers["Authorization"] == "Token secret-token"


# ---------------------------------------------------------------------------
# get_asset_info
# ---------------------------------------------------------------------------

@patch("portopt.data.providers.tiingo_provider.requests.get")
def test_get_asset_info_success(mock_get):
    """get_asset_info returns Asset from Tiingo metadata."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "ticker": "AAPL",
        "name": "Apple Inc",
        "exchangeCode": "NASDAQ",
        "startDate": "1980-12-12",
        "endDate": "2024-07-15",
        "description": "Apple designs and manufactures consumer electronics.",
    }
    mock_get.return_value = mock_resp

    provider = TiingoProvider(api_key="test-key")
    asset = provider.get_asset_info("aapl")

    assert isinstance(asset, Asset)
    assert asset.symbol == "AAPL"
    assert asset.name == "Apple Inc"
    assert asset.asset_type == AssetType.STOCK
    assert asset.exchange == "NASDAQ"
    assert asset.currency == "USD"


@patch("portopt.data.providers.tiingo_provider.requests.get")
def test_get_asset_info_missing_fields(mock_get):
    """Asset info with missing optional fields uses defaults."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "ticker": "XYZ",
        # No name, no exchangeCode
    }
    mock_get.return_value = mock_resp

    provider = TiingoProvider(api_key="test-key")
    asset = provider.get_asset_info("xyz")

    assert asset.symbol == "XYZ"
    assert asset.name == "xyz"  # Falls back to the input symbol
    assert asset.exchange == ""


@patch("portopt.data.providers.tiingo_provider.requests.get")
def test_get_asset_info_network_failure(mock_get):
    """Network failure returns minimal Asset with just symbol."""
    mock_get.side_effect = ConnectionError("Network unreachable")

    provider = TiingoProvider(api_key="test-key")
    asset = provider.get_asset_info("AAPL")

    assert isinstance(asset, Asset)
    assert asset.symbol == "AAPL"


# ---------------------------------------------------------------------------
# get_current_price
# ---------------------------------------------------------------------------

@patch("portopt.data.providers.tiingo_provider.requests.get")
def test_get_current_price_success(mock_get):
    """get_current_price returns the last traded price from IEX endpoint."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = [
        {
            "ticker": "AAPL",
            "last": 192.53,
            "tngoLast": 192.53,
            "timestamp": "2024-07-15T20:00:00+00:00",
        }
    ]
    mock_get.return_value = mock_resp

    provider = TiingoProvider(api_key="test-key")
    price = provider.get_current_price("AAPL")

    assert price == pytest.approx(192.53)


@patch("portopt.data.providers.tiingo_provider.requests.get")
def test_get_current_price_uses_tngo_last_fallback(mock_get):
    """When 'last' is None/0, falls back to tngoLast."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = [
        {
            "ticker": "AAPL",
            "last": None,
            "tngoLast": 191.00,
        }
    ]
    mock_get.return_value = mock_resp

    provider = TiingoProvider(api_key="test-key")
    price = provider.get_current_price("AAPL")

    assert price == pytest.approx(191.00)


@patch("portopt.data.providers.tiingo_provider.requests.get")
def test_get_current_price_empty_response(mock_get):
    """Empty list response returns 0.0."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = []
    mock_get.return_value = mock_resp

    provider = TiingoProvider(api_key="test-key")
    price = provider.get_current_price("UNKNOWN")

    assert price == 0.0


@patch("portopt.data.providers.tiingo_provider.requests.get")
def test_get_current_price_non_list_response(mock_get):
    """Non-list response (e.g. error dict) returns 0.0."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"detail": "Not found"}
    mock_get.return_value = mock_resp

    provider = TiingoProvider(api_key="test-key")
    price = provider.get_current_price("INVALID")

    assert price == 0.0


@patch("portopt.data.providers.tiingo_provider.requests.get")
def test_get_current_price_network_failure(mock_get):
    """Network failure returns 0.0."""
    mock_get.side_effect = ConnectionError("Network unreachable")

    provider = TiingoProvider(api_key="test-key")
    price = provider.get_current_price("AAPL")

    assert price == 0.0


# ---------------------------------------------------------------------------
# get_multiple_prices
# ---------------------------------------------------------------------------

@patch("portopt.data.providers.tiingo_provider.requests.get")
def test_get_multiple_prices(mock_get):
    """get_multiple_prices fetches each symbol sequentially."""
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = [
            {
                "date": "2024-01-02T00:00:00+00:00",
                "close": 100.0 + call_count,
                "high": 101.0, "low": 99.0, "open": 100.0,
                "volume": 1000,
                "adjClose": 100.0 + call_count,
                "adjOpen": 100.0, "adjHigh": 101.0,
                "adjLow": 99.0, "adjVolume": 1000,
            }
        ]
        return mock_resp

    mock_get.side_effect = side_effect

    provider = TiingoProvider(api_key="test-key")
    results = provider.get_multiple_prices(
        ["AAPL", "MSFT"], start=date(2024, 1, 1), end=date(2024, 1, 5)
    )

    assert "AAPL" in results
    assert "MSFT" in results
    assert not results["AAPL"].empty
    assert not results["MSFT"].empty


@patch("portopt.data.providers.tiingo_provider.requests.get")
def test_get_multiple_prices_partial_failure(mock_get):
    """If one symbol fails, others still return data."""
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("Failed")
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = [
            {
                "date": "2024-01-02T00:00:00+00:00",
                "close": 150.0, "high": 151.0, "low": 149.0, "open": 150.0,
                "volume": 2000,
                "adjClose": 150.0, "adjOpen": 150.0,
                "adjHigh": 151.0, "adjLow": 149.0, "adjVolume": 2000,
            }
        ]
        return mock_resp

    mock_get.side_effect = side_effect

    provider = TiingoProvider(api_key="test-key")
    results = provider.get_multiple_prices(
        ["BAD", "GOOD"], start=date(2024, 1, 1)
    )

    assert "BAD" in results
    assert "GOOD" in results
    assert results["BAD"].empty  # Failed symbol
    assert not results["GOOD"].empty  # Succeeded
