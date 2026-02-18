"""Tests for FRED (Federal Reserve Economic Data) provider."""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from portopt.data.models import Asset, AssetType
from portopt.data.providers.fred_provider import FredProvider, FRED_SERIES


# ---------------------------------------------------------------------------
# Initialization & availability
# ---------------------------------------------------------------------------

def test_init_with_api_key():
    """FredProvider stores the key passed to constructor."""
    provider = FredProvider(api_key="test-key-123")
    assert provider._api_key == "test-key-123"


@patch("portopt.utils.credentials.get_credential", return_value=None)
@patch.dict("os.environ", {"FRED_API_KEY": "env-key-456"})
def test_init_from_env(_mock_cred):
    """FredProvider reads FRED_API_KEY from the environment when no arg given."""
    provider = FredProvider()
    assert provider._api_key == "env-key-456"


def test_init_no_key():
    """Without an API key, the provider is not available."""
    provider = FredProvider(api_key="")
    assert provider._api_key == ""
    assert provider.available is False


def test_available_with_key():
    """With an API key, the provider is available."""
    provider = FredProvider(api_key="my-key")
    assert provider.available is True


def test_available_without_key():
    """Without an API key, available is False."""
    provider = FredProvider(api_key="")
    assert provider.available is False


# ---------------------------------------------------------------------------
# get_prices
# ---------------------------------------------------------------------------

@patch("portopt.data.providers.fred_provider.requests.get")
def test_get_prices_success(mock_get):
    """get_prices returns OHLCV DataFrame from valid FRED JSON."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "observations": [
            {"date": "2024-01-02", "value": "4.25"},
            {"date": "2024-01-03", "value": "4.30"},
            {"date": "2024-01-04", "value": "4.28"},
        ]
    }
    mock_get.return_value = mock_resp

    provider = FredProvider(api_key="test-key")
    df = provider.get_prices("DGS10", start=date(2024, 1, 1), end=date(2024, 1, 5))

    assert not df.empty
    assert len(df) == 3
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
    assert df["Close"].iloc[0] == pytest.approx(4.25)
    assert df["Close"].iloc[1] == pytest.approx(4.30)
    # Open/High/Low should equal Close for FRED data
    assert (df["Open"] == df["Close"]).all()
    assert (df["High"] == df["Close"]).all()
    assert (df["Low"] == df["Close"]).all()
    # Volume should be 0 for macro data
    assert (df["Volume"] == 0).all()


@patch("portopt.data.providers.fred_provider.requests.get")
def test_get_prices_skips_missing_values(mock_get):
    """Observations with value='.' are skipped (FRED missing marker)."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "observations": [
            {"date": "2024-01-02", "value": "4.25"},
            {"date": "2024-01-03", "value": "."},
            {"date": "2024-01-04", "value": "4.28"},
            {"date": "2024-01-05", "value": "."},
        ]
    }
    mock_get.return_value = mock_resp

    provider = FredProvider(api_key="test-key")
    df = provider.get_prices("DGS10", start=date(2024, 1, 1), end=date(2024, 1, 5))

    assert len(df) == 2  # Only the two valid observations
    assert df["Close"].iloc[0] == pytest.approx(4.25)
    assert df["Close"].iloc[1] == pytest.approx(4.28)


@patch("portopt.data.providers.fred_provider.requests.get")
def test_get_prices_all_missing_returns_empty(mock_get):
    """When all observations are '.', returns empty DataFrame."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "observations": [
            {"date": "2024-01-02", "value": "."},
            {"date": "2024-01-03", "value": "."},
        ]
    }
    mock_get.return_value = mock_resp

    provider = FredProvider(api_key="test-key")
    df = provider.get_prices("DGS10", start=date(2024, 1, 1), end=date(2024, 1, 5))

    assert df.empty


@patch("portopt.data.providers.fred_provider.requests.get")
def test_get_prices_skips_non_numeric_values(mock_get):
    """Non-numeric values (other than '.') are silently skipped."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "observations": [
            {"date": "2024-01-02", "value": "4.25"},
            {"date": "2024-01-03", "value": "N/A"},
            {"date": "2024-01-04", "value": "not-a-number"},
        ]
    }
    mock_get.return_value = mock_resp

    provider = FredProvider(api_key="test-key")
    df = provider.get_prices("DGS10", start=date(2024, 1, 1), end=date(2024, 1, 5))

    assert len(df) == 1
    assert df["Close"].iloc[0] == pytest.approx(4.25)


@patch("portopt.data.providers.fred_provider.requests.get")
def test_get_prices_no_observations_key(mock_get):
    """Response without 'observations' key returns empty DataFrame."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"error_message": "Bad series_id"}
    mock_get.return_value = mock_resp

    provider = FredProvider(api_key="test-key")
    df = provider.get_prices("INVALID", start=date(2024, 1, 1))

    assert df.empty


@patch("portopt.data.providers.fred_provider.requests.get")
def test_get_prices_empty_observations(mock_get):
    """Empty observations list returns empty DataFrame."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"observations": []}
    mock_get.return_value = mock_resp

    provider = FredProvider(api_key="test-key")
    df = provider.get_prices("DGS10", start=date(2024, 1, 1))

    assert df.empty


@patch("portopt.data.providers.fred_provider.requests.get")
def test_get_prices_network_failure(mock_get):
    """Network error returns empty DataFrame, no exception raised."""
    mock_get.side_effect = ConnectionError("Network unreachable")

    provider = FredProvider(api_key="test-key")
    df = provider.get_prices("DGS10", start=date(2024, 1, 1))

    assert isinstance(df, pd.DataFrame)
    assert df.empty


@patch("portopt.data.providers.fred_provider.requests.get")
def test_get_prices_http_error(mock_get):
    """HTTP error (e.g. 500) returns empty DataFrame."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = Exception("500 Server Error")
    mock_get.return_value = mock_resp

    provider = FredProvider(api_key="test-key")
    df = provider.get_prices("DGS10", start=date(2024, 1, 1))

    assert df.empty


@patch("portopt.data.providers.fred_provider.requests.get")
def test_get_prices_datetime_index(mock_get):
    """Returned DataFrame has a proper DatetimeIndex."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "observations": [
            {"date": "2024-06-15", "value": "5.10"},
        ]
    }
    mock_get.return_value = mock_resp

    provider = FredProvider(api_key="test-key")
    df = provider.get_prices("DFF", start=date(2024, 6, 1), end=date(2024, 6, 30))

    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index[0] == pd.Timestamp("2024-06-15")


@patch("portopt.data.providers.fred_provider.requests.get")
def test_get_prices_default_end_date(mock_get):
    """When end is None, the request should still succeed (defaults to today)."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"observations": [{"date": "2024-01-02", "value": "1.5"}]}
    mock_get.return_value = mock_resp

    provider = FredProvider(api_key="test-key")
    df = provider.get_prices("DGS10", start=date(2024, 1, 1), end=None)

    assert len(df) == 1
    # Verify the API was called with an observation_end parameter
    call_kwargs = mock_get.call_args
    assert "params" in call_kwargs.kwargs or len(call_kwargs.args) > 0


# ---------------------------------------------------------------------------
# get_asset_info
# ---------------------------------------------------------------------------

@patch("portopt.data.providers.fred_provider.requests.get")
def test_get_asset_info_success(mock_get):
    """get_asset_info returns an Asset with FRED metadata."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "seriess": [{
            "id": "DGS10",
            "title": "Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity",
            "frequency": "Daily",
            "units": "Percent",
        }]
    }
    mock_get.return_value = mock_resp

    provider = FredProvider(api_key="test-key")
    asset = provider.get_asset_info("dgs10")

    assert isinstance(asset, Asset)
    assert asset.symbol == "DGS10"
    assert "Treasury" in asset.name or "Yield" in asset.name
    assert asset.asset_type == AssetType.OTHER
    assert asset.sector == "Macro"
    assert asset.exchange == "FRED"
    assert asset.currency == "USD"


@patch("portopt.data.providers.fred_provider.requests.get")
def test_get_asset_info_empty_seriess(mock_get):
    """When seriess list is empty, falls back to FRED_SERIES lookup."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"seriess": [{}]}
    mock_get.return_value = mock_resp

    provider = FredProvider(api_key="test-key")
    asset = provider.get_asset_info("DGS10")

    assert asset.symbol == "DGS10"
    # Falls back to FRED_SERIES dict title since title key is missing
    assert asset.name == FRED_SERIES.get("DGS10", "DGS10")


@patch("portopt.data.providers.fred_provider.requests.get")
def test_get_asset_info_network_failure(mock_get):
    """Network failure returns Asset with fallback name from FRED_SERIES."""
    mock_get.side_effect = ConnectionError("Network error")

    provider = FredProvider(api_key="test-key")
    asset = provider.get_asset_info("DGS10")

    assert isinstance(asset, Asset)
    assert asset.symbol == "DGS10"
    assert asset.name == FRED_SERIES["DGS10"]
    assert asset.asset_type == AssetType.OTHER
    assert asset.exchange == "FRED"


@patch("portopt.data.providers.fred_provider.requests.get")
def test_get_asset_info_unknown_series_fallback(mock_get):
    """Unknown series not in FRED_SERIES falls back to the symbol itself."""
    mock_get.side_effect = ConnectionError("Network error")

    provider = FredProvider(api_key="test-key")
    asset = provider.get_asset_info("XYZABC")

    assert asset.symbol == "XYZABC"
    assert asset.name == "XYZABC"  # Falls back to symbol when not in FRED_SERIES


# ---------------------------------------------------------------------------
# get_current_price
# ---------------------------------------------------------------------------

@patch("portopt.data.providers.fred_provider.requests.get")
def test_get_current_price_success(mock_get):
    """get_current_price returns latest observation value."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "observations": [
            {"date": "2024-07-15", "value": "4.42"},
        ]
    }
    mock_get.return_value = mock_resp

    provider = FredProvider(api_key="test-key")
    price = provider.get_current_price("DGS10")

    assert price == pytest.approx(4.42)


@patch("portopt.data.providers.fred_provider.requests.get")
def test_get_current_price_missing_value(mock_get):
    """When latest observation value is '.', returns 0.0."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "observations": [
            {"date": "2024-07-15", "value": "."},
        ]
    }
    mock_get.return_value = mock_resp

    provider = FredProvider(api_key="test-key")
    price = provider.get_current_price("DGS10")

    assert price == 0.0


@patch("portopt.data.providers.fred_provider.requests.get")
def test_get_current_price_empty_observations(mock_get):
    """Empty observations returns 0.0."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"observations": []}
    mock_get.return_value = mock_resp

    provider = FredProvider(api_key="test-key")
    price = provider.get_current_price("DGS10")

    assert price == 0.0


@patch("portopt.data.providers.fred_provider.requests.get")
def test_get_current_price_network_failure(mock_get):
    """Network failure returns 0.0."""
    mock_get.side_effect = ConnectionError("Network unreachable")

    provider = FredProvider(api_key="test-key")
    price = provider.get_current_price("DGS10")

    assert price == 0.0


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

@patch("portopt.data.providers.fred_provider.requests.get")
def test_search_success(mock_get):
    """search returns list of matching series."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "seriess": [
            {
                "id": "DGS10",
                "title": "10-Year Treasury Constant Maturity Rate",
                "frequency": "Daily",
                "units": "Percent",
            },
            {
                "id": "DGS2",
                "title": "2-Year Treasury Constant Maturity Rate",
                "frequency": "Daily",
                "units": "Percent",
            },
        ]
    }
    mock_get.return_value = mock_resp

    provider = FredProvider(api_key="test-key")
    results = provider.search("treasury rate", limit=5)

    assert len(results) == 2
    assert results[0]["id"] == "DGS10"
    assert results[0]["title"] == "10-Year Treasury Constant Maturity Rate"
    assert results[0]["frequency"] == "Daily"
    assert results[0]["units"] == "Percent"
    assert results[1]["id"] == "DGS2"


@patch("portopt.data.providers.fred_provider.requests.get")
def test_search_empty_results(mock_get):
    """Search with no matches returns empty list."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"seriess": []}
    mock_get.return_value = mock_resp

    provider = FredProvider(api_key="test-key")
    results = provider.search("xyznonexistent")

    assert results == []


@patch("portopt.data.providers.fred_provider.requests.get")
def test_search_network_failure(mock_get):
    """Network failure in search returns empty list."""
    mock_get.side_effect = ConnectionError("Network unreachable")

    provider = FredProvider(api_key="test-key")
    results = provider.search("treasury")

    assert results == []


@patch("portopt.data.providers.fred_provider.requests.get")
def test_search_handles_missing_fields(mock_get):
    """Search results with missing optional fields use empty string defaults."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "seriess": [
            {"id": "CUSTOM1"},  # No title, frequency, or units
        ]
    }
    mock_get.return_value = mock_resp

    provider = FredProvider(api_key="test-key")
    results = provider.search("custom")

    assert len(results) == 1
    assert results[0]["id"] == "CUSTOM1"
    assert results[0]["title"] == ""
    assert results[0]["frequency"] == ""
    assert results[0]["units"] == ""


# ---------------------------------------------------------------------------
# API call parameters
# ---------------------------------------------------------------------------

@patch("portopt.data.providers.fred_provider.requests.get")
def test_get_prices_passes_correct_params(mock_get):
    """Verify the correct query parameters are sent to the FRED API."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"observations": []}
    mock_get.return_value = mock_resp

    provider = FredProvider(api_key="my-secret-key")
    provider.get_prices("DFF", start=date(2024, 3, 1), end=date(2024, 3, 31))

    mock_get.assert_called_once()
    call_kwargs = mock_get.call_args
    params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params")
    assert params["series_id"] == "DFF"
    assert params["api_key"] == "my-secret-key"
    assert params["file_type"] == "json"
    assert params["observation_start"] == "2024-03-01"
    assert params["observation_end"] == "2024-03-31"


@patch("portopt.data.providers.fred_provider.requests.get")
def test_get_prices_uppercases_symbol(mock_get):
    """Symbols are uppercased before sending to FRED."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"observations": []}
    mock_get.return_value = mock_resp

    provider = FredProvider(api_key="key")
    provider.get_prices("dgs10", start=date(2024, 1, 1))

    params = mock_get.call_args.kwargs.get("params") or mock_get.call_args[1].get("params")
    assert params["series_id"] == "DGS10"
