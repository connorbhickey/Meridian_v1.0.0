"""Tests for fundamental data provider (P/E, market cap, sector, etc.)."""

import math
from unittest.mock import MagicMock, patch

import pytest

from portopt.data.providers.fundamental_provider import (
    FundamentalData,
    FundamentalProvider,
    _safe_float,
)


# ---------------------------------------------------------------------------
# _safe_float helper
# ---------------------------------------------------------------------------

def test_safe_float_with_none():
    """None input returns None."""
    assert _safe_float(None) is None


def test_safe_float_with_nan():
    """NaN input returns None."""
    assert _safe_float(float("nan")) is None


def test_safe_float_with_valid_float():
    """Valid float is returned as-is."""
    assert _safe_float(3.14) == pytest.approx(3.14)


def test_safe_float_with_valid_int():
    """Integer is converted to float."""
    assert _safe_float(42) == pytest.approx(42.0)


def test_safe_float_with_numeric_string():
    """Numeric string is converted to float."""
    assert _safe_float("12.5") == pytest.approx(12.5)


def test_safe_float_with_invalid_string():
    """Non-numeric string returns None."""
    assert _safe_float("not-a-number") is None


def test_safe_float_with_empty_string():
    """Empty string returns None."""
    assert _safe_float("") is None


def test_safe_float_with_zero():
    """Zero is a valid float value."""
    assert _safe_float(0) == pytest.approx(0.0)


def test_safe_float_with_negative():
    """Negative floats are returned correctly."""
    assert _safe_float(-5.5) == pytest.approx(-5.5)


def test_safe_float_with_inf():
    """Infinity is returned as float (not filtered like NaN)."""
    result = _safe_float(float("inf"))
    assert result == float("inf")


# ---------------------------------------------------------------------------
# FundamentalData dataclass
# ---------------------------------------------------------------------------

def test_fundamental_data_defaults():
    """FundamentalData with only symbol has sensible defaults."""
    fd = FundamentalData(symbol="AAPL")
    assert fd.symbol == "AAPL"
    assert fd.name == ""
    assert fd.sector == ""
    assert fd.industry == ""
    assert fd.market_cap == 0.0
    assert fd.pe_ratio is None
    assert fd.forward_pe is None
    assert fd.pb_ratio is None
    assert fd.ps_ratio is None
    assert fd.dividend_yield is None
    assert fd.beta is None
    assert fd.eps is None
    assert fd.revenue is None
    assert fd.profit_margin is None
    assert fd.roe is None
    assert fd.debt_to_equity is None
    assert fd.free_cash_flow is None
    assert fd.fifty_two_week_high is None
    assert fd.fifty_two_week_low is None
    assert fd.avg_volume is None
    assert fd.description == ""


def test_fundamental_data_all_fields():
    """FundamentalData can be constructed with all fields populated."""
    fd = FundamentalData(
        symbol="MSFT",
        name="Microsoft Corporation",
        sector="Technology",
        industry="Software",
        market_cap=3_000_000_000_000.0,
        pe_ratio=35.2,
        forward_pe=30.1,
        pb_ratio=12.5,
        ps_ratio=13.8,
        dividend_yield=0.0072,
        beta=0.89,
        eps=11.20,
        revenue=230_000_000_000.0,
        profit_margin=0.36,
        roe=0.40,
        debt_to_equity=42.3,
        free_cash_flow=65_000_000_000.0,
        fifty_two_week_high=450.0,
        fifty_two_week_low=310.0,
        avg_volume=25_000_000.0,
        description="Microsoft develops software and cloud services.",
    )
    assert fd.symbol == "MSFT"
    assert fd.market_cap == pytest.approx(3e12)
    assert fd.pe_ratio == pytest.approx(35.2)
    assert fd.dividend_yield == pytest.approx(0.0072)
    assert fd.description == "Microsoft develops software and cloud services."


# ---------------------------------------------------------------------------
# FundamentalProvider.get_fundamentals
# ---------------------------------------------------------------------------

MOCK_AAPL_INFO = {
    "longName": "Apple Inc.",
    "shortName": "Apple Inc",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "marketCap": 2_800_000_000_000,
    "trailingPE": 28.5,
    "forwardPE": 26.0,
    "priceToBook": 45.2,
    "priceToSalesTrailing12Months": 8.3,
    "dividendYield": 0.0055,
    "beta": 1.21,
    "trailingEps": 6.42,
    "totalRevenue": 394_000_000_000,
    "profitMargins": 0.2535,
    "returnOnEquity": 1.608,
    "debtToEquity": 176.3,
    "freeCashflow": 111_000_000_000,
    "fiftyTwoWeekHigh": 199.62,
    "fiftyTwoWeekLow": 164.08,
    "averageVolume": 58_000_000,
    "longBusinessSummary": "Apple designs, manufactures, and markets smartphones.",
}


@patch("portopt.data.providers.fundamental_provider.yf.Ticker")
def test_get_fundamentals_success(mock_ticker_cls):
    """get_fundamentals returns fully populated FundamentalData."""
    mock_ticker = MagicMock()
    mock_ticker.info = MOCK_AAPL_INFO.copy()
    mock_ticker_cls.return_value = mock_ticker

    provider = FundamentalProvider()
    fd = provider.get_fundamentals("AAPL")

    assert isinstance(fd, FundamentalData)
    assert fd.symbol == "AAPL"
    assert fd.name == "Apple Inc."
    assert fd.sector == "Technology"
    assert fd.industry == "Consumer Electronics"
    assert fd.market_cap == pytest.approx(2.8e12)
    assert fd.pe_ratio == pytest.approx(28.5)
    assert fd.forward_pe == pytest.approx(26.0)
    assert fd.pb_ratio == pytest.approx(45.2)
    assert fd.ps_ratio == pytest.approx(8.3)
    assert fd.dividend_yield == pytest.approx(0.0055)
    assert fd.beta == pytest.approx(1.21)
    assert fd.eps == pytest.approx(6.42)
    assert fd.revenue == pytest.approx(394e9)
    assert fd.profit_margin == pytest.approx(0.2535)
    assert fd.roe == pytest.approx(1.608)
    assert fd.debt_to_equity == pytest.approx(176.3)
    assert fd.free_cash_flow == pytest.approx(111e9)
    assert fd.fifty_two_week_high == pytest.approx(199.62)
    assert fd.fifty_two_week_low == pytest.approx(164.08)
    assert fd.avg_volume == pytest.approx(58e6)
    assert "Apple" in fd.description


@patch("portopt.data.providers.fundamental_provider.yf.Ticker")
def test_get_fundamentals_uses_short_name_fallback(mock_ticker_cls):
    """When longName is missing, falls back to shortName."""
    mock_ticker = MagicMock()
    mock_ticker.info = {"shortName": "Apple Inc", "marketCap": 1e12}
    mock_ticker_cls.return_value = mock_ticker

    provider = FundamentalProvider()
    fd = provider.get_fundamentals("AAPL")

    assert fd.name == "Apple Inc"


@patch("portopt.data.providers.fundamental_provider.yf.Ticker")
def test_get_fundamentals_uses_symbol_as_name_fallback(mock_ticker_cls):
    """When both longName and shortName are missing, uses symbol."""
    mock_ticker = MagicMock()
    mock_ticker.info = {"marketCap": 500e6}
    mock_ticker_cls.return_value = mock_ticker

    provider = FundamentalProvider()
    fd = provider.get_fundamentals("XYZ")

    assert fd.name == "XYZ"


@patch("portopt.data.providers.fundamental_provider.yf.Ticker")
def test_get_fundamentals_none_values_become_none(mock_ticker_cls):
    """Fields with None values in yfinance info map to None via _safe_float."""
    mock_ticker = MagicMock()
    mock_ticker.info = {
        "longName": "Test Corp",
        "trailingPE": None,
        "forwardPE": None,
        "priceToBook": None,
        "dividendYield": None,
        "beta": None,
    }
    mock_ticker_cls.return_value = mock_ticker

    provider = FundamentalProvider()
    fd = provider.get_fundamentals("TEST")

    assert fd.pe_ratio is None
    assert fd.forward_pe is None
    assert fd.pb_ratio is None
    assert fd.dividend_yield is None
    assert fd.beta is None


@patch("portopt.data.providers.fundamental_provider.yf.Ticker")
def test_get_fundamentals_nan_values_become_none(mock_ticker_cls):
    """NaN values from yfinance are converted to None."""
    mock_ticker = MagicMock()
    mock_ticker.info = {
        "longName": "NaN Corp",
        "trailingPE": float("nan"),
        "beta": float("nan"),
    }
    mock_ticker_cls.return_value = mock_ticker

    provider = FundamentalProvider()
    fd = provider.get_fundamentals("NAN")

    assert fd.pe_ratio is None
    assert fd.beta is None


@patch("portopt.data.providers.fundamental_provider.yf.Ticker")
def test_get_fundamentals_empty_info(mock_ticker_cls):
    """Empty info dict returns FundamentalData with defaults."""
    mock_ticker = MagicMock()
    mock_ticker.info = {}
    mock_ticker_cls.return_value = mock_ticker

    provider = FundamentalProvider()
    fd = provider.get_fundamentals("EMPTY")

    assert fd.symbol == "EMPTY"
    assert fd.name == "EMPTY"  # Falls back to symbol
    assert fd.sector == ""
    assert fd.market_cap == 0.0
    assert fd.pe_ratio is None


@patch("portopt.data.providers.fundamental_provider.yf.Ticker")
def test_get_fundamentals_none_info(mock_ticker_cls):
    """When ticker.info is None, returns FundamentalData with defaults."""
    mock_ticker = MagicMock()
    mock_ticker.info = None
    mock_ticker_cls.return_value = mock_ticker

    provider = FundamentalProvider()
    fd = provider.get_fundamentals("NULLINFO")

    assert fd.symbol == "NULLINFO"
    assert fd.name == "NULLINFO"
    assert fd.market_cap == 0.0


@patch("portopt.data.providers.fundamental_provider.yf.Ticker")
def test_get_fundamentals_yfinance_exception(mock_ticker_cls):
    """When yfinance raises an exception, returns minimal FundamentalData."""
    mock_ticker_cls.side_effect = Exception("yfinance connection failed")

    provider = FundamentalProvider()
    fd = provider.get_fundamentals("FAIL")

    assert isinstance(fd, FundamentalData)
    assert fd.symbol == "FAIL"
    assert fd.name == ""
    assert fd.sector == ""
    assert fd.market_cap == 0.0
    assert fd.pe_ratio is None


@patch("portopt.data.providers.fundamental_provider.yf.Ticker")
def test_get_fundamentals_market_cap_none_becomes_zero(mock_ticker_cls):
    """When marketCap is None, market_cap defaults to 0.0."""
    mock_ticker = MagicMock()
    mock_ticker.info = {"longName": "SmallCap Inc", "marketCap": None}
    mock_ticker_cls.return_value = mock_ticker

    provider = FundamentalProvider()
    fd = provider.get_fundamentals("SMALL")

    assert fd.market_cap == 0.0


# ---------------------------------------------------------------------------
# FundamentalProvider.get_multiple_fundamentals
# ---------------------------------------------------------------------------

@patch("portopt.data.providers.fundamental_provider.yf.Ticker")
def test_get_multiple_fundamentals(mock_ticker_cls):
    """get_multiple_fundamentals returns dict keyed by symbol."""
    def create_ticker(symbol):
        mock = MagicMock()
        mock.info = {
            "longName": f"{symbol} Corp",
            "sector": "Technology",
            "marketCap": 1e9,
        }
        return mock

    mock_ticker_cls.side_effect = create_ticker

    provider = FundamentalProvider()
    results = provider.get_multiple_fundamentals(["AAPL", "MSFT", "GOOGL"])

    assert len(results) == 3
    assert "AAPL" in results
    assert "MSFT" in results
    assert "GOOGL" in results
    assert results["AAPL"].name == "AAPL Corp"
    assert results["MSFT"].name == "MSFT Corp"
    assert results["GOOGL"].name == "GOOGL Corp"


@patch("portopt.data.providers.fundamental_provider.yf.Ticker")
def test_get_multiple_fundamentals_partial_failure(mock_ticker_cls):
    """One symbol failing does not prevent others from succeeding."""
    call_count = 0

    def create_ticker(symbol):
        nonlocal call_count
        call_count += 1
        if symbol == "BAD":
            raise Exception("yfinance error for BAD")
        mock = MagicMock()
        mock.info = {"longName": f"{symbol} Inc", "marketCap": 5e9}
        return mock

    mock_ticker_cls.side_effect = create_ticker

    provider = FundamentalProvider()
    results = provider.get_multiple_fundamentals(["GOOD1", "BAD", "GOOD2"])

    assert len(results) == 3
    assert results["GOOD1"].name == "GOOD1 Inc"
    assert results["BAD"].name == ""  # Failed, so minimal data
    assert results["BAD"].market_cap == 0.0
    assert results["GOOD2"].name == "GOOD2 Inc"


@patch("portopt.data.providers.fundamental_provider.yf.Ticker")
def test_get_multiple_fundamentals_empty_list(mock_ticker_cls):
    """Empty symbol list returns empty dict."""
    provider = FundamentalProvider()
    results = provider.get_multiple_fundamentals([])

    assert results == {}
    mock_ticker_cls.assert_not_called()


@patch("portopt.data.providers.fundamental_provider.yf.Ticker")
def test_get_fundamentals_ticker_called_with_symbol(mock_ticker_cls):
    """Verify yf.Ticker is called with the correct symbol."""
    mock_ticker = MagicMock()
    mock_ticker.info = {"longName": "Test"}
    mock_ticker_cls.return_value = mock_ticker

    provider = FundamentalProvider()
    provider.get_fundamentals("TSLA")

    mock_ticker_cls.assert_called_once_with("TSLA")
