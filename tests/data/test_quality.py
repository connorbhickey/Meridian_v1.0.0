"""Tests for data quality analysis — coverage, anomalies, reporting."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from portopt.data.quality import (
    DataAnomaly,
    QualityReport,
    SymbolCoverage,
    analyze_coverage,
    build_quality_report,
    detect_anomalies,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(dates, close_values, volumes=None):
    """Build a minimal OHLCV DataFrame from dates and close values."""
    n = len(dates)
    close = np.array(close_values, dtype=float)
    if volumes is None:
        volumes = np.full(n, 1_000_000.0)
    return pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": volumes,
            "Adj Close": close,
        },
        index=pd.DatetimeIndex(dates),
    )


# ---------------------------------------------------------------------------
# analyze_coverage
# ---------------------------------------------------------------------------

def test_analyze_coverage_basic():
    """Coverage analysis returns correct metrics for normal data."""
    dates = pd.bdate_range("2024-01-02", periods=60)
    prices = {"AAPL": _make_ohlcv(dates, np.linspace(150, 160, 60))}

    results = analyze_coverage(prices, reference_date=date(2024, 3, 25))

    assert len(results) == 1
    cov = results[0]
    assert isinstance(cov, SymbolCoverage)
    assert cov.symbol == "AAPL"
    assert cov.first_date == dates[0].date()
    assert cov.last_date == dates[-1].date()
    assert cov.trading_days == 60
    assert cov.total_days > 0
    assert cov.coverage_pct > 0


def test_analyze_coverage_empty_dataframe():
    """Empty DataFrame results in zero-value SymbolCoverage."""
    prices = {"EMPTY": pd.DataFrame()}

    results = analyze_coverage(prices)

    assert len(results) == 1
    cov = results[0]
    assert cov.symbol == "EMPTY"
    assert cov.first_date is None
    assert cov.last_date is None
    assert cov.trading_days == 0
    assert cov.coverage_pct == 0.0


def test_analyze_coverage_staleness():
    """Staleness measures days since last observation."""
    dates = pd.bdate_range("2024-01-02", periods=10)
    prices = {"AAPL": _make_ohlcv(dates, np.linspace(100, 110, 10))}

    ref = date(2024, 2, 15)  # Well after last data point
    results = analyze_coverage(prices, reference_date=ref)

    cov = results[0]
    expected_staleness = (ref - dates[-1].date()).days
    assert cov.staleness_days == expected_staleness


def test_analyze_coverage_staleness_zero_when_current():
    """Staleness is 0 when reference date matches last observation."""
    dates = pd.bdate_range("2024-01-02", periods=5)
    prices = {"X": _make_ohlcv(dates, [100, 101, 102, 103, 104])}

    ref = dates[-1].date()
    results = analyze_coverage(prices, reference_date=ref)

    assert results[0].staleness_days == 0


def test_analyze_coverage_multiple_symbols():
    """Coverage works with multiple symbols."""
    dates1 = pd.bdate_range("2024-01-02", periods=20)
    dates2 = pd.bdate_range("2024-01-02", periods=10)
    prices = {
        "AAPL": _make_ohlcv(dates1, np.linspace(100, 120, 20)),
        "MSFT": _make_ohlcv(dates2, np.linspace(300, 310, 10)),
    }

    results = analyze_coverage(prices, reference_date=date(2024, 2, 28))

    assert len(results) == 2
    symbols = {r.symbol for r in results}
    assert symbols == {"AAPL", "MSFT"}

    aapl = next(r for r in results if r.symbol == "AAPL")
    msft = next(r for r in results if r.symbol == "MSFT")
    assert aapl.trading_days == 20
    assert msft.trading_days == 10


def test_analyze_coverage_missing_days_computed():
    """Missing days = expected trading days - actual trading days."""
    # Create data with gaps (only 5 trading days over 2 months)
    dates = pd.to_datetime(["2024-01-02", "2024-01-15", "2024-02-01",
                            "2024-02-15", "2024-03-01"])
    prices = {"SPARSE": _make_ohlcv(dates, [100, 101, 102, 103, 104])}

    results = analyze_coverage(prices, reference_date=date(2024, 3, 1))

    cov = results[0]
    assert cov.trading_days == 5
    assert cov.missing_days > 0  # Should detect missing days
    assert cov.coverage_pct < 100.0


def test_analyze_coverage_empty_dict():
    """Empty prices dict returns empty list."""
    results = analyze_coverage({})
    assert results == []


# ---------------------------------------------------------------------------
# detect_anomalies — outlier returns
# ---------------------------------------------------------------------------

def test_detect_outlier_returns():
    """Large daily returns are flagged as outlier_return anomalies."""
    dates = pd.bdate_range("2024-01-02", periods=10)
    # Normal prices except for a 20% jump on day 5
    close = [100, 101, 100, 102, 101, 121.2, 120, 119, 121, 120]
    prices = {"JUMP": _make_ohlcv(dates, close)}

    anomalies = detect_anomalies(prices, return_threshold=0.15)

    outliers = [a for a in anomalies if a.anomaly_type == "outlier_return"]
    assert len(outliers) >= 1
    assert any(a.symbol == "JUMP" for a in outliers)
    # The ~20% jump should be flagged
    jump_anomaly = next(
        a for a in outliers
        if a.date == dates[5].date()
    )
    assert "return" in jump_anomaly.description.lower() or "%" in jump_anomaly.description


def test_detect_outlier_returns_severity():
    """Returns >30% get severity='error', smaller get 'warning'."""
    dates = pd.bdate_range("2024-01-02", periods=6)
    # 20% return (warning) followed by 35% return (error)
    close = [100, 100, 120, 120, 162, 162]
    prices = {"SEV": _make_ohlcv(dates, close)}

    anomalies = detect_anomalies(prices, return_threshold=0.15)

    outliers = [a for a in anomalies if a.anomaly_type == "outlier_return"]
    warnings = [a for a in outliers if a.severity == "warning"]
    errors = [a for a in outliers if a.severity == "error"]
    assert len(warnings) >= 1
    assert len(errors) >= 1


def test_no_outlier_returns_for_small_moves():
    """Normal price moves below threshold are not flagged."""
    dates = pd.bdate_range("2024-01-02", periods=10)
    close = [100, 101, 100.5, 101.5, 100, 100.5, 101, 100.5, 101, 100]
    prices = {"CALM": _make_ohlcv(dates, close)}

    anomalies = detect_anomalies(prices, return_threshold=0.15)

    outliers = [a for a in anomalies if a.anomaly_type == "outlier_return"]
    assert len(outliers) == 0


# ---------------------------------------------------------------------------
# detect_anomalies — zero volume
# ---------------------------------------------------------------------------

def test_detect_zero_volume():
    """Days with zero volume are flagged."""
    dates = pd.bdate_range("2024-01-02", periods=10)
    close = np.linspace(100, 110, 10)
    volumes = [1e6, 1e6, 0, 1e6, 1e6, 50, 1e6, 1e6, 1e6, 1e6]
    prices = {"LOW_VOL": _make_ohlcv(dates, close, volumes=volumes)}

    anomalies = detect_anomalies(prices, min_volume=100)

    zero_vol = [a for a in anomalies if a.anomaly_type == "zero_volume"]
    assert len(zero_vol) >= 2  # volume=0 and volume=50 are both < 100
    assert all(a.severity == "info" for a in zero_vol)


def test_zero_volume_capped_at_ten():
    """Zero-volume anomalies are capped at 10 per symbol to avoid spam."""
    dates = pd.bdate_range("2024-01-02", periods=20)
    close = np.linspace(100, 120, 20)
    volumes = np.zeros(20)  # All zero volume
    prices = {"ALL_ZERO": _make_ohlcv(dates, close, volumes=volumes)}

    anomalies = detect_anomalies(prices, min_volume=100)

    zero_vol = [a for a in anomalies if a.anomaly_type == "zero_volume"]
    assert len(zero_vol) <= 10


# ---------------------------------------------------------------------------
# detect_anomalies — gaps
# ---------------------------------------------------------------------------

def test_detect_gaps():
    """Gaps longer than 7 days are flagged."""
    dates = pd.to_datetime([
        "2024-01-02", "2024-01-03", "2024-01-04",
        "2024-01-05", "2024-01-08",
        # 15-day gap
        "2024-01-23", "2024-01-24", "2024-01-25",
        "2024-01-26", "2024-01-29",
    ])
    close = np.linspace(100, 110, 10)
    prices = {"GAP": _make_ohlcv(dates, close)}

    anomalies = detect_anomalies(prices)

    gaps = [a for a in anomalies if a.anomaly_type == "gap"]
    assert len(gaps) >= 1
    assert any("gap" in a.description.lower() for a in gaps)


def test_detect_gap_severity():
    """Gaps < 14 days are warning, >= 14 days are error."""
    dates = pd.to_datetime([
        "2024-01-02",
        "2024-01-12",  # 10-day gap (warning)
        "2024-02-01",  # 20-day gap (error)
        "2024-02-02",
        "2024-02-05",
    ])
    close = [100, 101, 102, 103, 104]
    prices = {"GAPS": _make_ohlcv(dates, close)}

    anomalies = detect_anomalies(prices)

    gaps = [a for a in anomalies if a.anomaly_type == "gap"]
    warnings = [g for g in gaps if g.severity == "warning"]
    errors = [g for g in gaps if g.severity == "error"]
    assert len(warnings) >= 1
    assert len(errors) >= 1


def test_no_gaps_for_normal_weekdays():
    """Normal weekday trading data (3-day weekend gaps) are not flagged."""
    dates = pd.bdate_range("2024-01-02", periods=20)
    close = np.linspace(100, 120, 20)
    prices = {"NORMAL": _make_ohlcv(dates, close)}

    anomalies = detect_anomalies(prices)

    gaps = [a for a in anomalies if a.anomaly_type == "gap"]
    assert len(gaps) == 0


# ---------------------------------------------------------------------------
# detect_anomalies — suspected splits
# ---------------------------------------------------------------------------

def test_detect_suspected_split():
    """A 2:1 stock split pattern is flagged."""
    dates = pd.bdate_range("2024-01-02", periods=10)
    # Price halves on day 5 (2:1 split)
    close = [200, 201, 199, 200, 202, 101, 102, 100, 101, 103]
    prices = {"SPLIT": _make_ohlcv(dates, close)}

    anomalies = detect_anomalies(prices)

    splits = [a for a in anomalies if a.anomaly_type == "suspected_split"]
    assert len(splits) >= 1
    assert any("split" in a.description.lower() or "ratio" in a.description.lower()
               for a in splits)


def test_no_split_for_gradual_decline():
    """Gradual price declines are not flagged as splits."""
    dates = pd.bdate_range("2024-01-02", periods=10)
    close = np.linspace(200, 180, 10)  # Gradual decline
    prices = {"DECLINE": _make_ohlcv(dates, close)}

    anomalies = detect_anomalies(prices)

    splits = [a for a in anomalies if a.anomaly_type == "suspected_split"]
    assert len(splits) == 0


# ---------------------------------------------------------------------------
# detect_anomalies — edge cases
# ---------------------------------------------------------------------------

def test_detect_anomalies_empty_dataframe():
    """Empty DataFrame is skipped (no anomalies)."""
    prices = {"EMPTY": pd.DataFrame()}

    anomalies = detect_anomalies(prices)

    assert anomalies == []


def test_detect_anomalies_too_few_rows():
    """DataFrames with fewer than 5 rows are skipped."""
    dates = pd.bdate_range("2024-01-02", periods=3)
    close = [100, 101, 102]
    prices = {"SHORT": _make_ohlcv(dates, close)}

    anomalies = detect_anomalies(prices)

    assert anomalies == []


def test_detect_anomalies_empty_dict():
    """Empty prices dict returns empty list."""
    anomalies = detect_anomalies({})
    assert anomalies == []


def test_detect_anomalies_uses_adj_close():
    """When 'Adj Close' is present, it is used for return calculations."""
    dates = pd.bdate_range("2024-01-02", periods=10)
    close = [100, 101, 100, 102, 101, 121.2, 120, 119, 121, 120]
    prices = {"ADJ": _make_ohlcv(dates, close)}
    # The _make_ohlcv helper sets Adj Close = Close, so outlier detection
    # should find the same jump regardless of which column is used
    anomalies = detect_anomalies(prices, return_threshold=0.15)

    outliers = [a for a in anomalies if a.anomaly_type == "outlier_return"]
    assert len(outliers) >= 1


def test_detect_anomalies_multiple_symbols():
    """Anomalies are detected across multiple symbols."""
    dates = pd.bdate_range("2024-01-02", periods=10)
    # AAPL has a big jump
    aapl_close = [100, 101, 100, 102, 101, 121, 120, 119, 121, 120]
    # MSFT has zero volume on some days
    msft_close = np.linspace(300, 310, 10)
    msft_vol = [1e6, 1e6, 0, 0, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6]

    prices = {
        "AAPL": _make_ohlcv(dates, aapl_close),
        "MSFT": _make_ohlcv(dates, msft_close, volumes=msft_vol),
    }

    anomalies = detect_anomalies(prices, return_threshold=0.15, min_volume=100)

    symbols_with_anomalies = {a.symbol for a in anomalies}
    assert "AAPL" in symbols_with_anomalies
    assert "MSFT" in symbols_with_anomalies


def test_detect_anomalies_custom_threshold():
    """Custom return_threshold changes what is flagged."""
    dates = pd.bdate_range("2024-01-02", periods=6)
    # 5% daily return
    close = [100, 105, 100, 105, 100, 105]
    prices = {"SWING": _make_ohlcv(dates, close)}

    # With 15% threshold, 5% moves should not be flagged
    anomalies_high = detect_anomalies(prices, return_threshold=0.15)
    outliers_high = [a for a in anomalies_high if a.anomaly_type == "outlier_return"]
    assert len(outliers_high) == 0

    # With 3% threshold, 5% moves should be flagged
    anomalies_low = detect_anomalies(prices, return_threshold=0.03)
    outliers_low = [a for a in anomalies_low if a.anomaly_type == "outlier_return"]
    assert len(outliers_low) > 0


# ---------------------------------------------------------------------------
# DataAnomaly dataclass
# ---------------------------------------------------------------------------

def test_data_anomaly_fields():
    """DataAnomaly stores all expected fields."""
    a = DataAnomaly(
        symbol="AAPL",
        date=date(2024, 3, 15),
        anomaly_type="outlier_return",
        description="+25.0% daily return",
        severity="warning",
    )
    assert a.symbol == "AAPL"
    assert a.date == date(2024, 3, 15)
    assert a.anomaly_type == "outlier_return"
    assert a.description == "+25.0% daily return"
    assert a.severity == "warning"


def test_data_anomaly_default_severity():
    """Default severity is 'warning'."""
    a = DataAnomaly(
        symbol="X", date=date(2024, 1, 1),
        anomaly_type="gap", description="test",
    )
    assert a.severity == "warning"


# ---------------------------------------------------------------------------
# build_quality_report
# ---------------------------------------------------------------------------

def test_build_quality_report_basic():
    """build_quality_report aggregates coverage and anomalies."""
    dates = pd.bdate_range("2024-01-02", periods=60)
    close = np.linspace(100, 120, 60)
    prices = {
        "AAPL": _make_ohlcv(dates, close),
        "MSFT": _make_ohlcv(dates, close * 2),
    }

    report = build_quality_report(prices, cache_size_mb=15.5)

    assert isinstance(report, QualityReport)
    assert report.total_symbols == 2
    assert report.total_observations == 120  # 60 + 60
    assert report.cache_size_mb == pytest.approx(15.5)
    assert report.avg_coverage_pct > 0
    assert len(report.coverage) == 2
    assert isinstance(report.anomalies, list)


def test_build_quality_report_empty_prices():
    """Report with no prices has zero totals."""
    report = build_quality_report({})

    assert report.total_symbols == 0
    assert report.total_observations == 0
    assert report.avg_coverage_pct == 0.0
    assert report.coverage == []
    assert report.anomalies == []


def test_build_quality_report_with_empty_dataframes():
    """Symbols with empty DataFrames contribute to count but not observations."""
    prices = {
        "EMPTY1": pd.DataFrame(),
        "EMPTY2": pd.DataFrame(),
    }

    report = build_quality_report(prices)

    assert report.total_symbols == 2
    assert report.total_observations == 0
    assert report.avg_coverage_pct == 0.0
    assert len(report.coverage) == 2


def test_build_quality_report_mixed():
    """Report with both valid and empty data is computed correctly."""
    dates = pd.bdate_range("2024-01-02", periods=20)
    prices = {
        "AAPL": _make_ohlcv(dates, np.linspace(100, 120, 20)),
        "EMPTY": pd.DataFrame(),
    }

    report = build_quality_report(prices)

    assert report.total_symbols == 2
    assert report.total_observations == 20
    # avg_coverage_pct should only average symbols with data
    assert report.avg_coverage_pct > 0


def test_build_quality_report_includes_anomalies():
    """Report includes anomalies detected in the price data."""
    dates = pd.bdate_range("2024-01-02", periods=10)
    # Include a big jump to trigger outlier detection
    close = [100, 101, 100, 102, 101, 125, 120, 119, 121, 120]
    prices = {"JUMPY": _make_ohlcv(dates, close)}

    report = build_quality_report(prices)

    assert len(report.anomalies) > 0
    assert any(a.anomaly_type == "outlier_return" for a in report.anomalies)


def test_build_quality_report_default_cache_size():
    """Default cache_size_mb is 0.0."""
    report = build_quality_report({})
    assert report.cache_size_mb == 0.0


# ---------------------------------------------------------------------------
# QualityReport dataclass
# ---------------------------------------------------------------------------

def test_quality_report_defaults():
    """QualityReport has sensible defaults."""
    report = QualityReport()
    assert report.coverage == []
    assert report.anomalies == []
    assert report.cache_size_mb == 0.0
    assert report.total_symbols == 0
    assert report.total_observations == 0
    assert report.avg_coverage_pct == 0.0


# ---------------------------------------------------------------------------
# SymbolCoverage dataclass
# ---------------------------------------------------------------------------

def test_symbol_coverage_defaults():
    """SymbolCoverage with only symbol has zero values."""
    sc = SymbolCoverage(symbol="TEST")
    assert sc.symbol == "TEST"
    assert sc.first_date is None
    assert sc.last_date is None
    assert sc.total_days == 0
    assert sc.trading_days == 0
    assert sc.missing_days == 0
    assert sc.coverage_pct == 0.0
    assert sc.staleness_days == 0
