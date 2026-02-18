"""Data quality analysis — coverage, staleness, anomalies."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SymbolCoverage:
    """Coverage info for a single symbol."""
    symbol: str
    first_date: date | None = None
    last_date: date | None = None
    total_days: int = 0
    trading_days: int = 0
    missing_days: int = 0          # Expected trading days minus actual
    coverage_pct: float = 0.0      # trading_days / expected
    staleness_days: int = 0        # Days since last observation


@dataclass
class DataAnomaly:
    """A detected data quality anomaly."""
    symbol: str
    date: date
    anomaly_type: str              # "outlier_return", "zero_volume", "gap", "suspected_split"
    description: str
    severity: str = "warning"      # "info", "warning", "error"


@dataclass
class QualityReport:
    """Complete data quality report."""
    coverage: list[SymbolCoverage] = field(default_factory=list)
    anomalies: list[DataAnomaly] = field(default_factory=list)
    cache_size_mb: float = 0.0
    total_symbols: int = 0
    total_observations: int = 0
    avg_coverage_pct: float = 0.0


def analyze_coverage(
    prices: dict[str, pd.DataFrame],
    reference_date: date | None = None,
) -> list[SymbolCoverage]:
    """Analyze data coverage for each symbol.

    Args:
        prices: Dict of {symbol: OHLCV DataFrame with DatetimeIndex}.
        reference_date: Date to measure staleness from (default: today).

    Returns:
        List of SymbolCoverage for each symbol.
    """
    ref = reference_date or date.today()
    results = []

    for symbol, df in prices.items():
        if df.empty:
            results.append(SymbolCoverage(symbol=symbol))
            continue

        idx = df.index
        first = idx.min()
        last = idx.max()

        first_d = first.date() if hasattr(first, "date") else first
        last_d = last.date() if hasattr(last, "date") else last

        trading_days = len(idx)

        # Estimate expected trading days (~252/year)
        calendar_days = (last_d - first_d).days
        expected = max(1, int(calendar_days * 252 / 365))
        missing = max(0, expected - trading_days)
        coverage = min(1.0, trading_days / expected) if expected > 0 else 0.0
        staleness = (ref - last_d).days

        results.append(SymbolCoverage(
            symbol=symbol,
            first_date=first_d,
            last_date=last_d,
            total_days=calendar_days,
            trading_days=trading_days,
            missing_days=missing,
            coverage_pct=coverage * 100,
            staleness_days=max(0, staleness),
        ))

    return results


def detect_anomalies(
    prices: dict[str, pd.DataFrame],
    return_threshold: float = 0.15,
    min_volume: float = 100,
) -> list[DataAnomaly]:
    """Detect anomalies in price data.

    Args:
        prices: Dict of {symbol: OHLCV DataFrame}.
        return_threshold: Flag returns with absolute value above this (default 15%).
        min_volume: Flag days with volume below this.

    Returns:
        List of DataAnomaly objects.
    """
    anomalies = []

    for symbol, df in prices.items():
        if df.empty or len(df) < 5:
            continue

        close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        close = df[close_col].dropna()

        if close.empty:
            continue

        # 1. Outlier returns
        returns = close.pct_change().dropna()
        outliers = returns[returns.abs() > return_threshold]
        for dt, ret in outliers.items():
            d = dt.date() if hasattr(dt, "date") else dt
            anomalies.append(DataAnomaly(
                symbol=symbol,
                date=d,
                anomaly_type="outlier_return",
                description=f"{ret:+.1%} daily return",
                severity="warning" if abs(ret) < 0.30 else "error",
            ))

        # 2. Zero-volume days
        if "Volume" in df.columns:
            vol = df["Volume"].fillna(0)
            zero_vol = vol[vol < min_volume]
            for dt in zero_vol.index[:10]:  # Cap at 10 to avoid spam
                d = dt.date() if hasattr(dt, "date") else dt
                anomalies.append(DataAnomaly(
                    symbol=symbol,
                    date=d,
                    anomaly_type="zero_volume",
                    description=f"Volume = {vol.get(dt, 0):.0f}",
                    severity="info",
                ))

        # 3. Suspected stock splits (>40% daily move + close roughly N*prev or prev/N)
        for i in range(1, len(close)):
            prev = close.iloc[i - 1]
            curr = close.iloc[i]
            if prev == 0:
                continue
            ratio = curr / prev
            if ratio > 1.4 or ratio < 0.6:
                # Check if it's a clean split ratio
                for split in [2, 3, 4, 0.5, 0.25, 1/3]:
                    if abs(ratio - split) < 0.05:
                        d = close.index[i]
                        d = d.date() if hasattr(d, "date") else d
                        anomalies.append(DataAnomaly(
                            symbol=symbol,
                            date=d,
                            anomaly_type="suspected_split",
                            description=f"Price ratio {ratio:.2f}x (possible {split}:1 split)",
                            severity="warning",
                        ))
                        break

        # 4. Large gaps (>5 trading days between observations)
        dates = close.index.sort_values()
        for i in range(1, len(dates)):
            gap = (dates[i] - dates[i - 1]).days
            if gap > 7:  # More than a week = suspicious
                d = dates[i].date() if hasattr(dates[i], "date") else dates[i]
                anomalies.append(DataAnomaly(
                    symbol=symbol,
                    date=d,
                    anomaly_type="gap",
                    description=f"{gap}-day gap in data",
                    severity="warning" if gap < 14 else "error",
                ))

    return anomalies


def build_quality_report(
    prices: dict[str, pd.DataFrame],
    cache_size_mb: float = 0.0,
) -> QualityReport:
    """Build a complete data quality report."""
    coverage = analyze_coverage(prices)
    anomalies = detect_anomalies(prices)

    total_obs = sum(c.trading_days for c in coverage)
    avg_cov = (
        np.mean([c.coverage_pct for c in coverage if c.trading_days > 0])
        if any(c.trading_days > 0 for c in coverage)
        else 0.0
    )

    return QualityReport(
        coverage=coverage,
        anomalies=anomalies,
        cache_size_mb=cache_size_mb,
        total_symbols=len(prices),
        total_observations=total_obs,
        avg_coverage_pct=avg_cov,
    )
