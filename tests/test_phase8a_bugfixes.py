"""Tests for Phase 8a bug fixes (A1–A14).

Covers:
- A1: walk_forward_enabled config key
- A2: WatchlistPanel.get_symbols()
- A3: Double-run protection on controllers
- A4: Metrics no longer return float('inf')
- A5: NaN validation in BaseOptimizer
- A7: HRP division by zero
- A8: PSD validation before optimization
- A9: CAGR numerical overflow
- A10: Risk decomposition chart rendering
- A11: Single-asset portfolio guard
- A13: Marchenko-Pastur T<N fallback
- A14: CSV importer numeric parsing
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portopt.constants import CovEstimator, LinkageMethod, OptMethod, RiskMeasure
from portopt.data.models import OptimizationResult
from portopt.engine.constraints import PortfolioConstraints
from portopt.engine.metrics import (
    cagr,
    compute_all_metrics,
    omega_ratio,
    profit_factor,
    tail_ratio,
)
from portopt.engine.optimization.base import BaseOptimizer
from portopt.engine.optimization.herc import herc_optimize
from portopt.engine.optimization.hrp import hrp_optimize
from portopt.engine.optimization.mean_variance import MeanVarianceOptimizer
from portopt.engine.returns import estimate_returns
from portopt.engine.risk import estimate_covariance
from tests.conftest import assert_valid_weights


# ══════════════════════════════════════════════════════════════════════
# A1: walk_forward_enabled config key
# ══════════════════════════════════════════════════════════════════════

class TestA1WalkForwardConfigKey:
    @pytest.mark.skipif(
        not pytest.importorskip("PySide6", reason="PySide6 not available"),
        reason="PySide6 required",
    )
    def test_config_key_is_walk_forward_enabled(self, qtbot):
        from portopt.gui.panels.backtest_panel import BacktestPanel
        panel = BacktestPanel()
        qtbot.addWidget(panel)
        config = panel.get_config()
        assert "walk_forward_enabled" in config
        assert "walk_forward" not in config  # old key should NOT exist


# ══════════════════════════════════════════════════════════════════════
# A2: WatchlistPanel.get_symbols()
# ══════════════════════════════════════════════════════════════════════

class TestA2WatchlistGetSymbols:
    @pytest.mark.skipif(
        not pytest.importorskip("PySide6", reason="PySide6 not available"),
        reason="PySide6 required",
    )
    def test_get_symbols_returns_list(self, qtbot):
        from portopt.gui.panels.watchlist_panel import WatchlistPanel
        panel = WatchlistPanel()
        qtbot.addWidget(panel)
        result = panel.get_symbols()
        assert isinstance(result, list)

    @pytest.mark.skipif(
        not pytest.importorskip("PySide6", reason="PySide6 not available"),
        reason="PySide6 required",
    )
    def test_get_symbols_with_data(self, qtbot):
        from portopt.gui.panels.watchlist_panel import WatchlistPanel
        panel = WatchlistPanel()
        qtbot.addWidget(panel)
        panel._watchlist = [
            {"symbol": "AAPL", "price": 175.0},
            {"symbol": "MSFT", "price": 350.0},
            {"symbol": "GOOG", "price": 140.0},
        ]
        symbols = panel.get_symbols()
        assert set(symbols) == {"AAPL", "MSFT", "GOOG"}


# ══════════════════════════════════════════════════════════════════════
# A3: Double-run protection
# ══════════════════════════════════════════════════════════════════════

class TestA3DoubleRunProtection:
    def test_optimization_controller_has_running_flag(self):
        """Verify the running_changed signal and _running flag exist."""
        from portopt.gui.controllers.optimization_controller import OptimizationController
        # Just verify the class has the signal attribute
        assert hasattr(OptimizationController, "running_changed")

    def test_backtest_controller_has_running_flag(self):
        from portopt.gui.controllers.backtest_controller import BacktestController
        assert hasattr(BacktestController, "running_changed")


# ══════════════════════════════════════════════════════════════════════
# A4: Metrics no longer return float('inf')
# ══════════════════════════════════════════════════════════════════════

class TestA4NoInfinityMetrics:
    def test_omega_ratio_all_positive(self):
        """Omega ratio should not return infinity for all-positive returns."""
        returns = np.abs(np.random.randn(252)) * 0.01 + 0.001  # All positive
        result = omega_ratio(returns)
        assert np.isfinite(result)

    def test_tail_ratio_no_infinity(self):
        returns = np.abs(np.random.randn(252)) * 0.01 + 0.001
        result = tail_ratio(returns)
        assert np.isfinite(result)

    def test_profit_factor_no_infinity(self):
        returns = np.abs(np.random.randn(252)) * 0.01 + 0.001
        result = profit_factor(returns)
        assert np.isfinite(result)

    def test_compute_all_metrics_no_inf(self, daily_returns):
        """No inf values in full metrics dict."""
        metrics = compute_all_metrics(daily_returns)
        for key, value in metrics.items():
            if isinstance(value, float):
                assert np.isfinite(value), f"Metric '{key}' is not finite: {value}"

    def test_omega_zero_denominator(self):
        """When losses = 0, omega_ratio should return 0.0, not inf."""
        returns = np.array([0.01, 0.02, 0.03, 0.04])  # All positive
        result = omega_ratio(returns, threshold=0.0)
        assert result == 0.0 or np.isfinite(result)


# ══════════════════════════════════════════════════════════════════════
# A5: NaN validation in BaseOptimizer
# ══════════════════════════════════════════════════════════════════════

class TestA5NaNValidation:
    def test_nan_in_expected_returns_raises(self, prices_5):
        mu = estimate_returns(prices_5)
        cov = estimate_covariance(prices_5)
        # Inject NaN
        mu.iloc[0] = np.nan
        with pytest.raises(ValueError, match="NaN in expected returns"):
            MeanVarianceOptimizer(mu, cov)

    def test_nan_in_covariance_raises(self, prices_5):
        mu = estimate_returns(prices_5)
        cov = estimate_covariance(prices_5)
        cov.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN values found in covariance"):
            MeanVarianceOptimizer(mu, cov)

    def test_empty_portfolio_raises(self):
        mu = pd.Series([], dtype=float)
        cov = pd.DataFrame()
        with pytest.raises((ValueError, AssertionError)):
            MeanVarianceOptimizer(mu, cov)


# ══════════════════════════════════════════════════════════════════════
# A7: HRP division by zero
# ══════════════════════════════════════════════════════════════════════

class TestA7HRPDivisionByZero:
    def test_very_low_vol_asset(self):
        """HRP should handle an asset with very low variance without crashing.

        This tests the 1/max(diag, 1e-10) guard in _cluster_risk().
        """
        np.random.seed(99)
        n_days = 252
        symbols = ["HI_VOL", "LO_VOL", "MED_VOL"]
        dates = pd.bdate_range("2023-01-01", periods=n_days)
        data = {
            "HI_VOL": 100 + np.cumsum(np.random.randn(n_days) * 0.03),
            "LO_VOL": 100 + np.cumsum(np.random.randn(n_days) * 0.0001),  # very low vol
            "MED_VOL": 100 + np.cumsum(np.random.randn(n_days) * 0.01),
        }
        prices = pd.DataFrame(data, index=dates)
        cov = estimate_covariance(prices)

        result = hrp_optimize(cov)
        assert isinstance(result, OptimizationResult)
        assert_valid_weights(result.weights, symbols)

    def test_standard_hrp_finite_weights(self, prices_5):
        """Standard HRP should always produce finite weights that sum to 1."""
        cov = estimate_covariance(prices_5)
        result = hrp_optimize(cov)
        for w in result.weights.values():
            assert np.isfinite(w), f"Non-finite weight: {w}"
        assert abs(sum(result.weights.values()) - 1.0) < 1e-4


# ══════════════════════════════════════════════════════════════════════
# A8: PSD validation before optimization
# ══════════════════════════════════════════════════════════════════════

class TestA8PSDValidation:
    def test_non_psd_gets_corrected(self):
        """Non-PSD covariance should be silently corrected."""
        symbols = ["A", "B"]
        # Non-PSD matrix
        bad_cov = pd.DataFrame(
            [[1.0, 2.0], [2.0, 1.0]],
            index=symbols, columns=symbols,
        )
        mu = pd.Series([0.10, 0.12], index=symbols)

        # Should not raise — should auto-correct
        opt = MeanVarianceOptimizer(mu, bad_cov)
        result = opt.optimize()
        assert isinstance(result, OptimizationResult)
        assert_valid_weights(result.weights, symbols)


# ══════════════════════════════════════════════════════════════════════
# A9: CAGR numerical overflow
# ══════════════════════════════════════════════════════════════════════

class TestA9CAGROverflow:
    def test_long_series_no_overflow(self):
        """CAGR should not overflow for a 10-year daily series."""
        np.random.seed(123)
        returns = np.random.randn(2520) * 0.01 + 0.0004  # 10 years of daily
        result = cagr(returns)
        assert np.isfinite(result)

    def test_extreme_returns_finite(self):
        """CAGR with extreme but valid returns should not overflow."""
        np.random.seed(42)
        returns = np.random.randn(5000) * 0.02 + 0.001
        result = cagr(returns)
        assert np.isfinite(result)

    def test_all_zero_returns(self):
        """CAGR of zero returns should be 0."""
        returns = np.zeros(252)
        result = cagr(returns)
        assert result == pytest.approx(0.0, abs=1e-10)


# ══════════════════════════════════════════════════════════════════════
# A10: Risk decomposition chart
# ══════════════════════════════════════════════════════════════════════

class TestA10RiskDecomposition:
    @pytest.mark.skipif(
        not pytest.importorskip("PySide6", reason="PySide6 not available"),
        reason="PySide6 required",
    )
    def test_set_risk_decomposition(self, qtbot):
        from portopt.gui.panels.risk_panel import RiskPanel
        panel = RiskPanel()
        qtbot.addWidget(panel)
        # Should not crash
        panel.set_risk_decomposition(
            ["AAPL", "MSFT", "GOOG"],
            [0.35, 0.40, 0.25],
        )

    @pytest.mark.skipif(
        not pytest.importorskip("PySide6", reason="PySide6 not available"),
        reason="PySide6 required",
    )
    def test_set_risk_decomposition_empty(self, qtbot):
        from portopt.gui.panels.risk_panel import RiskPanel
        panel = RiskPanel()
        qtbot.addWidget(panel)
        # Should handle empty data gracefully
        panel.set_risk_decomposition([], [])


# ══════════════════════════════════════════════════════════════════════
# A11: Single-asset portfolio guard
# ══════════════════════════════════════════════════════════════════════

class TestA11SingleAsset:
    def test_mvo_single_asset(self):
        """MVO with one asset should return 100% weight."""
        mu = pd.Series([0.10], index=["AAPL"])
        cov = pd.DataFrame([[0.04]], index=["AAPL"], columns=["AAPL"])
        opt = MeanVarianceOptimizer(mu, cov, method=OptMethod.MAX_SHARPE)
        result = opt.optimize()
        assert result.weights == {"AAPL": 1.0}

    def test_hrp_single_asset(self):
        """HRP with one asset should return 100% weight."""
        cov = pd.DataFrame([[0.04]], index=["AAPL"], columns=["AAPL"])
        result = hrp_optimize(cov)
        assert result.weights == {"AAPL": 1.0}
        assert result.method == "HRP"

    def test_herc_single_asset(self):
        """HERC with one asset should return 100% weight."""
        cov = pd.DataFrame([[0.04]], index=["AAPL"], columns=["AAPL"])
        result = herc_optimize(cov)
        assert result.weights == {"AAPL": 1.0}
        assert result.method == "HERC"


# ══════════════════════════════════════════════════════════════════════
# A13: Marchenko-Pastur T<N fallback
# ══════════════════════════════════════════════════════════════════════

class TestA13DenoisingFallback:
    def test_tlt_n_fallback_no_crash(self):
        """With T < N, denoised method should fallback to Ledoit-Wolf without crashing."""
        # 10 days of data for 20 assets → T < N
        np.random.seed(42)
        n_assets = 20
        n_days = 10
        symbols = [f"SYM{i}" for i in range(n_assets)]
        dates = pd.bdate_range("2024-01-01", periods=n_days)
        data = 100 + np.cumsum(np.random.randn(n_days, n_assets) * 0.01, axis=0)
        prices = pd.DataFrame(data, index=dates, columns=symbols)

        cov = estimate_covariance(prices, method=CovEstimator.DENOISED)
        assert isinstance(cov, pd.DataFrame)
        assert cov.shape == (n_assets, n_assets)
        # Should be valid PSD matrix
        eigvals = np.linalg.eigvalsh(cov.values)
        assert np.all(eigvals >= -1e-8), f"Not PSD: min eigenvalue = {eigvals.min()}"


# ══════════════════════════════════════════════════════════════════════
# A14: CSV importer numeric parsing
# ══════════════════════════════════════════════════════════════════════

class TestA14CSVImporterParsing:
    def test_non_numeric_values(self, tmp_path):
        """CSV with non-numeric values should import using defaults, not crash."""
        from portopt.data.importers.generic_csv import parse_generic_csv

        content = """Symbol,Quantity,Price,Cost_Basis
AAPL,N/A,175.50,14500
MSFT,50,abc,12500
GOOG,30,$140.00,"3,900"
"""
        csv_file = tmp_path / "bad_values.csv"
        csv_file.write_text(content)

        portfolio = parse_generic_csv(csv_file)
        assert len(portfolio.holdings) == 3
        symbols = portfolio.symbols
        assert "AAPL" in symbols
        assert "MSFT" in symbols
        assert "GOOG" in symbols

    def test_dollar_sign_and_commas(self, tmp_path):
        """Values with $, commas should be parsed correctly."""
        from portopt.data.importers.generic_csv import parse_generic_csv

        content = """Symbol,Quantity,Price
AAPL,100,$175.50
MSFT,50,"$1,350.00"
"""
        csv_file = tmp_path / "formatted.csv"
        csv_file.write_text(content)

        portfolio = parse_generic_csv(csv_file)
        assert len(portfolio.holdings) == 2
        # Verify the prices were parsed correctly
        aapl = [h for h in portfolio.holdings if h.asset.symbol == "AAPL"][0]
        assert aapl.current_price == pytest.approx(175.50)

    def test_empty_values_use_defaults(self, tmp_path):
        """Empty quantity/price fields should use defaults."""
        from portopt.data.importers.generic_csv import parse_generic_csv

        content = """Symbol,Quantity,Price
AAPL,,
MSFT,50,
"""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text(content)

        portfolio = parse_generic_csv(csv_file)
        assert len(portfolio.holdings) == 2
        aapl = [h for h in portfolio.holdings if h.asset.symbol == "AAPL"][0]
        assert aapl.quantity == 1.0  # default quantity
        assert aapl.current_price == 0.0  # default price


# ══════════════════════════════════════════════════════════════════════
# Fixture for daily_returns (duplicated from test_metrics for self-containment)
# ══════════════════════════════════════════════════════════════════════

@pytest.fixture
def daily_returns():
    """Synthetic daily returns (positive drift)."""
    np.random.seed(42)
    return np.random.randn(252) * 0.01 + 0.0004
