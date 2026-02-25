"""Stress tests — verify performance with large portfolios and edge cases.

Tests exercise the engine and controller layers with 500+ holdings,
large price matrices, and boundary conditions.
"""

from __future__ import annotations

from datetime import date, timedelta
import time

import numpy as np
import pandas as pd
import pytest

from portopt.constants import CovEstimator, OptMethod, ReturnEstimator
from portopt.data.models import Asset, AssetType, Holding, Portfolio
from portopt.engine.constraints import PortfolioConstraints


# ── Helpers ───────────────────────────────────────────────────────────

def _make_large_prices(n_assets: int, n_days: int = 504) -> tuple[list[str], pd.DataFrame]:
    """Generate synthetic prices for many assets."""
    np.random.seed(42)
    symbols = [f"SYM{i:04d}" for i in range(n_assets)]
    dt = 1 / 252
    dates = pd.bdate_range(end=date.today(), periods=n_days)
    data = {}
    for i, sym in enumerate(symbols):
        mu = 0.03 + 0.001 * (i % 50)
        sigma = 0.10 + 0.005 * (i % 40)
        log_rets = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn(n_days)
        data[sym] = 100.0 * np.exp(np.cumsum(log_rets))
    return symbols, pd.DataFrame(data, index=dates)


def _make_large_portfolio(n_holdings: int) -> Portfolio:
    """Create a portfolio with many holdings."""
    holdings = []
    for i in range(n_holdings):
        sym = f"SYM{i:04d}"
        holdings.append(Holding(
            asset=Asset(symbol=sym, name=f"Company {i}"),
            quantity=100 + i,
            cost_basis=10000 + i * 100,
            current_price=100 + np.random.randn() * 10,
            account="Individual" if i % 2 == 0 else "Retirement",
        ))
    return Portfolio(name="Large Portfolio", holdings=holdings)


# ── Large Portfolio Stress Tests ─────────────────────────────────────

class TestLargePortfolioCreation:
    """Test creating and manipulating large portfolios."""

    def test_create_500_holding_portfolio(self):
        """500-holding portfolio creates without issues."""
        portfolio = _make_large_portfolio(500)
        assert len(portfolio.holdings) == 500
        assert portfolio.total_value > 0
        assert len(portfolio.symbols) == 500

    def test_create_1000_holding_portfolio(self):
        """1000-holding portfolio creates without issues."""
        portfolio = _make_large_portfolio(1000)
        assert len(portfolio.holdings) == 1000
        assert portfolio.total_value > 0

    def test_portfolio_weight_computation_500(self):
        """Weight computation works for 500 holdings."""
        portfolio = _make_large_portfolio(500)
        total = portfolio.total_value
        for h in portfolio.holdings:
            assert h.market_value >= 0 or h.market_value < 0  # no NaN


# ── Covariance Estimation Stress ─────────────────────────────────────

class TestCovarianceStress:
    """Test covariance estimation with large matrices."""

    def test_sample_cov_50_assets(self):
        """Sample covariance for 50 assets computes in < 2s."""
        from portopt.engine.risk import estimate_covariance
        _, prices = _make_large_prices(50)

        t0 = time.time()
        cov = estimate_covariance(prices, method=CovEstimator.SAMPLE)
        elapsed = time.time() - t0

        assert cov.shape == (50, 50)
        assert elapsed < 2.0, f"Sample cov took {elapsed:.1f}s (expected < 2s)"

    def test_sample_cov_100_assets(self):
        """Sample covariance for 100 assets computes in < 5s."""
        from portopt.engine.risk import estimate_covariance
        _, prices = _make_large_prices(100)

        t0 = time.time()
        cov = estimate_covariance(prices, method=CovEstimator.SAMPLE)
        elapsed = time.time() - t0

        assert cov.shape == (100, 100)
        assert elapsed < 5.0, f"Sample cov took {elapsed:.1f}s (expected < 5s)"

    def test_ledoit_wolf_100_assets(self):
        """Ledoit-Wolf shrinkage for 100 assets computes in < 5s."""
        from portopt.engine.risk import estimate_covariance
        _, prices = _make_large_prices(100)

        t0 = time.time()
        cov = estimate_covariance(prices, method=CovEstimator.LEDOIT_WOLF)
        elapsed = time.time() - t0

        assert cov.shape == (100, 100)
        assert elapsed < 5.0, f"Ledoit-Wolf took {elapsed:.1f}s (expected < 5s)"
        # Should be positive semi-definite
        eigvals = np.linalg.eigvalsh(cov.values)
        assert np.all(eigvals >= -1e-8)

    def test_covariance_matrix_symmetry(self):
        """Large covariance matrix is symmetric."""
        from portopt.engine.risk import estimate_covariance
        _, prices = _make_large_prices(50)
        cov = estimate_covariance(prices, method=CovEstimator.SAMPLE)
        np.testing.assert_array_almost_equal(cov.values, cov.values.T)


# ── Optimization Stress Tests ────────────────────────────────────────

class TestOptimizationStress:
    """Test optimization with many assets."""

    def test_max_sharpe_20_assets(self):
        """Max Sharpe optimization with 20 assets."""
        from portopt.engine.optimization.mean_variance import MeanVarianceOptimizer
        from portopt.engine.risk import estimate_covariance
        from portopt.engine.returns import estimate_returns

        _, prices = _make_large_prices(20)
        cov = estimate_covariance(prices, method=CovEstimator.SAMPLE)
        mu = estimate_returns(prices, method=ReturnEstimator.HISTORICAL_MEAN)

        t0 = time.time()
        optimizer = MeanVarianceOptimizer(
            expected_returns=mu, covariance=cov, method=OptMethod.MAX_SHARPE,
        )
        result = optimizer.optimize()
        elapsed = time.time() - t0

        assert abs(sum(result.weights.values()) - 1.0) < 1e-4
        assert elapsed < 10.0, f"Max Sharpe took {elapsed:.1f}s (expected < 10s)"

    def test_min_vol_50_assets(self):
        """Min volatility optimization with 50 assets."""
        from portopt.engine.optimization.mean_variance import MeanVarianceOptimizer
        from portopt.engine.risk import estimate_covariance
        from portopt.engine.returns import estimate_returns

        _, prices = _make_large_prices(50)
        cov = estimate_covariance(prices, method=CovEstimator.SAMPLE)
        mu = estimate_returns(prices, method=ReturnEstimator.HISTORICAL_MEAN)

        t0 = time.time()
        optimizer = MeanVarianceOptimizer(
            expected_returns=mu, covariance=cov, method=OptMethod.MIN_VOLATILITY,
        )
        result = optimizer.optimize()
        elapsed = time.time() - t0

        assert abs(sum(result.weights.values()) - 1.0) < 1e-4
        assert elapsed < 15.0, f"Min vol 50 took {elapsed:.1f}s (expected < 15s)"

    def test_hrp_100_assets(self):
        """HRP optimization with 100 assets (should be faster than MVO)."""
        from portopt.engine.optimization.hrp import hrp_optimize
        from portopt.engine.risk import estimate_covariance

        _, prices = _make_large_prices(100)
        cov = estimate_covariance(prices, method=CovEstimator.SAMPLE)

        t0 = time.time()
        result = hrp_optimize(covariance=cov, returns=prices.pct_change().dropna())
        elapsed = time.time() - t0

        assert abs(sum(result.weights.values()) - 1.0) < 1e-4
        assert elapsed < 10.0, f"HRP 100 took {elapsed:.1f}s (expected < 10s)"
        assert all(w >= 0 for w in result.weights.values())

    def test_herc_50_assets(self):
        """HERC optimization with 50 assets."""
        from portopt.engine.optimization.herc import herc_optimize
        from portopt.engine.risk import estimate_covariance

        _, prices = _make_large_prices(50)
        cov = estimate_covariance(prices, method=CovEstimator.SAMPLE)

        t0 = time.time()
        result = herc_optimize(covariance=cov, returns=prices.pct_change().dropna())
        elapsed = time.time() - t0

        assert abs(sum(result.weights.values()) - 1.0) < 1e-4
        assert elapsed < 15.0, f"HERC 50 took {elapsed:.1f}s (expected < 15s)"

    def test_inverse_variance_100_assets(self):
        """Inverse variance is fast for 100+ assets."""
        from portopt.engine.optimization.mean_variance import MeanVarianceOptimizer
        from portopt.engine.risk import estimate_covariance
        from portopt.engine.returns import estimate_returns

        _, prices = _make_large_prices(100)
        cov = estimate_covariance(prices, method=CovEstimator.SAMPLE)
        mu = estimate_returns(prices, method=ReturnEstimator.HISTORICAL_MEAN)

        t0 = time.time()
        optimizer = MeanVarianceOptimizer(
            expected_returns=mu, covariance=cov, method=OptMethod.INVERSE_VARIANCE,
        )
        result = optimizer.optimize()
        elapsed = time.time() - t0

        assert abs(sum(result.weights.values()) - 1.0) < 1e-4
        assert elapsed < 5.0, f"Inv var 100 took {elapsed:.1f}s (expected < 5s)"


# ── Backtest Stress Tests ────────────────────────────────────────────

class TestBacktestStress:
    """Test backtesting with large portfolios and long time series."""

    def test_backtest_20_assets_2years(self):
        """Backtest 20 assets over 2 years of data."""
        from portopt.backtest.engine import BacktestConfig, BacktestEngine
        from portopt.constants import RebalanceFreq

        _, prices = _make_large_prices(20, n_days=504)

        config = BacktestConfig(
            initial_value=100000,
            rebalance_freq=RebalanceFreq.MONTHLY,
        )
        engine = BacktestEngine(prices, config)

        t0 = time.time()
        output = engine.run()
        elapsed = time.time() - t0

        assert output.result is not None
        assert len(output.result.portfolio_values) > 0
        assert elapsed < 10.0, f"Backtest 20 took {elapsed:.1f}s (expected < 10s)"

    def test_backtest_50_assets_3years(self):
        """Backtest 50 assets over 3 years of data."""
        from portopt.backtest.engine import BacktestConfig, BacktestEngine
        from portopt.constants import RebalanceFreq

        _, prices = _make_large_prices(50, n_days=756)

        config = BacktestConfig(
            initial_value=500000,
            rebalance_freq=RebalanceFreq.QUARTERLY,
        )
        engine = BacktestEngine(prices, config)

        t0 = time.time()
        output = engine.run()
        elapsed = time.time() - t0

        assert output.result is not None
        assert len(output.result.portfolio_values) > 0
        assert elapsed < 20.0, f"Backtest 50 took {elapsed:.1f}s (expected < 20s)"


# ── Metrics Stress Tests ─────────────────────────────────────────────

class TestMetricsStress:
    """Test metric computation with large datasets."""

    def test_metrics_1000_days(self):
        """Compute metrics on 1000-day return series."""
        from portopt.engine.metrics import compute_all_metrics

        np.random.seed(42)
        returns = pd.Series(
            np.random.randn(1000) * 0.01 + 0.0003,
            index=pd.bdate_range(end=date.today(), periods=1000),
        )

        t0 = time.time()
        metrics = compute_all_metrics(returns)
        elapsed = time.time() - t0

        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert elapsed < 2.0, f"Metrics took {elapsed:.1f}s (expected < 2s)"

    def test_metrics_5000_days(self):
        """Compute metrics on 20-year return series."""
        from portopt.engine.metrics import compute_all_metrics

        np.random.seed(42)
        returns = pd.Series(
            np.random.randn(5000) * 0.01 + 0.0003,
            index=pd.bdate_range(end=date.today(), periods=5000),
        )

        t0 = time.time()
        metrics = compute_all_metrics(returns)
        elapsed = time.time() - t0

        assert "sharpe_ratio" in metrics
        assert elapsed < 5.0, f"Metrics took {elapsed:.1f}s (expected < 5s)"


# ── Edge Cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_single_asset_optimization(self):
        """Single-asset optimization returns 100% weight."""
        from portopt.engine.optimization.mean_variance import MeanVarianceOptimizer
        from portopt.engine.risk import estimate_covariance
        from portopt.engine.returns import estimate_returns

        _, prices = _make_large_prices(1)
        cov = estimate_covariance(prices, method=CovEstimator.SAMPLE)
        mu = estimate_returns(prices, method=ReturnEstimator.HISTORICAL_MEAN)

        optimizer = MeanVarianceOptimizer(
            expected_returns=mu, covariance=cov,
            method=OptMethod.INVERSE_VARIANCE,
        )
        result = optimizer.optimize()
        assert abs(sum(result.weights.values()) - 1.0) < 1e-4

    def test_two_asset_optimization(self):
        """Two-asset optimization works correctly."""
        from portopt.engine.optimization.hrp import hrp_optimize
        from portopt.engine.risk import estimate_covariance

        _, prices = _make_large_prices(2)
        cov = estimate_covariance(prices, method=CovEstimator.SAMPLE)
        result = hrp_optimize(covariance=cov, returns=prices.pct_change().dropna())
        assert abs(sum(result.weights.values()) - 1.0) < 1e-4
        assert len(result.weights) == 2

    def test_identical_prices_handled(self):
        """Assets with identical prices don't crash optimization."""
        from portopt.engine.optimization.mean_variance import MeanVarianceOptimizer
        from portopt.engine.risk import estimate_covariance
        from portopt.engine.returns import estimate_returns

        np.random.seed(42)
        n_days = 252
        dates = pd.bdate_range(end=date.today(), periods=n_days)
        base = 100.0 * np.exp(np.cumsum(np.random.randn(n_days) * 0.01))
        prices = pd.DataFrame({
            "A": base,
            "B": base * 1.001,  # Nearly identical
            "C": base * 1.5,     # Different level, same pattern
        }, index=dates)

        cov = estimate_covariance(prices, method=CovEstimator.LEDOIT_WOLF)
        mu = estimate_returns(prices, method=ReturnEstimator.HISTORICAL_MEAN)
        optimizer = MeanVarianceOptimizer(
            expected_returns=mu, covariance=cov,
            method=OptMethod.MIN_VOLATILITY,
        )
        result = optimizer.optimize()
        assert abs(sum(result.weights.values()) - 1.0) < 1e-4

    def test_very_short_price_history(self):
        """Optimization handles short history (30 days)."""
        from portopt.engine.optimization.hrp import hrp_optimize
        from portopt.engine.risk import estimate_covariance

        _, prices = _make_large_prices(5, n_days=30)
        cov = estimate_covariance(prices, method=CovEstimator.SAMPLE)
        result = hrp_optimize(covariance=cov, returns=prices.pct_change().dropna())
        assert abs(sum(result.weights.values()) - 1.0) < 1e-4

    def test_portfolio_with_zero_price_assets(self):
        """Portfolio handles assets with zero current price."""
        portfolio = Portfolio(
            name="Test",
            holdings=[
                Holding(
                    asset=Asset(symbol="DELISTED", name="Delisted Corp"),
                    quantity=100, cost_basis=5000, current_price=0.0,
                    account="Individual",
                ),
                Holding(
                    asset=Asset(symbol="AAPL", name="Apple"),
                    quantity=50, cost_basis=7500, current_price=175.0,
                    account="Individual",
                ),
            ],
        )
        assert len(portfolio.holdings) == 2
        # Total value should still work
        assert portfolio.total_value > 0


# ── GUI Panel Stress (if PySide6 available) ──────────────────────────

@pytest.fixture
def _skip_no_gui():
    """Skip if PySide6 not available."""
    pytest.importorskip("PySide6", reason="PySide6 not installed")


class TestPanelStress:
    """Test GUI panels with large datasets."""

    def test_portfolio_panel_500_holdings(self, qtbot, _skip_no_gui):
        """Portfolio panel renders 500 holdings."""
        from portopt.gui.panels.portfolio_panel import PortfolioPanel

        panel = PortfolioPanel()
        qtbot.addWidget(panel)

        portfolio = _make_large_portfolio(500)

        t0 = time.time()
        panel.set_portfolio(portfolio)
        elapsed = time.time() - t0

        assert panel._table.rowCount() == 500
        assert elapsed < 5.0, f"Panel render took {elapsed:.1f}s (expected < 5s)"

    def test_correlation_panel_50x50(self, qtbot, _skip_no_gui):
        """Correlation panel renders 50x50 heatmap."""
        from portopt.gui.panels.correlation_panel import CorrelationPanel

        panel = CorrelationPanel()
        qtbot.addWidget(panel)

        n = 50
        symbols = [f"SYM{i:03d}" for i in range(n)]
        corr = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                val = 0.3 + 0.4 * np.random.random()
                corr[i, j] = val
                corr[j, i] = val

        t0 = time.time()
        panel.set_correlation(corr, symbols)
        elapsed = time.time() - t0

        assert elapsed < 5.0, f"Heatmap render took {elapsed:.1f}s (expected < 5s)"

    def test_trade_blotter_1000_trades(self, qtbot, _skip_no_gui):
        """Trade blotter renders 1000 trades."""
        from portopt.gui.panels.trade_blotter_panel import TradeBlotterPanel

        panel = TradeBlotterPanel()
        qtbot.addWidget(panel)

        trades = []
        for i in range(1000):
            trades.append({
                "date": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                "symbol": f"SYM{i % 50:03d}",
                "side": "BUY" if i % 3 != 0 else "SELL",
                "quantity": 10 + i % 100,
                "price": 50 + (i % 200),
                "cost": 0.05 * (10 + i % 100),
                "weight_after": 0.02,
            })

        t0 = time.time()
        panel.set_trades(trades)
        elapsed = time.time() - t0

        assert panel._table.rowCount() == 1000
        assert elapsed < 5.0, f"Blotter render took {elapsed:.1f}s (expected < 5s)"

    def test_weights_panel_100_assets(self, qtbot, _skip_no_gui):
        """Weights panel renders 100 assets."""
        from portopt.gui.panels.weights_panel import WeightsPanel

        panel = WeightsPanel()
        qtbot.addWidget(panel)

        n = 100
        current = {f"SYM{i:03d}": 1.0 / n for i in range(n)}
        optimized = {f"SYM{i:03d}": max(0, 1.0 / n + np.random.randn() * 0.005) for i in range(n)}
        # Normalize
        total = sum(optimized.values())
        optimized = {k: v / total for k, v in optimized.items()}

        t0 = time.time()
        panel.set_weights(current, optimized)
        elapsed = time.time() - t0

        assert panel._table.rowCount() == 100
        assert elapsed < 3.0, f"Weights render took {elapsed:.1f}s (expected < 3s)"
