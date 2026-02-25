"""Integration tests — exercise real GUI workflows end-to-end.

These tests verify that controllers, signals, and panels work together
to produce correct results. All data is synthetic (no network calls).
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6", reason="PySide6 not installed")

from PySide6.QtCore import Qt, QTimer

from portopt.constants import (
    Colors, CovEstimator, OptMethod, RebalanceFreq, ReturnEstimator,
)
from portopt.data.models import (
    Asset, AssetType, Holding, OptimizationResult, Portfolio,
)
from portopt.engine.constraints import PortfolioConstraints


# ── Helpers ───────────────────────────────────────────────────────────

def _make_prices(symbols, n_days=504):
    """Generate synthetic GBM prices for testing."""
    np.random.seed(42)
    dt = 1 / 252
    dates = pd.bdate_range(end=date.today(), periods=n_days)
    data = {}
    for i, sym in enumerate(symbols):
        mu = 0.05 + 0.02 * i
        sigma = 0.15 + 0.03 * i
        log_rets = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn(n_days)
        data[sym] = 100.0 * np.exp(np.cumsum(log_rets))
    return pd.DataFrame(data, index=dates)


def _make_portfolio(symbols):
    """Create a test portfolio from symbol list."""
    holdings = []
    for sym in symbols:
        holdings.append(Holding(
            asset=Asset(symbol=sym, name=f"{sym} Corp"),
            quantity=100,
            cost_basis=10000,
            current_price=150.0,
            account="Individual",
        ))
    return Portfolio(name="Test", holdings=holdings)


# ── Optimization Controller Integration ──────────────────────────────

class TestOptimizationFlow:
    """Test optimization controller → signal → panel data flow.

    We call _do_optimization() synchronously (it returns a dict),
    then feed it to _on_optimization_done() which emits the signals.
    """

    def _run_opt(self, opt_ctrl, config):
        """Run optimization synchronously and feed through signal chain."""
        output = opt_ctrl._do_optimization(config)
        opt_ctrl._on_optimization_done(output)
        return output

    def test_optimize_max_sharpe_produces_valid_weights(self, qtbot):
        """Full flow: set prices → optimize → receive valid weights via signal."""
        from portopt.gui.controllers.data_controller import DataController
        from portopt.gui.controllers.optimization_controller import OptimizationController

        data_ctrl = DataController()
        opt_ctrl = OptimizationController(data_ctrl)

        symbols = ["AAPL", "MSFT", "GOOG"]
        prices = _make_prices(symbols)

        results = []
        opt_ctrl.optimization_complete.connect(lambda r: results.append(r))

        errors = []
        opt_ctrl.error.connect(lambda e: errors.append(e))

        opt_ctrl.set_prices(prices)
        opt_ctrl._symbols = symbols
        config = {
            "method": OptMethod.MAX_SHARPE,
            "risk_free_rate": 0.05,
            "risk_aversion": 1.0,
            "long_only": True,
            "min_weight": 0.0,
            "max_weight": 1.0,
        }
        self._run_opt(opt_ctrl, config)

        assert len(errors) == 0, f"Optimization error: {errors}"
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, OptimizationResult)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-4
        assert all(w >= -1e-6 for w in result.weights.values())

    def test_optimize_min_vol_produces_valid_weights(self, qtbot):
        """Min volatility optimization produces valid weights."""
        from portopt.gui.controllers.data_controller import DataController
        from portopt.gui.controllers.optimization_controller import OptimizationController

        data_ctrl = DataController()
        opt_ctrl = OptimizationController(data_ctrl)

        symbols = ["AAPL", "MSFT", "GOOG", "AMZN"]
        prices = _make_prices(symbols)

        results = []
        opt_ctrl.optimization_complete.connect(lambda r: results.append(r))

        opt_ctrl.set_prices(prices)
        opt_ctrl._symbols = symbols
        config = {
            "method": OptMethod.MIN_VOLATILITY,
            "risk_free_rate": 0.04,
            "risk_aversion": 1.0,
            "long_only": True,
            "min_weight": 0.0,
            "max_weight": 1.0,
        }
        self._run_opt(opt_ctrl, config)
        assert len(results) == 1
        assert abs(sum(results[0].weights.values()) - 1.0) < 1e-4

    def test_optimize_hrp_produces_valid_weights(self, qtbot):
        """HRP optimization produces valid weights."""
        from portopt.gui.controllers.data_controller import DataController
        from portopt.gui.controllers.optimization_controller import OptimizationController

        data_ctrl = DataController()
        opt_ctrl = OptimizationController(data_ctrl)

        symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
        prices = _make_prices(symbols)

        results = []
        opt_ctrl.optimization_complete.connect(lambda r: results.append(r))

        opt_ctrl.set_prices(prices)
        opt_ctrl._symbols = symbols
        config = {
            "method": OptMethod.HRP,
            "risk_free_rate": 0.04,
            "risk_aversion": 1.0,
            "long_only": True,
            "min_weight": 0.0,
            "max_weight": 1.0,
        }
        self._run_opt(opt_ctrl, config)
        assert len(results) == 1
        assert abs(sum(results[0].weights.values()) - 1.0) < 1e-4

    def test_frontier_generation(self, qtbot):
        """Optimization emits frontier alongside main result."""
        from portopt.gui.controllers.data_controller import DataController
        from portopt.gui.controllers.optimization_controller import OptimizationController

        data_ctrl = DataController()
        opt_ctrl = OptimizationController(data_ctrl)

        symbols = ["AAPL", "MSFT", "GOOG"]
        prices = _make_prices(symbols)

        frontiers = []
        opt_ctrl.frontier_complete.connect(
            lambda risks, rets, wts: frontiers.append((risks, rets, wts))
        )

        opt_ctrl.set_prices(prices)
        opt_ctrl._symbols = symbols
        self._run_opt(opt_ctrl, {
            "method": OptMethod.MAX_SHARPE,
            "risk_free_rate": 0.04,
            "long_only": True,
            "min_weight": 0.0,
            "max_weight": 1.0,
        })

        assert len(frontiers) >= 1
        risks, rets, wts = frontiers[0]
        assert len(risks) > 0
        assert len(rets) > 0
        assert all(r >= 0 for r in risks)

    def test_correlation_emitted(self, qtbot):
        """Optimization emits correlation matrix."""
        from portopt.gui.controllers.data_controller import DataController
        from portopt.gui.controllers.optimization_controller import OptimizationController

        data_ctrl = DataController()
        opt_ctrl = OptimizationController(data_ctrl)

        symbols = ["AAPL", "MSFT", "GOOG", "AMZN"]
        prices = _make_prices(symbols)

        corr_results = []
        opt_ctrl.correlation_ready.connect(
            lambda corr, labels: corr_results.append((corr, labels))
        )

        opt_ctrl.set_prices(prices)
        opt_ctrl._symbols = symbols
        self._run_opt(opt_ctrl, {
            "method": OptMethod.MIN_VOLATILITY,
            "risk_free_rate": 0.04,
            "long_only": True,
            "min_weight": 0.0,
            "max_weight": 1.0,
        })

        assert len(corr_results) == 1
        corr, labels = corr_results[0]
        assert len(labels) == 4
        for i in range(len(labels)):
            assert abs(corr[i, i] - 1.0) < 1e-6


# ── Backtest Controller Integration ──────────────────────────────────

class TestBacktestFlow:
    """Test backtest controller → signal → data flow.

    We call _do_backtest() synchronously (returns dict),
    then feed it to _on_backtest_done() which emits signals.
    """

    def _run_bt(self, bt_ctrl, config):
        """Run backtest synchronously and feed through signal chain."""
        output = bt_ctrl._do_backtest(config)
        bt_ctrl._on_backtest_done(output)
        return output

    def test_backtest_produces_equity_curve(self, qtbot):
        """Full backtest flow produces equity curve and metrics."""
        from portopt.gui.controllers.data_controller import DataController
        from portopt.gui.controllers.backtest_controller import BacktestController

        data_ctrl = DataController()
        bt_ctrl = BacktestController(data_ctrl)

        symbols = ["AAPL", "MSFT", "GOOG"]
        prices = _make_prices(symbols)

        equity_curves = []
        bt_ctrl.equity_curve_ready.connect(
            lambda dates, vals: equity_curves.append((dates, vals))
        )

        errors = []
        bt_ctrl.error.connect(lambda e: errors.append(e))

        bt_ctrl._prices = prices
        config = {
            "rebalance_freq": "Monthly",
            "cost_model": "Proportional",
            "cost_rate": 0.001,
            "initial_value": 100000,
            "walk_forward_enabled": False,
            "lookback": 252,
            "drift_threshold": 0.0,
        }
        self._run_bt(bt_ctrl, config)

        assert len(errors) == 0, f"Backtest error: {errors}"
        assert len(equity_curves) == 1

        dates, vals = equity_curves[0]
        assert len(dates) > 0
        assert len(vals) > 0
        # Starting value should be approximately the initial value
        assert abs(vals[0] - 100000) / 100000 < 0.05  # 5% tolerance for rebalance cost

    def test_backtest_with_walk_forward(self, qtbot):
        """Walk-forward backtest produces equity curve."""
        from portopt.gui.controllers.data_controller import DataController
        from portopt.gui.controllers.backtest_controller import BacktestController

        data_ctrl = DataController()
        bt_ctrl = BacktestController(data_ctrl)

        symbols = ["AAPL", "MSFT"]
        prices = _make_prices(symbols, n_days=756)  # 3 years for walk-forward

        equity_curves = []
        bt_ctrl.equity_curve_ready.connect(
            lambda dates, vals: equity_curves.append((dates, vals))
        )

        errors = []
        bt_ctrl.error.connect(lambda e: errors.append(e))

        bt_ctrl._prices = prices
        config = {
            "rebalance_freq": "Quarterly",
            "cost_model": "Zero",
            "cost_rate": 0.0,
            "initial_value": 50000,
            "walk_forward_enabled": True,
            "wf_train_months": 6,
            "wf_test_months": 3,
            "wf_anchored": False,
            "lookback": 252,
            "drift_threshold": 0.0,
        }
        self._run_bt(bt_ctrl, config)

        assert len(errors) == 0, f"Walk-forward error: {errors}"
        assert len(equity_curves) == 1


# ── Panel Data Display Integration ───────────────────────────────────

class TestPanelDataFlow:
    """Test that panels correctly display data from controllers."""

    def test_weights_panel_displays_optimization_result(self, qtbot):
        """Weights panel renders optimization output correctly."""
        from portopt.gui.panels.weights_panel import WeightsPanel

        panel = WeightsPanel()
        qtbot.addWidget(panel)

        current = {"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2}
        optimized = {"AAPL": 0.4, "MSFT": 0.35, "GOOG": 0.25}
        panel.set_weights(current, optimized)

        # Table should have 3 rows
        assert panel._table.rowCount() == 3

    def test_frontier_panel_renders_curve(self, qtbot):
        """Frontier panel accepts risk/return arrays."""
        from portopt.gui.panels.frontier_panel import FrontierPanel

        panel = FrontierPanel()
        qtbot.addWidget(panel)

        risks = np.linspace(0.05, 0.30, 20)
        returns = np.linspace(0.03, 0.15, 20)
        panel.set_frontier(risks, returns)

    def test_metrics_panel_displays_all_metrics(self, qtbot):
        """Metrics panel shows all metric categories."""
        from portopt.gui.panels.metrics_panel import MetricsPanel

        panel = MetricsPanel()
        qtbot.addWidget(panel)

        metrics = {
            "annual_return": 0.12,
            "annual_volatility": 0.18,
            "sharpe_ratio": 1.2,
            "sortino_ratio": 1.5,
            "max_drawdown": -0.15,
            "calmar_ratio": 0.8,
            "var_95": -0.02,
            "cvar_95": -0.03,
            "skewness": -0.2,
            "kurtosis": 3.5,
            "beta": 1.1,
            "alpha": 0.02,
            "treynor_ratio": 0.065,
            "information_ratio": 0.5,
        }
        panel.set_metrics(metrics)

    def test_correlation_panel_heatmap(self, qtbot):
        """Correlation panel renders heatmap from matrix."""
        from portopt.gui.panels.correlation_panel import CorrelationPanel

        panel = CorrelationPanel()
        qtbot.addWidget(panel)

        symbols = ["AAPL", "MSFT", "GOOG", "AMZN"]
        corr = np.array([
            [1.0, 0.7, 0.5, 0.6],
            [0.7, 1.0, 0.4, 0.5],
            [0.5, 0.4, 1.0, 0.3],
            [0.6, 0.5, 0.3, 1.0],
        ])
        panel.set_correlation(corr, symbols)

    def test_portfolio_panel_with_multiple_accounts(self, qtbot):
        """Portfolio panel handles multi-account portfolios."""
        from portopt.gui.panels.portfolio_panel import PortfolioPanel
        from portopt.data.models import AccountSummary

        panel = PortfolioPanel()
        qtbot.addWidget(panel)

        portfolio = Portfolio(
            name="Multi-Account",
            holdings=[
                Holding(
                    asset=Asset(symbol="AAPL", name="Apple", sector="Technology"),
                    quantity=50, cost_basis=7500, current_price=175.0,
                    account="Taxable",
                ),
                Holding(
                    asset=Asset(symbol="MSFT", name="Microsoft", sector="Technology"),
                    quantity=30, cost_basis=9000, current_price=350.0,
                    account="Roth IRA",
                ),
                Holding(
                    asset=Asset(symbol="JNJ", name="J&J", sector="Healthcare"),
                    quantity=40, cost_basis=6000, current_price=155.0,
                    account="Taxable",
                ),
                Holding(
                    asset=Asset(symbol="V", name="Visa", sector="Financials"),
                    quantity=20, cost_basis=5000, current_price=280.0,
                    account="401k",
                ),
            ],
            accounts=[
                AccountSummary(account_id="1", account_name="Taxable", total_value=14950),
                AccountSummary(account_id="2", account_name="Roth IRA", total_value=10500),
                AccountSummary(account_id="3", account_name="401k", total_value=5600),
            ],
        )
        panel.set_portfolio(portfolio)

        # All 4 holdings visible
        assert panel._table.rowCount() == 4
        # 4 filter options: All + 3 accounts
        assert panel._account_filter.count() == 4

        # Filter to Taxable
        panel._account_filter.setCurrentText("Taxable")
        assert panel._table.rowCount() == 2

        # Filter to 401k
        panel._account_filter.setCurrentText("401k")
        assert panel._table.rowCount() == 1


# ── Strategy Lab Integration ─────────────────────────────────────────

class TestStrategyLabFlow:
    """Test Strategy Lab panel's internal optimization/backtest flow."""

    def test_strategy_lab_creates_with_controllers(self, qtbot):
        """Strategy Lab initializes with controller hookups."""
        from portopt.gui.panels.strategy_lab_panel import StrategyLabPanel
        from portopt.gui.controllers.data_controller import DataController
        from portopt.gui.controllers.optimization_controller import OptimizationController
        from portopt.gui.controllers.backtest_controller import BacktestController

        panel = StrategyLabPanel()
        qtbot.addWidget(panel)

        data_ctrl = DataController()
        opt_ctrl = OptimizationController(data_ctrl)
        bt_ctrl = BacktestController(data_ctrl)

        panel.set_controllers(data_ctrl, opt_ctrl, bt_ctrl)
        assert panel._opt_controller is not None
        assert panel._bt_controller is not None

    def test_strategy_lab_add_tickers(self, qtbot):
        """Strategy Lab can add tickers to its list."""
        from portopt.gui.panels.strategy_lab_panel import StrategyLabPanel

        panel = StrategyLabPanel()
        qtbot.addWidget(panel)

        panel._ticker_input.setText("AAPL")
        panel._add_ticker()
        panel._ticker_input.setText("MSFT")
        panel._add_ticker()
        panel._ticker_input.setText("GOOG")
        panel._add_ticker()

        assert panel._ticker_list.count() == 3

    def test_strategy_lab_deduplicates_tickers(self, qtbot):
        """Strategy Lab doesn't add duplicate tickers."""
        from portopt.gui.panels.strategy_lab_panel import StrategyLabPanel

        panel = StrategyLabPanel()
        qtbot.addWidget(panel)

        panel._ticker_input.setText("AAPL")
        panel._add_ticker()
        panel._ticker_input.setText("AAPL")
        panel._add_ticker()

        assert panel._ticker_list.count() == 1


# ── Export Dialog Integration ────────────────────────────────────────

class TestExportFlow:
    """Test export dialog with real data."""

    def test_export_dialog_initializes(self, qtbot):
        """Export dialog creates without errors."""
        from portopt.gui.dialogs.export_dialog import ExportDialog

        dialog = ExportDialog()
        qtbot.addWidget(dialog)

    def test_export_dialog_csv_format(self, qtbot):
        """Export dialog lists CSV as an option."""
        from portopt.gui.dialogs.export_dialog import ExportDialog

        dialog = ExportDialog()
        qtbot.addWidget(dialog)

        # Should have format options
        formats = [dialog._format_combo.itemText(i)
                    for i in range(dialog._format_combo.count())]
        assert any("CSV" in f for f in formats)


# ── Risk Analysis Integration ────────────────────────────────────────

class TestRiskAnalysisFlow:
    """Test risk-related panels with computed data."""

    def test_risk_panel_displays_var_metrics(self, qtbot):
        """Risk panel correctly shows VaR and CVaR."""
        from portopt.gui.panels.risk_panel import RiskPanel

        panel = RiskPanel()
        qtbot.addWidget(panel)

        risk_data = {
            "var_95": -0.0234,
            "cvar_95": -0.0345,
            "var_99": -0.0456,
            "cvar_99": -0.0567,
            "max_drawdown": -0.152,
            "current_drawdown": -0.032,
        }
        panel.set_risk_metrics(risk_data)

    def test_backtest_trade_blotter(self, qtbot):
        """Trade blotter panel shows trade history correctly."""
        from portopt.gui.panels.trade_blotter_panel import TradeBlotterPanel

        panel = TradeBlotterPanel()
        qtbot.addWidget(panel)

        trades = [
            {"date": "2024-01-02", "symbol": "AAPL", "side": "BUY",
             "quantity": 100, "price": 175.0, "cost": 0.175, "weight_after": 0.33},
            {"date": "2024-01-02", "symbol": "MSFT", "side": "BUY",
             "quantity": 50, "price": 350.0, "cost": 0.175, "weight_after": 0.33},
            {"date": "2024-01-02", "symbol": "GOOG", "side": "BUY",
             "quantity": 80, "price": 130.0, "cost": 0.104, "weight_after": 0.33},
            {"date": "2024-04-01", "symbol": "AAPL", "side": "SELL",
             "quantity": 20, "price": 185.0, "cost": 0.037, "weight_after": 0.28},
            {"date": "2024-04-01", "symbol": "GOOG", "side": "BUY",
             "quantity": 15, "price": 142.0, "cost": 0.021, "weight_after": 0.38},
        ]
        panel.set_trades(trades)
        assert panel._table.rowCount() == 5


# ── Console Panel Integration ────────────────────────────────────────

class TestConsolePanelFlow:
    """Test console panel logging at different levels."""

    def test_console_log_levels(self, qtbot):
        """Console panel handles info, warning, error, success."""
        from portopt.gui.panels.console_panel import ConsolePanel

        panel = ConsolePanel()
        qtbot.addWidget(panel)

        panel.log_info("Data loaded for 5 symbols")
        panel.log_warning("Missing data for TSLA")
        panel.log_error("Optimization failed: singular matrix")
        panel.log_success("Backtest complete: Sharpe=1.45")

        # Console uses QTextEdit, check content contains all messages
        text = panel._text.toPlainText()
        assert "Data loaded" in text
        assert "Missing data" in text
        assert "Optimization failed" in text
        assert "Backtest complete" in text

    def test_console_clear(self, qtbot):
        """Console can be cleared."""
        from portopt.gui.panels.console_panel import ConsolePanel

        panel = ConsolePanel()
        qtbot.addWidget(panel)

        panel.log_info("message 1")
        panel.log_info("message 2")
        assert "message 1" in panel._text.toPlainText()

        panel._clear()
        assert panel._text.toPlainText() == ""
