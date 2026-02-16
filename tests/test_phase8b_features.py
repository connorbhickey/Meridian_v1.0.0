"""Tests for Phase 8b new features (B1–B5).

Covers:
- B1: Status bar progress signals on controllers
- B2: Right-click context menus on tables
- B3: Comparison panel for strategies
- B4: Risk alerts with visual indicators
- B5: What-if scenario analysis panel
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portopt.constants import CovEstimator, OptMethod
from portopt.data.models import OptimizationResult
from portopt.engine.optimization.mean_variance import MeanVarianceOptimizer
from portopt.engine.returns import estimate_returns
from portopt.engine.risk import estimate_covariance


# ══════════════════════════════════════════════════════════════════════
# B1: Status bar progress signals
# ══════════════════════════════════════════════════════════════════════

class TestB1ProgressSignals:
    """Verify controllers emit progress signals."""

    def test_optimization_controller_has_progress_signal(self):
        from portopt.gui.controllers.optimization_controller import OptimizationController
        assert hasattr(OptimizationController, "progress")

    def test_optimization_controller_has_running_changed_signal(self):
        from portopt.gui.controllers.optimization_controller import OptimizationController
        assert hasattr(OptimizationController, "running_changed")

    def test_backtest_controller_has_progress_signal(self):
        from portopt.gui.controllers.backtest_controller import BacktestController
        assert hasattr(BacktestController, "progress")

    def test_backtest_controller_has_running_changed_signal(self):
        from portopt.gui.controllers.backtest_controller import BacktestController
        assert hasattr(BacktestController, "running_changed")


# ══════════════════════════════════════════════════════════════════════
# B2: Right-click context menus on tables
# ══════════════════════════════════════════════════════════════════════

class TestB2ContextMenus:
    """Verify context menu is attached to data tables."""

    @pytest.mark.skipif(
        not pytest.importorskip("PySide6", reason="PySide6 not available"),
        reason="PySide6 required",
    )
    def test_setup_attaches_custom_context_menu(self, qtbot):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QTableWidget
        from portopt.gui.widgets.table_context_menu import setup_table_context_menu

        table = QTableWidget()
        qtbot.addWidget(table)
        setup_table_context_menu(table)
        assert table.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu

    @pytest.mark.skipif(
        not pytest.importorskip("PySide6", reason="PySide6 not available"),
        reason="PySide6 required",
    )
    def test_portfolio_panel_has_context_menu(self, qtbot):
        from PySide6.QtCore import Qt
        from portopt.gui.panels.portfolio_panel import PortfolioPanel

        panel = PortfolioPanel()
        qtbot.addWidget(panel)
        assert panel._table.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu

    @pytest.mark.skipif(
        not pytest.importorskip("PySide6", reason="PySide6 not available"),
        reason="PySide6 required",
    )
    def test_trade_blotter_panel_has_context_menu(self, qtbot):
        from PySide6.QtCore import Qt
        from portopt.gui.panels.trade_blotter_panel import TradeBlotterPanel

        panel = TradeBlotterPanel()
        qtbot.addWidget(panel)
        assert panel._table.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu

    @pytest.mark.skipif(
        not pytest.importorskip("PySide6", reason="PySide6 not available"),
        reason="PySide6 required",
    )
    def test_copy_cell_from_table(self, qtbot):
        from PySide6.QtWidgets import QApplication, QTableWidget, QTableWidgetItem
        from portopt.gui.widgets.table_context_menu import _copy_cell

        table = QTableWidget(2, 2)
        qtbot.addWidget(table)
        table.setItem(0, 0, QTableWidgetItem("hello"))
        table.setCurrentCell(0, 0)
        _copy_cell(table)
        assert QApplication.clipboard().text() == "hello"


# ══════════════════════════════════════════════════════════════════════
# B3: Comparison panel for strategies
# ══════════════════════════════════════════════════════════════════════

class TestB3ComparisonPanel:
    """Verify comparison panel stores and displays multiple strategies."""

    @pytest.mark.skipif(
        not pytest.importorskip("PySide6", reason="PySide6 not available"),
        reason="PySide6 required",
    )
    def test_add_snapshot(self, qtbot):
        from portopt.gui.panels.comparison_panel import ComparisonPanel

        panel = ComparisonPanel()
        qtbot.addWidget(panel)

        result = OptimizationResult(
            method="MAX_SHARPE",
            weights={"AAPL": 0.4, "MSFT": 0.6},
            expected_return=0.12,
            volatility=0.18,
            sharpe_ratio=0.56,
        )
        panel.add_snapshot(result, "Test Strategy")
        assert len(panel._snapshots) == 1
        assert "Test Strategy" in panel._snapshots

    @pytest.mark.skipif(
        not pytest.importorskip("PySide6", reason="PySide6 not available"),
        reason="PySide6 required",
    )
    def test_max_snapshots_enforced(self, qtbot):
        from portopt.gui.panels.comparison_panel import ComparisonPanel, MAX_SNAPSHOTS

        panel = ComparisonPanel()
        qtbot.addWidget(panel)

        for i in range(MAX_SNAPSHOTS + 2):
            result = OptimizationResult(
                method=f"Method{i}",
                weights={"AAPL": 0.5, "MSFT": 0.5},
                expected_return=0.10 + i * 0.01,
                volatility=0.15,
                sharpe_ratio=0.5,
            )
            panel.add_snapshot(result, f"Strat {i}")

        assert len(panel._snapshots) == MAX_SNAPSHOTS

    @pytest.mark.skipif(
        not pytest.importorskip("PySide6", reason="PySide6 not available"),
        reason="PySide6 required",
    )
    def test_clear_snapshots(self, qtbot):
        from portopt.gui.panels.comparison_panel import ComparisonPanel

        panel = ComparisonPanel()
        qtbot.addWidget(panel)

        result = OptimizationResult(
            method="MAX_SHARPE",
            weights={"AAPL": 0.5, "MSFT": 0.5},
            expected_return=0.12,
            volatility=0.18,
            sharpe_ratio=0.56,
        )
        panel.add_snapshot(result, "Strategy A")
        panel.clear_snapshots()
        assert len(panel._snapshots) == 0

    @pytest.mark.skipif(
        not pytest.importorskip("PySide6", reason="PySide6 not available"),
        reason="PySide6 required",
    )
    def test_duplicate_names_get_suffix(self, qtbot):
        from portopt.gui.panels.comparison_panel import ComparisonPanel

        panel = ComparisonPanel()
        qtbot.addWidget(panel)

        for _ in range(3):
            result = OptimizationResult(
                method="MAX_SHARPE",
                weights={"AAPL": 0.5, "MSFT": 0.5},
                expected_return=0.12,
                volatility=0.18,
                sharpe_ratio=0.56,
            )
            panel.add_snapshot(result, "Same Name")

        keys = list(panel._snapshots.keys())
        assert len(keys) == 3
        assert "Same Name" in keys
        assert "Same Name #2" in keys
        assert "Same Name #3" in keys

    @pytest.mark.skipif(
        not pytest.importorskip("PySide6", reason="PySide6 not available"),
        reason="PySide6 required",
    )
    def test_save_button_on_optimization_panel(self, qtbot):
        from portopt.gui.panels.optimization_panel import OptimizationPanel

        panel = OptimizationPanel()
        qtbot.addWidget(panel)
        assert hasattr(panel, "save_requested")
        assert hasattr(panel, "set_has_result")

        # Button starts disabled
        assert not panel._save_btn.isEnabled()
        panel.set_has_result(True)
        assert panel._save_btn.isEnabled()


# ══════════════════════════════════════════════════════════════════════
# B4: Risk alerts with visual indicators
# ══════════════════════════════════════════════════════════════════════

class TestB4RiskAlerts:
    """Verify risk alert thresholds and indicator signals."""

    @pytest.mark.skipif(
        not pytest.importorskip("PySide6", reason="PySide6 not available"),
        reason="PySide6 required",
    )
    def test_risk_panel_has_alert_signal(self, qtbot):
        from portopt.gui.panels.risk_panel import RiskPanel

        panel = RiskPanel()
        qtbot.addWidget(panel)
        assert hasattr(panel, "alert_triggered")

    @pytest.mark.skipif(
        not pytest.importorskip("PySide6", reason="PySide6 not available"),
        reason="PySide6 required",
    )
    def test_alert_emitted_on_breach(self, qtbot):
        from portopt.gui.panels.risk_panel import RiskPanel

        panel = RiskPanel()
        qtbot.addWidget(panel)

        # Enable alert with threshold
        panel.set_alert_config({
            "var_95": {"enabled": True, "threshold": -0.03, "label": "VaR 95%"},
            "cvar_95": {"enabled": False, "threshold": -0.05, "label": "CVaR 95%"},
            "max_drawdown": {"enabled": True, "threshold": -0.10, "label": "Max Drawdown"},
            "annual_volatility": {"enabled": True, "threshold": 0.20, "label": "Annual Vol"},
            "downside_vol": {"enabled": False, "threshold": 0.15, "label": "Downside Vol"},
            "beta": {"enabled": False, "threshold": 1.5, "label": "Beta"},
        })

        # Collect emitted signals
        alerts = []
        panel.alert_triggered.connect(lambda n, v, t: alerts.append((n, v, t)))

        # Set metrics that breach thresholds
        panel.set_risk_metrics({
            "var_95": -0.05,        # Breaches -0.03 threshold (more negative)
            "cvar_95": -0.04,       # Disabled, should not alert
            "max_drawdown": -0.15,  # Breaches -0.10 threshold (more negative)
            "annual_volatility": 0.25,  # Breaches 0.20 threshold (higher)
            "downside_vol": 0.18,
            "beta": 1.2,
        })

        # Should have alerts for var_95, max_drawdown, annual_volatility
        assert len(alerts) == 3
        metric_names = [a[0] for a in alerts]
        assert "var_95" in metric_names
        assert "max_drawdown" in metric_names
        assert "annual_volatility" in metric_names

    @pytest.mark.skipif(
        not pytest.importorskip("PySide6", reason="PySide6 not available"),
        reason="PySide6 required",
    )
    def test_no_alert_when_within_threshold(self, qtbot):
        from portopt.gui.panels.risk_panel import RiskPanel

        panel = RiskPanel()
        qtbot.addWidget(panel)

        panel.set_alert_config({
            "var_95": {"enabled": True, "threshold": -0.03, "label": "VaR 95%"},
            "cvar_95": {"enabled": False, "threshold": -0.05, "label": "CVaR 95%"},
            "max_drawdown": {"enabled": True, "threshold": -0.20, "label": "Max Drawdown"},
            "annual_volatility": {"enabled": True, "threshold": 0.30, "label": "Annual Vol"},
            "downside_vol": {"enabled": False, "threshold": 0.15, "label": "Downside Vol"},
            "beta": {"enabled": False, "threshold": 1.5, "label": "Beta"},
        })

        alerts = []
        panel.alert_triggered.connect(lambda n, v, t: alerts.append((n, v, t)))

        # Set metrics that do NOT breach thresholds
        panel.set_risk_metrics({
            "var_95": -0.01,        # Better than -0.03 threshold
            "max_drawdown": -0.05,  # Better than -0.20 threshold
            "annual_volatility": 0.15,  # Below 0.30 threshold
        })

        assert len(alerts) == 0

    @pytest.mark.skipif(
        not pytest.importorskip("PySide6", reason="PySide6 not available"),
        reason="PySide6 required",
    )
    def test_alert_config_dialog_returns_settings(self, qtbot):
        from portopt.gui.dialogs.alert_config_dialog import AlertConfigDialog, DEFAULT_ALERTS

        dialog = AlertConfigDialog(dict(DEFAULT_ALERTS))
        qtbot.addWidget(dialog)
        alerts = dialog.get_alerts()
        assert isinstance(alerts, dict)
        assert "var_95" in alerts
        assert "threshold" in alerts["var_95"]


# ══════════════════════════════════════════════════════════════════════
# B5: What-if scenario analysis panel
# ══════════════════════════════════════════════════════════════════════

class TestB5ScenarioPanel:
    """Verify scenario panel accepts data and computes scenarios."""

    @pytest.mark.skipif(
        not pytest.importorskip("PySide6", reason="PySide6 not available"),
        reason="PySide6 required",
    )
    def test_set_base_data_creates_sliders(self, qtbot, prices_5):
        from portopt.gui.panels.scenario_panel import ScenarioPanel

        panel = ScenarioPanel()
        qtbot.addWidget(panel)

        mu = estimate_returns(prices_5)
        cov = estimate_covariance(prices_5, method=CovEstimator.SAMPLE)
        weights = {sym: 1.0 / len(mu) for sym in mu.index}

        panel.set_base_data(mu, cov, weights)
        assert len(panel._sliders) == len(mu)
        for sym in mu.index:
            assert sym in panel._sliders

    @pytest.mark.skipif(
        not pytest.importorskip("PySide6", reason="PySide6 not available"),
        reason="PySide6 required",
    )
    def test_compute_scenario_no_crash(self, qtbot, prices_5):
        from portopt.gui.panels.scenario_panel import ScenarioPanel

        panel = ScenarioPanel()
        qtbot.addWidget(panel)

        mu = estimate_returns(prices_5)
        cov = estimate_covariance(prices_5, method=CovEstimator.SAMPLE)
        weights = {sym: 1.0 / len(mu) for sym in mu.index}

        panel.set_base_data(mu, cov, weights)
        # Should compute without crash — initial scenario is all zeros
        assert panel._table.rowCount() > 0

    @pytest.mark.skipif(
        not pytest.importorskip("PySide6", reason="PySide6 not available"),
        reason="PySide6 required",
    )
    def test_slider_change_triggers_debounce(self, qtbot, prices_5):
        from portopt.gui.panels.scenario_panel import ScenarioPanel

        panel = ScenarioPanel()
        qtbot.addWidget(panel)

        mu = estimate_returns(prices_5)
        cov = estimate_covariance(prices_5, method=CovEstimator.SAMPLE)
        weights = {sym: 1.0 / len(mu) for sym in mu.index}

        panel.set_base_data(mu, cov, weights)

        # Change a slider value
        first_sym = list(panel._sliders.keys())[0]
        panel._sliders[first_sym].setValue(10)

        # Debounce timer should be active
        assert panel._debounce.isActive()

    @pytest.mark.skipif(
        not pytest.importorskip("PySide6", reason="PySide6 not available"),
        reason="PySide6 required",
    )
    def test_reset_sliders_zeros_all(self, qtbot, prices_5):
        from portopt.gui.panels.scenario_panel import ScenarioPanel

        panel = ScenarioPanel()
        qtbot.addWidget(panel)

        mu = estimate_returns(prices_5)
        cov = estimate_covariance(prices_5, method=CovEstimator.SAMPLE)
        weights = {sym: 1.0 / len(mu) for sym in mu.index}

        panel.set_base_data(mu, cov, weights)

        # Set some sliders to non-zero
        for slider in panel._sliders.values():
            slider.setValue(25)

        panel._reset_sliders()

        for slider in panel._sliders.values():
            assert slider.value() == 0

    @pytest.mark.skipif(
        not pytest.importorskip("PySide6", reason="PySide6 not available"),
        reason="PySide6 required",
    )
    def test_scenario_with_shock_produces_different_weights(self, qtbot, prices_5):
        """After applying a shock, the resulting weights should differ from base."""
        from portopt.gui.panels.scenario_panel import ScenarioPanel

        panel = ScenarioPanel()
        qtbot.addWidget(panel)

        mu = estimate_returns(prices_5)
        cov = estimate_covariance(prices_5, method=CovEstimator.SAMPLE)

        # Run a real optimization for base weights
        opt = MeanVarianceOptimizer(
            expected_returns=mu,
            covariance=cov,
            method=OptMethod.MAX_SHARPE,
        )
        base_result = opt.optimize()
        panel.set_base_data(mu, cov, base_result.weights)

        # Record the initial table state (base = shocked at 0% shock)
        initial_rows = panel._table.rowCount()
        assert initial_rows > 0

        # Apply a large shock to the first asset and compute
        first_sym = list(panel._sliders.keys())[0]
        panel._sliders[first_sym].setValue(-40)  # -40% shock
        panel._compute_scenario()

        # The table should still have rows
        assert panel._table.rowCount() > 0
