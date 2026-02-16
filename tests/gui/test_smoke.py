"""GUI smoke tests — verify panels and dialogs instantiate without crash.

Requires: pytest-qt (provides qtbot fixture)
"""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from portopt.constants import Colors
from portopt.engine.constraints import PortfolioConstraints


# ── Panel Smoke Tests ─────────────────────────────────────────────────

class TestPanelInstantiation:
    """Verify each panel can be created without errors."""

    def test_console_panel(self, qtbot):
        from portopt.gui.panels.console_panel import ConsolePanel
        panel = ConsolePanel()
        qtbot.addWidget(panel)
        panel.log_info("test message")

    def test_portfolio_panel(self, qtbot):
        from portopt.gui.panels.portfolio_panel import PortfolioPanel
        panel = PortfolioPanel()
        qtbot.addWidget(panel)

    def test_watchlist_panel(self, qtbot):
        from portopt.gui.panels.watchlist_panel import WatchlistPanel
        panel = WatchlistPanel()
        qtbot.addWidget(panel)

    def test_optimization_panel(self, qtbot):
        from portopt.gui.panels.optimization_panel import OptimizationPanel
        panel = OptimizationPanel()
        qtbot.addWidget(panel)
        config = panel.get_config()
        assert isinstance(config, dict)

    def test_weights_panel(self, qtbot):
        from portopt.gui.panels.weights_panel import WeightsPanel
        panel = WeightsPanel()
        qtbot.addWidget(panel)
        panel.set_weights(
            {"AAPL": 0.5, "MSFT": 0.5},
            {"AAPL": 0.6, "MSFT": 0.4},
        )

    def test_frontier_panel(self, qtbot):
        import numpy as np
        from portopt.gui.panels.frontier_panel import FrontierPanel
        panel = FrontierPanel()
        qtbot.addWidget(panel)
        panel.set_frontier(
            np.array([0.10, 0.15, 0.20]),
            np.array([0.05, 0.10, 0.15]),
        )

    def test_correlation_panel(self, qtbot):
        import numpy as np
        from portopt.gui.panels.correlation_panel import CorrelationPanel
        panel = CorrelationPanel()
        qtbot.addWidget(panel)
        corr = np.array([[1, 0.5], [0.5, 1]])
        panel.set_correlation(corr, ["AAPL", "MSFT"])

    def test_metrics_panel(self, qtbot):
        from portopt.gui.panels.metrics_panel import MetricsPanel
        panel = MetricsPanel()
        qtbot.addWidget(panel)
        panel.set_metrics({"sharpe_ratio": 1.5, "max_drawdown": -0.15})

    def test_risk_panel(self, qtbot):
        from portopt.gui.panels.risk_panel import RiskPanel
        panel = RiskPanel()
        qtbot.addWidget(panel)
        panel.set_risk_metrics({"var_95": -0.02, "cvar_95": -0.03})

    def test_backtest_panel(self, qtbot):
        from portopt.gui.panels.backtest_panel import BacktestPanel
        panel = BacktestPanel()
        qtbot.addWidget(panel)
        config = panel.get_config()
        assert isinstance(config, dict)

    def test_trade_blotter_panel(self, qtbot):
        from portopt.gui.panels.trade_blotter_panel import TradeBlotterPanel
        panel = TradeBlotterPanel()
        qtbot.addWidget(panel)
        panel.set_trades([
            {"date": "2024-01-02", "symbol": "AAPL", "side": "BUY",
             "quantity": 100, "price": 175.0, "cost": 1.0, "weight_after": 0.5},
        ])


# ── Dialog Smoke Tests ────────────────────────────────────────────────

class TestDialogInstantiation:
    def test_constraint_dialog(self, qtbot):
        from portopt.gui.dialogs.constraint_dialog import ConstraintDialog
        dialog = ConstraintDialog(["AAPL", "MSFT", "GOOG"])
        qtbot.addWidget(dialog)

    def test_bl_views_dialog(self, qtbot):
        from portopt.gui.dialogs.bl_views_dialog import BLViewsDialog
        dialog = BLViewsDialog(["AAPL", "MSFT", "GOOG"])
        qtbot.addWidget(dialog)

    def test_export_dialog(self, qtbot):
        from portopt.gui.dialogs.export_dialog import ExportDialog
        dialog = ExportDialog()
        qtbot.addWidget(dialog)

    def test_layout_dialog(self, qtbot):
        from portopt.gui.dialogs.layout_dialog import LayoutDialog
        dialog = LayoutDialog(["Default", "Backtest"])
        qtbot.addWidget(dialog)
