"""Main trading terminal window with dockable panel system."""

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QAction, QFont, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QMainWindow, QMenuBar, QMenu, QStatusBar, QLabel,
    QDockWidget, QHBoxLayout, QVBoxLayout, QWidget, QFileDialog, QMessageBox,
)

from portopt.constants import APP_NAME, APP_VERSION, Colors, Fonts, PanelID
from portopt.gui.dock_manager import DockManager
from portopt.gui.widgets.ticker_bar import TickerBar
from portopt.gui.panels.portfolio_panel import PortfolioPanel
from portopt.gui.panels.watchlist_panel import WatchlistPanel
from portopt.gui.panels.price_chart_panel import PriceChartPanel
from portopt.gui.panels.correlation_panel import CorrelationPanel
from portopt.gui.panels.optimization_panel import OptimizationPanel
from portopt.gui.panels.weights_panel import WeightsPanel
from portopt.gui.panels.frontier_panel import FrontierPanel
from portopt.gui.panels.backtest_panel import BacktestPanel
from portopt.gui.panels.metrics_panel import MetricsPanel
from portopt.gui.panels.attribution_panel import AttributionPanel
from portopt.gui.panels.network_panel import NetworkPanel
from portopt.gui.panels.dendrogram_panel import DendrogramPanel
from portopt.gui.panels.trade_blotter_panel import TradeBlotterPanel
from portopt.gui.panels.risk_panel import RiskPanel
from portopt.gui.panels.comparison_panel import ComparisonPanel
from portopt.gui.panels.scenario_panel import ScenarioPanel
from portopt.gui.panels.strategy_lab_panel import StrategyLabPanel
from portopt.gui.panels.monte_carlo_panel import MonteCarloPanel
from portopt.gui.panels.stress_test_panel import StressTestPanel
from portopt.gui.panels.rolling_panel import RollingAnalyticsPanel
from portopt.gui.panels.copilot_panel import CopilotPanel
from portopt.gui.panels.factor_panel import FactorAnalysisPanel
from portopt.gui.panels.regime_panel import RegimePanel
from portopt.gui.panels.risk_budget_panel import RiskBudgetPanel
from portopt.gui.panels.tax_harvest_panel import TaxHarvestPanel
from portopt.gui.panels.console_panel import ConsolePanel, ConsoleLogHandler
from portopt.gui.controllers.fidelity_controller import FidelityController
from portopt.gui.controllers.copilot_controller import CopilotController
from portopt.gui.controllers.data_controller import DataController
from portopt.gui.controllers.optimization_controller import OptimizationController
from portopt.gui.controllers.backtest_controller import BacktestController
from portopt.gui.controllers.monte_carlo_controller import MonteCarloController
from portopt.gui.dialogs.fidelity_login_dialog import FidelityLoginDialog
from portopt.gui.dialogs.bl_views_dialog import BLViewsDialog
from portopt.gui.dialogs.constraint_dialog import ConstraintDialog
from portopt.gui.dialogs.export_dialog import (
    ExportDialog, export_weights_csv, export_trades_csv, export_metrics_csv,
    export_optimization_json, export_session_json, export_excel_report, export_charts_png,
    FMT_CSV_WEIGHTS, FMT_CSV_TRADES, FMT_CSV_METRICS,
    FMT_JSON_RESULTS, FMT_JSON_SESSION, FMT_EXCEL_REPORT, FMT_PNG_CHARTS,
)
from portopt.gui.dialogs.layout_dialog import LayoutDialog
from portopt.gui.dialogs.api_key_dialog import ApiKeyDialog
from portopt.gui.dialogs.report_dialog import ReportDialog
from portopt.gui.dialogs.preferences_dialog import PreferencesDialog
from portopt.config import get_settings
from portopt.data.importers.fidelity_csv import parse_fidelity_csv
from portopt.data.importers.generic_csv import parse_generic_csv
from portopt.engine.constraints import PortfolioConstraints

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Trading terminal main window with dockable panels."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION} — Quantitative Portfolio Terminal")
        self._set_app_icon()
        self.setMinimumSize(QSize(1280, 800))
        self.resize(1600, 1000)

        # Enable nested docking for Bloomberg-style tiling
        self.setDockNestingEnabled(True)
        self.setDockOptions(
            QMainWindow.DockOption.AnimatedDocks
            | QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
        )

        # Dock manager for layout presets
        self.dock_manager = DockManager(self)

        # Panels registry
        self.panels: dict[str, QDockWidget] = {}

        # State
        self._portfolio = None
        self._constraints = PortfolioConstraints()
        self._settings = get_settings()

        self._setup_ticker_bar()
        self._setup_panels()
        self._setup_controllers()
        self._setup_menu_bar()
        self._setup_status_bar()
        self._setup_logging()
        self._setup_focus_layout()

        # Restore window geometry (with safety check for off-screen)
        geo = self._settings.value("window/geometry")
        if geo:
            self.restoreGeometry(geo)
        state = self._settings.value("window/state")
        if state:
            self.restoreState(state)

        # Safety: ensure window is visible on screen
        self._ensure_on_screen()

        # Try restoring last session layout
        if not self.dock_manager.restore_session():
            self._setup_focus_layout()

        # Startup: try auto-connect to Fidelity after window is shown
        QTimer.singleShot(500, self._on_startup)

    def _ensure_on_screen(self):
        """Reset window position if it's not visible on any screen."""
        from PySide6.QtWidgets import QApplication
        window_geo = self.frameGeometry()
        visible = False
        for screen in QApplication.screens():
            if screen.availableGeometry().intersects(window_geo):
                visible = True
                break
        if not visible:
            self.resize(1600, 1000)
            screen = QApplication.primaryScreen()
            if screen:
                center = screen.availableGeometry().center()
                self.move(center.x() - 800, center.y() - 500)

    def _set_app_icon(self):
        """Set the window icon and taskbar icon."""
        from pathlib import Path
        icon_path = Path(__file__).resolve().parent.parent / "assets" / "icon.png"
        if icon_path.exists():
            icon = QIcon(str(icon_path))
            self.setWindowIcon(icon)
            # Also set on the QApplication for taskbar
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                app.setWindowIcon(icon)

    # ── Startup ──────────────────────────────────────────────────────
    def _on_startup(self):
        """Called shortly after window shows — attempt Fidelity auto-connect."""
        self.console_panel.log_info(f"{APP_NAME} terminal started")
        cache_mb = self.data_controller.get_cache_size()
        self.set_cache_status(f"CACHE: {cache_mb:.1f} MB")

        if False and self.fidelity_controller.has_saved_session:
            # Auto-connect disabled — too fragile with Playwright session persistence.
            # Users should connect manually via Data > Fidelity Connection.
            self.console_panel.log_info("Found saved Fidelity session, attempting auto-connect...")
            self.set_fidelity_status(None)  # amber = connecting
            self.fidelity_controller.try_auto_connect()
        else:
            # Try loading cached portfolio for offline startup
            cached = self.fidelity_controller.load_cached_portfolio()
            if cached and cached.holdings:
                self._portfolio = cached
                self.portfolio_panel.set_portfolio(cached)
                self.console_panel.log_info(
                    f"Loaded cached portfolio: {len(cached.holdings)} positions "
                    f"(last updated: {cached.last_updated.strftime('%Y-%m-%d %H:%M') if cached.last_updated else 'unknown'})"
                )
            else:
                self.console_panel.log_info("No saved Fidelity session. Use Data > Fidelity Connection to link your account.")

    # ── Ticker Bar ───────────────────────────────────────────────────
    def _setup_ticker_bar(self):
        """Add the scrolling ticker bar at the top."""
        self.ticker_bar = TickerBar(self)
        ticker_dock = QDockWidget("", self)
        ticker_dock.setObjectName("ticker_bar_dock")
        ticker_dock.setWidget(self.ticker_bar)
        ticker_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        ticker_dock.setTitleBarWidget(QWidget())
        ticker_dock.setFixedHeight(28)
        self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, ticker_dock)

    # ── Panels ───────────────────────────────────────────────────────
    def _setup_panels(self):
        """Create all dockable panels."""
        self.portfolio_panel = PortfolioPanel(self)
        self.watchlist_panel = WatchlistPanel(self)
        self.price_chart_panel = PriceChartPanel(self)
        self.correlation_panel = CorrelationPanel(self)
        self.optimization_panel = OptimizationPanel(self)
        self.weights_panel = WeightsPanel(self)
        self.frontier_panel = FrontierPanel(self)
        self.backtest_panel = BacktestPanel(self)
        self.metrics_panel = MetricsPanel(self)
        self.attribution_panel = AttributionPanel(self)
        self.network_panel = NetworkPanel(self)
        self.dendrogram_panel = DendrogramPanel(self)
        self.trade_blotter_panel = TradeBlotterPanel(self)
        self.risk_panel = RiskPanel(self)
        self.comparison_panel = ComparisonPanel(self)
        self.scenario_panel = ScenarioPanel(self)
        self.console_panel = ConsolePanel(self)
        self.strategy_lab_panel = StrategyLabPanel(self)
        self.monte_carlo_panel = MonteCarloPanel(self)
        self.stress_test_panel = StressTestPanel(self)
        self.rolling_panel = RollingAnalyticsPanel(self)
        self.copilot_panel = CopilotPanel(self)
        self.factor_panel = FactorAnalysisPanel(self)
        self.regime_panel = RegimePanel(self)
        self.risk_budget_panel = RiskBudgetPanel(self)
        self.tax_harvest_panel = TaxHarvestPanel(self)

        for panel in [
            self.portfolio_panel, self.watchlist_panel, self.price_chart_panel,
            self.correlation_panel, self.optimization_panel, self.weights_panel,
            self.frontier_panel, self.backtest_panel, self.metrics_panel,
            self.attribution_panel, self.network_panel, self.dendrogram_panel,
            self.trade_blotter_panel, self.risk_panel, self.comparison_panel,
            self.scenario_panel, self.strategy_lab_panel, self.monte_carlo_panel,
            self.stress_test_panel, self.rolling_panel, self.copilot_panel,
            self.factor_panel, self.regime_panel, self.risk_budget_panel,
            self.tax_harvest_panel, self.console_panel,
        ]:
            self.panels[panel.panel_id] = panel

    # ── Controllers ──────────────────────────────────────────────────
    def _setup_controllers(self):
        """Initialize controllers and connect signals."""
        # Fidelity controller
        self.fidelity_controller = FidelityController(self)
        self.fidelity_controller.connected.connect(self._on_fidelity_connected)
        self.fidelity_controller.disconnected.connect(self._on_fidelity_disconnected)
        self.fidelity_controller.connection_error.connect(self._on_fidelity_error)
        self.fidelity_controller.needs_2fa.connect(self._on_fidelity_needs_2fa)
        self.fidelity_controller.status_changed.connect(
            lambda msg: self.console_panel.log_info(f"Fidelity: {msg}")
        )
        self.fidelity_controller.playwright_missing.connect(self._on_playwright_missing)

        # Data controller
        self.data_controller = DataController(self)
        self.data_controller.status_changed.connect(
            lambda msg: self.console_panel.log_data(f"Data: {msg}")
        )
        self.data_controller.error.connect(
            lambda msg: self.console_panel.log_error(f"Data error: {msg}")
        )

        # Optimization controller
        self.opt_controller = OptimizationController(self.data_controller, self)
        self.opt_controller.optimization_complete.connect(self._on_optimization_complete)
        self.opt_controller.frontier_complete.connect(self._on_frontier_complete)
        self.opt_controller.frontier_assets.connect(self._on_frontier_assets)
        self.opt_controller.correlation_ready.connect(self._on_correlation_ready)
        self.opt_controller.mst_ready.connect(self._on_mst_ready)
        self.opt_controller.dendrogram_ready.connect(self._on_dendrogram_ready)
        self.opt_controller.risk_metrics_ready.connect(self._on_risk_metrics)
        self.opt_controller.status_changed.connect(
            lambda msg: self.console_panel.log_info(f"Opt: {msg}")
        )
        self.opt_controller.error.connect(
            lambda msg: self.console_panel.log_error(f"Opt error: {msg}")
        )
        # B1: Wire progress + running signals to status bar
        self.opt_controller.progress.connect(lambda msg: self._op_status.setText(msg))
        self.opt_controller.running_changed.connect(
            lambda running: self._start_elapsed() if running else self._stop_elapsed()
        )

        # Backtest controller
        self.bt_controller = BacktestController(self.data_controller, self)
        self.bt_controller.equity_curve_ready.connect(self._on_equity_curve)
        self.bt_controller.drawdown_ready.connect(self._on_drawdown)
        self.bt_controller.metrics_ready.connect(self._on_bt_metrics)
        self.bt_controller.trades_ready.connect(self._on_trades)
        self.bt_controller.status_changed.connect(
            lambda msg: self.console_panel.log_info(f"Backtest: {msg}")
        )
        self.bt_controller.error.connect(
            lambda msg: self.console_panel.log_error(f"Backtest error: {msg}")
        )
        self.bt_controller.benchmark_curve_ready.connect(self._on_benchmark_curve)
        self.bt_controller.benchmark_metrics_ready.connect(self._on_benchmark_metrics)
        # B1: Wire progress + running signals to status bar
        self.bt_controller.progress.connect(lambda msg: self._op_status.setText(msg))
        self.bt_controller.running_changed.connect(
            lambda running: self._start_elapsed() if running else self._stop_elapsed()
        )

        # Monte Carlo controller
        self.mc_controller = MonteCarloController(self)
        self.mc_controller.simulation_complete.connect(self._on_mc_complete)
        self.mc_controller.status_changed.connect(
            lambda msg: self.console_panel.log_info(f"MC: {msg}")
        )
        self.mc_controller.error.connect(
            lambda msg: self.console_panel.log_error(f"MC error: {msg}")
        )
        self.mc_controller.progress.connect(lambda msg: self._op_status.setText(msg))
        self.mc_controller.running_changed.connect(
            lambda running: self._start_elapsed() if running else self._stop_elapsed()
        )
        self.monte_carlo_panel.run_requested.connect(self._run_mc)

        # Stress test panel
        self.stress_test_panel.run_requested.connect(self._run_stress_test)

        # Rolling analytics panel
        self.rolling_panel.compute_requested.connect(self._run_rolling)

        # Factor analysis panel
        self.factor_panel.run_requested.connect(self._run_factor_analysis)

        # Regime detection panel
        self.regime_panel.run_requested.connect(self._run_regime_detection)

        # Risk budget panel
        self.risk_budget_panel.run_requested.connect(self._run_risk_budget)

        # Tax harvest panel
        self.tax_harvest_panel.run_requested.connect(self._run_tax_harvest)

        # Copilot controller
        self.copilot_controller = CopilotController(self)
        self.copilot_panel.message_submitted.connect(self._on_copilot_message)
        self.copilot_controller.response_chunk.connect(
            self.copilot_panel.append_assistant_chunk
        )
        self.copilot_controller.tool_use_started.connect(
            self.copilot_panel.show_tool_use
        )
        self.copilot_controller.response_complete.connect(self._on_copilot_response)
        self.copilot_controller.error.connect(self.copilot_panel.show_error)

        # Strategy Lab: own controllers (isolated from main portfolio)
        self._lab_opt_controller = OptimizationController(self.data_controller, self)
        self._lab_bt_controller = BacktestController(self.data_controller, self)
        self.strategy_lab_panel.set_controllers(
            self.data_controller, self._lab_opt_controller, self._lab_bt_controller,
            console=self.console_panel,
        )
        self.strategy_lab_panel.import_portfolio_requested.connect(self._on_lab_import)

        # Panel signals
        self.portfolio_panel.connect_requested.connect(self._show_fidelity_login)
        self.portfolio_panel.refresh_requested.connect(self._refresh_fidelity)
        self.optimization_panel.run_requested.connect(self._run_optimization)
        self.backtest_panel.run_requested.connect(self._run_backtest)

        # Watchlist: add ticker → fetch price
        self.watchlist_panel.add_ticker_requested.connect(self._on_watchlist_add)

        # B3: Save result for comparison
        self.optimization_panel.save_requested.connect(self._save_to_comparison)

        # B4: Risk alert → console warning
        self.risk_panel.alert_triggered.connect(
            lambda name, val, thresh: self.console_panel.log_warning(
                f"ALERT: {name} = {val:.4f} breaches threshold {thresh:.4f}"
            )
        )

    # ── Logging ──────────────────────────────────────────────────────
    def _setup_logging(self):
        """Route Python logging to the console panel."""
        handler = ConsoleLogHandler(self.console_panel)
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        root = logging.getLogger("portopt")
        root.addHandler(handler)
        root.setLevel(logging.INFO)

    # ── Optimization Flow ────────────────────────────────────────────
    def _run_optimization(self, config: dict):
        """Handle optimization run from panel or menu."""
        symbols = self._get_active_symbols()
        if not symbols:
            self.console_panel.log_warning("No symbols selected. Import a portfolio or add symbols to watchlist.")
            return

        self.optimization_panel.set_running(True)
        self.console_panel.log_info(f"Running optimization on {len(symbols)} symbols...")

        # Feed config with current constraints
        config["long_only"] = self._constraints.long_only
        config["min_weight"] = self._constraints.min_weight
        config["max_weight"] = self._constraints.max_weight

        self.opt_controller.fetch_and_optimize(symbols, config)

    def _on_optimization_complete(self, result):
        """Handle optimization result."""
        self.optimization_panel.set_running(False)

        # Update weights panel
        current = self._portfolio.weights if self._portfolio else {}
        self.weights_panel.set_weights(current, result.weights)

        # Update frontier with optimal point
        self.frontier_panel.set_optimal_portfolio(
            result.volatility, result.expected_return, result.method,
        )

        # B3: Enable save-to-compare button
        self.optimization_panel.set_has_result(True)

        # B5: Feed scenario panel with base data
        if self.opt_controller._last_mu is not None and self.opt_controller._last_cov is not None:
            self.scenario_panel.set_base_data(
                self.opt_controller._last_mu,
                self.opt_controller._last_cov,
                result.weights,
            )

        # Feed price chart with the price data used for optimization
        self._update_price_chart(self.opt_controller._prices)

        # Feed Monte Carlo controller with optimization data
        if self.opt_controller._prices is not None:
            self.mc_controller.set_prices(self.opt_controller._prices)
            self.mc_controller.set_weights(result.weights)

        # Feed stress test panel with portfolio weights
        self.stress_test_panel.set_weights(result.weights)

        # Feed rolling analytics panel with symbols
        self.rolling_panel.set_symbols(list(result.weights.keys()))

        # Feed risk budget panel with symbols
        self.risk_budget_panel.set_symbols(list(result.weights.keys()))

        # Feed copilot with portfolio context
        self.copilot_controller.set_context(
            prices=self.opt_controller._prices,
            weights=result.weights,
            mu=self.opt_controller._last_mu,
            cov=self.opt_controller._last_cov,
            result=result,
        )

        # Store last result for comparison save
        self._last_opt_result = result

        self.console_panel.log_success(
            f"Optimization complete: {result.method} | "
            f"Sharpe={result.sharpe_ratio:.3f}"
        )

    def _save_to_comparison(self):
        """B3: Save current optimization result to comparison panel."""
        result = getattr(self, "_last_opt_result", None)
        if result is None:
            self.console_panel.log_warning("No optimization result to save.")
            return
        name = result.method if isinstance(result.method, str) else result.method.name
        self.comparison_panel.add_snapshot(result, name)
        self.console_panel.log_info(f"Saved '{name}' to comparison panel.")

    # ── Monte Carlo Flow ─────────────────────────────────────────────
    def _run_mc(self, config):
        """Handle Monte Carlo run from panel."""
        if not self.mc_controller.is_ready:
            self.console_panel.log_warning("Run optimization first to provide data for Monte Carlo.")
            return
        self.monte_carlo_panel.set_running(True)
        self.console_panel.log_info(f"Running Monte Carlo: {config.n_sims} sims, {config.horizon_days} days...")
        self.mc_controller.run_simulation(config)

    def _on_mc_complete(self, result):
        """Handle Monte Carlo simulation result."""
        self.monte_carlo_panel.set_running(False)
        self.monte_carlo_panel.set_result(result)
        self.console_panel.log_success(
            f"Monte Carlo complete: {result.n_sims} sims | "
            f"P(shortfall)={result.shortfall_probability:.1%}"
        )

    def _on_frontier_complete(self, risks, returns, weights_list=None):
        self.frontier_panel.clear_plot()
        self.frontier_panel.set_frontier(risks, returns, weights_list)

    def _on_frontier_assets(self, symbols, vols, mus):
        self.frontier_panel.set_individual_assets(symbols, vols, mus)

    def _on_correlation_ready(self, corr_matrix, labels):
        self.correlation_panel.set_correlation(corr_matrix, labels)

    def _on_mst_ready(self, nodes, edges, sectors):
        self.network_panel.set_mst(nodes, edges, sectors)

    def _on_dendrogram_ready(self, linkage_matrix, labels):
        self.dendrogram_panel.set_dendrogram(linkage_matrix, labels)

    def _on_risk_metrics(self, metrics):
        self.risk_panel.set_risk_metrics(metrics)

    # ── Backtest Flow ────────────────────────────────────────────────
    def _run_backtest(self, config: dict):
        """Handle backtest run from panel or menu."""
        symbols = self._get_active_symbols()
        if not symbols:
            self.console_panel.log_warning("No symbols selected for backtest.")
            return

        self.console_panel.log_info(f"Running backtest on {len(symbols)} symbols...")
        self.bt_controller.fetch_and_backtest(symbols, config)

    def _on_equity_curve(self, dates_epoch, values):
        self.backtest_panel.set_equity_curve(dates_epoch, values)

    def _on_drawdown(self, dates_epoch, drawdowns):
        self.backtest_panel.set_drawdown(dates_epoch, drawdowns)

    def _on_bt_metrics(self, metrics):
        self.metrics_panel.set_metrics(metrics)

    def _on_trades(self, trades):
        self.trade_blotter_panel.set_trades(trades)

    def _on_benchmark_curve(self, dates_epoch, values, label):
        self.backtest_panel.set_benchmark(dates_epoch, values, label)

    def _on_benchmark_metrics(self, port_metrics, bench_metrics):
        self.backtest_panel.set_benchmark_metrics(port_metrics, bench_metrics)

    # ── Stress Test Flow ─────────────────────────────────────────────
    def _run_stress_test(self, config: dict):
        """Run stress tests on current portfolio weights."""
        from portopt.engine.stress import (
            HISTORICAL_SCENARIOS, StressScenario,
            run_stress_test, run_all_stress_tests,
        )
        from portopt.utils.threading import run_in_thread

        weights = config.get("weights")
        if not weights:
            self.console_panel.log_warning("No portfolio weights. Run optimization first.")
            return

        self.stress_test_panel.set_running(True)
        self.console_panel.log_info("Running stress tests...")

        def _do_stress():
            selected = config.get("selected_scenarios", list(HISTORICAL_SCENARIOS.keys()))
            scenarios = [HISTORICAL_SCENARIOS[n] for n in selected if n in HISTORICAL_SCENARIOS]

            # Add custom scenario if provided
            custom = config.get("custom")
            if custom:
                scenarios.append(StressScenario(
                    name=custom["name"],
                    description="Custom scenario",
                    shocks=custom["shocks"],
                ))

            # Get covariance matrix if available
            cov = getattr(self.opt_controller, "_last_cov", None)

            return run_all_stress_tests(
                weights, scenarios=scenarios, cov=cov,
            )

        run_in_thread(
            _do_stress,
            on_result=self._on_stress_complete,
            on_error=lambda msg: (
                self.stress_test_panel.set_running(False),
                self.console_panel.log_error(f"Stress test error: {msg}"),
            ),
        )

    def _on_stress_complete(self, results):
        self.stress_test_panel.set_running(False)
        self.stress_test_panel.set_results(results)
        worst = min(results, key=lambda r: r.portfolio_impact) if results else None
        if worst:
            self.console_panel.log_success(
                f"Stress test complete: {len(results)} scenarios | "
                f"Worst: {worst.scenario.name} ({worst.portfolio_impact:+.2%})"
            )

    # ── Rolling Analytics Flow ────────────────────────────────────────
    def _run_rolling(self, config: dict):
        """Run rolling metric computation on background thread."""
        from portopt.engine.rolling import (
            rolling_sharpe, rolling_sortino, rolling_volatility,
            rolling_max_drawdown, rolling_beta, rolling_correlation,
        )
        from portopt.utils.threading import run_in_thread

        prices = getattr(self.opt_controller, "_prices", None)
        if prices is None or prices.empty:
            self.console_panel.log_warning("No price data. Run optimization first.")
            return

        metric = config["metric"]
        window = config["window"]
        asset_a = config["asset_a"]
        asset_b = config.get("asset_b")

        if asset_a not in prices.columns:
            self.console_panel.log_warning(f"Asset {asset_a} not found in price data.")
            return

        self.rolling_panel.set_running(True)
        self.console_panel.log_info(f"Computing rolling {metric} (window={window})...")

        def _compute():
            returns = prices.pct_change().dropna()
            ret_a = returns[asset_a]

            if metric == "Sharpe Ratio":
                values = rolling_sharpe(ret_a, window=window)
            elif metric == "Sortino Ratio":
                values = rolling_sortino(ret_a, window=window)
            elif metric == "Volatility":
                values = rolling_volatility(ret_a, window=window)
            elif metric == "Max Drawdown":
                values = rolling_max_drawdown(ret_a, window=window)
            elif metric == "Beta":
                if not asset_b or asset_b not in returns.columns:
                    raise ValueError(f"Asset B '{asset_b}' not available")
                values = rolling_beta(ret_a, returns[asset_b], window=window)
            elif metric == "Correlation":
                if not asset_b or asset_b not in returns.columns:
                    raise ValueError(f"Asset B '{asset_b}' not available")
                values = rolling_correlation(ret_a, returns[asset_b], window=window)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            # Convert dates to epoch seconds for DateAxisItem
            dates_epoch = np.array([
                d.timestamp() for d in values.index
            ], dtype=float)
            return dates_epoch, values.values, metric

        def _on_result(result):
            dates_epoch, vals, name = result
            self.rolling_panel.set_running(False)
            self.rolling_panel.set_result(dates_epoch, vals, name)
            self.console_panel.log_success(f"Rolling {name} computed ({window}-day window)")

        def _on_error(msg):
            self.rolling_panel.set_running(False)
            self.console_panel.log_error(f"Rolling analytics error: {msg}")

        run_in_thread(_compute, on_result=_on_result, on_error=_on_error)

    # ── Factor Analysis Flow ────────────────────────────────────────
    def _run_factor_analysis(self):
        """Run Fama-French factor analysis on current portfolio."""
        from portopt.engine.factors import run_factor_analysis
        from portopt.utils.threading import run_in_thread

        prices = getattr(self.opt_controller, "_prices", None)
        result = getattr(self, "_last_opt_result", None)
        if prices is None or prices.empty or result is None:
            self.console_panel.log_warning("Run optimization first to provide data for factor analysis.")
            return

        self.factor_panel.set_running(True)
        self.console_panel.log_info("Running Fama-French factor analysis...")

        weights = result.weights

        def _compute():
            return run_factor_analysis(prices, weights)

        def _on_result(factor_result):
            self.factor_panel.set_running(False)
            self.factor_panel.set_result(factor_result)
            self.console_panel.log_success(
                f"Factor analysis complete: {len(factor_result.asset_exposures)} assets"
            )

        def _on_error(msg):
            self.factor_panel.set_running(False)
            self.console_panel.log_error(f"Factor analysis error: {msg}")

        run_in_thread(_compute, on_result=_on_result, on_error=_on_error)

    # ── Regime Detection Flow ────────────────────────────────────────
    def _run_regime_detection(self, n_regimes: int):
        """Run HMM regime detection on portfolio market returns."""
        from portopt.engine.regime import detect_regimes
        from portopt.utils.threading import run_in_thread

        prices = getattr(self.opt_controller, "_prices", None)
        if prices is None or prices.empty:
            self.console_panel.log_warning("Run optimization first to provide data for regime detection.")
            return

        self.regime_panel.set_running(True)
        self.console_panel.log_info(f"Detecting {n_regimes} market regimes...")

        def _compute():
            # Use equal-weighted portfolio returns as market proxy
            returns = prices.pct_change().dropna().mean(axis=1)
            return detect_regimes(returns, n_regimes=n_regimes)

        def _on_result(regime_result):
            self.regime_panel.set_running(False)
            self.regime_panel.set_result(regime_result)
            self.console_panel.log_success(
                f"Regime detection complete: current={regime_result.current_regime_name}"
            )

        def _on_error(msg):
            self.regime_panel.set_running(False)
            self.console_panel.log_error(f"Regime detection error: {msg}")

        run_in_thread(_compute, on_result=_on_result, on_error=_on_error)

    # ── Risk Budget Flow ─────────────────────────────────────────────
    def _run_risk_budget(self, config: dict):
        """Run risk budget or ERC optimization."""
        from portopt.engine.risk_budgeting import (
            equal_risk_contribution, optimize_risk_budget,
        )
        from portopt.utils.threading import run_in_thread

        mu = getattr(self.opt_controller, "_last_mu", None)
        cov = getattr(self.opt_controller, "_last_cov", None)
        if mu is None or cov is None:
            self.console_panel.log_warning("Run optimization first to provide data for risk budgeting.")
            return

        is_erc = config.get("erc", True)
        budgets = config.get("budgets", {})

        self.risk_budget_panel.set_running(True)
        label = "Equal Risk Contribution" if is_erc else "Risk Budget"
        self.console_panel.log_info(f"Running {label} optimization...")

        def _compute():
            if is_erc:
                return equal_risk_contribution(mu, cov)
            else:
                return optimize_risk_budget(mu, cov, budgets)

        def _on_result(result):
            self.risk_budget_panel.set_running(False)
            self.risk_budget_panel.set_result(result)
            self.console_panel.log_success(
                f"{label} complete: Sharpe={result.sharpe_ratio:.3f}"
            )

        def _on_error(msg):
            self.risk_budget_panel.set_running(False)
            self.console_panel.log_error(f"Risk budget error: {msg}")

        run_in_thread(_compute, on_result=_on_result, on_error=_on_error)

    # ── Tax Harvest Flow ─────────────────────────────────────────────
    def _run_tax_harvest(self, tax_rate: float):
        """Run tax-loss harvesting analysis on current portfolio."""
        from portopt.engine.tax_harvest import compute_harvest_recommendation
        from portopt.utils.threading import run_in_thread

        if not self._portfolio or not self._portfolio.holdings:
            self.console_panel.log_warning("Import a portfolio first for tax-loss harvesting analysis.")
            return

        prices = getattr(self.opt_controller, "_prices", None)
        self.tax_harvest_panel.set_running(True)
        self.console_panel.log_info(f"Analyzing tax-loss harvesting opportunities (rate={tax_rate:.0%})...")

        holdings = self._portfolio.holdings

        def _compute():
            return compute_harvest_recommendation(
                holdings, prices=prices, tax_rate=tax_rate,
            )

        def _on_result(recommendation):
            self.tax_harvest_panel.set_running(False)
            self.tax_harvest_panel.set_result(recommendation)
            n = len(recommendation.candidates)
            self.console_panel.log_success(
                f"Tax harvest analysis: {n} candidates, "
                f"${recommendation.total_tax_savings:,.0f} potential savings"
            )

        def _on_error(msg):
            self.tax_harvest_panel.set_running(False)
            self.console_panel.log_error(f"Tax harvest error: {msg}")

        run_in_thread(_compute, on_result=_on_result, on_error=_on_error)

    # ── Copilot Flow ─────────────────────────────────────────────────
    def _on_copilot_message(self, text: str):
        """Handle user message from copilot panel."""
        self.copilot_panel.append_user_message(text)
        self.copilot_panel.start_assistant_message()
        self.copilot_panel.set_waiting(True)
        self.copilot_controller.send_message(text)

    def _on_copilot_response(self, full_text: str):
        """Handle completed copilot response."""
        self.copilot_panel.finish_assistant_message()
        self.copilot_panel.set_waiting(False)

    def _show_copilot(self):
        """Raise and show the copilot panel."""
        self.copilot_panel.show()
        self.copilot_panel.raise_()

    def _show_api_key_dialog(self):
        """Show the Anthropic API key configuration dialog."""
        dialog = ApiKeyDialog(self)
        if dialog.exec() == ApiKeyDialog.DialogCode.Accepted:
            key = dialog.get_api_key()
            if key:
                self.console_panel.log_success("Anthropic API key saved.")
            else:
                self.console_panel.log_info("Anthropic API key removed.")

    def _show_report_dialog(self):
        """Show the report generation dialog."""
        result = getattr(self, "_last_opt_result", None)
        weights_dict = result.weights if result else None
        metrics = getattr(self.metrics_panel, "_metrics", None)

        panels = {
            "frontier": self.frontier_panel,
            "weights": self.weights_panel,
            "correlation": self.correlation_panel,
        }

        dialog = ReportDialog(
            parent=self,
            weights=weights_dict,
            metrics=metrics,
            panels=panels,
            copilot_controller=self.copilot_controller,
        )
        dialog.exec()

    # ── Dialogs ──────────────────────────────────────────────────────
    def _show_bl_views(self):
        """Show Black-Litterman views dialog."""
        symbols = self._get_active_symbols()
        if not symbols:
            self.console_panel.log_warning("Load a portfolio first to set BL views.")
            return
        dialog = BLViewsDialog(symbols, self)
        dialog.views_submitted.connect(self.opt_controller.set_bl_views)
        dialog.exec()

    def _show_constraints(self):
        """Show constraints editor dialog."""
        symbols = self._get_active_symbols()
        dialog = ConstraintDialog(symbols, self._constraints, self)
        dialog.constraints_updated.connect(self._on_constraints_updated)
        dialog.exec()

    def _on_constraints_updated(self, constraints):
        self._constraints = constraints
        self.console_panel.log_info(
            f"Constraints updated: long_only={constraints.long_only}, "
            f"bounds=[{constraints.min_weight:.3f}, {constraints.max_weight:.3f}]"
        )

    def _show_export(self):
        """Show export dialog."""
        has_weights = self.opt_controller.last_result is not None
        has_bt = self.bt_controller.last_output is not None
        dialog = ExportDialog(self, has_weights=has_weights, has_backtest=has_bt)
        dialog.export_requested.connect(self._do_export)
        dialog.exec()

    def _do_export(self, config: dict):
        fmt = config["format"]
        path = config["path"]
        try:
            if fmt == FMT_CSV_WEIGHTS and self.opt_controller.last_result:
                result = self.opt_controller.last_result
                metadata = {"method": result.method, "sharpe": f"{result.sharpe_ratio:.4f}"}
                if config.get("include_metadata"):
                    export_weights_csv(result.weights, path, metadata)
                else:
                    export_weights_csv(result.weights, path)
                self.console_panel.log_success(f"Weights exported to {path}")

            elif fmt == FMT_CSV_TRADES and self.bt_controller.last_output:
                output = self.bt_controller.last_output
                trades = []
                if output.result:
                    trades = self.bt_controller._trades_to_dicts(output.result.trades)
                export_trades_csv(trades, path)
                self.console_panel.log_success(f"Trades exported to {path}")

            elif fmt == FMT_CSV_METRICS:
                metrics = {}
                if self.bt_controller.last_output:
                    metrics = self.bt_controller.last_output.metrics
                export_metrics_csv(metrics, path)
                self.console_panel.log_success(f"Metrics exported to {path}")

            elif fmt == FMT_JSON_RESULTS and self.opt_controller.last_result:
                export_optimization_json(self.opt_controller.last_result, path)
                self.console_panel.log_success(f"Optimization JSON exported to {path}")

            elif fmt == FMT_JSON_SESSION:
                export_session_json(
                    opt_result=self.opt_controller.last_result,
                    bt_output=self.bt_controller.last_output,
                    portfolio=self._portfolio,
                    path=path,
                )
                self.console_panel.log_success(f"Session state exported to {path}")

            elif fmt == FMT_EXCEL_REPORT:
                weights = self.opt_controller.last_result.weights if self.opt_controller.last_result else None
                metrics = self.bt_controller.last_output.metrics if self.bt_controller.last_output else None
                trades = None
                if self.bt_controller.last_output and self.bt_controller.last_output.result:
                    trades = self.bt_controller._trades_to_dicts(self.bt_controller.last_output.result.trades)
                export_excel_report(weights=weights, metrics=metrics, trades=trades, path=path)
                self.console_panel.log_success(f"Excel report exported to {path}")

            elif fmt == FMT_PNG_CHARTS:
                chart_panels = {
                    "frontier": self.frontier_panel,
                    "correlation": self.correlation_panel,
                    "backtest": self.backtest_panel,
                    "network": self.network_panel,
                    "monte_carlo": self.monte_carlo_panel,
                    "stress_test": self.stress_test_panel,
                    "rolling": self.rolling_panel,
                    "factor_analysis": self.factor_panel,
                    "regime": self.regime_panel,
                    "risk_budget": self.risk_budget_panel,
                }
                n = export_charts_png(chart_panels, path)
                self.console_panel.log_success(f"Exported {n} charts to {path}")

            else:
                self.console_panel.log_warning(f"No data available for {fmt}")
        except Exception as e:
            self.console_panel.log_error(f"Export failed: {e}")

    def _show_layout_manager(self):
        """Show layout save/load dialog."""
        saved = self.dock_manager.list_layouts()
        dialog = LayoutDialog(saved, self)
        dialog.layout_save_requested.connect(
            lambda name: self.dock_manager.save_layout(name)
        )
        dialog.layout_load_requested.connect(
            lambda name: self.dock_manager.restore_layout(name)
        )
        dialog.layout_delete_requested.connect(
            lambda name: self.dock_manager.delete_layout(name)
        )
        dialog.exec()

    # ── Fidelity Connection Flow ─────────────────────────────────────
    def _show_fidelity_login(self, show_playwright_setup: bool = False):
        """Show the Fidelity login dialog."""
        # Disconnect any stale controller signals from previous dialogs
        # to prevent race conditions with background auto-connect
        try:
            self.fidelity_controller.needs_2fa.disconnect()
        except RuntimeError:
            pass
        try:
            self.fidelity_controller.connection_error.disconnect()
        except RuntimeError:
            pass

        self._fid_dialog = FidelityLoginDialog(self, show_playwright_setup=show_playwright_setup)
        self._fid_dialog.login_requested.connect(self._on_fidelity_login)
        self._fid_dialog.interactive_login_requested.connect(self._on_fidelity_browser_login)
        self._fid_dialog.csv_imported.connect(self._on_fidelity_csv_imported)
        self._fid_dialog.twofa_submitted.connect(self._on_fidelity_2fa)
        self._fid_dialog.skip_requested.connect(
            lambda: self.console_panel.log_info("Fidelity connection skipped")
        )

        # Pre-fill saved credentials
        if self.fidelity_controller.has_saved_credentials:
            user, pwd, totp = self.fidelity_controller.get_saved_credentials()
            self._fid_dialog.prefill_credentials(user, pwd, totp)

        # Wire controller signals to dialog
        self.fidelity_controller.needs_2fa.connect(self._fid_dialog.show_2fa)
        self.fidelity_controller.connected.connect(
            lambda p: self._fid_dialog.show_success(f"{len(p.holdings)} positions loaded")
        )
        self.fidelity_controller.connection_error.connect(self._fid_dialog.show_error)

        self._fid_dialog.exec()

    def _on_fidelity_login(self, username: str, password: str, totp: str):
        self.set_fidelity_status(None)  # amber = connecting
        self.fidelity_controller.login(
            username, password, totp,
            save_credentials=self._fid_dialog.remember_credentials,
        )

    def _on_fidelity_browser_login(self):
        self.set_fidelity_status(None)  # amber = connecting
        self.fidelity_controller.login_interactive()

    def _on_fidelity_csv_imported(self, path: str):
        """Handle CSV file imported from the Fidelity dialog."""
        try:
            portfolio = parse_fidelity_csv(path)
            if not portfolio.holdings:
                msg = "CSV parsed but no holdings found. Check the file format."
                self.console_panel.log_warning(msg)
                if hasattr(self, '_fid_dialog') and self._fid_dialog.isVisible():
                    self._fid_dialog.show_error(msg)
                return

            self._portfolio = portfolio
            self.portfolio_panel.set_portfolio(portfolio)
            # Raise portfolio panel to front (it may be tabbed behind another panel)
            self.portfolio_panel.show()
            self.portfolio_panel.raise_()
            self.set_fidelity_status(True)
            self.console_panel.log_success(
                f"Imported Fidelity CSV: {len(portfolio.holdings)} positions, "
                f"${portfolio.total_value:,.2f} total value"
            )
            # Update ticker bar with top holdings
            items = []
            for h in portfolio.holdings[:20]:
                items.append({
                    "symbol": h.asset.symbol,
                    "price": h.current_price,
                    "change_pct": h.unrealized_pnl_pct if h.cost_basis else 0.0,
                })
            if items:
                self.ticker_bar.set_items(items)
            self._populate_watchlist_from_portfolio()
            # Show success in dialog
            if hasattr(self, '_fid_dialog') and self._fid_dialog.isVisible():
                self._fid_dialog.show_success(f"{len(portfolio.holdings)} positions loaded from CSV")
        except Exception as e:
            logger.exception("Fidelity CSV import failed")
            self.console_panel.log_error(f"Fidelity CSV import failed: {e}")
            if hasattr(self, '_fid_dialog') and self._fid_dialog.isVisible():
                self._fid_dialog.show_error(f"CSV import failed: {e}")

    def _on_fidelity_2fa(self, code: str):
        self.fidelity_controller.submit_2fa(code)

    def _on_fidelity_needs_2fa(self):
        if not hasattr(self, '_fid_dialog') or not self._fid_dialog.isVisible():
            self._show_fidelity_login()
            self._fid_dialog.show_2fa()

    def _on_fidelity_connected(self, portfolio):
        self._portfolio = portfolio
        self.portfolio_panel.set_portfolio(portfolio)
        self.set_fidelity_status(True)
        self.console_panel.log_success(
            f"Fidelity connected: {len(portfolio.holdings)} positions, "
            f"${portfolio.total_value:,.2f} total value"
        )
        items = []
        for h in portfolio.holdings[:20]:
            items.append({
                "symbol": h.asset.symbol,
                "price": h.current_price,
                "change_pct": h.unrealized_pnl_pct if h.cost_basis else 0.0,
            })
        if items:
            self.ticker_bar.set_items(items)
        self._populate_watchlist_from_portfolio()

    def _on_fidelity_disconnected(self):
        self.set_fidelity_status(False)
        self.console_panel.log_warning("Fidelity disconnected")

    def _on_fidelity_error(self, msg: str):
        self.console_panel.log_error(f"Fidelity error: {msg}")

    def _on_playwright_missing(self):
        self.console_panel.log_warning(
            "Playwright Firefox not installed. Fidelity connection requires a browser engine. "
            "Use Data > Fidelity Connection to install it."
        )
        self.set_fidelity_status(False)

    def _refresh_fidelity(self):
        if self.fidelity_controller.is_connected:
            self.fidelity_controller.refresh_positions()
        else:
            self.console_panel.log_warning("Not connected to Fidelity. Use Data > Fidelity Connection.")

    def _set_refresh_interval(self, minutes: int):
        """Update auto-refresh interval and check the right menu item."""
        self.fidelity_controller.set_refresh_interval(minutes)
        # Update checked state of menu actions
        intervals = [0, 1, 5, 15, 30]
        for i, action in enumerate(self._refresh_actions):
            action.setChecked(intervals[i] == minutes)
        label = f"{minutes} min" if minutes > 0 else "off"
        self.console_panel.log_info(f"Fidelity auto-refresh: {label}")

    # ── CSV Import ───────────────────────────────────────────────────
    def _import_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Portfolio CSV", "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return
        try:
            try:
                portfolio = parse_fidelity_csv(path)
                self.console_panel.log_success(f"Imported Fidelity CSV: {len(portfolio.holdings)} positions")
            except Exception:
                portfolio = parse_generic_csv(path)
                self.console_panel.log_success(f"Imported CSV: {len(portfolio.holdings)} positions")

            self._portfolio = portfolio
            self.portfolio_panel.set_portfolio(portfolio)
            self.portfolio_panel.show()
            self.portfolio_panel.raise_()
        except Exception as e:
            self.console_panel.log_error(f"CSV import failed: {e}")

    # ── Helpers ───────────────────────────────────────────────────────
    def _on_watchlist_add(self, symbol: str):
        """Handle adding a symbol to the watchlist — fetch its current price."""
        self.console_panel.log_info(f"Adding {symbol} to watchlist...")
        existing = self.watchlist_panel.get_symbols()
        if symbol in existing:
            self.console_panel.log_warning(f"{symbol} already in watchlist")
            return
        # Add placeholder entry immediately, then fetch price
        items = [{"symbol": s} for s in existing]
        items.append({"symbol": symbol, "price": 0.0, "change": 0.0, "change_pct": 0.0, "volume": 0})
        self.watchlist_panel.set_watchlist(items)
        self.data_controller.fetch_current_price(symbol)

    def _populate_watchlist_from_portfolio(self):
        """Populate watchlist with portfolio holdings."""
        if not self._portfolio:
            return
        items = []
        for h in self._portfolio.holdings:
            items.append({
                "symbol": h.asset.symbol,
                "price": h.current_price,
                "change": h.unrealized_pnl / h.quantity if h.quantity else 0.0,
                "change_pct": h.unrealized_pnl_pct,
                "volume": 0,
            })
        if items:
            self.watchlist_panel.set_watchlist(items[:30])  # top 30

    def _update_price_chart(self, prices: pd.DataFrame | None):
        """Feed price data to the price chart panel."""
        if prices is None or prices.empty:
            return
        self.price_chart_panel.clear_all()
        for col in prices.columns:
            self.price_chart_panel.set_prices(col, prices.index, prices[col].values)

    def _get_active_symbols(self) -> list[str]:
        """Get the current list of symbols to optimize/backtest."""
        if self._portfolio and self._portfolio.symbols:
            return self._portfolio.symbols
        # Fallback: check watchlist
        if hasattr(self.watchlist_panel, 'get_symbols'):
            syms = self.watchlist_panel.get_symbols()
            if syms:
                return syms
        return []

    def _on_lab_import(self):
        """Import current portfolio holdings into the Strategy Lab."""
        from portopt.data.models import AssetType
        if self._portfolio and self._portfolio.holdings:
            # Only send tradeable positions (exclude money market / cash)
            tradeable = [
                h for h in self._portfolio.holdings
                if h.asset.asset_type != AssetType.MONEY_MARKET
            ]
            self.strategy_lab_panel.import_holdings(tradeable)
            self.console_panel.log_info(
                f"Imported {len(tradeable)} tradeable positions into Strategy Lab"
            )
        else:
            self.console_panel.log_warning("No portfolio loaded. Import a CSV first.")

    # ── Layout ───────────────────────────────────────────────────────
    def _setup_focus_layout(self):
        """Clean 2-panel Focus layout: Portfolio | Strategy Lab."""
        for panel in self.panels.values():
            self.removeDockWidget(panel)
            panel.hide()

        # Portfolio (left) | Strategy Lab (right)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.portfolio_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.strategy_lab_panel)

        self.portfolio_panel.show()
        self.strategy_lab_panel.show()

    def _setup_full_layout(self):
        """Full 17-panel trading terminal layout (all panels visible)."""
        for panel in self.panels.values():
            self.removeDockWidget(panel)

        # Row 1: Portfolio (left) | Price Chart (center) | Metrics (right)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.portfolio_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.price_chart_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.metrics_panel)
        self.splitDockWidget(self.price_chart_panel, self.metrics_panel, Qt.Orientation.Horizontal)

        # Row 2: Optimization (left) | Weights/Frontier (center) | Correlation (right)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.optimization_panel)
        self.splitDockWidget(self.portfolio_panel, self.optimization_panel, Qt.Orientation.Vertical)

        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.weights_panel)
        self.splitDockWidget(self.price_chart_panel, self.weights_panel, Qt.Orientation.Vertical)

        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.correlation_panel)
        self.splitDockWidget(self.metrics_panel, self.correlation_panel, Qt.Orientation.Vertical)

        # Tab stacked panels
        self.tabifyDockWidget(self.weights_panel, self.frontier_panel)
        self.weights_panel.raise_()

        self.tabifyDockWidget(self.correlation_panel, self.backtest_panel)
        self.tabifyDockWidget(self.correlation_panel, self.rolling_panel)
        self.tabifyDockWidget(self.correlation_panel, self.risk_panel)
        self.correlation_panel.raise_()

        self.tabifyDockWidget(self.optimization_panel, self.network_panel)
        self.tabifyDockWidget(self.optimization_panel, self.dendrogram_panel)
        self.optimization_panel.raise_()

        self.tabifyDockWidget(self.portfolio_panel, self.watchlist_panel)
        self.tabifyDockWidget(self.portfolio_panel, self.attribution_panel)
        self.tabifyDockWidget(self.portfolio_panel, self.trade_blotter_panel)
        self.tabifyDockWidget(self.portfolio_panel, self.strategy_lab_panel)
        self.portfolio_panel.raise_()

        # Comparison, Scenario, Monte Carlo, Stress Test, Factor, Regime, Risk Budget, Tax Harvest
        self.tabifyDockWidget(self.correlation_panel, self.comparison_panel)
        self.tabifyDockWidget(self.correlation_panel, self.scenario_panel)
        self.tabifyDockWidget(self.correlation_panel, self.monte_carlo_panel)
        self.tabifyDockWidget(self.correlation_panel, self.stress_test_panel)
        self.tabifyDockWidget(self.correlation_panel, self.factor_panel)
        self.tabifyDockWidget(self.correlation_panel, self.regime_panel)
        self.tabifyDockWidget(self.correlation_panel, self.risk_budget_panel)
        self.tabifyDockWidget(self.correlation_panel, self.tax_harvest_panel)
        self.correlation_panel.raise_()

        # Bottom: Console + Copilot (tabbed)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.console_panel)
        self.tabifyDockWidget(self.console_panel, self.copilot_panel)
        self.console_panel.raise_()

        for panel in self.panels.values():
            panel.show()

    # ── Menu Bar ─────────────────────────────────────────────────────
    def _setup_menu_bar(self):
        menubar = self.menuBar()

        # Logo + brand in menu bar corner
        from pathlib import Path
        logo_widget = QWidget()
        logo_layout = QHBoxLayout(logo_widget)
        logo_layout.setContentsMargins(8, 0, 12, 0)
        logo_layout.setSpacing(6)
        icon_path = Path(__file__).resolve().parent.parent / "assets" / "icon.png"
        if icon_path.exists():
            icon_label = QLabel()
            pixmap = QPixmap(str(icon_path)).scaled(
                20, 20, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            icon_label.setPixmap(pixmap)
            logo_layout.addWidget(icon_label)
        brand_label = QLabel(APP_NAME.upper())
        brand_label.setFont(QFont(Fonts.SANS, Fonts.SIZE_SMALL, QFont.Weight.Bold))
        brand_label.setStyleSheet(f"color: {Colors.ACCENT}; letter-spacing: 3px;")
        logo_layout.addWidget(brand_label)
        menubar.setCornerWidget(logo_widget, Qt.Corner.TopLeftCorner)

        # File
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self._action("&Import CSV...", "Ctrl+I", self._import_csv))
        file_menu.addSeparator()
        file_menu.addAction(self._action("E&xport...", "Ctrl+Shift+E", self._show_export))
        file_menu.addSeparator()
        file_menu.addAction(self._action("&Preferences...", "Ctrl+,", self._show_preferences))
        file_menu.addSeparator()
        file_menu.addAction(self._action("&Exit", "Ctrl+Q", self.close))

        # View
        view_menu = menubar.addMenu("&View")
        panels_menu = view_menu.addMenu("&Panels")
        for panel in self.panels.values():
            panels_menu.addAction(panel.toggleViewAction())
        view_menu.addSeparator()
        view_menu.addAction(self._action("Strategy &Lab", "Ctrl+L", self._show_strategy_lab))
        view_menu.addSeparator()
        view_menu.addAction(self._action("Focus View (default)", callback=self._setup_focus_layout))
        view_menu.addAction(self._action("Full View (all panels)", callback=self._setup_full_layout))
        view_menu.addSeparator()
        view_menu.addAction(self._action("&Save Current View...", "Ctrl+Shift+S", self._quick_save_layout))
        view_menu.addAction(self._action("&Manage Views...", "Ctrl+Shift+L", self._show_layout_manager))

        # Data
        data_menu = menubar.addMenu("&Data")
        data_menu.addAction(self._action("&Fidelity Connection...", "Ctrl+F", self._show_fidelity_login))
        data_menu.addAction(self._action("&Refresh Positions", "F5", self._refresh_fidelity))
        data_menu.addSeparator()

        # Auto-refresh submenu
        refresh_menu = data_menu.addMenu("Auto-Refresh &Interval")
        for label, mins in [("Off", 0), ("1 min", 1), ("5 min", 5), ("15 min", 15), ("30 min", 30)]:
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(mins == 0)
            m = mins  # capture for closure
            action.triggered.connect(lambda checked, m=m: self._set_refresh_interval(m))
            refresh_menu.addAction(action)
        self._refresh_actions = refresh_menu.actions()

        # Optimize
        opt_menu = menubar.addMenu("&Optimize")
        opt_menu.addAction(self._action("&Run Optimization", "Ctrl+O", lambda: self._run_optimization(self.optimization_panel.get_config())))
        opt_menu.addAction(self._action("&Black-Litterman Views...", "Ctrl+Shift+B", self._show_bl_views))
        opt_menu.addAction(self._action("&Constraints...", "Ctrl+Shift+C", self._show_constraints))

        # Backtest
        bt_menu = menubar.addMenu("&Backtest")
        bt_menu.addAction(self._action("&Run Backtest", "Ctrl+B", lambda: self._run_backtest(self.backtest_panel.get_config())))

        # AI
        ai_menu = menubar.addMenu("&AI")
        ai_menu.addAction(self._action("&Copilot", "Ctrl+Shift+A", self._show_copilot))
        ai_menu.addAction(self._action("&API Key...", callback=self._show_api_key_dialog))
        ai_menu.addSeparator()
        ai_menu.addAction(self._action("Generate &Report...", "Ctrl+R", self._show_report_dialog))

        # Help
        help_menu = menubar.addMenu("&Help")
        help_menu.addAction(self._action("&About", callback=self._show_about))

    def _action(self, text: str, shortcut: str = None, callback=None) -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        if callback:
            action.triggered.connect(callback)
        return action

    def _show_strategy_lab(self):
        """Show and raise the Strategy Lab panel."""
        self.strategy_lab_panel.show()
        self.strategy_lab_panel.raise_()

    def _quick_save_layout(self):
        """Quick-save current panel arrangement as a named view."""
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(
            self, "Save View", "View name:",
        )
        if ok and name.strip():
            self.dock_manager.save_layout(name.strip())
            self.console_panel.log_success(f"View '{name.strip()}' saved")

    def _show_preferences(self):
        """Show the application preferences dialog."""
        dialog = PreferencesDialog(self)
        dialog.settings_changed.connect(
            lambda: self.console_panel.log_info("Preferences updated — some changes take effect on restart")
        )
        dialog.exec()

    def _show_about(self):
        QMessageBox.about(
            self, f"About {APP_NAME}",
            f"<b>{APP_NAME} v{APP_VERSION}</b><br>"
            f"<i>Quantitative Portfolio Terminal</i><br><br>"
            f"Professional portfolio optimization and backtesting terminal.<br>"
            f"Implements all methods from Hudson & Thames guide.<br><br>"
            f"<b>Shortcuts:</b><br>"
            f"Ctrl+O — Run Optimization<br>"
            f"Ctrl+B — Run Backtest<br>"
            f"Ctrl+I — Import CSV<br>"
            f"Ctrl+L — Strategy Lab<br>"
            f"Ctrl+F — Fidelity Connection<br>"
            f"F5 — Refresh Positions<br>"
            f"Ctrl+Q — Exit"
        )

    # ── Status Bar ───────────────────────────────────────────────────
    def _setup_status_bar(self):
        status = self.statusBar()
        status.setFixedHeight(22)

        self._fidelity_status = QLabel("FIDELITY: DISCONNECTED")
        self._fidelity_status.setFont(QFont(Fonts.MONO, Fonts.SIZE_SMALL))
        self._fidelity_status.setStyleSheet(f"color: {Colors.LOSS}; padding: 0 8px;")
        status.addWidget(self._fidelity_status)

        self._data_status = QLabel("DATA: IDLE")
        self._data_status.setFont(QFont(Fonts.MONO, Fonts.SIZE_SMALL))
        self._data_status.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; padding: 0 8px;")
        status.addWidget(self._data_status)

        # B1: Operation progress label
        self._op_status = QLabel("")
        self._op_status.setFont(QFont(Fonts.MONO, Fonts.SIZE_SMALL))
        self._op_status.setStyleSheet(f"color: {Colors.ACCENT}; padding: 0 8px;")
        status.addWidget(self._op_status)

        # B1: Elapsed timer label
        self._elapsed_label = QLabel("")
        self._elapsed_label.setFont(QFont(Fonts.MONO, Fonts.SIZE_SMALL))
        self._elapsed_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; padding: 0 8px;")
        status.addPermanentWidget(self._elapsed_label)

        self._cache_status = QLabel("")
        self._cache_status.setFont(QFont(Fonts.MONO, Fonts.SIZE_SMALL))
        self._cache_status.setStyleSheet(f"color: {Colors.TEXT_MUTED}; padding: 0 8px;")
        status.addPermanentWidget(self._cache_status)

        # B1: Elapsed timer
        self._elapsed_seconds = 0
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._tick_elapsed)

    def _tick_elapsed(self):
        self._elapsed_seconds += 1
        self._elapsed_label.setText(f"⏱ {self._elapsed_seconds}s")

    def _start_elapsed(self):
        self._elapsed_seconds = 0
        self._elapsed_label.setText("⏱ 0s")
        self._elapsed_timer.start()

    def _stop_elapsed(self):
        self._elapsed_timer.stop()
        if self._elapsed_seconds > 0:
            self._elapsed_label.setText(f"⏱ {self._elapsed_seconds}s (done)")
        QTimer.singleShot(5000, lambda: self._elapsed_label.setText(""))
        QTimer.singleShot(5000, lambda: self._op_status.setText(""))

    def set_fidelity_status(self, connected: bool | None):
        """Update Fidelity status indicator.

        Args:
            connected: True=green, False=red, None=amber (connecting)
        """
        if connected is True:
            self._fidelity_status.setText("FIDELITY: CONNECTED")
            self._fidelity_status.setStyleSheet(f"color: {Colors.PROFIT}; padding: 0 8px;")
        elif connected is None:
            self._fidelity_status.setText("FIDELITY: CONNECTING...")
            self._fidelity_status.setStyleSheet(f"color: {Colors.WARNING}; padding: 0 8px;")
        else:
            self._fidelity_status.setText("FIDELITY: DISCONNECTED")
            self._fidelity_status.setStyleSheet(f"color: {Colors.LOSS}; padding: 0 8px;")

    def set_data_status(self, text: str):
        self._data_status.setText(f"DATA: {text.upper()}")

    def set_cache_status(self, text: str):
        self._cache_status.setText(text)

    # ── Lifecycle ────────────────────────────────────────────────────
    def closeEvent(self, event):
        """Save session state and clean up on close."""
        self._settings.setValue("window/geometry", self.saveGeometry())
        self._settings.setValue("window/state", self.saveState())
        self.dock_manager.save_session()
        self.fidelity_controller.close()
        self.data_controller.close()
        super().closeEvent(event)
