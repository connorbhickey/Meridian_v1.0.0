"""Main trading terminal window with dockable panel system."""

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QSize, QTimer, QSettings
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
from portopt.gui.panels.console_panel import ConsolePanel, ConsoleLogHandler
from portopt.gui.controllers.fidelity_controller import FidelityController
from portopt.gui.controllers.data_controller import DataController
from portopt.gui.controllers.optimization_controller import OptimizationController
from portopt.gui.controllers.backtest_controller import BacktestController
from portopt.gui.dialogs.fidelity_login_dialog import FidelityLoginDialog
from portopt.gui.dialogs.bl_views_dialog import BLViewsDialog
from portopt.gui.dialogs.constraint_dialog import ConstraintDialog
from portopt.gui.dialogs.export_dialog import ExportDialog, export_weights_csv, export_trades_csv, export_metrics_csv
from portopt.gui.dialogs.layout_dialog import LayoutDialog
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
        self._settings = QSettings(APP_NAME, APP_NAME)

        self._setup_ticker_bar()
        self._setup_panels()
        self._setup_controllers()
        self._setup_menu_bar()
        self._setup_status_bar()
        self._setup_logging()
        self._setup_default_layout()

        # Restore window geometry
        geo = self._settings.value("window/geometry")
        if geo:
            self.restoreGeometry(geo)
        state = self._settings.value("window/state")
        if state:
            self.restoreState(state)

        # Try restoring last session layout
        if not self.dock_manager.restore_session():
            self._setup_default_layout()

        # Startup: try auto-connect to Fidelity after window is shown
        QTimer.singleShot(500, self._on_startup)

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

        for panel in [
            self.portfolio_panel, self.watchlist_panel, self.price_chart_panel,
            self.correlation_panel, self.optimization_panel, self.weights_panel,
            self.frontier_panel, self.backtest_panel, self.metrics_panel,
            self.attribution_panel, self.network_panel, self.dendrogram_panel,
            self.trade_blotter_panel, self.risk_panel, self.comparison_panel,
            self.scenario_panel, self.console_panel,
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
        # B1: Wire progress + running signals to status bar
        self.bt_controller.progress.connect(lambda msg: self._op_status.setText(msg))
        self.bt_controller.running_changed.connect(
            lambda running: self._start_elapsed() if running else self._stop_elapsed()
        )

        # Panel signals
        self.portfolio_panel.connect_requested.connect(self._show_fidelity_login)
        self.portfolio_panel.refresh_requested.connect(self._refresh_fidelity)
        self.optimization_panel.run_requested.connect(self._run_optimization)
        self.backtest_panel.run_requested.connect(self._run_backtest)

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

    def _on_frontier_complete(self, risks, returns):
        self.frontier_panel.clear_plot()
        self.frontier_panel.set_frontier(risks, returns)

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
            if "Weights" in fmt and self.opt_controller.last_result:
                result = self.opt_controller.last_result
                metadata = {"method": result.method, "sharpe": f"{result.sharpe_ratio:.4f}"}
                if config.get("include_metadata"):
                    export_weights_csv(result.weights, path, metadata)
                else:
                    export_weights_csv(result.weights, path)
                self.console_panel.log_success(f"Weights exported to {path}")

            elif "Trades" in fmt and self.bt_controller.last_output:
                output = self.bt_controller.last_output
                trades = []
                if output.result:
                    trades = self.bt_controller._trades_to_dicts(output.result.trades)
                export_trades_csv(trades, path)
                self.console_panel.log_success(f"Trades exported to {path}")

            elif "Metrics" in fmt:
                metrics = {}
                if self.bt_controller.last_output:
                    metrics = self.bt_controller.last_output.metrics
                export_metrics_csv(metrics, path)
                self.console_panel.log_success(f"Metrics exported to {path}")
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
            self._portfolio = portfolio
            self.portfolio_panel.set_portfolio(portfolio)
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
            # Close dialog on success
            if hasattr(self, '_fid_dialog') and self._fid_dialog.isVisible():
                self._fid_dialog.show_success(f"{len(portfolio.holdings)} positions loaded from CSV")
        except Exception as e:
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
        except Exception as e:
            self.console_panel.log_error(f"CSV import failed: {e}")

    # ── Helpers ───────────────────────────────────────────────────────
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

    # ── Layout ───────────────────────────────────────────────────────
    def _setup_default_layout(self):
        """Arrange panels in the default trading terminal layout."""
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
        self.tabifyDockWidget(self.correlation_panel, self.risk_panel)
        self.correlation_panel.raise_()

        self.tabifyDockWidget(self.optimization_panel, self.network_panel)
        self.tabifyDockWidget(self.optimization_panel, self.dendrogram_panel)
        self.optimization_panel.raise_()

        self.tabifyDockWidget(self.portfolio_panel, self.watchlist_panel)
        self.tabifyDockWidget(self.portfolio_panel, self.attribution_panel)
        self.tabifyDockWidget(self.portfolio_panel, self.trade_blotter_panel)
        self.portfolio_panel.raise_()

        # Bottom: Console
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.console_panel)

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
        file_menu.addAction(self._action("&Exit", "Ctrl+Q", self.close))

        # View
        view_menu = menubar.addMenu("&View")
        panels_menu = view_menu.addMenu("&Panels")
        for panel in self.panels.values():
            panels_menu.addAction(panel.toggleViewAction())
        view_menu.addSeparator()
        layout_menu = view_menu.addMenu("&Layouts")
        layout_menu.addAction(self._action("Default Layout", callback=self._setup_default_layout))
        layout_menu.addAction(self._action("Save/Load Layout...", "Ctrl+Shift+L", self._show_layout_manager))

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
