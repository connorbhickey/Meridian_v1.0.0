"""Main trading terminal window with dockable panel system."""

import logging

from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QAction, QFont, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow, QMenuBar, QMenu, QStatusBar, QLabel,
    QDockWidget, QVBoxLayout, QWidget, QFileDialog,
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
from portopt.gui.panels.console_panel import ConsolePanel, ConsoleLogHandler
from portopt.gui.controllers.fidelity_controller import FidelityController
from portopt.gui.controllers.data_controller import DataController
from portopt.gui.dialogs.fidelity_login_dialog import FidelityLoginDialog
from portopt.data.importers.fidelity_csv import parse_fidelity_csv
from portopt.data.importers.generic_csv import parse_generic_csv

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Trading terminal main window with dockable panels."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION} — Portfolio Terminal")
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

        # Current portfolio
        self._portfolio = None

        self._setup_ticker_bar()
        self._setup_panels()
        self._setup_controllers()
        self._setup_menu_bar()
        self._setup_status_bar()
        self._setup_logging()
        self._setup_default_layout()

        # Try restoring last session layout
        if not self.dock_manager.restore_session():
            self._setup_default_layout()

        # Startup: try auto-connect to Fidelity after window is shown
        QTimer.singleShot(500, self._on_startup)

    # ── Startup ──────────────────────────────────────────────────────
    def _on_startup(self):
        """Called shortly after window shows — attempt Fidelity auto-connect."""
        self.console_panel.log_info("PortOpt terminal started")
        cache_mb = self.data_controller.get_cache_size()
        self.set_cache_status(f"CACHE: {cache_mb:.1f} MB")

        if self.fidelity_controller.has_saved_session:
            self.console_panel.log_info("Found saved Fidelity session, attempting auto-connect...")
            self.fidelity_controller.try_auto_connect()
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
        self.console_panel = ConsolePanel(self)

        for panel in [
            self.portfolio_panel, self.watchlist_panel, self.price_chart_panel,
            self.correlation_panel, self.optimization_panel, self.weights_panel,
            self.frontier_panel, self.backtest_panel, self.metrics_panel,
            self.attribution_panel, self.network_panel, self.dendrogram_panel,
            self.trade_blotter_panel, self.risk_panel, self.console_panel,
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

        # Data controller
        self.data_controller = DataController(self)
        self.data_controller.status_changed.connect(
            lambda msg: self.console_panel.log_data(f"Data: {msg}")
        )
        self.data_controller.error.connect(
            lambda msg: self.console_panel.log_error(f"Data error: {msg}")
        )

        # Portfolio panel signals
        self.portfolio_panel.connect_requested.connect(self._show_fidelity_login)
        self.portfolio_panel.refresh_requested.connect(self._refresh_fidelity)

    # ── Logging ──────────────────────────────────────────────────────
    def _setup_logging(self):
        """Route Python logging to the console panel."""
        handler = ConsoleLogHandler(self.console_panel)
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        root = logging.getLogger("portopt")
        root.addHandler(handler)
        root.setLevel(logging.INFO)

    # ── Fidelity Connection Flow ─────────────────────────────────────
    def _show_fidelity_login(self):
        """Show the Fidelity login dialog."""
        self._fid_dialog = FidelityLoginDialog(self)
        self._fid_dialog.login_requested.connect(self._on_fidelity_login)
        self._fid_dialog.twofa_submitted.connect(self._on_fidelity_2fa)
        self._fid_dialog.skip_requested.connect(
            lambda: self.console_panel.log_info("Fidelity connection skipped")
        )

        # Wire controller signals to dialog
        self.fidelity_controller.needs_2fa.connect(self._fid_dialog.show_2fa)
        self.fidelity_controller.connected.connect(
            lambda p: self._fid_dialog.show_success(f"{len(p.holdings)} positions loaded")
        )
        self.fidelity_controller.connection_error.connect(self._fid_dialog.show_error)

        self._fid_dialog.exec()

    def _on_fidelity_login(self, username: str, password: str, totp: str):
        self.fidelity_controller.login(
            username, password, totp,
            save_credentials=self._fid_dialog.remember_credentials,
        )

    def _on_fidelity_2fa(self, code: str):
        self.fidelity_controller.submit_2fa(code)

    def _on_fidelity_needs_2fa(self):
        # If dialog isn't open, open it at 2FA page
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
        # Update ticker bar with portfolio symbols
        items = []
        for h in portfolio.holdings[:20]:  # Top 20 for ticker bar
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

    def _refresh_fidelity(self):
        if self.fidelity_controller.is_connected:
            self.fidelity_controller.refresh_positions()
        else:
            self.console_panel.log_warning("Not connected to Fidelity. Use Data > Fidelity Connection.")

    # ── CSV Import ───────────────────────────────────────────────────
    def _import_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Portfolio CSV", "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return
        try:
            # Try Fidelity format first, fall back to generic
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

        # File
        file_menu = menubar.addMenu("&File")
        import_action = self._action("&Import CSV...", "Ctrl+I", self._import_csv)
        file_menu.addAction(import_action)
        file_menu.addSeparator()
        file_menu.addAction(self._action("E&xport Weights CSV...", "Ctrl+Shift+W"))
        file_menu.addAction(self._action("Export &Report PDF...", "Ctrl+Shift+R"))
        file_menu.addSeparator()
        exit_action = self._action("&Exit", "Ctrl+Q", self.close)
        file_menu.addAction(exit_action)

        # View
        view_menu = menubar.addMenu("&View")
        panels_menu = view_menu.addMenu("&Panels")
        for panel in self.panels.values():
            panels_menu.addAction(panel.toggleViewAction())
        view_menu.addSeparator()
        layout_menu = view_menu.addMenu("&Layouts")
        layout_menu.addAction(self._action("Default Layout", callback=self._setup_default_layout))
        layout_menu.addAction(self._action("Save Layout...", "Ctrl+Shift+L"))

        # Data
        data_menu = menubar.addMenu("&Data")
        data_menu.addAction(self._action("&Fidelity Connection...", "Ctrl+F", self._show_fidelity_login))
        data_menu.addAction(self._action("&Refresh Positions", "F5", self._refresh_fidelity))
        data_menu.addSeparator()
        data_menu.addAction(self._action("&Cache Management..."))

        # Optimize
        opt_menu = menubar.addMenu("&Optimize")
        opt_menu.addAction(self._action("&Run Optimization", "Ctrl+O"))
        opt_menu.addAction(self._action("&Black-Litterman Views...", "Ctrl+Shift+B"))
        opt_menu.addAction(self._action("&Constraints...", "Ctrl+Shift+C"))

        # Backtest
        bt_menu = menubar.addMenu("&Backtest")
        bt_menu.addAction(self._action("&Run Backtest", "Ctrl+B"))
        bt_menu.addAction(self._action("&Walk-Forward Config..."))

        # Help
        help_menu = menubar.addMenu("&Help")
        help_menu.addAction(self._action("&About"))
        help_menu.addAction(self._action("&Keyboard Shortcuts"))

    def _action(self, text: str, shortcut: str = None, callback=None) -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        if callback:
            action.triggered.connect(callback)
        return action

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

        self._cache_status = QLabel("")
        self._cache_status.setFont(QFont(Fonts.MONO, Fonts.SIZE_SMALL))
        self._cache_status.setStyleSheet(f"color: {Colors.TEXT_MUTED}; padding: 0 8px;")
        status.addPermanentWidget(self._cache_status)

    def set_fidelity_status(self, connected: bool):
        if connected:
            self._fidelity_status.setText("FIDELITY: CONNECTED")
            self._fidelity_status.setStyleSheet(f"color: {Colors.PROFIT}; padding: 0 8px;")
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
        self.dock_manager.save_session()
        self.fidelity_controller.close()
        self.data_controller.close()
        super().closeEvent(event)
