"""Dockable STRATEGY LAB panel — build & test custom portfolios independently.

Users can enter up to 100 tickers, set weights, choose a date range, then
optimize or backtest without affecting the main portfolio.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox, QDateEdit, QDoubleSpinBox, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QProgressBar, QPushButton, QSpinBox, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from portopt.constants import (
    Colors, Fonts, OptMethod, RebalanceFreq, CostModel, MAX_ASSETS,
)
from portopt.gui.panels.base_panel import BasePanel
from portopt.gui.widgets.table_context_menu import setup_table_context_menu

logger = logging.getLogger(__name__)

# Reuse method labels from optimization panel
METHOD_LABELS = {
    OptMethod.INVERSE_VARIANCE: "Inverse Variance",
    OptMethod.MIN_VOLATILITY: "Minimum Volatility",
    OptMethod.MAX_SHARPE: "Maximum Sharpe Ratio",
    OptMethod.EFFICIENT_RISK: "Efficient Risk",
    OptMethod.EFFICIENT_RETURN: "Efficient Return",
    OptMethod.MAX_QUADRATIC_UTILITY: "Max Quadratic Utility",
    OptMethod.MAX_DIVERSIFICATION: "Max Diversification",
    OptMethod.MAX_DECORRELATION: "Max Decorrelation",
    OptMethod.BLACK_LITTERMAN: "Black-Litterman",
    OptMethod.HRP: "Hierarchical Risk Parity",
    OptMethod.HERC: "Hierarchical Equal Risk Contribution",
    OptMethod.TIC: "Theory-Implied Correlation",
}

METRIC_DEFS = [
    ("sharpe_ratio", "Sharpe"),
    ("annual_return", "Ann. Return"),
    ("annual_volatility", "Ann. Vol"),
    ("max_drawdown", "Max DD"),
    ("sortino_ratio", "Sortino"),
    ("calmar_ratio", "Calmar"),
    ("win_rate", "Win Rate"),
    ("total_return", "Total Return"),
]


class StrategyLabPanel(BasePanel):
    panel_id = "strategy_lab"
    panel_title = "STRATEGY LAB"

    # Signal to request portfolio holdings from MainWindow
    import_portfolio_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._opt_controller = None
        self._bt_controller = None
        self._data_controller = None
        self._build_ui()

    def set_controllers(self, data_controller, opt_controller, bt_controller):
        """Set the lab's own controller instances (created by MainWindow)."""
        self._data_controller = data_controller
        self._opt_controller = opt_controller
        self._bt_controller = bt_controller

        # Wire optimization signals
        self._opt_controller.optimization_complete.connect(self._on_optimization_complete)
        self._opt_controller.status_changed.connect(self._on_status)
        self._opt_controller.error.connect(self._on_error)
        self._opt_controller.running_changed.connect(
            lambda running: self._opt_progress.setVisible(running)
        )

        # Wire backtest signals
        self._bt_controller.equity_curve_ready.connect(self._on_equity_curve)
        self._bt_controller.metrics_ready.connect(self._on_bt_metrics)
        self._bt_controller.status_changed.connect(self._on_status)
        self._bt_controller.error.connect(self._on_error)
        self._bt_controller.running_changed.connect(
            lambda running: self._bt_progress.setVisible(running)
        )

    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)

        # ── LEFT: Configuration ──────────────────────────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(4)

        # -- Tickers group --
        ticker_group = QGroupBox("Tickers")
        ticker_group.setStyleSheet(self._group_style())
        ticker_layout = QVBoxLayout()
        ticker_layout.setSpacing(4)

        add_row = QHBoxLayout()
        self._ticker_input = QLineEdit()
        self._ticker_input.setPlaceholderText("Enter ticker (e.g. AAPL)")
        self._ticker_input.setFixedHeight(26)
        self._ticker_input.returnPressed.connect(self._add_ticker)
        add_row.addWidget(self._ticker_input)

        add_btn = QPushButton("Add")
        add_btn.setFixedSize(50, 26)
        add_btn.setStyleSheet(self._accent_btn_style())
        add_btn.clicked.connect(self._add_ticker)
        add_row.addWidget(add_btn)
        ticker_layout.addLayout(add_row)

        self._ticker_list = QListWidget()
        self._ticker_list.setMaximumHeight(120)
        self._ticker_list.setStyleSheet(f"""
            QListWidget {{
                background: {Colors.BG_INPUT};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                font-family: {Fonts.MONO};
                font-size: 10px;
            }}
            QListWidget::item:selected {{
                background: {Colors.ACCENT_DIM};
            }}
        """)
        ticker_layout.addWidget(self._ticker_list)

        btn_row = QHBoxLayout()
        remove_btn = QPushButton("Remove Selected")
        remove_btn.setFixedHeight(22)
        remove_btn.clicked.connect(self._remove_selected_ticker)
        btn_row.addWidget(remove_btn)

        import_btn = QPushButton("Import from Portfolio")
        import_btn.setFixedHeight(22)
        import_btn.setStyleSheet(self._accent_btn_style())
        import_btn.clicked.connect(lambda: self.import_portfolio_requested.emit())
        btn_row.addWidget(import_btn)
        ticker_layout.addLayout(btn_row)

        ticker_group.setLayout(ticker_layout)
        left_layout.addWidget(ticker_group)

        # -- Weights group --
        weights_group = QGroupBox("Weights (optional, leave blank for equal-weight)")
        weights_group.setStyleSheet(self._group_style())
        weights_layout = QVBoxLayout()

        self._weights_table = QTableWidget()
        self._weights_table.setColumnCount(2)
        self._weights_table.setHorizontalHeaderLabels(["Symbol", "Weight %"])
        self._weights_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._weights_table.verticalHeader().setVisible(False)
        self._weights_table.setMaximumHeight(100)
        self._weights_table.setStyleSheet(f"""
            QTableWidget {{
                background: {Colors.BG_INPUT};
                color: {Colors.TEXT_PRIMARY};
                font-family: {Fonts.MONO};
                font-size: 10px;
                border: 1px solid {Colors.BORDER};
            }}
            QHeaderView::section {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_SECONDARY};
                border: 1px solid {Colors.BORDER};
                font-size: 9px;
                font-weight: bold;
            }}
        """)
        weights_layout.addWidget(self._weights_table)
        weights_group.setLayout(weights_layout)
        left_layout.addWidget(weights_group)

        # -- Date Range --
        date_group = QGroupBox("Date Range")
        date_group.setStyleSheet(self._group_style())
        date_layout = QHBoxLayout()
        date_layout.setSpacing(8)

        self._start_date = QDateEdit()
        self._start_date.setCalendarPopup(True)
        self._start_date.setDate(date.today() - timedelta(days=756))
        self._start_date.setDisplayFormat("yyyy-MM-dd")
        date_layout.addWidget(QLabel("Start:"))
        date_layout.addWidget(self._start_date)

        self._end_date = QDateEdit()
        self._end_date.setCalendarPopup(True)
        self._end_date.setDate(date.today())
        self._end_date.setDisplayFormat("yyyy-MM-dd")
        date_layout.addWidget(QLabel("End:"))
        date_layout.addWidget(self._end_date)

        date_group.setLayout(date_layout)
        left_layout.addWidget(date_group)

        # -- Optimization config --
        opt_group = QGroupBox("Optimization")
        opt_group.setStyleSheet(self._group_style())
        opt_layout = QHBoxLayout()
        opt_layout.setSpacing(6)

        self._method_combo = QComboBox()
        for method, label in METHOD_LABELS.items():
            self._method_combo.addItem(label, method)
        # Default to MAX_SHARPE
        self._method_combo.setCurrentIndex(2)
        opt_layout.addWidget(self._method_combo)

        opt_layout.addWidget(QLabel("Rf:"))
        self._rf_spin = QDoubleSpinBox()
        self._rf_spin.setRange(0, 0.20)
        self._rf_spin.setValue(0.02)
        self._rf_spin.setDecimals(3)
        self._rf_spin.setSingleStep(0.005)
        self._rf_spin.setFixedWidth(70)
        opt_layout.addWidget(self._rf_spin)

        opt_group.setLayout(opt_layout)
        left_layout.addWidget(opt_group)

        # -- Backtest config --
        bt_group = QGroupBox("Backtest")
        bt_group.setStyleSheet(self._group_style())
        bt_layout = QHBoxLayout()
        bt_layout.setSpacing(6)

        bt_layout.addWidget(QLabel("Rebal:"))
        self._rebal_combo = QComboBox()
        for f in RebalanceFreq:
            self._rebal_combo.addItem(f.name.title(), f)
        self._rebal_combo.setCurrentIndex(2)  # Monthly
        bt_layout.addWidget(self._rebal_combo)

        bt_layout.addWidget(QLabel("Init $:"))
        self._initial_spin = QSpinBox()
        self._initial_spin.setRange(1000, 100_000_000)
        self._initial_spin.setValue(100_000)
        self._initial_spin.setSingleStep(10000)
        self._initial_spin.setFixedWidth(90)
        bt_layout.addWidget(self._initial_spin)

        bt_group.setLayout(bt_layout)
        left_layout.addWidget(bt_group)

        # -- Action buttons --
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)

        self._opt_btn = QPushButton("OPTIMIZE")
        self._opt_btn.setFixedHeight(32)
        self._opt_btn.setStyleSheet(self._run_btn_style())
        self._opt_btn.clicked.connect(self._run_optimization)
        btn_layout.addWidget(self._opt_btn)

        self._bt_btn = QPushButton("BACKTEST")
        self._bt_btn.setFixedHeight(32)
        self._bt_btn.setStyleSheet(self._run_btn_style())
        self._bt_btn.clicked.connect(self._run_backtest)
        btn_layout.addWidget(self._bt_btn)

        left_layout.addLayout(btn_layout)

        # Progress bars
        self._opt_progress = QProgressBar()
        self._opt_progress.setFixedHeight(3)
        self._opt_progress.setTextVisible(False)
        self._opt_progress.setRange(0, 0)
        self._opt_progress.hide()
        self._opt_progress.setStyleSheet(f"""
            QProgressBar {{ background: {Colors.BG_INPUT}; border: none; }}
            QProgressBar::chunk {{ background: {Colors.ACCENT}; }}
        """)
        left_layout.addWidget(self._opt_progress)

        self._bt_progress = QProgressBar()
        self._bt_progress.setFixedHeight(3)
        self._bt_progress.setTextVisible(False)
        self._bt_progress.setRange(0, 0)
        self._bt_progress.hide()
        self._bt_progress.setStyleSheet(f"""
            QProgressBar {{ background: {Colors.BG_INPUT}; border: none; }}
            QProgressBar::chunk {{ background: {Colors.PROFIT}; }}
        """)
        left_layout.addWidget(self._bt_progress)

        # Status label
        self._status_label = QLabel("Add tickers to begin")
        self._status_label.setStyleSheet(
            f"color: {Colors.TEXT_MUTED}; font-size: {Fonts.SIZE_SMALL}pt;"
        )
        left_layout.addWidget(self._status_label)

        left_layout.addStretch()
        splitter.addWidget(left)

        # ── RIGHT: Results ───────────────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(4)

        # Weights result table
        weights_label = QLabel("OPTIMIZED WEIGHTS")
        weights_label.setStyleSheet(
            f"color: {Colors.TEXT_MUTED}; font-family: {Fonts.SANS}; "
            f"font-size: 9px; font-weight: bold;"
        )
        right_layout.addWidget(weights_label)

        self._result_table = QTableWidget()
        self._result_table.setColumnCount(3)
        self._result_table.setHorizontalHeaderLabels(["Symbol", "Current %", "Optimized %"])
        self._result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._result_table.verticalHeader().setVisible(False)
        self._result_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._result_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._result_table.setMaximumHeight(180)
        self._result_table.setFont(QFont(Fonts.MONO, Fonts.SIZE_SMALL))
        self._result_table.setStyleSheet(f"""
            QTableWidget {{
                background: {Colors.BG_SECONDARY};
                gridline-color: {Colors.BORDER};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
            }}
            QHeaderView::section {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_SECONDARY};
                border: 1px solid {Colors.BORDER};
                font-size: 9px;
                font-weight: bold;
            }}
        """)
        setup_table_context_menu(self._result_table)
        right_layout.addWidget(self._result_table)

        # Equity curve chart
        equity_label = QLabel("EQUITY CURVE")
        equity_label.setStyleSheet(
            f"color: {Colors.TEXT_MUTED}; font-family: {Fonts.SANS}; "
            f"font-size: 9px; font-weight: bold;"
        )
        right_layout.addWidget(equity_label)

        self._equity_plot = pg.PlotWidget()
        self._equity_plot.setBackground(Colors.BG_SECONDARY)
        self._equity_plot.showGrid(x=True, y=True, alpha=0.15)
        self._equity_plot.setLabel("left", "Value ($)", color=Colors.TEXT_SECONDARY)
        self._equity_plot.setMinimumHeight(160)
        right_layout.addWidget(self._equity_plot, stretch=1)

        # Key metrics grid
        metrics_label = QLabel("KEY METRICS")
        metrics_label.setStyleSheet(
            f"color: {Colors.TEXT_MUTED}; font-family: {Fonts.SANS}; "
            f"font-size: 9px; font-weight: bold;"
        )
        right_layout.addWidget(metrics_label)

        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(2)
        self._metric_labels: dict[str, QLabel] = {}

        for i, (key, display_name) in enumerate(METRIC_DEFS):
            row, col = i // 4, (i % 4) * 2

            name_lbl = QLabel(display_name)
            name_lbl.setStyleSheet(
                f"color: {Colors.TEXT_MUTED}; font-family: {Fonts.SANS}; "
                f"font-size: 8px; font-weight: bold;"
            )
            metrics_grid.addWidget(name_lbl, row, col)

            val_lbl = QLabel("--")
            val_lbl.setStyleSheet(
                f"color: {Colors.TEXT_PRIMARY}; font-family: {Fonts.MONO}; "
                f"font-size: 11px; font-weight: bold;"
            )
            val_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            metrics_grid.addWidget(val_lbl, row, col + 1)
            self._metric_labels[key] = val_lbl

        right_layout.addLayout(metrics_grid)
        splitter.addWidget(right)
        splitter.setSizes([320, 480])

        self.content_layout.addWidget(splitter)

    # ── Ticker Management ────────────────────────────────────────────

    def _add_ticker(self):
        """Add a ticker from the input field to the list."""
        text = self._ticker_input.text().strip().upper()
        if not text:
            return
        if self._ticker_list.count() >= MAX_ASSETS:
            self._status_label.setText(f"Max {MAX_ASSETS} tickers")
            return
        # Check for duplicates
        for i in range(self._ticker_list.count()):
            if self._ticker_list.item(i).text() == text:
                self._ticker_input.clear()
                return
        self._ticker_list.addItem(text)
        self._ticker_input.clear()
        self._sync_weights_table()
        self._status_label.setText(f"{self._ticker_list.count()} tickers")

    def _remove_selected_ticker(self):
        """Remove the selected ticker from the list."""
        for item in self._ticker_list.selectedItems():
            self._ticker_list.takeItem(self._ticker_list.row(item))
        self._sync_weights_table()
        self._status_label.setText(f"{self._ticker_list.count()} tickers")

    def _sync_weights_table(self):
        """Keep the weights table in sync with the ticker list."""
        symbols = self._get_symbols()
        self._weights_table.setRowCount(len(symbols))
        for i, sym in enumerate(symbols):
            sym_item = QTableWidgetItem(sym)
            sym_item.setFlags(sym_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._weights_table.setItem(i, 0, sym_item)
            # Preserve existing weight if present
            existing = self._weights_table.item(i, 1)
            if existing is None:
                self._weights_table.setItem(i, 1, QTableWidgetItem(""))

    def _get_symbols(self) -> list[str]:
        """Get the list of ticker symbols."""
        return [
            self._ticker_list.item(i).text()
            for i in range(self._ticker_list.count())
        ]

    def _get_custom_weights(self) -> dict[str, float] | None:
        """Get user-specified weights, or None for equal-weight."""
        symbols = self._get_symbols()
        weights = {}
        has_any = False
        for i, sym in enumerate(symbols):
            item = self._weights_table.item(i, 1)
            if item and item.text().strip():
                try:
                    w = float(item.text().strip()) / 100.0
                    weights[sym] = w
                    has_any = True
                except ValueError:
                    pass
            else:
                weights[sym] = 0.0

        if not has_any:
            return None  # Use equal-weight
        return weights

    # ── Import from Portfolio ────────────────────────────────────────

    def import_holdings(self, holdings: list):
        """Populate the lab with holdings from the real portfolio.

        Args:
            holdings: list of Holding objects with .asset.symbol, .market_value
        """
        self._ticker_list.clear()
        total_value = sum(h.market_value for h in holdings)

        for h in holdings:
            sym = h.asset.symbol
            if sym and sym not in [
                self._ticker_list.item(i).text()
                for i in range(self._ticker_list.count())
            ]:
                self._ticker_list.addItem(sym)

        self._sync_weights_table()

        # Fill weights from current allocation
        if total_value > 0:
            symbols = self._get_symbols()
            for i, sym in enumerate(symbols):
                for h in holdings:
                    if h.asset.symbol == sym:
                        pct = (h.market_value / total_value) * 100
                        self._weights_table.setItem(i, 1, QTableWidgetItem(f"{pct:.1f}"))
                        break

        self._status_label.setText(f"Imported {self._ticker_list.count()} tickers from portfolio")

    # ── Optimization ─────────────────────────────────────────────────

    def _run_optimization(self):
        """Run optimization on the lab's tickers."""
        symbols = self._get_symbols()
        if not symbols:
            self._status_label.setText("Add tickers first")
            return
        if self._opt_controller is None:
            self._status_label.setText("Controllers not initialized")
            return

        method = self._method_combo.currentData()
        config = {
            "method": method,
            "risk_free_rate": self._rf_spin.value(),
            "risk_aversion": 1.0,
            "long_only": True,
            "min_weight": 0.0,
            "max_weight": 1.0,
        }

        # Calculate lookback from date range
        start = self._start_date.date().toPython()
        end = self._end_date.date().toPython()
        lookback = (end - start).days

        self._status_label.setText(f"Optimizing {len(symbols)} symbols...")
        self._opt_controller.fetch_and_optimize(symbols, config, lookback_days=lookback)

    def _on_optimization_complete(self, result):
        """Handle optimization result."""
        current_weights = self._get_custom_weights()
        symbols = self._get_symbols()

        # Build equal-weight if no custom weights
        if current_weights is None and symbols:
            eq = 1.0 / len(symbols)
            current_weights = {s: eq for s in symbols}

        # Populate result table
        opt_weights = result.weights
        all_syms = sorted(set(list(opt_weights.keys()) + symbols))
        self._result_table.setRowCount(len(all_syms))

        for i, sym in enumerate(all_syms):
            self._result_table.setItem(i, 0, QTableWidgetItem(sym))

            cur = (current_weights or {}).get(sym, 0.0) * 100
            cur_item = QTableWidgetItem(f"{cur:.1f}%")
            cur_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._result_table.setItem(i, 1, cur_item)

            opt = opt_weights.get(sym, 0.0) * 100
            opt_item = QTableWidgetItem(f"{opt:.1f}%")
            opt_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if opt > 0.1:
                from PySide6.QtGui import QColor
                opt_item.setForeground(QColor(Colors.ACCENT))
            self._result_table.setItem(i, 2, opt_item)

        self._status_label.setText(
            f"Optimization complete: {result.method} | Sharpe={result.sharpe_ratio:.3f}"
        )

    # ── Backtest ─────────────────────────────────────────────────────

    def _run_backtest(self):
        """Run backtest on the lab's tickers."""
        symbols = self._get_symbols()
        if not symbols:
            self._status_label.setText("Add tickers first")
            return
        if self._bt_controller is None:
            self._status_label.setText("Controllers not initialized")
            return

        rebal = self._rebal_combo.currentData()
        config = {
            "rebalance_freq": rebal.name.title(),
            "cost_model": "Proportional",
            "cost_rate": 0.001,
            "initial_value": self._initial_spin.value(),
            "walk_forward_enabled": False,
            "lookback": 252,
            "drift_threshold": 0.0,
        }

        # Calculate lookback from date range
        start = self._start_date.date().toPython()
        end = self._end_date.date().toPython()
        lookback = (end - start).days

        self._status_label.setText(f"Backtesting {len(symbols)} symbols...")
        self._bt_controller.fetch_and_backtest(symbols, config, lookback_days=lookback)

    def _on_equity_curve(self, dates_epoch, values):
        """Plot equity curve from backtest."""
        self._equity_plot.clear()
        pen = pg.mkPen(Colors.ACCENT, width=2)
        self._equity_plot.plot(dates_epoch, values, pen=pen)

    def _on_bt_metrics(self, metrics: dict):
        """Display backtest metrics in the grid."""
        fmt_map = {
            "sharpe_ratio": lambda v: f"{v:.3f}",
            "annual_return": lambda v: f"{v:+.2%}",
            "annual_volatility": lambda v: f"{v:.2%}",
            "max_drawdown": lambda v: f"{v:.2%}",
            "sortino_ratio": lambda v: f"{v:.3f}",
            "calmar_ratio": lambda v: f"{v:.3f}",
            "win_rate": lambda v: f"{v:.1%}",
            "total_return": lambda v: f"{v:+.2%}",
        }
        colored_keys = {
            "sharpe_ratio", "annual_return", "sortino_ratio",
            "calmar_ratio", "total_return",
        }

        for key, label_widget in self._metric_labels.items():
            value = metrics.get(key)
            if value is None:
                label_widget.setText("--")
                label_widget.setStyleSheet(
                    f"color: {Colors.TEXT_MUTED}; font-family: {Fonts.MONO}; "
                    f"font-size: 11px; font-weight: bold;"
                )
                continue

            fmt = fmt_map.get(key, lambda v: f"{v:.4f}")
            try:
                text = fmt(value)
            except (TypeError, ValueError):
                text = str(value)

            color = Colors.TEXT_PRIMARY
            if key in colored_keys:
                if isinstance(value, (int, float)):
                    color = Colors.PROFIT if value > 0 else Colors.LOSS
            elif key == "max_drawdown" and isinstance(value, (int, float)) and value < 0:
                color = Colors.LOSS

            label_widget.setText(text)
            label_widget.setStyleSheet(
                f"color: {color}; font-family: {Fonts.MONO}; "
                f"font-size: 11px; font-weight: bold;"
            )

        self._status_label.setText("Backtest complete")

    # ── Callbacks ────────────────────────────────────────────────────

    def _on_status(self, msg: str):
        self._status_label.setText(msg)

    def _on_error(self, msg: str):
        self._status_label.setText(f"Error: {msg}")
        self._status_label.setStyleSheet(
            f"color: {Colors.LOSS}; font-size: {Fonts.SIZE_SMALL}pt;"
        )

    # ── Styles ───────────────────────────────────────────────────────

    def _group_style(self) -> str:
        return f"""
            QGroupBox {{
                color: {Colors.TEXT_SECONDARY};
                font-family: {Fonts.SANS};
                font-size: 10px;
                font-weight: bold;
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 12px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }}
        """

    def _accent_btn_style(self) -> str:
        return f"""
            QPushButton {{
                background: {Colors.BG_INPUT};
                color: {Colors.ACCENT};
                border: 1px solid {Colors.BORDER};
                border-radius: 3px;
                font-family: {Fonts.SANS};
                font-size: 9px;
                font-weight: bold;
                padding: 0 8px;
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT_DIM};
                border-color: {Colors.ACCENT};
            }}
        """

    def _run_btn_style(self) -> str:
        return f"""
            QPushButton {{
                background: {Colors.ACCENT_DIM};
                color: {Colors.ACCENT};
                border: 1px solid {Colors.ACCENT};
                border-radius: 4px;
                font-family: {Fonts.SANS};
                font-size: 11px;
                font-weight: bold;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT};
                color: {Colors.BG_PRIMARY};
            }}
            QPushButton:disabled {{
                background: {Colors.BG_INPUT};
                color: {Colors.TEXT_MUTED};
                border-color: {Colors.BORDER};
            }}
        """
