"""Dockable MONTE CARLO panel — fan chart, config, metrics distribution."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox,
    QSpinBox, QDoubleSpinBox, QPushButton, QProgressBar,
    QGroupBox, QSplitter, QLabel, QTableWidget, QTableWidgetItem,
    QHeaderView,
)
import pyqtgraph as pg
import numpy as np

from portopt.gui.panels.base_panel import BasePanel
from portopt.constants import Colors, Fonts, MCSimMethod
from portopt.data.models import MonteCarloConfig, MonteCarloResult
from portopt.gui.widgets.table_context_menu import setup_table_context_menu


class MonteCarloPanel(BasePanel):
    panel_id = "monte_carlo"
    panel_title = "MONTE CARLO"

    run_requested = Signal(object)  # emits MonteCarloConfig

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        splitter = QSplitter(Qt.Horizontal)

        # ── Config section (left) ─────────────────────────────────────
        config_widget = pg.QtWidgets.QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.setSpacing(4)

        # Simulation settings
        sim_group = QGroupBox("Simulation")
        sim_group.setStyleSheet(self._group_style())
        sim_layout = QFormLayout()
        sim_layout.setSpacing(4)

        self._method_combo = QComboBox()
        self._method_combo.addItem("Parametric (GBM)", MCSimMethod.PARAMETRIC)
        self._method_combo.addItem("Block Bootstrap", MCSimMethod.BOOTSTRAP)
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        sim_layout.addRow("Method:", self._method_combo)

        self._nsims_spin = QSpinBox()
        self._nsims_spin.setRange(100, 10000)
        self._nsims_spin.setValue(1000)
        self._nsims_spin.setSingleStep(100)
        sim_layout.addRow("# Simulations:", self._nsims_spin)

        self._horizon_spin = QSpinBox()
        self._horizon_spin.setRange(21, 1260)
        self._horizon_spin.setValue(252)
        self._horizon_spin.setSingleStep(21)
        self._horizon_spin.setSuffix(" days")
        sim_layout.addRow("Horizon:", self._horizon_spin)

        self._block_spin = QSpinBox()
        self._block_spin.setRange(5, 60)
        self._block_spin.setValue(20)
        self._block_spin.setSuffix(" days")
        self._block_spin.setEnabled(False)  # only for bootstrap
        sim_layout.addRow("Block Size:", self._block_spin)

        sim_group.setLayout(sim_layout)
        config_layout.addWidget(sim_group)

        # Parameters
        params_group = QGroupBox("Parameters")
        params_group.setStyleSheet(self._group_style())
        params_layout = QFormLayout()
        params_layout.setSpacing(4)

        self._initial_spin = QSpinBox()
        self._initial_spin.setRange(1000, 100_000_000)
        self._initial_spin.setValue(100_000)
        self._initial_spin.setSingleStep(10000)
        self._initial_spin.setPrefix("$")
        params_layout.addRow("Initial Value:", self._initial_spin)

        self._spending_spin = QDoubleSpinBox()
        self._spending_spin.setRange(0, 0.20)
        self._spending_spin.setValue(0.04)
        self._spending_spin.setSingleStep(0.01)
        self._spending_spin.setDecimals(2)
        self._spending_spin.setSuffix(" ")
        params_layout.addRow("Spending Rate:", self._spending_spin)

        self._rfr_spin = QDoubleSpinBox()
        self._rfr_spin.setRange(0, 0.20)
        self._rfr_spin.setValue(0.04)
        self._rfr_spin.setSingleStep(0.005)
        self._rfr_spin.setDecimals(3)
        params_layout.addRow("Risk-Free Rate:", self._rfr_spin)

        params_group.setLayout(params_layout)
        config_layout.addWidget(params_group)

        # Run button
        self._run_btn = QPushButton("RUN SIMULATION")
        self._run_btn.setFixedHeight(36)
        self._run_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT_DIM};
                color: {Colors.ACCENT};
                border: 1px solid {Colors.ACCENT};
                border-radius: 4px;
                font-family: {Fonts.SANS};
                font-size: 12px;
                font-weight: bold;
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
        """)
        self._run_btn.clicked.connect(self._on_run)
        config_layout.addWidget(self._run_btn)

        self._progress = QProgressBar()
        self._progress.setFixedHeight(4)
        self._progress.setTextVisible(False)
        self._progress.setRange(0, 0)
        self._progress.hide()
        self._progress.setStyleSheet(f"""
            QProgressBar {{ background: {Colors.BG_INPUT}; border: none; }}
            QProgressBar::chunk {{ background: {Colors.ACCENT}; }}
        """)
        config_layout.addWidget(self._progress)
        config_layout.addStretch()

        splitter.addWidget(config_widget)

        # ── Results section (right) ───────────────────────────────────
        results_widget = pg.QtWidgets.QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(4)

        # Fan chart
        self._fan_plot = pg.PlotWidget(title="Wealth Projection")
        self._fan_plot.setBackground(Colors.BG_SECONDARY)
        self._fan_plot.showGrid(x=True, y=True, alpha=0.15)
        self._fan_plot.setLabel("left", "Portfolio Value ($)", color=Colors.TEXT_SECONDARY)
        self._fan_plot.setLabel("bottom", "Trading Days", color=Colors.TEXT_SECONDARY)
        results_layout.addWidget(self._fan_plot, stretch=3)

        # Shortfall label
        self._shortfall_label = QLabel("Run simulation to see results")
        self._shortfall_label.setStyleSheet(f"""
            color: {Colors.TEXT_MUTED};
            font-family: {Fonts.MONO};
            font-size: 10px;
            padding: 4px 8px;
            background: {Colors.BG_TERTIARY};
            border: 1px solid {Colors.BORDER};
            border-radius: 3px;
        """)
        self._shortfall_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        results_layout.addWidget(self._shortfall_label)

        # Metrics distribution table
        self._metrics_table = QTableWidget()
        self._metrics_table.setColumnCount(6)
        self._metrics_table.setHorizontalHeaderLabels(
            ["Metric", "P5", "P25", "Median", "P75", "P95"]
        )
        self._metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._metrics_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._metrics_table.verticalHeader().setVisible(False)
        self._metrics_table.setAlternatingRowColors(True)
        self._metrics_table.setStyleSheet(f"""
            QTableWidget {{
                background: {Colors.BG_SECONDARY};
                gridline-color: {Colors.BORDER};
                color: {Colors.TEXT_PRIMARY};
                font-family: {Fonts.MONO};
                font-size: 10px;
                border: 1px solid {Colors.BORDER};
            }}
            QTableWidget::item {{
                padding: 2px 6px;
            }}
            QHeaderView::section {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_SECONDARY};
                border: 1px solid {Colors.BORDER};
                padding: 3px;
                font-size: 9px;
                font-weight: bold;
            }}
        """)
        self._metrics_table.setMaximumHeight(180)
        setup_table_context_menu(self._metrics_table)
        results_layout.addWidget(self._metrics_table, stretch=1)

        splitter.addWidget(results_widget)
        splitter.setSizes([280, 550])

        layout.addWidget(splitter)
        self.content_layout.addLayout(layout)

    # ── Public API ────────────────────────────────────────────────────

    def get_config(self) -> MonteCarloConfig:
        """Read current UI widgets and return a MonteCarloConfig."""
        return MonteCarloConfig(
            n_sims=self._nsims_spin.value(),
            horizon_days=self._horizon_spin.value(),
            method=self._method_combo.currentData(),
            block_size=self._block_spin.value(),
            initial_value=float(self._initial_spin.value()),
            spending_rate=self._spending_spin.value(),
            risk_free_rate=self._rfr_spin.value(),
        )

    def set_result(self, result: MonteCarloResult):
        """Update fan chart, shortfall label, and metrics table."""
        self._draw_fan_chart(result)
        self._update_shortfall_label(result)
        self._update_metrics_table(result)

    def set_running(self, running: bool):
        self._run_btn.setEnabled(not running)
        self._progress.setVisible(running)

    # ── Fan Chart ─────────────────────────────────────────────────────

    def _draw_fan_chart(self, result: MonteCarloResult):
        self._fan_plot.clear()
        x = np.arange(result.equity_percentiles.shape[0])
        labels = list(result.percentile_labels)

        # Helper to get column
        def col(pct):
            return result.equity_percentiles[:, labels.index(pct)]

        # Outer band: 5th-95th percentile
        p5 = pg.PlotDataItem(x, col(5))
        p95 = pg.PlotDataItem(x, col(95))
        outer_fill = pg.FillBetweenItem(
            p5, p95, brush=pg.mkBrush(0, 212, 255, 20)
        )
        self._fan_plot.addItem(outer_fill)

        # Inner band: 25th-75th percentile
        p25 = pg.PlotDataItem(x, col(25))
        p75 = pg.PlotDataItem(x, col(75))
        inner_fill = pg.FillBetweenItem(
            p25, p75, brush=pg.mkBrush(0, 212, 255, 50)
        )
        self._fan_plot.addItem(inner_fill)

        # Boundary lines (dashed)
        dash_pen = pg.mkPen(Colors.ACCENT_LIGHT, width=1, style=Qt.DashLine)
        self._fan_plot.plot(x, col(5), pen=dash_pen)
        self._fan_plot.plot(x, col(95), pen=dash_pen)

        # Median line (solid)
        median_pen = pg.mkPen(Colors.ACCENT, width=2)
        self._fan_plot.plot(x, col(50), pen=median_pen)

        # Shortfall threshold horizontal line
        threshold = result.shortfall_threshold
        thresh_pen = pg.mkPen(Colors.LOSS, width=1, style=Qt.DashDotLine)
        self._fan_plot.addItem(
            pg.InfiniteLine(pos=threshold, angle=0, pen=thresh_pen,
                            label=f"Shortfall ${threshold:,.0f}",
                            labelOpts={"color": Colors.LOSS, "position": 0.05})
        )

    # ── Shortfall Label ───────────────────────────────────────────────

    def _update_shortfall_label(self, result: MonteCarloResult):
        prob = result.shortfall_probability
        threshold = result.shortfall_threshold
        median_term = result.metadata.get("median_terminal", 0)
        mean_term = result.metadata.get("mean_terminal", 0)

        color = Colors.PROFIT if prob < 0.10 else Colors.WARNING if prob < 0.25 else Colors.LOSS
        self._shortfall_label.setText(
            f"P(SHORTFALL): {prob:.1%}  |  "
            f"Threshold: ${threshold:,.0f}  |  "
            f"Median Terminal: ${median_term:,.0f}  |  "
            f"Mean Terminal: ${mean_term:,.0f}"
        )
        self._shortfall_label.setStyleSheet(f"""
            color: {color};
            font-family: {Fonts.MONO};
            font-size: 10px;
            padding: 4px 8px;
            background: {Colors.BG_TERTIARY};
            border: 1px solid {Colors.BORDER};
            border-radius: 3px;
            font-weight: bold;
        """)

    # ── Metrics Table ─────────────────────────────────────────────────

    def _update_metrics_table(self, result: MonteCarloResult):
        display_names = {
            "sharpe_ratio": "Sharpe Ratio",
            "annualized_return": "Ann. Return",
            "annualized_volatility": "Ann. Volatility",
            "max_drawdown": "Max Drawdown",
            "cvar_95": "CVaR 95%",
        }
        metrics = result.metrics_distributions
        rows = [k for k in display_names if k in metrics]
        self._metrics_table.setRowCount(len(rows))

        pct_indices = [5, 25, 50, 75, 95]

        for i, key in enumerate(rows):
            arr = metrics[key]
            n = len(arr)

            # Metric name
            name_item = QTableWidgetItem(display_names[key])
            name_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self._metrics_table.setItem(i, 0, name_item)

            # Percentile values
            for j, pct in enumerate(pct_indices):
                idx = int(np.clip(pct / 100 * n, 0, n - 1))
                val = arr[idx]

                # Format based on metric type
                if key in ("annualized_return", "annualized_volatility",
                           "max_drawdown", "cvar_95"):
                    text = f"{val:.2%}"
                else:
                    text = f"{val:.3f}"

                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

                # Color coding for return/sharpe
                if key in ("sharpe_ratio", "annualized_return"):
                    if val > 0:
                        item.setForeground(pg.mkColor(Colors.PROFIT))
                    elif val < 0:
                        item.setForeground(pg.mkColor(Colors.LOSS))

                self._metrics_table.setItem(i, j + 1, item)

    # ── Internal ──────────────────────────────────────────────────────

    def _on_run(self):
        self.run_requested.emit(self.get_config())

    def _on_method_changed(self, index):
        is_bootstrap = self._method_combo.currentData() == MCSimMethod.BOOTSTRAP
        self._block_spin.setEnabled(is_bootstrap)

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
