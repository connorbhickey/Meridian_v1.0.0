"""Dockable BACKTEST panel — configuration, equity curve, and benchmark comparison."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox,
    QSpinBox, QDoubleSpinBox, QPushButton, QProgressBar,
    QGroupBox, QCheckBox, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView, QLabel,
)
import pyqtgraph as pg
import numpy as np

from portopt.gui.panels.base_panel import BasePanel
from portopt.constants import Colors, Fonts, RebalanceFreq, CostModel


class BacktestPanel(BasePanel):
    panel_id = "backtest"
    panel_title = "BACKTEST"

    run_requested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        splitter = QSplitter(Qt.Horizontal)

        # ── Config section (left) ────────────────────────────────────
        config_widget = pg.QtWidgets.QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.setSpacing(4)

        # Rebalancing
        reb_group = QGroupBox("Rebalancing")
        reb_group.setStyleSheet(self._group_style())
        reb_layout = QFormLayout()
        reb_layout.setSpacing(4)

        self._freq_combo = QComboBox()
        for f in RebalanceFreq:
            self._freq_combo.addItem(f.name.title(), f)
        self._freq_combo.setCurrentIndex(2)  # Monthly default
        reb_layout.addRow("Frequency:", self._freq_combo)

        self._drift_spin = QDoubleSpinBox()
        self._drift_spin.setRange(0, 0.50)
        self._drift_spin.setSingleStep(0.01)
        self._drift_spin.setValue(0.05)
        self._drift_spin.setDecimals(2)
        self._drift_spin.setSuffix(" ")
        reb_layout.addRow("Drift Threshold:", self._drift_spin)

        reb_group.setLayout(reb_layout)
        config_layout.addWidget(reb_group)

        # Costs
        cost_group = QGroupBox("Transaction Costs")
        cost_group.setStyleSheet(self._group_style())
        cost_layout = QFormLayout()
        cost_layout.setSpacing(4)

        self._cost_combo = QComboBox()
        for cm in CostModel:
            self._cost_combo.addItem(cm.name.title(), cm)
        self._cost_combo.setCurrentIndex(2)  # Proportional default
        cost_layout.addRow("Model:", self._cost_combo)

        self._cost_rate_spin = QDoubleSpinBox()
        self._cost_rate_spin.setRange(0, 0.05)
        self._cost_rate_spin.setSingleStep(0.0001)
        self._cost_rate_spin.setValue(0.001)
        self._cost_rate_spin.setDecimals(4)
        cost_layout.addRow("Rate:", self._cost_rate_spin)

        cost_group.setLayout(cost_layout)
        config_layout.addWidget(cost_group)

        # Walk-forward
        wf_group = QGroupBox("Walk-Forward")
        wf_group.setStyleSheet(self._group_style())
        wf_layout = QFormLayout()
        wf_layout.setSpacing(4)

        self._wf_check = QCheckBox()
        self._wf_check.setChecked(False)
        wf_layout.addRow("Enabled:", self._wf_check)

        self._train_spin = QSpinBox()
        self._train_spin.setRange(60, 2520)
        self._train_spin.setValue(504)
        self._train_spin.setSuffix(" days")
        wf_layout.addRow("Train Window:", self._train_spin)

        self._test_spin = QSpinBox()
        self._test_spin.setRange(20, 504)
        self._test_spin.setValue(126)
        self._test_spin.setSuffix(" days")
        wf_layout.addRow("Test Window:", self._test_spin)

        self._anchored_check = QCheckBox()
        self._anchored_check.setChecked(False)
        wf_layout.addRow("Anchored:", self._anchored_check)

        wf_group.setLayout(wf_layout)
        config_layout.addWidget(wf_group)

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

        self._lookback_spin = QSpinBox()
        self._lookback_spin.setRange(60, 2520)
        self._lookback_spin.setValue(252)
        self._lookback_spin.setSuffix(" days")
        params_layout.addRow("Lookback:", self._lookback_spin)

        params_group.setLayout(params_layout)
        config_layout.addWidget(params_group)

        # Benchmark
        bench_group = QGroupBox("Benchmark")
        bench_group.setStyleSheet(self._group_style())
        bench_layout = QFormLayout()
        bench_layout.setSpacing(4)

        self._benchmark_combo = QComboBox()
        self._benchmark_combo.addItems([
            "None", "SPY", "QQQ", "IWM", "AGG",
            "60/40 (SPY/AGG)", "Equal-Weight",
        ])
        bench_layout.addRow("Compare:", self._benchmark_combo)

        bench_group.setLayout(bench_layout)
        config_layout.addWidget(bench_group)

        # Run button
        self._run_btn = QPushButton("RUN BACKTEST")
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

        # ── Chart section (right) ────────────────────────────────────
        chart_widget = pg.QtWidgets.QWidget()
        chart_layout = QVBoxLayout(chart_widget)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.setSpacing(2)

        # Equity curve
        self._equity_plot = pg.PlotWidget(title="Equity Curve")
        self._equity_plot.setBackground(Colors.BG_SECONDARY)
        self._equity_plot.showGrid(x=True, y=True, alpha=0.15)
        self._equity_plot.setLabel("left", "Portfolio Value ($)", color=Colors.TEXT_SECONDARY)
        chart_layout.addWidget(self._equity_plot, stretch=3)

        # Drawdown subplot
        self._dd_plot = pg.PlotWidget(title="Drawdown")
        self._dd_plot.setBackground(Colors.BG_SECONDARY)
        self._dd_plot.showGrid(x=True, y=True, alpha=0.15)
        self._dd_plot.setLabel("left", "Drawdown %", color=Colors.TEXT_SECONDARY)
        chart_layout.addWidget(self._dd_plot, stretch=1)

        # Link x-axes
        self._dd_plot.setXLink(self._equity_plot)

        # Benchmark comparison table
        self._bench_label = QLabel("BENCHMARK COMPARISON")
        self._bench_label.setStyleSheet(f"""
            color: {Colors.TEXT_MUTED}; font-family: {Fonts.SANS};
            font-size: 10px; font-weight: bold; padding: 4px 0 2px 4px;
        """)
        self._bench_label.hide()
        chart_layout.addWidget(self._bench_label)

        self._bench_table = QTableWidget(8, 3)
        self._bench_table.setHorizontalHeaderLabels(["Portfolio", "Benchmark", "Diff"])
        self._bench_table.setVerticalHeaderLabels([
            "Return", "Volatility", "Sharpe", "Max DD",
            "Alpha", "Beta", "Track. Error", "Info Ratio",
        ])
        self._bench_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._bench_table.verticalHeader().setDefaultSectionSize(22)
        self._bench_table.setFixedHeight(210)
        self._bench_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._bench_table.setStyleSheet(f"""
            QTableWidget {{
                background: {Colors.BG_SECONDARY};
                color: {Colors.TEXT_PRIMARY};
                gridline-color: {Colors.BORDER};
                font-family: {Fonts.MONO};
                font-size: 10px;
                border: 1px solid {Colors.BORDER};
            }}
            QHeaderView::section {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_MUTED};
                border: 1px solid {Colors.BORDER};
                padding: 2px 4px;
                font-family: {Fonts.SANS};
                font-size: 9px;
                font-weight: bold;
            }}
        """)
        self._bench_table.hide()
        chart_layout.addWidget(self._bench_table)

        splitter.addWidget(chart_widget)
        splitter.setSizes([250, 550])

        layout.addWidget(splitter)
        self.content_layout.addLayout(layout)

    # ── Public API ───────────────────────────────────────────────────

    def get_config(self) -> dict:
        bench = self._benchmark_combo.currentText()
        return {
            "rebalance_freq": self._freq_combo.currentData(),
            "drift_threshold": self._drift_spin.value(),
            "cost_model": self._cost_combo.currentData(),
            "cost_rate": self._cost_rate_spin.value(),
            "walk_forward_enabled": self._wf_check.isChecked(),
            "train_window": self._train_spin.value(),
            "test_window": self._test_spin.value(),
            "anchored": self._anchored_check.isChecked(),
            "initial_value": self._initial_spin.value(),
            "lookback": self._lookback_spin.value(),
            "benchmark": bench if bench != "None" else None,
        }

    def set_equity_curve(self, dates_epoch, values):
        """Plot equity curve from epoch timestamps and portfolio values."""
        self._equity_plot.clear()
        pen = pg.mkPen(Colors.ACCENT, width=2)
        self._equity_plot.plot(dates_epoch, values, pen=pen)

    def set_drawdown(self, dates_epoch, drawdowns):
        """Plot drawdown curve (values should be negative percentages)."""
        self._dd_plot.clear()
        fill = pg.FillBetweenItem(
            pg.PlotDataItem(dates_epoch, np.zeros_like(drawdowns)),
            pg.PlotDataItem(dates_epoch, drawdowns * 100),
            brush=pg.mkBrush(Colors.LOSS_DIM),
        )
        self._dd_plot.addItem(fill)
        pen = pg.mkPen(Colors.LOSS, width=1)
        self._dd_plot.plot(dates_epoch, drawdowns * 100, pen=pen)

    def set_benchmark(self, dates_epoch, values, label="Benchmark"):
        """Overlay a benchmark equity curve."""
        pen = pg.mkPen(Colors.TEXT_MUTED, width=1, style=Qt.DashLine)
        self._equity_plot.plot(dates_epoch, values, pen=pen, name=label)

    def set_benchmark_metrics(self, port_metrics: dict, bench_metrics: dict):
        """Populate the benchmark comparison table with color-coded diffs."""
        self._bench_label.show()
        self._bench_table.show()

        # Metric keys in order matching table rows
        rows = [
            ("total_return", "{:.2%}", True),       # higher is better
            ("annual_volatility", "{:.2%}", False),  # lower is better
            ("sharpe_ratio", "{:.3f}", True),
            ("max_drawdown", "{:.2%}", False),       # less negative is better
            ("alpha", "{:.4f}", True),
            ("beta", "{:.3f}", None),                # neutral
            ("tracking_error", "{:.4f}", None),
            ("information_ratio", "{:.3f}", True),
        ]

        for i, (key, fmt, higher_better) in enumerate(rows):
            pv = port_metrics.get(key, 0.0)
            bv = bench_metrics.get(key, 0.0)
            diff = pv - bv

            p_item = QTableWidgetItem(fmt.format(pv))
            b_item = QTableWidgetItem(fmt.format(bv))
            d_item = QTableWidgetItem(fmt.format(diff))

            p_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            b_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            d_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

            # Color the diff column
            if higher_better is not None and abs(diff) > 1e-6:
                is_good = (diff > 0) == higher_better
                color = Colors.PROFIT if is_good else Colors.LOSS
                from PySide6.QtGui import QColor
                d_item.setForeground(QColor(color))

            self._bench_table.setItem(i, 0, p_item)
            self._bench_table.setItem(i, 1, b_item)
            self._bench_table.setItem(i, 2, d_item)

    def set_running(self, running: bool):
        self._run_btn.setEnabled(not running)
        self._progress.setVisible(running)

    # ── Internal ─────────────────────────────────────────────────────

    def _on_run(self):
        self.run_requested.emit(self.get_config())

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
