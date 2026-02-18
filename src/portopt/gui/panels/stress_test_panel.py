"""Dockable STRESS TEST panel — scenario config, impact table, and bar chart."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox,
    QDoubleSpinBox, QLineEdit, QPushButton, QProgressBar,
    QGroupBox, QCheckBox, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView, QScrollArea, QWidget,
)
import pyqtgraph as pg
import numpy as np

from portopt.gui.panels.base_panel import BasePanel
from portopt.constants import Colors, Fonts
from portopt.engine.stress import HISTORICAL_SCENARIOS, StressResult


def _pg_color(hex_color: str) -> QColor:
    """Convert hex string to QColor."""
    return QColor(hex_color)


class StressTestPanel(BasePanel):
    panel_id = "stress_test"
    panel_title = "STRESS TEST"

    run_requested = Signal(object)  # dict with scenarios config

    def __init__(self, parent=None):
        super().__init__(parent)
        self._weights: dict[str, float] | None = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        splitter = QSplitter(Qt.Horizontal)

        # ── Config section (left, 280px) ──────────────────────────────
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.setSpacing(4)

        # Named scenarios group
        scenarios_group = QGroupBox("Historical Scenarios")
        scenarios_group.setStyleSheet(self._group_style())
        scenarios_layout = QVBoxLayout()
        scenarios_layout.setSpacing(2)

        self._scenario_checks: dict[str, QCheckBox] = {}
        for name, scenario in HISTORICAL_SCENARIOS.items():
            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.setToolTip(scenario.description)
            scenarios_layout.addWidget(cb)
            self._scenario_checks[name] = cb

        scenarios_group.setLayout(scenarios_layout)
        config_layout.addWidget(scenarios_group)

        # Custom scenario group
        custom_group = QGroupBox("Custom Scenario")
        custom_group.setStyleSheet(self._group_style())
        custom_layout = QFormLayout()
        custom_layout.setSpacing(4)

        self._custom_name = QLineEdit()
        self._custom_name.setPlaceholderText("e.g. 'Recession 2025'")
        custom_layout.addRow("Name:", self._custom_name)

        self._equity_shock = QDoubleSpinBox()
        self._equity_shock.setRange(-100, 50)
        self._equity_shock.setValue(-20.0)
        self._equity_shock.setSuffix("%")
        self._equity_shock.setDecimals(1)
        custom_layout.addRow("Equity Shock:", self._equity_shock)

        self._bond_shock = QDoubleSpinBox()
        self._bond_shock.setRange(-100, 50)
        self._bond_shock.setValue(5.0)
        self._bond_shock.setSuffix("%")
        self._bond_shock.setDecimals(1)
        custom_layout.addRow("Bond Shock:", self._bond_shock)

        self._gold_shock = QDoubleSpinBox()
        self._gold_shock.setRange(-100, 50)
        self._gold_shock.setValue(0.0)
        self._gold_shock.setSuffix("%")
        self._gold_shock.setDecimals(1)
        custom_layout.addRow("Gold Shock:", self._gold_shock)

        custom_group.setLayout(custom_layout)
        config_layout.addWidget(custom_group)

        # Run button
        self._run_btn = QPushButton("RUN STRESS TEST")
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
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(4)

        # Impact table
        self._impact_table = QTableWidget(0, 4)
        self._impact_table.setHorizontalHeaderLabels([
            "Scenario", "Impact %", "Stressed Value", "Description",
        ])
        self._impact_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._impact_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._impact_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._impact_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self._impact_table.verticalHeader().hide()
        self._impact_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._impact_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._impact_table.setStyleSheet(f"""
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
                padding: 4px;
                font-family: {Fonts.SANS};
                font-size: 9px;
                font-weight: bold;
            }}
            QTableWidget::item:selected {{
                background: {Colors.ACCENT_DIM};
            }}
        """)
        results_layout.addWidget(self._impact_table, stretch=2)

        # Bar chart
        self._bar_plot = pg.PlotWidget(title="Scenario Impact")
        self._bar_plot.setBackground(Colors.BG_SECONDARY)
        self._bar_plot.showGrid(x=False, y=True, alpha=0.15)
        self._bar_plot.setLabel("bottom", "Impact (%)", color=Colors.TEXT_SECONDARY)
        self._bar_plot.getAxis("left").setStyle(tickLength=0)
        results_layout.addWidget(self._bar_plot, stretch=1)

        splitter.addWidget(results_widget)
        splitter.setSizes([280, 520])

        layout.addWidget(splitter)
        self.content_layout.addLayout(layout)

    # ── Public API ────────────────────────────────────────────────────

    def set_weights(self, weights: dict[str, float]):
        """Set current portfolio weights (from optimization)."""
        self._weights = weights

    def set_running(self, running: bool):
        self._run_btn.setEnabled(not running)
        self._progress.setVisible(running)

    def set_results(self, results: list[StressResult]):
        """Populate the impact table and bar chart with stress test results."""
        # Table
        self._impact_table.setRowCount(len(results))
        for i, r in enumerate(results):
            name_item = QTableWidgetItem(r.scenario.name)
            impact_item = QTableWidgetItem(f"{r.portfolio_impact:+.2%}")
            value_item = QTableWidgetItem(f"${r.stressed_value:,.0f}")
            desc_item = QTableWidgetItem(r.scenario.description)

            # Color impact
            color = Colors.LOSS if r.portfolio_impact < 0 else Colors.PROFIT
            impact_item.setForeground(_pg_color(color))
            value_item.setForeground(_pg_color(color))

            impact_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

            self._impact_table.setItem(i, 0, name_item)
            self._impact_table.setItem(i, 1, impact_item)
            self._impact_table.setItem(i, 2, value_item)
            self._impact_table.setItem(i, 3, desc_item)

        # Bar chart — horizontal bars sorted by severity
        self._bar_plot.clear()
        if not results:
            return

        names = [r.scenario.name for r in results]
        impacts = [r.portfolio_impact * 100 for r in results]
        y_pos = np.arange(len(results))

        colors = [
            _pg_color(Colors.LOSS) if imp < 0 else _pg_color(Colors.PROFIT)
            for imp in impacts
        ]
        brushes = [pg.mkBrush(c) for c in colors]

        bar = pg.BarGraphItem(
            x0=0, y=y_pos, width=impacts, height=0.6,
            brushes=brushes, pens=[pg.mkPen(None)] * len(results),
        )
        self._bar_plot.addItem(bar)

        # Y-axis labels
        left_axis = self._bar_plot.getAxis("left")
        left_axis.setTicks([list(zip(y_pos, names))])

        # Add zero line
        zero_line = pg.InfiniteLine(
            pos=0, angle=90,
            pen=pg.mkPen(Colors.TEXT_MUTED, width=1, style=Qt.DashLine),
        )
        self._bar_plot.addItem(zero_line)

    # ── Internal ──────────────────────────────────────────────────────

    def _on_run(self):
        """Emit run_requested with current configuration."""
        selected_scenarios = [
            name for name, cb in self._scenario_checks.items() if cb.isChecked()
        ]

        custom = None
        custom_name = self._custom_name.text().strip()
        if custom_name:
            custom = {
                "name": custom_name,
                "shocks": {
                    "equity": self._equity_shock.value() / 100,
                    "bond": self._bond_shock.value() / 100,
                    "gold": self._gold_shock.value() / 100,
                },
            }

        self.run_requested.emit({
            "selected_scenarios": selected_scenarios,
            "custom": custom,
            "weights": self._weights,
        })

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
