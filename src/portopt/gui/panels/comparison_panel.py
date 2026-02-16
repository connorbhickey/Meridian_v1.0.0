"""Dockable COMPARISON panel — save & compare multiple optimization strategies.

B3 feature: stores up to 5 OptimizationResult snapshots and displays
a grouped bar chart (weights) and metrics comparison table side by side.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QHBoxLayout, QHeaderView, QLabel, QPushButton, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from portopt.constants import Colors, Fonts
from portopt.data.models import OptimizationResult
from portopt.gui.panels.base_panel import BasePanel
from portopt.gui.widgets.table_context_menu import setup_table_context_menu

MAX_SNAPSHOTS = 5

METRIC_KEYS = [
    ("expected_return", "Expected Return", "{:.2%}"),
    ("volatility", "Volatility", "{:.2%}"),
    ("sharpe_ratio", "Sharpe Ratio", "{:.3f}"),
    ("max_drawdown", "Max Drawdown", "{:.2%}"),
]


class ComparisonPanel(BasePanel):
    panel_id = "comparison"
    panel_title = "COMPARE"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._snapshots: OrderedDict[str, OptimizationResult] = OrderedDict()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Toolbar
        toolbar = QHBoxLayout()
        self._count_label = QLabel("0 / 5 strategies saved")
        self._count_label.setStyleSheet(
            f"color: {Colors.TEXT_MUTED}; font-family: {Fonts.SANS}; font-size: 9px;"
        )
        toolbar.addWidget(self._count_label)
        toolbar.addStretch()

        clear_btn = QPushButton("CLEAR ALL")
        clear_btn.setFixedHeight(22)
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_INPUT};
                color: {Colors.LOSS};
                border: 1px solid {Colors.BORDER};
                border-radius: 3px;
                font-family: {Fonts.SANS};
                font-size: 9px;
                font-weight: bold;
                padding: 0 10px;
            }}
            QPushButton:hover {{
                background: {Colors.LOSS_DIM};
            }}
        """)
        clear_btn.clicked.connect(self.clear_snapshots)
        toolbar.addWidget(clear_btn)
        layout.addLayout(toolbar)

        # Splitter: chart top, table bottom
        splitter = QSplitter(Qt.Vertical)

        # ── Grouped bar chart ─────────────────────────────────────
        self._chart = pg.PlotWidget(title="Weight Comparison")
        self._chart.setBackground(Colors.BG_SECONDARY)
        self._chart.showGrid(x=False, y=True, alpha=0.15)
        self._chart.setLabel("left", "Weight %", color=Colors.TEXT_SECONDARY)
        splitter.addWidget(self._chart)

        # ── Metrics table ─────────────────────────────────────────
        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.NoSelection)
        self._table.verticalHeader().setVisible(True)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(f"""
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
        setup_table_context_menu(self._table)
        splitter.addWidget(self._table)
        splitter.setSizes([250, 150])

        layout.addWidget(splitter)
        self.content_layout.addLayout(layout)

    # ── Public API ───────────────────────────────────────────────

    def add_snapshot(self, result: OptimizationResult, name: str | None = None):
        """Save an optimization result for comparison.

        If the maximum number of snapshots is exceeded, the oldest is removed.
        """
        label = name or result.method or "Strategy"
        # Make label unique if duplicate
        base = label
        counter = 2
        while label in self._snapshots:
            label = f"{base} #{counter}"
            counter += 1

        self._snapshots[label] = result

        # Enforce max
        while len(self._snapshots) > MAX_SNAPSHOTS:
            self._snapshots.popitem(last=False)

        self._count_label.setText(f"{len(self._snapshots)} / {MAX_SNAPSHOTS} strategies saved")
        self._refresh_chart()
        self._refresh_table()

    def clear_snapshots(self):
        """Remove all saved snapshots."""
        self._snapshots.clear()
        self._count_label.setText("0 / 5 strategies saved")
        self._chart.clear()
        self._table.clear()
        self._table.setRowCount(0)
        self._table.setColumnCount(0)

    # ── Chart ────────────────────────────────────────────────────

    def _refresh_chart(self):
        """Redraw the grouped bar chart."""
        self._chart.clear()

        if not self._snapshots:
            return

        # Collect all unique symbols across all snapshots
        all_symbols: list[str] = []
        seen = set()
        for result in self._snapshots.values():
            for sym in result.weights:
                if sym not in seen:
                    all_symbols.append(sym)
                    seen.add(sym)

        n_assets = len(all_symbols)
        n_strategies = len(self._snapshots)
        if n_assets == 0:
            return

        bar_width = 0.8 / n_strategies
        x_base = np.arange(n_assets)

        for i, (label, result) in enumerate(self._snapshots.items()):
            weights = [result.weights.get(sym, 0.0) * 100 for sym in all_symbols]
            x_pos = x_base + i * bar_width - (n_strategies - 1) * bar_width / 2

            color = Colors.CHART_PALETTE[i % len(Colors.CHART_PALETTE)]
            bar = pg.BarGraphItem(
                x=x_pos, height=weights, width=bar_width * 0.9,
                brush=pg.mkBrush(color),
                name=label,
            )
            self._chart.addItem(bar)

        # X-axis ticks
        x_axis = self._chart.getAxis("bottom")
        ticks = [(i, sym) for i, sym in enumerate(all_symbols)]
        x_axis.setTicks([ticks])

    # ── Table ────────────────────────────────────────────────────

    def _refresh_table(self):
        """Redraw the metrics comparison table."""
        if not self._snapshots:
            self._table.clear()
            return

        strategies = list(self._snapshots.keys())
        n_cols = len(strategies)
        n_rows = len(METRIC_KEYS)

        self._table.setRowCount(n_rows)
        self._table.setColumnCount(n_cols)
        self._table.setHorizontalHeaderLabels(strategies)
        self._table.setVerticalHeaderLabels([mk[1] for mk in METRIC_KEYS])

        for col, label in enumerate(strategies):
            result = self._snapshots[label]
            color = Colors.CHART_PALETTE[col % len(Colors.CHART_PALETTE)]

            for row, (attr, _display_name, fmt) in enumerate(METRIC_KEYS):
                value = getattr(result, attr, None)
                if value is None:
                    text = "—"
                else:
                    try:
                        text = fmt.format(value)
                    except (ValueError, TypeError):
                        text = str(value)

                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                item.setForeground(pg.mkColor(color))
                self._table.setItem(row, col, item)
