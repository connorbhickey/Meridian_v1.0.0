"""Dockable WEIGHTS panel — current vs target weight comparison."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QComboBox, QLabel,
)
import pyqtgraph as pg
import numpy as np

from portopt.gui.panels.base_panel import BasePanel
from portopt.constants import Colors, Fonts

PALETTE = [
    "#00d4ff", "#00ff88", "#ff4444", "#f0b429", "#a855f7",
    "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#6366f1",
]


class WeightsPanel(BasePanel):
    panel_id = "weights"
    panel_title = "WEIGHTS"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current = {}   # symbol -> float
        self._target = {}    # symbol -> float
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Toolbar
        toolbar = QHBoxLayout()
        self._view_combo = QComboBox()
        self._view_combo.addItems(["Table", "Bar Chart"])
        self._view_combo.setFixedWidth(100)
        self._view_combo.currentIndexChanged.connect(self._on_view_changed)
        toolbar.addWidget(QLabel("View:"))
        toolbar.addWidget(self._view_combo)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Table view
        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["Symbol", "Current", "Target", "Diff"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
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
        layout.addWidget(self._table)

        # Bar chart view
        self._chart_widget = pg.PlotWidget()
        self._chart_widget.setBackground(Colors.BG_SECONDARY)
        self._chart_widget.showGrid(x=False, y=True, alpha=0.15)
        self._chart_widget.setLabel("left", "Weight %", color=Colors.TEXT_SECONDARY)
        self._chart_widget.hide()
        layout.addWidget(self._chart_widget)

        self.content_layout.addLayout(layout)

    # ── Public API ───────────────────────────────────────────────────

    def set_weights(self, current: dict[str, float], target: dict[str, float]):
        """Set current and target weights and refresh display."""
        self._current = dict(current)
        self._target = dict(target)
        self._refresh()

    def set_target_weights(self, target: dict[str, float]):
        self._target = dict(target)
        self._refresh()

    def set_current_weights(self, current: dict[str, float]):
        self._current = dict(current)
        self._refresh()

    # ── Internal ─────────────────────────────────────────────────────

    def _refresh(self):
        symbols = sorted(set(list(self._current.keys()) + list(self._target.keys())))
        self._refresh_table(symbols)
        self._refresh_chart(symbols)

    def _refresh_table(self, symbols):
        self._table.setRowCount(len(symbols))
        for i, sym in enumerate(symbols):
            cur = self._current.get(sym, 0.0)
            tgt = self._target.get(sym, 0.0)
            diff = tgt - cur

            self._table.setItem(i, 0, QTableWidgetItem(sym))

            cur_item = QTableWidgetItem(f"{cur:.2%}")
            cur_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(i, 1, cur_item)

            tgt_item = QTableWidgetItem(f"{tgt:.2%}")
            tgt_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            tgt_item.setForeground(pg.mkColor(Colors.ACCENT))
            self._table.setItem(i, 2, tgt_item)

            diff_item = QTableWidgetItem(f"{diff:+.2%}")
            diff_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            color = Colors.PROFIT if diff > 0.001 else Colors.LOSS if diff < -0.001 else Colors.TEXT_MUTED
            diff_item.setForeground(pg.mkColor(color))
            self._table.setItem(i, 3, diff_item)

    def _refresh_chart(self, symbols):
        self._chart_widget.clear()
        if not symbols:
            return

        n = len(symbols)
        x = np.arange(n)
        width = 0.35

        # Current bars (left)
        cur_vals = [self._current.get(s, 0.0) * 100 for s in symbols]
        cur_bar = pg.BarGraphItem(
            x=x - width / 2, height=cur_vals, width=width,
            brush=pg.mkBrush(Colors.TEXT_MUTED),
            pen=pg.mkPen(Colors.BORDER),
        )
        self._chart_widget.addItem(cur_bar)

        # Target bars (right)
        tgt_vals = [self._target.get(s, 0.0) * 100 for s in symbols]
        tgt_bar = pg.BarGraphItem(
            x=x + width / 2, height=tgt_vals, width=width,
            brush=pg.mkBrush(Colors.ACCENT),
            pen=pg.mkPen(Colors.ACCENT),
        )
        self._chart_widget.addItem(tgt_bar)

        # X-axis labels
        axis = self._chart_widget.getAxis("bottom")
        ticks = [(i, symbols[i]) for i in range(n)]
        axis.setTicks([ticks])
        axis.setStyle(tickFont=pg.QtGui.QFont(Fonts.MONO, 8))

        # Legend
        legend = self._chart_widget.addLegend(
            offset=(10, 10),
            labelTextColor=Colors.TEXT_SECONDARY,
        )
        legend.addItem(cur_bar, "Current")
        legend.addItem(tgt_bar, "Target")

    def _on_view_changed(self, idx):
        if idx == 0:
            self._table.show()
            self._chart_widget.hide()
        else:
            self._table.hide()
            self._chart_widget.show()
