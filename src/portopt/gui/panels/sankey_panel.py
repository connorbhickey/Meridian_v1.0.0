"""Dockable SANKEY panel — visualize portfolio rebalancing flows.

Shows weight transitions from current → target as a Sankey-style flow diagram
using pyqtgraph rectangles and paths. Also shows a summary table of changes.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainterPath
from PySide6.QtWidgets import (
    QHBoxLayout, QHeaderView, QLabel, QPushButton, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from portopt.constants import Colors, Fonts
from portopt.gui.panels.base_panel import BasePanel
from portopt.gui.widgets.table_context_menu import setup_table_context_menu


class SankeyPanel(BasePanel):
    """Visualize portfolio rebalancing as a flow diagram."""

    panel_id = "sankey"
    panel_title = "REBALANCE FLOW"

    refresh_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_weights: dict[str, float] = {}
        self._target_weights: dict[str, float] = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Toolbar
        toolbar = QHBoxLayout()
        self._info_label = QLabel("Load current and target weights to visualize rebalancing flow")
        self._info_label.setStyleSheet(
            f"color: {Colors.TEXT_MUTED}; font-family: {Fonts.SANS}; font-size: 9px;"
        )
        toolbar.addWidget(self._info_label)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Splitter: flow chart top, table bottom
        splitter = QSplitter(Qt.Vertical)

        # Flow chart area
        self._chart = pg.PlotWidget()
        self._chart.setBackground(Colors.BG_SECONDARY)
        self._chart.hideAxis("bottom")
        self._chart.hideAxis("left")
        self._chart.setMouseEnabled(x=False, y=False)
        self._chart.setAspectLocked(False)
        splitter.addWidget(self._chart)

        # Changes table
        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.NoSelection)
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels([
            "Symbol", "Current %", "Target %", "Change %", "Action",
        ])
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
        splitter.setSizes([300, 200])

        layout.addWidget(splitter)
        self.content_layout.addLayout(layout)

    # ── Public API ───────────────────────────────────────────────

    def set_weights(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ):
        """Set current and target weights and render the flow diagram."""
        self._current_weights = current_weights or {}
        self._target_weights = target_weights or {}
        self._render_flow()
        self._render_table()

    def set_rebalance_event(self, event):
        """Set weights from a RebalanceEvent object."""
        self.set_weights(event.weights_before, event.weights_after)

    # ── Rendering ────────────────────────────────────────────────

    def _render_flow(self):
        """Draw the Sankey-style flow diagram."""
        self._chart.clear()

        current = self._current_weights
        target = self._target_weights

        # Collect all symbols
        all_symbols = sorted(set(list(current.keys()) + list(target.keys())))
        if not all_symbols:
            return

        n = len(all_symbols)

        # Layout params
        left_x = 0.0
        right_x = 10.0
        bar_width = 1.5
        total_height = n * 2.0

        # Draw left bars (current), right bars (target), and flows
        y_current = total_height
        y_target = total_height

        left_positions = {}
        right_positions = {}

        for i, sym in enumerate(all_symbols):
            cw = current.get(sym, 0.0) * 100
            tw = target.get(sym, 0.0) * 100

            # Scale bar heights proportionally (min 0.3 for visibility)
            ch = max(cw / 10, 0.3) if cw > 0.01 else 0.0
            th = max(tw / 10, 0.3) if tw > 0.01 else 0.0

            # Left bar
            if ch > 0:
                y_current -= ch + 0.3
                color_idx = i % len(Colors.CHART_PALETTE)
                color = QColor(Colors.CHART_PALETTE[color_idx])
                color.setAlpha(180)

                bar = pg.BarGraphItem(
                    x=[left_x], height=[ch], width=bar_width,
                    y0=y_current, brush=pg.mkBrush(color),
                )
                self._chart.addItem(bar)

                # Label
                text = pg.TextItem(f"{sym} {cw:.1f}%", anchor=(1.0, 0.5))
                text.setPos(left_x - 0.3, y_current + ch / 2)
                text.setColor(Colors.TEXT_SECONDARY)
                text.setFont(QFont(Fonts.MONO, 8))
                self._chart.addItem(text)

                left_positions[sym] = (left_x + bar_width, y_current + ch / 2)

            # Right bar
            if th > 0:
                y_target -= th + 0.3
                color_idx = i % len(Colors.CHART_PALETTE)
                color = QColor(Colors.CHART_PALETTE[color_idx])
                color.setAlpha(180)

                bar = pg.BarGraphItem(
                    x=[right_x], height=[th], width=bar_width,
                    y0=y_target, brush=pg.mkBrush(color),
                )
                self._chart.addItem(bar)

                text = pg.TextItem(f"{tw:.1f}% {sym}", anchor=(0.0, 0.5))
                text.setPos(right_x + bar_width + 0.3, y_target + th / 2)
                text.setColor(Colors.TEXT_SECONDARY)
                text.setFont(QFont(Fonts.MONO, 8))
                self._chart.addItem(text)

                right_positions[sym] = (right_x, y_target + th / 2)

        # Draw flow curves
        for sym in all_symbols:
            if sym in left_positions and sym in right_positions:
                lx, ly = left_positions[sym]
                rx, ry = right_positions[sym]

                cw = current.get(sym, 0.0)
                tw = target.get(sym, 0.0)
                delta = tw - cw

                # Color: green for increase, red for decrease, muted for same
                if delta > 0.005:
                    color = QColor(Colors.PROFIT)
                elif delta < -0.005:
                    color = QColor(Colors.LOSS)
                else:
                    color = QColor(Colors.TEXT_MUTED)
                color.setAlpha(100)

                # Draw a bezier curve
                mid_x = (lx + rx) / 2
                path = QPainterPath()
                path.moveTo(lx, ly)
                path.cubicTo(mid_x, ly, mid_x, ry, rx, ry)

                from PySide6.QtWidgets import QGraphicsPathItem
                curve_item = QGraphicsPathItem(path)
                pen = pg.mkPen(color=color, width=max(1, abs(delta) * 200))
                curve_item.setPen(pen)
                self._chart.addItem(curve_item)

        # Column headers
        header_y = total_height + 1.0
        header_left = pg.TextItem("CURRENT", anchor=(0.5, 0.0))
        header_left.setPos(left_x + bar_width / 2, header_y)
        header_left.setColor(Colors.ACCENT)
        header_left.setFont(QFont(Fonts.SANS, 10, QFont.Weight.Bold))
        self._chart.addItem(header_left)

        header_right = pg.TextItem("TARGET", anchor=(0.5, 0.0))
        header_right.setPos(right_x + bar_width / 2, header_y)
        header_right.setColor(Colors.ACCENT)
        header_right.setFont(QFont(Fonts.SANS, 10, QFont.Weight.Bold))
        self._chart.addItem(header_right)

        self._chart.autoRange()

        # Update info label
        total_turnover = sum(
            abs(target.get(s, 0.0) - current.get(s, 0.0))
            for s in all_symbols
        ) / 2
        n_buys = sum(1 for s in all_symbols if target.get(s, 0.0) > current.get(s, 0.0) + 0.001)
        n_sells = sum(1 for s in all_symbols if target.get(s, 0.0) < current.get(s, 0.0) - 0.001)
        self._info_label.setText(
            f"{n} symbols | Turnover: {total_turnover:.1%} | "
            f"{n_buys} buys, {n_sells} sells"
        )

    def _render_table(self):
        """Populate the changes table."""
        current = self._current_weights
        target = self._target_weights
        all_symbols = sorted(set(list(current.keys()) + list(target.keys())))

        # Sort by absolute change descending
        all_symbols.sort(
            key=lambda s: abs(target.get(s, 0.0) - current.get(s, 0.0)),
            reverse=True,
        )

        self._table.setRowCount(len(all_symbols))

        for row, sym in enumerate(all_symbols):
            cw = current.get(sym, 0.0)
            tw = target.get(sym, 0.0)
            delta = tw - cw

            # Symbol
            item = QTableWidgetItem(sym)
            item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, 0, item)

            # Current %
            item = QTableWidgetItem(f"{cw:.2%}")
            item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, 1, item)

            # Target %
            item = QTableWidgetItem(f"{tw:.2%}")
            item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, 2, item)

            # Change %
            item = QTableWidgetItem(f"{delta:+.2%}")
            item.setTextAlignment(Qt.AlignCenter)
            if delta > 0.001:
                item.setForeground(pg.mkColor(Colors.PROFIT))
            elif delta < -0.001:
                item.setForeground(pg.mkColor(Colors.LOSS))
            else:
                item.setForeground(pg.mkColor(Colors.TEXT_MUTED))
            self._table.setItem(row, 3, item)

            # Action
            if delta > 0.001:
                action = "BUY"
                color = Colors.PROFIT
            elif delta < -0.001:
                action = "SELL"
                color = Colors.LOSS
            elif tw < 0.001 and cw < 0.001:
                action = "—"
                color = Colors.TEXT_MUTED
            else:
                action = "HOLD"
                color = Colors.TEXT_MUTED

            item = QTableWidgetItem(action)
            item.setTextAlignment(Qt.AlignCenter)
            item.setForeground(pg.mkColor(color))
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            self._table.setItem(row, 4, item)
