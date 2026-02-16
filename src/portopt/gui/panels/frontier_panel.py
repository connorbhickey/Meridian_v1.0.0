"""Dockable EFFICIENT FRONTIER panel — scatter plot of risk vs return."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout
import pyqtgraph as pg
import numpy as np

from portopt.gui.panels.base_panel import BasePanel
from portopt.constants import Colors, Fonts

PALETTE = [
    "#00d4ff", "#00ff88", "#ff4444", "#f0b429", "#a855f7",
    "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#6366f1",
]


class FrontierPanel(BasePanel):
    panel_id = "frontier"
    panel_title = "EFFICIENT FRONTIER"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground(Colors.BG_SECONDARY)
        self._plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self._plot_widget.setLabel("left", "Expected Return", color=Colors.TEXT_SECONDARY)
        self._plot_widget.setLabel("bottom", "Risk (Std Dev)", color=Colors.TEXT_SECONDARY)

        # Crosshair
        self._vline = pg.InfiniteLine(angle=90, pen=pg.mkPen(Colors.TEXT_MUTED, width=1, style=Qt.DashLine))
        self._hline = pg.InfiniteLine(angle=0, pen=pg.mkPen(Colors.TEXT_MUTED, width=1, style=Qt.DashLine))
        self._plot_widget.addItem(self._vline, ignoreBounds=True)
        self._plot_widget.addItem(self._hline, ignoreBounds=True)
        self._proxy = pg.SignalProxy(
            self._plot_widget.scene().sigMouseMoved,
            rateLimit=60, slot=self._on_mouse_moved,
        )

        self._legend = self._plot_widget.addLegend(
            offset=(10, 10),
            labelTextColor=Colors.TEXT_SECONDARY,
            labelTextSize="9pt",
        )

        layout.addWidget(self._plot_widget)
        self.content_layout.addLayout(layout)

    # ── Public API ───────────────────────────────────────────────────

    def set_frontier(self, risks: np.ndarray, returns: np.ndarray):
        """Plot the efficient frontier curve."""
        self._plot_widget.clear()
        self._plot_widget.addItem(self._vline)
        self._plot_widget.addItem(self._hline)

        pen = pg.mkPen(Colors.ACCENT, width=2)
        self._plot_widget.plot(
            risks * 100, returns * 100,
            pen=pen, name="Efficient Frontier",
        )

    def set_individual_assets(self, symbols: list[str], risks: np.ndarray, returns: np.ndarray):
        """Plot individual asset positions as scatter dots."""
        for i, sym in enumerate(symbols):
            color = PALETTE[i % len(PALETTE)]
            scatter = pg.ScatterPlotItem(
                [risks[i] * 100], [returns[i] * 100],
                pen=pg.mkPen(color, width=1),
                brush=pg.mkBrush(color),
                size=10,
                symbol="o",
                name=sym,
            )
            self._plot_widget.addItem(scatter)

    def set_optimal_portfolio(self, risk: float, ret: float, label: str = "Optimal"):
        """Plot the optimal portfolio as a star marker."""
        scatter = pg.ScatterPlotItem(
            [risk * 100], [ret * 100],
            pen=pg.mkPen("#ffffff", width=2),
            brush=pg.mkBrush(Colors.WARNING),
            size=16,
            symbol="star",
            name=label,
        )
        self._plot_widget.addItem(scatter)

    def clear_plot(self):
        self._plot_widget.clear()
        self._plot_widget.addItem(self._vline)
        self._plot_widget.addItem(self._hline)

    # ── Internal ─────────────────────────────────────────────────────

    def _on_mouse_moved(self, evt):
        pos = evt[0]
        if self._plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self._plot_widget.plotItem.vb.mapSceneToView(pos)
            self._vline.setPos(mouse_point.x())
            self._hline.setPos(mouse_point.y())
