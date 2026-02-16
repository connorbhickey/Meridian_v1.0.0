"""Dockable ATTRIBUTION panel — Brinson attribution and factor decomposition charts."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QComboBox, QLabel
import pyqtgraph as pg
import numpy as np

from portopt.gui.panels.base_panel import BasePanel
from portopt.constants import Colors, Fonts

PALETTE = [
    "#00d4ff", "#00ff88", "#ff4444", "#f0b429", "#a855f7",
    "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#6366f1",
]


class AttributionPanel(BasePanel):
    panel_id = "attribution"
    panel_title = "ATTRIBUTION"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        lbl = QLabel("View:")
        lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 10px;")
        toolbar.addWidget(lbl)

        self._view_combo = QComboBox()
        self._view_combo.addItems(["Brinson", "Factor", "Contribution"])
        self._view_combo.setFixedWidth(120)
        self._view_combo.currentIndexChanged.connect(self._on_view_changed)
        toolbar.addWidget(self._view_combo)
        toolbar.addStretch()

        layout.addLayout(toolbar)

        # Chart
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground(Colors.BG_SECONDARY)
        self._plot_widget.showGrid(x=False, y=True, alpha=0.15)
        layout.addWidget(self._plot_widget)

        self.content_layout.addLayout(layout)

        # Data storage
        self._brinson_data = None
        self._factor_data = None
        self._contribution_data = None

    # ── Public API ───────────────────────────────────────────────────

    def set_brinson(self, sectors: list[str], allocation: list[float],
                    selection: list[float], interaction: list[float]):
        """Set Brinson attribution data (per sector)."""
        self._brinson_data = {
            "sectors": sectors,
            "allocation": allocation,
            "selection": selection,
            "interaction": interaction,
        }
        if self._view_combo.currentIndex() == 0:
            self._draw_brinson()

    def set_factor_attribution(self, factors: list[str], contributions: list[float],
                               specific_return: float):
        """Set factor attribution data."""
        self._factor_data = {
            "factors": factors,
            "contributions": contributions,
            "specific_return": specific_return,
        }
        if self._view_combo.currentIndex() == 1:
            self._draw_factor()

    def set_contribution(self, symbols: list[str], contributions: list[float]):
        """Set asset contribution data (total contribution of each asset)."""
        self._contribution_data = {
            "symbols": symbols,
            "contributions": contributions,
        }
        if self._view_combo.currentIndex() == 2:
            self._draw_contribution()

    # ── Internal ─────────────────────────────────────────────────────

    def _on_view_changed(self, idx):
        if idx == 0:
            self._draw_brinson()
        elif idx == 1:
            self._draw_factor()
        else:
            self._draw_contribution()

    def _draw_brinson(self):
        self._plot_widget.clear()
        if not self._brinson_data:
            return

        data = self._brinson_data
        sectors = data["sectors"]
        n = len(sectors)
        x = np.arange(n)
        width = 0.25

        # Allocation bars
        alloc_bar = pg.BarGraphItem(
            x=x - width, height=[v * 100 for v in data["allocation"]], width=width,
            brush=pg.mkBrush(Colors.ACCENT), pen=pg.mkPen(Colors.ACCENT),
        )
        self._plot_widget.addItem(alloc_bar)

        # Selection bars
        sel_bar = pg.BarGraphItem(
            x=x, height=[v * 100 for v in data["selection"]], width=width,
            brush=pg.mkBrush(Colors.PROFIT), pen=pg.mkPen(Colors.PROFIT),
        )
        self._plot_widget.addItem(sel_bar)

        # Interaction bars
        inter_bar = pg.BarGraphItem(
            x=x + width, height=[v * 100 for v in data["interaction"]], width=width,
            brush=pg.mkBrush(Colors.WARNING), pen=pg.mkPen(Colors.WARNING),
        )
        self._plot_widget.addItem(inter_bar)

        # X-axis
        axis = self._plot_widget.getAxis("bottom")
        axis.setTicks([[(i, s) for i, s in enumerate(sectors)]])
        axis.setStyle(tickFont=pg.QtGui.QFont(Fonts.MONO, 8))
        self._plot_widget.setLabel("left", "Return (%)", color=Colors.TEXT_SECONDARY)

        # Legend
        legend = self._plot_widget.addLegend(offset=(10, 10), labelTextColor=Colors.TEXT_SECONDARY)
        legend.addItem(alloc_bar, "Allocation")
        legend.addItem(sel_bar, "Selection")
        legend.addItem(inter_bar, "Interaction")

    def _draw_factor(self):
        self._plot_widget.clear()
        if not self._factor_data:
            return

        data = self._factor_data
        labels = data["factors"] + ["Specific"]
        values = data["contributions"] + [data["specific_return"]]
        n = len(labels)
        x = np.arange(n)

        colors = [
            Colors.PROFIT if v >= 0 else Colors.LOSS for v in values
        ]
        bars = pg.BarGraphItem(
            x=x, height=[v * 100 for v in values], width=0.6,
            brushes=[pg.mkBrush(c) for c in colors],
            pens=[pg.mkPen(c) for c in colors],
        )
        self._plot_widget.addItem(bars)

        axis = self._plot_widget.getAxis("bottom")
        axis.setTicks([[(i, l) for i, l in enumerate(labels)]])
        axis.setStyle(tickFont=pg.QtGui.QFont(Fonts.MONO, 8))
        self._plot_widget.setLabel("left", "Contribution (%)", color=Colors.TEXT_SECONDARY)

    def _draw_contribution(self):
        self._plot_widget.clear()
        if not self._contribution_data:
            return

        data = self._contribution_data
        symbols = data["symbols"]
        values = data["contributions"]
        n = len(symbols)
        x = np.arange(n)

        colors = [
            PALETTE[i % len(PALETTE)] for i in range(n)
        ]
        bars = pg.BarGraphItem(
            x=x, height=[v * 100 for v in values], width=0.6,
            brushes=[pg.mkBrush(c) for c in colors],
            pens=[pg.mkPen(c) for c in colors],
        )
        self._plot_widget.addItem(bars)

        axis = self._plot_widget.getAxis("bottom")
        axis.setTicks([[(i, s) for i, s in enumerate(symbols)]])
        axis.setStyle(tickFont=pg.QtGui.QFont(Fonts.MONO, 8))
        self._plot_widget.setLabel("left", "Contribution (%)", color=Colors.TEXT_SECONDARY)
