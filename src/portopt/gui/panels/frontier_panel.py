"""Dockable EFFICIENT FRONTIER panel — interactive scatter plot of risk vs return."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout
import pyqtgraph as pg
import numpy as np

from portopt.gui.panels.base_panel import BasePanel
from portopt.constants import Colors, Fonts


class FrontierPanel(BasePanel):
    panel_id = "frontier"
    panel_title = "EFFICIENT FRONTIER"

    frontier_point_clicked = Signal(dict)  # {"risk", "return", "weights", "sharpe"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frontier_points: list[dict] = []
        self._frontier_scatter: pg.ScatterPlotItem | None = None
        self._weights_label: pg.TextItem | None = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground(Colors.BG_SECONDARY)
        self._plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self._plot_widget.setLabel("left", "Expected Return (%)", color=Colors.TEXT_SECONDARY)
        self._plot_widget.setLabel("bottom", "Risk / Std Dev (%)", color=Colors.TEXT_SECONDARY)

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

    def set_frontier(self, risks: np.ndarray, returns: np.ndarray,
                     weights_list: list[dict] | None = None):
        """Plot the efficient frontier curve with optional click targets."""
        self._plot_widget.clear()
        self._plot_widget.addItem(self._vline)
        self._plot_widget.addItem(self._hline)
        self._weights_label = None

        # Store frontier point data for click inspection
        self._frontier_points = []
        if weights_list is not None:
            for i in range(len(risks)):
                w = weights_list[i] if i < len(weights_list) else {}
                r_val = float(returns[i])
                v_val = float(risks[i])
                sharpe = r_val / v_val if v_val > 0 else 0.0
                self._frontier_points.append({
                    "risk": v_val,
                    "return": r_val,
                    "weights": w,
                    "sharpe": sharpe,
                })

        # Draw the frontier line
        pen = pg.mkPen(Colors.ACCENT, width=2)
        self._plot_widget.plot(
            risks * 100, returns * 100,
            pen=pen, name="Efficient Frontier",
        )

        # Add invisible scatter for click detection along the frontier
        if self._frontier_points:
            self._frontier_scatter = pg.ScatterPlotItem(
                risks * 100, returns * 100,
                pen=pg.mkPen(None),
                brush=pg.mkBrush(0, 0, 0, 0),  # fully transparent
                size=18,
                hoverable=True,
                hoverPen=pg.mkPen(Colors.ACCENT_HOVER, width=2),
                hoverBrush=pg.mkBrush(Colors.ACCENT_DIM),
                hoverSize=20,
                tip=self._frontier_tip,
            )
            self._frontier_scatter.sigClicked.connect(self._on_frontier_click)
            self._plot_widget.addItem(self._frontier_scatter)

    def set_individual_assets(self, symbols: list[str], risks: np.ndarray, returns: np.ndarray):
        """Plot individual asset positions as scatter dots with hover tooltips."""
        for i, sym in enumerate(symbols):
            color = Colors.CHART_PALETTE[i % len(Colors.CHART_PALETTE)]
            scatter = pg.ScatterPlotItem(
                [risks[i] * 100], [returns[i] * 100],
                pen=pg.mkPen(color, width=1),
                brush=pg.mkBrush(color),
                size=10,
                symbol="o",
                name=sym,
                hoverable=True,
                hoverSize=14,
                tip=lambda x, y, data=None, s=sym: f"{s}: {x:.1f}% risk, {y:.1f}% return",
            )
            self._plot_widget.addItem(scatter)

    def set_optimal_portfolio(self, risk: float, ret: float, label: str = "Optimal"):
        """Plot the optimal portfolio as a star marker."""
        scatter = pg.ScatterPlotItem(
            [risk * 100], [ret * 100],
            pen=pg.mkPen(Colors.TEXT_PRIMARY, width=2),
            brush=pg.mkBrush(Colors.WARNING),
            size=16,
            symbol="star",
            name=label,
        )
        self._plot_widget.addItem(scatter)

    def set_current_portfolio(self, risk: float, ret: float, label: str = "Current"):
        """Plot the current portfolio position as a diamond marker."""
        scatter = pg.ScatterPlotItem(
            [risk * 100], [ret * 100],
            pen=pg.mkPen(Colors.TEXT_PRIMARY, width=2),
            brush=pg.mkBrush(Colors.PURPLE),
            size=14,
            symbol="d",
            name=label,
        )
        self._plot_widget.addItem(scatter)

    def clear_plot(self):
        self._plot_widget.clear()
        self._plot_widget.addItem(self._vline)
        self._plot_widget.addItem(self._hline)
        self._frontier_points = []
        self._frontier_scatter = None
        self._weights_label = None

    # ── Internal ─────────────────────────────────────────────────────

    def _frontier_tip(self, x, y, data=None):
        """Tooltip for frontier scatter points."""
        # Find nearest frontier point
        idx = self._find_nearest_frontier(x / 100, y / 100)
        if idx is None:
            return f"Risk: {x:.1f}%  Return: {y:.1f}%"
        pt = self._frontier_points[idx]
        sharpe = pt["sharpe"]
        # Top-3 weights
        top_w = sorted(pt["weights"].items(), key=lambda kv: kv[1], reverse=True)[:3]
        w_str = ", ".join(f"{s}:{w:.0%}" for s, w in top_w)
        return f"Risk: {x:.1f}%  Return: {y:.1f}%\nSharpe: {sharpe:.3f}\nTop: {w_str}"

    def _on_frontier_click(self, scatter, points, ev):
        """Handle click on frontier scatter — show weights label and emit signal."""
        if not points or not self._frontier_points:
            return

        # Get clicked point position
        pt = points[0]
        x_pct = pt.pos().x()
        y_pct = pt.pos().y()

        idx = self._find_nearest_frontier(x_pct / 100, y_pct / 100)
        if idx is None:
            return

        fp = self._frontier_points[idx]

        # Remove old label
        if self._weights_label is not None:
            self._plot_widget.removeItem(self._weights_label)

        # Build weights text (top 5)
        top_w = sorted(fp["weights"].items(), key=lambda kv: kv[1], reverse=True)[:5]
        lines = [f"Sharpe: {fp['sharpe']:.3f}"]
        for sym, w in top_w:
            lines.append(f"  {sym}: {w:.1%}")
        text = "\n".join(lines)

        self._weights_label = pg.TextItem(
            text=text,
            color=Colors.TEXT_PRIMARY,
            fill=pg.mkBrush(Colors.BG_TERTIARY),
            border=pg.mkPen(Colors.ACCENT, width=1),
        )
        self._weights_label.setPos(x_pct, y_pct)
        self._plot_widget.addItem(self._weights_label)

        self.frontier_point_clicked.emit(fp)

    def _find_nearest_frontier(self, risk: float, ret: float) -> int | None:
        """Find index of nearest frontier point to given risk/return."""
        if not self._frontier_points:
            return None

        risks = np.array([p["risk"] for p in self._frontier_points])
        rets = np.array([p["return"] for p in self._frontier_points])

        # Normalize by range for distance calculation
        r_range = risks.ptp() or 1.0
        t_range = rets.ptp() or 1.0
        dist = ((risks - risk) / r_range) ** 2 + ((rets - ret) / t_range) ** 2
        return int(np.argmin(dist))

    def _on_mouse_moved(self, evt):
        pos = evt[0]
        if self._plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self._plot_widget.plotItem.vb.mapSceneToView(pos)
            self._vline.setPos(mouse_point.x())
            self._hline.setPos(mouse_point.y())
