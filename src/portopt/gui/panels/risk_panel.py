"""Dockable RISK panel — VaR/CVaR display, drawdown chart, risk decomposition."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QFrame, QSplitter,
)
import pyqtgraph as pg
import numpy as np

from portopt.gui.panels.base_panel import BasePanel
from portopt.constants import Colors, Fonts


class RiskPanel(BasePanel):
    panel_id = "risk"
    panel_title = "RISK"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._gauges = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        splitter = QSplitter(Qt.Vertical)

        # ── Top: Risk metrics gauges ─────────────────────────────────
        gauges_frame = QFrame()
        gauges_frame.setStyleSheet(f"background: {Colors.BG_SECONDARY}; border: 1px solid {Colors.BORDER};")
        gauge_layout = QGridLayout(gauges_frame)
        gauge_layout.setContentsMargins(6, 6, 6, 6)
        gauge_layout.setSpacing(4)

        gauge_defs = [
            ("var_95", "VaR 95%"),
            ("cvar_95", "CVaR 95%"),
            ("max_drawdown", "Max Drawdown"),
            ("annual_volatility", "Annual Vol"),
            ("downside_vol", "Downside Vol"),
            ("beta", "Beta"),
        ]

        for i, (key, label) in enumerate(gauge_defs):
            row, col = i // 3, i % 3
            cell = self._make_gauge(key, label)
            gauge_layout.addWidget(cell, row, col)

        splitter.addWidget(gauges_frame)

        # ── Middle: Risk decomposition chart ─────────────────────────
        self._decomp_plot = pg.PlotWidget(title="Risk Decomposition")
        self._decomp_plot.setBackground(Colors.BG_SECONDARY)
        self._decomp_plot.showGrid(x=False, y=True, alpha=0.15)
        self._decomp_plot.setLabel("left", "Contribution %", color=Colors.TEXT_SECONDARY)
        splitter.addWidget(self._decomp_plot)

        # ── Bottom: Drawdown chart ───────────────────────────────────
        self._dd_plot = pg.PlotWidget(title="Drawdown History")
        self._dd_plot.setBackground(Colors.BG_SECONDARY)
        self._dd_plot.showGrid(x=True, y=True, alpha=0.15)
        self._dd_plot.setLabel("left", "Drawdown %", color=Colors.TEXT_SECONDARY)
        self._dd_plot.setLabel("bottom", "Date", color=Colors.TEXT_SECONDARY)
        splitter.addWidget(self._dd_plot)

        splitter.setSizes([120, 160, 220])
        layout.addWidget(splitter)
        self.content_layout.addLayout(layout)

    def _make_gauge(self, key: str, label: str) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background: {Colors.BG_TERTIARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 3px;
            }}
        """)
        frame.setFixedHeight(52)

        vlayout = QVBoxLayout(frame)
        vlayout.setContentsMargins(8, 4, 8, 4)
        vlayout.setSpacing(0)

        lbl = QLabel(label)
        lbl.setStyleSheet(
            f"color: {Colors.TEXT_MUTED}; font-family: {Fonts.SANS}; "
            f"font-size: 8px; font-weight: bold; border: none;"
        )
        vlayout.addWidget(lbl)

        val = QLabel("—")
        val.setStyleSheet(
            f"color: {Colors.LOSS}; font-family: {Fonts.MONO}; "
            f"font-size: 14px; font-weight: bold; border: none;"
        )
        vlayout.addWidget(val)

        self._gauges[key] = val
        return frame

    # ── Public API ───────────────────────────────────────────────────

    def set_risk_metrics(self, metrics: dict):
        """Update risk gauge values from a dict of key -> float."""
        format_map = {
            "var_95": lambda v: f"{v:.2%}",
            "cvar_95": lambda v: f"{v:.2%}",
            "max_drawdown": lambda v: f"{v:.2%}",
            "annual_volatility": lambda v: f"{v:.2%}",
            "downside_vol": lambda v: f"{v:.2%}",
            "beta": lambda v: f"{v:.3f}",
        }

        for key, val_widget in self._gauges.items():
            value = metrics.get(key)
            if value is None:
                val_widget.setText("—")
                continue

            fmt = format_map.get(key, lambda v: f"{v:.4f}")
            try:
                text = fmt(value)
            except (TypeError, ValueError):
                text = str(value)

            # Color: red for negative risk metrics, cyan for beta-like
            if key == "beta":
                color = Colors.ACCENT
            elif isinstance(value, (int, float)) and value < 0:
                color = Colors.LOSS
            else:
                color = Colors.WARNING

            val_widget.setText(text)
            val_widget.setStyleSheet(
                f"color: {color}; font-family: {Fonts.MONO}; "
                f"font-size: 14px; font-weight: bold; border: none;"
            )

    def set_drawdown_chart(self, dates_epoch, drawdowns):
        """Plot drawdown series (values should be negative fractions)."""
        self._dd_plot.clear()

        dd_pct = np.asarray(drawdowns) * 100

        # Fill area
        fill = pg.FillBetweenItem(
            pg.PlotDataItem(dates_epoch, np.zeros_like(dd_pct)),
            pg.PlotDataItem(dates_epoch, dd_pct),
            brush=pg.mkBrush(Colors.LOSS_DIM),
        )
        self._dd_plot.addItem(fill)

        # Line
        pen = pg.mkPen(Colors.LOSS, width=1)
        self._dd_plot.plot(dates_epoch, dd_pct, pen=pen)

    def set_risk_decomposition(self, symbols: list[str], contributions: list[float]):
        """Display risk contribution per asset as horizontal bar chart."""
        self._decomp_plot.clear()

        if not symbols or not contributions:
            return

        n = len(symbols)
        contribs = np.asarray(contributions) * 100  # Convert to %

        # Horizontal bar chart using BarGraphItem
        y_pos = np.arange(n)
        colors = []
        for c in contribs:
            if c > 20:
                colors.append(Colors.LOSS)
            elif c > 10:
                colors.append(Colors.WARNING)
            else:
                colors.append(Colors.ACCENT)

        brushes = [pg.mkBrush(c) for c in colors]

        bar = pg.BarGraphItem(
            x0=0, y=y_pos, height=0.6,
            width=contribs,
            brushes=brushes,
        )
        self._decomp_plot.addItem(bar)

        # Set Y-axis ticks to symbol names
        y_axis = self._decomp_plot.getAxis("left")
        ticks = [(i, sym) for i, sym in enumerate(symbols)]
        y_axis.setTicks([ticks])
        self._decomp_plot.setYRange(-0.5, n - 0.5)

    def clear_all(self):
        for val_widget in self._gauges.values():
            val_widget.setText("—")
        self._dd_plot.clear()
        self._decomp_plot.clear()
