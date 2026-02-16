"""Dockable RISK panel — VaR/CVaR display, drawdown chart, risk decomposition.

Enhanced with B4: alert indicators on gauges + configurable thresholds.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QFrame, QSplitter,
    QPushButton,
)
import pyqtgraph as pg
import numpy as np

from portopt.gui.panels.base_panel import BasePanel
from portopt.constants import Colors, Fonts
from portopt.gui.dialogs.alert_config_dialog import DEFAULT_ALERTS


class RiskPanel(BasePanel):
    panel_id = "risk"
    panel_title = "RISK"

    # B4: emitted when a metric breaches its alert threshold
    alert_triggered = Signal(str, float, float)  # metric_name, value, threshold

    def __init__(self, parent=None):
        super().__init__(parent)
        self._gauges = {}
        self._indicators = {}   # B4: key -> QLabel (dot indicator)
        self._alerts = dict(DEFAULT_ALERTS)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        splitter = QSplitter(Qt.Vertical)

        # ── Top: Risk metrics gauges ─────────────────────────────────
        gauges_container = QVBoxLayout()

        # Toolbar with gear icon
        toolbar = QHBoxLayout()
        toolbar.addStretch()
        gear_btn = QPushButton("⚙ ALERTS")
        gear_btn.setFixedHeight(20)
        gear_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {Colors.TEXT_MUTED};
                border: none;
                font-family: {Fonts.SANS};
                font-size: 9px;
                padding: 0 6px;
            }}
            QPushButton:hover {{
                color: {Colors.ACCENT};
            }}
        """)
        gear_btn.clicked.connect(self._open_alert_config)
        toolbar.addWidget(gear_btn)

        gauges_frame = QFrame()
        gauges_frame.setStyleSheet(f"background: {Colors.BG_SECONDARY}; border: 1px solid {Colors.BORDER};")

        gauges_inner = QVBoxLayout(gauges_frame)
        gauges_inner.setContentsMargins(0, 0, 0, 0)
        gauges_inner.setSpacing(0)

        # Toolbar inside the frame
        toolbar_widget = QFrame()
        toolbar_widget.setFixedHeight(20)
        toolbar_widget.setStyleSheet("border: none;")
        toolbar_layout = QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(4, 0, 4, 0)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(gear_btn)
        gauges_inner.addWidget(toolbar_widget)

        gauge_grid = QGridLayout()
        gauge_grid.setContentsMargins(6, 2, 6, 6)
        gauge_grid.setSpacing(4)

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
            gauge_grid.addWidget(cell, row, col)

        gauges_inner.addLayout(gauge_grid)
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

        splitter.setSizes([140, 160, 220])
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

        # Top row: label + indicator dot
        top_row = QHBoxLayout()
        top_row.setSpacing(4)

        lbl = QLabel(label)
        lbl.setStyleSheet(
            f"color: {Colors.TEXT_MUTED}; font-family: {Fonts.SANS}; "
            f"font-size: 8px; font-weight: bold; border: none;"
        )
        top_row.addWidget(lbl)
        top_row.addStretch()

        # B4: Alert indicator dot
        indicator = QLabel("●")
        indicator.setFixedWidth(12)
        indicator.setStyleSheet(
            f"color: {Colors.PROFIT}; font-size: 8px; border: none;"
        )
        indicator.setToolTip("OK")
        indicator.hide()  # Hidden until alerts are configured
        self._indicators[key] = indicator
        top_row.addWidget(indicator)

        vlayout.addLayout(top_row)

        val = QLabel("—")
        val.setStyleSheet(
            f"color: {Colors.LOSS}; font-family: {Fonts.MONO}; "
            f"font-size: 14px; font-weight: bold; border: none;"
        )
        vlayout.addWidget(val)

        self._gauges[key] = val
        return frame

    # ── Public API ───────────────────────────────────────────────────

    def set_alert_config(self, alerts: dict):
        """Update alert thresholds from the config dialog."""
        self._alerts = alerts

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

            # B4: Check alert thresholds
            self._check_alert(key, value)

    def _check_alert(self, key: str, value: float):
        """Check if a metric breaches its alert threshold."""
        indicator = self._indicators.get(key)
        if not indicator:
            return

        alert_cfg = self._alerts.get(key, {})
        if not alert_cfg.get("enabled", False):
            indicator.hide()
            return

        indicator.show()
        threshold = alert_cfg.get("threshold", 0.0)

        # For negative metrics (VaR, CVaR, drawdown), breach means value < threshold
        # For positive metrics (volatility), breach means value > abs(threshold)
        breached = False
        if key in ("var_95", "cvar_95", "max_drawdown"):
            breached = value < threshold
        else:
            breached = abs(value) > abs(threshold)

        if breached:
            indicator.setStyleSheet(
                f"color: {Colors.LOSS}; font-size: 8px; border: none;"
            )
            indicator.setToolTip(f"⚠ ALERT: {value:.4f} breaches threshold {threshold:.4f}")
            self.alert_triggered.emit(key, value, threshold)
        else:
            indicator.setStyleSheet(
                f"color: {Colors.PROFIT}; font-size: 8px; border: none;"
            )
            indicator.setToolTip("OK")

    def _open_alert_config(self):
        """Open the alert configuration dialog."""
        from portopt.gui.dialogs.alert_config_dialog import AlertConfigDialog
        dialog = AlertConfigDialog(self._alerts, self)
        if dialog.exec() == AlertConfigDialog.DialogCode.Accepted:
            self._alerts = dialog.get_alerts()

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
