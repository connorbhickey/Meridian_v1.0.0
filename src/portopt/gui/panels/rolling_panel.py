"""Dockable ROLLING ANALYTICS panel — rolling window metric charts."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QComboBox, QSpinBox,
    QPushButton, QProgressBar, QLabel,
)
import pyqtgraph as pg
import numpy as np

from portopt.gui.panels.base_panel import BasePanel
from portopt.constants import Colors, Fonts


METRIC_OPTIONS = [
    "Sharpe Ratio",
    "Sortino Ratio",
    "Volatility",
    "Max Drawdown",
    "Beta",
    "Correlation",
]

# Metrics that need a second asset
DUAL_ASSET_METRICS = {"Beta", "Correlation"}


class RollingAnalyticsPanel(BasePanel):
    panel_id = "rolling_analytics"
    panel_title = "ROLLING ANALYTICS"

    compute_requested = Signal(object)  # dict with metric config

    def __init__(self, parent=None):
        super().__init__(parent)
        self._symbols: list[str] = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Controls bar ─────────────────────────────────────────────
        controls = QHBoxLayout()
        controls.setSpacing(6)

        # Metric selector
        lbl_metric = QLabel("Metric:")
        lbl_metric.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 9px;")
        controls.addWidget(lbl_metric)

        self._metric_combo = QComboBox()
        self._metric_combo.addItems(METRIC_OPTIONS)
        self._metric_combo.setFixedWidth(120)
        self._metric_combo.currentTextChanged.connect(self._on_metric_changed)
        controls.addWidget(self._metric_combo)

        # Window size
        lbl_window = QLabel("Window:")
        lbl_window.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 9px;")
        controls.addWidget(lbl_window)

        self._window_spin = QSpinBox()
        self._window_spin.setRange(21, 504)
        self._window_spin.setValue(63)
        self._window_spin.setSuffix(" days")
        self._window_spin.setFixedWidth(90)
        controls.addWidget(self._window_spin)

        # Asset A
        lbl_a = QLabel("Asset:")
        lbl_a.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 9px;")
        controls.addWidget(lbl_a)

        self._asset_a_combo = QComboBox()
        self._asset_a_combo.setFixedWidth(80)
        controls.addWidget(self._asset_a_combo)

        # Asset B (for dual-asset metrics)
        self._lbl_b = QLabel("vs:")
        self._lbl_b.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 9px;")
        controls.addWidget(self._lbl_b)

        self._asset_b_combo = QComboBox()
        self._asset_b_combo.setFixedWidth(80)
        controls.addWidget(self._asset_b_combo)

        # Initially hide Asset B controls
        self._lbl_b.hide()
        self._asset_b_combo.hide()

        # Compute button
        self._compute_btn = QPushButton("COMPUTE")
        self._compute_btn.setFixedHeight(28)
        self._compute_btn.setFixedWidth(90)
        self._compute_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT_DIM};
                color: {Colors.ACCENT};
                border: 1px solid {Colors.ACCENT};
                border-radius: 4px;
                font-family: {Fonts.SANS};
                font-size: 10px;
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
        self._compute_btn.clicked.connect(self._on_compute)
        controls.addWidget(self._compute_btn)

        # Progress bar (hidden by default)
        self._progress = QProgressBar()
        self._progress.setFixedHeight(4)
        self._progress.setFixedWidth(60)
        self._progress.setTextVisible(False)
        self._progress.setRange(0, 0)
        self._progress.hide()
        self._progress.setStyleSheet(f"""
            QProgressBar {{ background: {Colors.BG_INPUT}; border: none; }}
            QProgressBar::chunk {{ background: {Colors.ACCENT}; }}
        """)
        controls.addWidget(self._progress)

        controls.addStretch()
        layout.addLayout(controls)

        # ── Chart ────────────────────────────────────────────────────
        date_axis = pg.DateAxisItem(orientation="bottom")
        self._plot_widget = pg.PlotWidget(axisItems={"bottom": date_axis})
        self._plot_widget.setBackground(Colors.BG_SECONDARY)
        self._plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self._plot_widget.setLabel("left", "Value", color=Colors.TEXT_SECONDARY)
        self._plot_widget.setLabel("bottom", "Date", color=Colors.TEXT_SECONDARY)

        # Zero reference line
        self._zero_line = pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen(Colors.TEXT_MUTED, width=1, style=Qt.DashLine),
        )
        self._plot_widget.addItem(self._zero_line, ignoreBounds=True)

        # Crosshair
        self._vline = pg.InfiniteLine(angle=90, pen=pg.mkPen(Colors.TEXT_MUTED, width=1, style=Qt.DashLine))
        self._hline = pg.InfiniteLine(angle=0, pen=pg.mkPen(Colors.TEXT_MUTED, width=1, style=Qt.DashLine))
        self._plot_widget.addItem(self._vline, ignoreBounds=True)
        self._plot_widget.addItem(self._hline, ignoreBounds=True)
        self._proxy = pg.SignalProxy(
            self._plot_widget.scene().sigMouseMoved,
            rateLimit=60, slot=self._on_mouse_moved,
        )

        layout.addWidget(self._plot_widget)
        self.content_layout.addLayout(layout)

    # ── Public API ────────────────────────────────────────────────────

    def set_symbols(self, symbols: list[str]):
        """Populate the asset combo boxes with available symbols."""
        self._symbols = list(symbols)
        self._asset_a_combo.clear()
        self._asset_b_combo.clear()
        self._asset_a_combo.addItems(symbols)
        self._asset_b_combo.addItems(symbols)
        if len(symbols) > 1:
            self._asset_b_combo.setCurrentIndex(1)

    def set_result(self, dates_epoch: np.ndarray, values: np.ndarray, metric_name: str):
        """Plot the rolling metric result."""
        self._plot_widget.clear()
        self._plot_widget.addItem(self._vline)
        self._plot_widget.addItem(self._hline)

        # Decide whether to show zero line
        show_zero = metric_name in ("Sharpe Ratio", "Sortino Ratio", "Beta", "Correlation")
        if show_zero:
            self._plot_widget.addItem(self._zero_line)

        # Filter out NaN values
        mask = ~np.isnan(values)
        if not mask.any():
            return
        dates_clean = dates_epoch[mask]
        values_clean = values[mask]

        pen = pg.mkPen(Colors.ACCENT, width=2)
        self._plot_widget.plot(dates_clean, values_clean, pen=pen, name=metric_name)
        self._plot_widget.setLabel("left", metric_name, color=Colors.TEXT_SECONDARY)

    def set_running(self, running: bool):
        self._compute_btn.setEnabled(not running)
        self._progress.setVisible(running)

    # ── Internal ──────────────────────────────────────────────────────

    def _on_metric_changed(self, text: str):
        """Show/hide Asset B controls based on selected metric."""
        needs_b = text in DUAL_ASSET_METRICS
        self._lbl_b.setVisible(needs_b)
        self._asset_b_combo.setVisible(needs_b)

    def _on_compute(self):
        """Emit compute_requested with current configuration."""
        metric = self._metric_combo.currentText()
        asset_a = self._asset_a_combo.currentText()
        asset_b = self._asset_b_combo.currentText() if metric in DUAL_ASSET_METRICS else None

        if not asset_a:
            return

        self.compute_requested.emit({
            "metric": metric,
            "window": self._window_spin.value(),
            "asset_a": asset_a,
            "asset_b": asset_b,
        })

    def _on_mouse_moved(self, evt):
        pos = evt[0]
        if self._plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self._plot_widget.plotItem.vb.mapSceneToView(pos)
            self._vline.setPos(mouse_point.x())
            self._hline.setPos(mouse_point.y())
