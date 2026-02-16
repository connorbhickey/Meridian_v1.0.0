"""Dockable PRICE CHART panel with multi-asset PyQtGraph charting."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QComboBox, QPushButton,
    QCheckBox, QButtonGroup, QWidget,
)
import pyqtgraph as pg
import numpy as np

from portopt.gui.panels.base_panel import BasePanel
from portopt.constants import Colors, Fonts



class PriceChartPanel(BasePanel):
    panel_id = "price_chart"
    panel_title = "PRICE CHART"

    asset_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._prices = {}       # symbol -> (dates_epoch, values)
        self._plot_items = {}   # symbol -> PlotDataItem
        self._mode = "line"     # "line" or "normalized"
        self._log_scale = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # --- Toolbar ---
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        # Mode selector
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Line", "Normalized (base=100)"])
        self._mode_combo.setFixedWidth(160)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        toolbar.addWidget(self._mode_combo)

        # Date range buttons
        self._range_group = QButtonGroup(self)
        self._range_group.setExclusive(True)
        for label in ["1M", "3M", "6M", "1Y", "3Y", "5Y", "All"]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedWidth(36)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {Colors.BG_INPUT};
                    color: {Colors.TEXT_SECONDARY};
                    border: 1px solid {Colors.BORDER};
                    border-radius: 3px;
                    font-size: 9px;
                    padding: 2px;
                }}
                QPushButton:checked {{
                    background: {Colors.ACCENT_DIM};
                    color: {Colors.ACCENT};
                    border-color: {Colors.ACCENT};
                }}
            """)
            self._range_group.addButton(btn)
            toolbar.addWidget(btn)
        # default to "All"
        all_btn = self._range_group.buttons()[-1]
        all_btn.setChecked(True)
        self._range_group.buttonClicked.connect(self._on_range_changed)

        toolbar.addStretch()

        # Log scale toggle
        self._log_check = QCheckBox("Log")
        self._log_check.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 10px;")
        self._log_check.toggled.connect(self._on_log_toggled)
        toolbar.addWidget(self._log_check)

        layout.addLayout(toolbar)

        # --- Chart ---
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground(Colors.BG_SECONDARY)
        self._plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self._plot_widget.setLabel("left", "Price", color=Colors.TEXT_SECONDARY)
        self._plot_widget.setLabel("bottom", "Date", color=Colors.TEXT_SECONDARY)

        # Crosshair
        self._vline = pg.InfiniteLine(angle=90, pen=pg.mkPen(Colors.TEXT_MUTED, width=1, style=Qt.DashLine))
        self._hline = pg.InfiniteLine(angle=0, pen=pg.mkPen(Colors.TEXT_MUTED, width=1, style=Qt.DashLine))
        self._plot_widget.addItem(self._vline, ignoreBounds=True)
        self._plot_widget.addItem(self._hline, ignoreBounds=True)
        self._proxy = pg.SignalProxy(
            self._plot_widget.scene().sigMouseMoved,
            rateLimit=60, slot=self._on_mouse_moved,
        )

        # Legend
        self._legend = self._plot_widget.addLegend(
            offset=(10, 10),
            labelTextColor=Colors.TEXT_SECONDARY,
            labelTextSize="9pt",
        )

        layout.addWidget(self._plot_widget)
        self.content_layout.addLayout(layout)

    # ── Public API ───────────────────────────────────────────────────

    def set_prices(self, symbol: str, dates, values):
        """Add or update a price series.

        Parameters
        ----------
        symbol : str
        dates : array-like of datetime64 or Timestamps
        values : array-like of float (prices)
        """
        import pandas as pd
        epochs = np.array([pd.Timestamp(d).timestamp() for d in dates], dtype=float)
        vals = np.asarray(values, dtype=float)
        self._prices[symbol] = (epochs, vals)
        self._redraw()

    def remove_symbol(self, symbol: str):
        self._prices.pop(symbol, None)
        if symbol in self._plot_items:
            self._plot_widget.removeItem(self._plot_items.pop(symbol))
        self._redraw()

    def clear_all(self):
        self._prices.clear()
        for item in self._plot_items.values():
            self._plot_widget.removeItem(item)
        self._plot_items.clear()

    # ── Internal ─────────────────────────────────────────────────────

    def _redraw(self):
        # Remove old items
        for item in self._plot_items.values():
            self._plot_widget.removeItem(item)
        self._plot_items.clear()
        self._legend.clear()

        normalized = self._mode_combo.currentIndex() == 1

        for i, (symbol, (epochs, vals)) in enumerate(self._prices.items()):
            color = Colors.CHART_PALETTE[i % len(Colors.CHART_PALETTE)]
            display_vals = vals.copy()

            if normalized and len(display_vals) > 0 and display_vals[0] != 0:
                display_vals = (display_vals / display_vals[0]) * 100.0

            # Apply date range filter
            filtered_epochs, filtered_vals = self._apply_range(epochs, display_vals)

            pen = pg.mkPen(color, width=2)
            item = self._plot_widget.plot(
                filtered_epochs, filtered_vals,
                pen=pen, name=symbol,
            )
            self._plot_items[symbol] = item

        # Date axis formatting
        axis = self._plot_widget.getAxis("bottom")
        axis.setStyle(tickFont=pg.QtGui.QFont(Fonts.MONO, 8))

        if self._log_scale:
            self._plot_widget.setLogMode(y=True)
        else:
            self._plot_widget.setLogMode(y=False)

        ylabel = "Normalized (base=100)" if normalized else "Price"
        self._plot_widget.setLabel("left", ylabel, color=Colors.TEXT_SECONDARY)

    def _apply_range(self, epochs, vals):
        """Filter data to selected date range."""
        import time
        btn = self._range_group.checkedButton()
        if btn is None or btn.text() == "All":
            return epochs, vals

        now = time.time()
        range_map = {
            "1M": 30, "3M": 90, "6M": 180,
            "1Y": 365, "3Y": 1095, "5Y": 1825,
        }
        days = range_map.get(btn.text(), 0)
        if days == 0:
            return epochs, vals

        cutoff = now - days * 86400
        mask = epochs >= cutoff
        return epochs[mask], vals[mask]

    def _on_mode_changed(self, _idx):
        self._redraw()

    def _on_range_changed(self, _btn):
        self._redraw()

    def _on_log_toggled(self, checked):
        self._log_scale = checked
        self._redraw()

    def _on_mouse_moved(self, evt):
        pos = evt[0]
        if self._plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self._plot_widget.plotItem.vb.mapSceneToView(pos)
            self._vline.setPos(mouse_point.x())
            self._hline.setPos(mouse_point.y())
