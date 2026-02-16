"""Dockable CORRELATION panel with heatmap visualization."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QComboBox, QLabel
import pyqtgraph as pg
import numpy as np

from portopt.gui.panels.base_panel import BasePanel
from portopt.constants import Colors, Fonts


class CorrelationPanel(BasePanel):
    panel_id = "correlation"
    panel_title = "CORRELATION"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._corr_matrix = None
        self._labels = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        lbl = QLabel("Method:")
        lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 10px;")
        toolbar.addWidget(lbl)

        self._method_combo = QComboBox()
        self._method_combo.addItems(["Pearson", "Spearman", "Kendall"])
        self._method_combo.setFixedWidth(100)
        self._method_combo.currentTextChanged.connect(self._on_method_changed)
        toolbar.addWidget(self._method_combo)

        toolbar.addStretch()

        self._hover_label = QLabel("")
        self._hover_label.setStyleSheet(
            f"color: {Colors.ACCENT}; font-family: {Fonts.MONO}; font-size: 10px;"
        )
        toolbar.addWidget(self._hover_label)

        layout.addLayout(toolbar)

        # Heatmap
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground(Colors.BG_SECONDARY)
        self._plot_widget.setAspectLocked(True)
        self._plot_widget.hideButtons()

        self._image_item = pg.ImageItem()
        self._plot_widget.addItem(self._image_item)

        # Colormap: red (-1) -> dark (0) -> green (+1)
        positions = [0.0, 0.5, 1.0]
        colors_rgb = [
            (255, 68, 68),    # red
            (28, 33, 40),     # dark (BG_TERTIARY approx)
            (0, 255, 136),    # green
        ]
        cmap = pg.ColorMap(positions, colors_rgb)
        lut = cmap.getLookupTable(nPts=256)
        self._image_item.setLookupTable(lut)

        # Color bar
        self._colorbar = pg.ColorBarItem(
            values=(-1, 1), colorMap=cmap, orientation="right",
            label="Correlation",
        )
        self._colorbar.setImageItem(self._image_item, insert_in=self._plot_widget.plotItem)

        # Mouse tracking for hover
        self._proxy = pg.SignalProxy(
            self._plot_widget.scene().sigMouseMoved,
            rateLimit=30, slot=self._on_mouse_moved,
        )

        layout.addWidget(self._plot_widget)
        self.content_layout.addLayout(layout)

    # ── Public API ───────────────────────────────────────────────────

    def set_correlation(self, corr_matrix: np.ndarray, labels: list[str]):
        """Set correlation matrix data and redraw heatmap."""
        self._corr_matrix = corr_matrix.copy()
        self._labels = list(labels)
        self._redraw()

    def get_method(self) -> str:
        return self._method_combo.currentText().lower()

    method_changed = None  # Will be connected by controller

    # ── Internal ─────────────────────────────────────────────────────

    def _redraw(self):
        if self._corr_matrix is None:
            return

        n = len(self._labels)
        # Map correlation [-1, 1] to [0, 1] for colormap
        data = (self._corr_matrix + 1.0) / 2.0
        # Flip vertically so row 0 is at top
        data_flipped = np.flipud(data)
        self._image_item.setImage(data_flipped.T)

        # Axis ticks
        x_axis = self._plot_widget.getAxis("bottom")
        y_axis = self._plot_widget.getAxis("left")

        ticks_x = [(i + 0.5, self._labels[i]) for i in range(n)]
        ticks_y = [(i + 0.5, self._labels[n - 1 - i]) for i in range(n)]

        x_axis.setTicks([ticks_x])
        y_axis.setTicks([ticks_y])

        x_axis.setStyle(tickFont=pg.QtGui.QFont(Fonts.MONO, 8))
        y_axis.setStyle(tickFont=pg.QtGui.QFont(Fonts.MONO, 8))

        self._plot_widget.setRange(xRange=(0, n), yRange=(0, n))

    def _on_method_changed(self, _text):
        # Controller should re-compute and call set_correlation
        pass

    def _on_mouse_moved(self, evt):
        if self._corr_matrix is None:
            return

        pos = evt[0]
        if not self._plot_widget.sceneBoundingRect().contains(pos):
            return

        mouse_point = self._plot_widget.plotItem.vb.mapSceneToView(pos)
        n = len(self._labels)
        col = int(mouse_point.x())
        row = n - 1 - int(mouse_point.y())

        if 0 <= row < n and 0 <= col < n:
            val = self._corr_matrix[row, col]
            sym_r = self._labels[row]
            sym_c = self._labels[col]
            self._hover_label.setText(f"{sym_r} / {sym_c}: {val:+.3f}")
        else:
            self._hover_label.setText("")
