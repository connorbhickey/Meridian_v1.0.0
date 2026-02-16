"""Dockable DENDROGRAM panel — HRP/HERC cluster visualization via matplotlib."""

from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QComboBox, QLabel
import numpy as np

from portopt.gui.panels.base_panel import BasePanel
from portopt.constants import Colors, Fonts


class DendrogramPanel(BasePanel):
    panel_id = "dendrogram"
    panel_title = "DENDROGRAM"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._canvas = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        lbl = QLabel("Linkage:")
        lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 10px;")
        toolbar.addWidget(lbl)

        self._linkage_combo = QComboBox()
        self._linkage_combo.addItems(["Single", "Complete", "Average", "Ward"])
        self._linkage_combo.setCurrentIndex(2)  # Average default
        self._linkage_combo.setFixedWidth(100)
        toolbar.addWidget(self._linkage_combo)

        toolbar.addStretch()

        self._info_label = QLabel("")
        self._info_label.setStyleSheet(
            f"color: {Colors.TEXT_MUTED}; font-family: {Fonts.MONO}; font-size: 9px;"
        )
        toolbar.addWidget(self._info_label)

        layout.addLayout(toolbar)

        # Matplotlib canvas
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self._fig = Figure(facecolor=Colors.BG_SECONDARY, dpi=100)
            self._ax = self._fig.add_subplot(111)
            self._canvas = FigureCanvasQTAgg(self._fig)
            self._style_axes()
            layout.addWidget(self._canvas)
        except ImportError:
            placeholder = QLabel("matplotlib required for dendrogram")
            placeholder.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
            layout.addWidget(placeholder)

        self.content_layout.addLayout(layout)

    # ── Public API ───────────────────────────────────────────────────

    def set_dendrogram(self, linkage_matrix: np.ndarray, labels: list[str]):
        """Draw dendrogram from a scipy linkage matrix."""
        if self._canvas is None:
            return

        from scipy.cluster.hierarchy import dendrogram

        self._ax.clear()
        self._style_axes()

        dendrogram(
            linkage_matrix,
            labels=labels,
            ax=self._ax,
            leaf_rotation=90,
            leaf_font_size=8,
            color_threshold=0,
            above_threshold_color=Colors.ACCENT,
        )

        self._ax.set_ylabel("Distance", color=Colors.TEXT_SECONDARY, fontsize=9)
        self._ax.tick_params(axis="x", colors=Colors.TEXT_PRIMARY, labelsize=8)
        self._ax.tick_params(axis="y", colors=Colors.TEXT_SECONDARY, labelsize=8)

        self._fig.tight_layout()
        self._canvas.draw()

        self._info_label.setText(f"{len(labels)} assets")

    def set_linkage_and_labels(self, linkage_matrix: np.ndarray, labels: list[str],
                               n_clusters: int | None = None):
        """Draw dendrogram with optional cluster count highlight."""
        if self._canvas is None:
            return

        from scipy.cluster.hierarchy import dendrogram

        self._ax.clear()
        self._style_axes()

        # Determine color threshold for n_clusters
        color_thresh = 0
        if n_clusters is not None and n_clusters > 1:
            sorted_dists = sorted(linkage_matrix[:, 2])
            idx = len(sorted_dists) - n_clusters + 1
            if 0 <= idx < len(sorted_dists):
                color_thresh = sorted_dists[idx]

        dendrogram(
            linkage_matrix,
            labels=labels,
            ax=self._ax,
            leaf_rotation=90,
            leaf_font_size=8,
            color_threshold=color_thresh,
        )

        self._ax.set_ylabel("Distance", color=Colors.TEXT_SECONDARY, fontsize=9)
        self._ax.tick_params(axis="x", colors=Colors.TEXT_PRIMARY, labelsize=8)
        self._ax.tick_params(axis="y", colors=Colors.TEXT_SECONDARY, labelsize=8)

        if n_clusters:
            self._ax.axhline(y=color_thresh, color=Colors.WARNING, linestyle="--", linewidth=1)

        self._fig.tight_layout()
        self._canvas.draw()

        cluster_text = f", {n_clusters} clusters" if n_clusters else ""
        self._info_label.setText(f"{len(labels)} assets{cluster_text}")

    def clear_plot(self):
        if self._canvas is None:
            return
        self._ax.clear()
        self._style_axes()
        self._canvas.draw()
        self._info_label.setText("")

    # ── Internal ─────────────────────────────────────────────────────

    def _style_axes(self):
        self._ax.set_facecolor(Colors.BG_SECONDARY)
        for spine in self._ax.spines.values():
            spine.set_color(Colors.BORDER)
        self._ax.tick_params(colors=Colors.TEXT_SECONDARY)
