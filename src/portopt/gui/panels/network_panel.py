"""Dockable NETWORK / MST panel — interactive graph visualization."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QComboBox, QLabel
import pyqtgraph as pg
import numpy as np

from portopt.gui.panels.base_panel import BasePanel
from portopt.constants import Colors, Fonts

# Sector color mapping
SECTOR_COLORS = {
    "Technology": "#00d4ff",
    "Healthcare": "#00ff88",
    "Financials": "#f0b429",
    "Consumer Discretionary": "#ec4899",
    "Consumer Staples": "#84cc16",
    "Industrials": "#6366f1",
    "Energy": "#ff4444",
    "Materials": "#f97316",
    "Utilities": "#06b6d4",
    "Real Estate": "#a855f7",
    "Communication Services": "#d946ef",
    "Other": "#8b949e",
}

DEFAULT_COLOR = "#8b949e"


class NetworkPanel(BasePanel):
    panel_id = "network"
    panel_title = "NETWORK / MST"

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

        lbl = QLabel("Layout:")
        lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 10px;")
        toolbar.addWidget(lbl)

        self._layout_combo = QComboBox()
        self._layout_combo.addItems(["Spring", "Circular", "Spectral"])
        self._layout_combo.setFixedWidth(100)
        self._layout_combo.currentTextChanged.connect(self._on_layout_changed)
        toolbar.addWidget(self._layout_combo)

        toolbar.addStretch()

        self._info_label = QLabel("")
        self._info_label.setStyleSheet(
            f"color: {Colors.TEXT_MUTED}; font-family: {Fonts.MONO}; font-size: 9px;"
        )
        toolbar.addWidget(self._info_label)

        layout.addLayout(toolbar)

        # Graph plot area
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground(Colors.BG_SECONDARY)
        self._plot_widget.hideAxis("left")
        self._plot_widget.hideAxis("bottom")
        self._plot_widget.hideButtons()
        self._plot_widget.setAspectLocked(True)
        layout.addWidget(self._plot_widget)

        self.content_layout.addLayout(layout)

        # Data storage
        self._nodes = []
        self._edges = []
        self._sectors = {}
        self._positions = {}

    # ── Public API ───────────────────────────────────────────────────

    def set_mst(self, nodes: list[str], edges: list[tuple[str, str, float]],
                sectors: dict[str, str] | None = None,
                market_caps: dict[str, float] | None = None):
        """Display MST graph.

        Parameters
        ----------
        nodes : list of symbol strings
        edges : list of (src, dst, weight) tuples
        sectors : optional {symbol: sector_name} for coloring
        market_caps : optional {symbol: cap} for node sizing
        """
        self._nodes = list(nodes)
        self._edges = list(edges)
        self._sectors = sectors or {}
        self._market_caps = market_caps or {}
        self._compute_layout()
        self._draw()

    def clear_graph(self):
        self._plot_widget.clear()
        self._nodes.clear()
        self._edges.clear()
        self._info_label.setText("")

    # ── Internal ─────────────────────────────────────────────────────

    def _compute_layout(self):
        """Compute node positions using selected layout algorithm."""
        n = len(self._nodes)
        if n == 0:
            self._positions = {}
            return

        layout_type = self._layout_combo.currentText().lower()

        if layout_type == "circular":
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            self._positions = {
                node: (np.cos(a), np.sin(a))
                for node, a in zip(self._nodes, angles)
            }
        elif layout_type == "spectral":
            # Simple spectral-like: use index-based placement
            side = int(np.ceil(np.sqrt(n)))
            self._positions = {
                node: (i % side, i // side)
                for i, node in enumerate(self._nodes)
            }
        else:
            # Spring layout (Fruchterman-Reingold approximation)
            try:
                import networkx as nx
                G = nx.Graph()
                G.add_nodes_from(self._nodes)
                for src, dst, w in self._edges:
                    G.add_edge(src, dst, weight=w)
                pos = nx.spring_layout(G, seed=42)
                self._positions = {k: (v[0], v[1]) for k, v in pos.items()}
            except ImportError:
                # Fallback to circular
                angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
                self._positions = {
                    node: (np.cos(a), np.sin(a))
                    for node, a in zip(self._nodes, angles)
                }

    def _draw(self):
        self._plot_widget.clear()
        if not self._nodes:
            return

        # Draw edges
        for src, dst, weight in self._edges:
            if src in self._positions and dst in self._positions:
                x0, y0 = self._positions[src]
                x1, y1 = self._positions[dst]
                pen = pg.mkPen(Colors.BORDER_LIGHT, width=1)
                self._plot_widget.plot(
                    [x0, x1], [y0, y1], pen=pen,
                )

        # Draw nodes
        xs, ys, sizes, colors, labels = [], [], [], [], []
        max_cap = max(self._market_caps.values()) if self._market_caps else 1
        for node in self._nodes:
            if node not in self._positions:
                continue
            x, y = self._positions[node]
            xs.append(x)
            ys.append(y)

            sector = self._sectors.get(node, "Other")
            colors.append(pg.mkBrush(SECTOR_COLORS.get(sector, DEFAULT_COLOR)))

            cap = self._market_caps.get(node, max_cap * 0.5)
            size = 8 + 12 * (cap / max_cap) if max_cap > 0 else 12
            sizes.append(size)

            labels.append(node)

        scatter = pg.ScatterPlotItem(
            xs, ys,
            size=sizes,
            brush=colors,
            pen=pg.mkPen("#ffffff", width=1),
            symbol="o",
        )
        self._plot_widget.addItem(scatter)

        # Node labels
        for node in self._nodes:
            if node not in self._positions:
                continue
            x, y = self._positions[node]
            text = pg.TextItem(node, color=Colors.TEXT_PRIMARY, anchor=(0.5, -0.5))
            text.setFont(pg.QtGui.QFont(Fonts.MONO, 8))
            text.setPos(x, y)
            self._plot_widget.addItem(text)

        self._info_label.setText(
            f"{len(self._nodes)} nodes, {len(self._edges)} edges"
        )

    def _on_layout_changed(self, _text):
        self._compute_layout()
        self._draw()
