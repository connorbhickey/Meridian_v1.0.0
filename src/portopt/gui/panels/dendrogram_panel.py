"""Dockable DENDROGRAM panel."""

from portopt.gui.panels.base_panel import BasePanel


class DendrogramPanel(BasePanel):
    panel_id = "dendrogram"
    panel_title = "DENDROGRAM"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.add_placeholder()
