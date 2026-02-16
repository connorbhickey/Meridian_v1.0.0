"""Dockable EFFICIENT FRONTIER panel."""

from portopt.gui.panels.base_panel import BasePanel


class FrontierPanel(BasePanel):
    panel_id = "frontier"
    panel_title = "EFFICIENT FRONTIER"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.add_placeholder()
