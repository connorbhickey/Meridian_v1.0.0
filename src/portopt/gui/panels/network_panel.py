"""Dockable NETWORK / MST panel."""

from portopt.gui.panels.base_panel import BasePanel


class NetworkPanel(BasePanel):
    panel_id = "network"
    panel_title = "NETWORK / MST"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.add_placeholder()
