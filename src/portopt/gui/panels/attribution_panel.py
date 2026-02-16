"""Dockable ATTRIBUTION panel."""

from portopt.gui.panels.base_panel import BasePanel


class AttributionPanel(BasePanel):
    panel_id = "attribution"
    panel_title = "ATTRIBUTION"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.add_placeholder()
