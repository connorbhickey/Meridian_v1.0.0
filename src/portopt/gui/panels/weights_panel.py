"""Dockable WEIGHTS panel."""

from portopt.gui.panels.base_panel import BasePanel


class WeightsPanel(BasePanel):
    panel_id = "weights"
    panel_title = "WEIGHTS"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.add_placeholder()
