"""Dockable OPTIMIZATION panel."""

from portopt.gui.panels.base_panel import BasePanel


class OptimizationPanel(BasePanel):
    panel_id = "optimization"
    panel_title = "OPTIMIZATION"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.add_placeholder()
