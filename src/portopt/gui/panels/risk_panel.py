"""Dockable RISK panel."""

from portopt.gui.panels.base_panel import BasePanel


class RiskPanel(BasePanel):
    panel_id = "risk"
    panel_title = "RISK"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.add_placeholder()
