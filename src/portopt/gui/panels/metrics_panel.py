"""Dockable METRICS panel."""

from portopt.gui.panels.base_panel import BasePanel


class MetricsPanel(BasePanel):
    panel_id = "metrics"
    panel_title = "METRICS"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.add_placeholder()
