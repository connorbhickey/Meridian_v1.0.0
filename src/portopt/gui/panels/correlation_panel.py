"""Dockable CORRELATION panel."""

from portopt.gui.panels.base_panel import BasePanel


class CorrelationPanel(BasePanel):
    panel_id = "correlation"
    panel_title = "CORRELATION"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.add_placeholder()
