"""Dockable TRADE BLOTTER panel."""

from portopt.gui.panels.base_panel import BasePanel


class TradeBlotterPanel(BasePanel):
    panel_id = "trade_blotter"
    panel_title = "TRADE BLOTTER"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.add_placeholder()
