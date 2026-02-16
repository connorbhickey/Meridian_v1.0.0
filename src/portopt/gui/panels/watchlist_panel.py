"""Dockable WATCHLIST panel."""

from portopt.gui.panels.base_panel import BasePanel


class WatchlistPanel(BasePanel):
    panel_id = "watchlist"
    panel_title = "WATCHLIST"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.add_placeholder()
