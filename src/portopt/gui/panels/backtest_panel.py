"""Dockable BACKTEST panel."""

from portopt.gui.panels.base_panel import BasePanel


class BacktestPanel(BasePanel):
    panel_id = "backtest"
    panel_title = "BACKTEST"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.add_placeholder()
