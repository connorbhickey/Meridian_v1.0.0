"""Dockable PRICE CHART panel."""

from portopt.gui.panels.base_panel import BasePanel


class PriceChartPanel(BasePanel):
    panel_id = "price_chart"
    panel_title = "PRICE CHART"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.add_placeholder()
