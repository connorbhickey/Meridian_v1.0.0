"""Dockable ORDER panel — generate and review trade orders from optimization results.

Shows pending orders computed from the difference between current portfolio
and target optimization weights. Allows editing order type and reviewing
before execution.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from portopt.constants import Colors, Fonts
from portopt.gui.panels.base_panel import BasePanel
from portopt.gui.widgets.table_context_menu import setup_table_context_menu


class OrderPanel(BasePanel):
    """Generate and review trade orders from optimization results."""

    panel_id = "orders"
    panel_title = "ORDERS"

    generate_requested = Signal()
    send_to_blotter = Signal(list)  # list[dict]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._orders = []  # list of TradeOrder-like dicts
        self._batch_stats = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        self._generate_btn = QPushButton("Generate Orders")
        self._generate_btn.setFixedHeight(26)
        self._generate_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT_DIM};
                color: {Colors.ACCENT};
                border: 1px solid {Colors.ACCENT};
                border-radius: 3px;
                padding: 0 12px;
                font-family: {Fonts.SANS};
                font-size: 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background: {Colors.ACCENT}; color: {Colors.BG_PRIMARY}; }}
        """)
        self._generate_btn.clicked.connect(self.generate_requested.emit)
        toolbar.addWidget(self._generate_btn)

        # Order type selector
        lbl = QLabel("Type:")
        lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 10px;")
        toolbar.addWidget(lbl)

        self._order_type_combo = QComboBox()
        self._order_type_combo.addItems(["MARKET", "LIMIT"])
        self._order_type_combo.setFixedWidth(80)
        toolbar.addWidget(self._order_type_combo)

        toolbar.addStretch()

        self._info_label = QLabel("No orders generated")
        self._info_label.setStyleSheet(
            f"color: {Colors.TEXT_MUTED}; font-family: {Fonts.MONO}; font-size: 9px;"
        )
        toolbar.addWidget(self._info_label)

        layout.addLayout(toolbar)

        # Splitter: order table top, summary bottom
        splitter = QSplitter(Qt.Vertical)

        # Orders table
        self._table = QTableWidget()
        self._table.setColumnCount(8)
        self._table.setHorizontalHeaderLabels([
            "Symbol", "Side", "Shares", "Price", "Limit",
            "Value", "Est. Cost", "Status",
        ])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        self._table.setStyleSheet(f"""
            QTableWidget {{
                background: {Colors.BG_SECONDARY};
                gridline-color: {Colors.BORDER};
                color: {Colors.TEXT_PRIMARY};
                font-family: {Fonts.MONO};
                font-size: 10px;
                border: 1px solid {Colors.BORDER};
            }}
            QTableWidget::item {{
                padding: 2px 6px;
            }}
            QHeaderView::section {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_SECONDARY};
                border: 1px solid {Colors.BORDER};
                padding: 3px;
                font-size: 9px;
                font-weight: bold;
            }}
        """)
        setup_table_context_menu(self._table)
        splitter.addWidget(self._table)

        # Summary area
        summary_widget = QWidget()
        summary_layout = QVBoxLayout(summary_widget)
        summary_layout.setContentsMargins(8, 4, 8, 4)
        summary_layout.setSpacing(2)

        self._summary_labels = {}
        for key, label_text in [
            ("total_buy", "Total Buy Value:"),
            ("total_sell", "Total Sell Value:"),
            ("net_cash", "Net Cash Flow:"),
            ("est_cost", "Estimated Cost:"),
            ("turnover", "Turnover:"),
        ]:
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setStyleSheet(
                f"color: {Colors.TEXT_SECONDARY}; font-family: {Fonts.SANS}; font-size: 10px;"
            )
            val = QLabel("—")
            val.setStyleSheet(
                f"color: {Colors.TEXT_PRIMARY}; font-family: {Fonts.MONO}; font-size: 10px;"
            )
            val.setAlignment(Qt.AlignRight)
            row.addWidget(lbl)
            row.addStretch()
            row.addWidget(val)
            summary_layout.addLayout(row)
            self._summary_labels[key] = val

        # Send to blotter button
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._blotter_btn = QPushButton("Send to Blotter")
        self._blotter_btn.setFixedHeight(24)
        self._blotter_btn.setEnabled(False)
        self._blotter_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_SECONDARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 3px;
                padding: 0 12px;
                font-family: {Fonts.SANS};
                font-size: 9px;
            }}
            QPushButton:hover {{ background: {Colors.BG_ELEVATED}; color: {Colors.TEXT_PRIMARY}; }}
            QPushButton:disabled {{ color: {Colors.TEXT_DISABLED}; }}
        """)
        self._blotter_btn.clicked.connect(self._on_send_to_blotter)
        btn_row.addWidget(self._blotter_btn)
        summary_layout.addLayout(btn_row)

        splitter.addWidget(summary_widget)
        splitter.setSizes([350, 150])

        layout.addWidget(splitter)
        self.content_layout.addLayout(layout)

    # ── Public API ───────────────────────────────────────────────────

    @property
    def order_type(self) -> str:
        return self._order_type_combo.currentText().lower()

    def set_orders(self, orders: list[dict], batch_stats: dict | None = None):
        """Set orders to display.

        Args:
            orders: List of order dicts with keys: symbol, side, quantity,
                    price, limit_price, value, estimated_cost, status.
            batch_stats: Optional dict with total_buy_value, total_sell_value,
                        net_cash_flow, total_estimated_cost, turnover.
        """
        self._orders = list(orders)
        self._batch_stats = batch_stats or {}
        self._render_table()
        self._render_summary()
        self._blotter_btn.setEnabled(bool(orders))

    def clear_orders(self):
        self._orders.clear()
        self._batch_stats.clear()
        self._table.setRowCount(0)
        self._info_label.setText("No orders generated")
        self._blotter_btn.setEnabled(False)
        for lbl in self._summary_labels.values():
            lbl.setText("—")

    # ── Rendering ────────────────────────────────────────────────────

    def _render_table(self):
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(self._orders))

        n_buys = 0
        n_sells = 0

        for row, order in enumerate(self._orders):
            # Symbol
            item = QTableWidgetItem(order.get("symbol", ""))
            item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, 0, item)

            # Side
            side = order.get("side", "")
            item = QTableWidgetItem(side)
            item.setTextAlignment(Qt.AlignCenter)
            if side.upper() == "BUY":
                item.setForeground(QColor(Colors.PROFIT))
                n_buys += 1
            else:
                item.setForeground(QColor(Colors.LOSS))
                n_sells += 1
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            self._table.setItem(row, 1, item)

            # Shares
            qty = order.get("quantity", 0)
            item = QTableWidgetItem(f"{qty:,.0f}")
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(row, 2, item)

            # Market price
            price = order.get("price", 0)
            item = QTableWidgetItem(f"${price:,.2f}")
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(row, 3, item)

            # Limit price
            limit = order.get("limit_price")
            item = QTableWidgetItem(f"${limit:,.2f}" if limit else "—")
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(row, 4, item)

            # Value
            value = order.get("value", qty * price)
            item = QTableWidgetItem(f"${value:,.0f}")
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(row, 5, item)

            # Estimated cost
            cost = order.get("estimated_cost", 0)
            item = QTableWidgetItem(f"${cost:,.2f}")
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            item.setForeground(QColor(Colors.WARNING))
            self._table.setItem(row, 6, item)

            # Status
            status = order.get("status", "PENDING")
            item = QTableWidgetItem(status.upper())
            item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, 7, item)

        self._table.setSortingEnabled(True)
        self._info_label.setText(
            f"{len(self._orders)} orders | {n_buys} buys, {n_sells} sells"
        )

    def _render_summary(self):
        stats = self._batch_stats
        self._summary_labels["total_buy"].setText(
            f"${stats.get('total_buy_value', 0):,.0f}"
        )
        self._summary_labels["total_sell"].setText(
            f"${stats.get('total_sell_value', 0):,.0f}"
        )

        net = stats.get("net_cash_flow", 0)
        net_label = self._summary_labels["net_cash"]
        net_label.setText(f"${net:+,.0f}")
        if net > 0:
            net_label.setStyleSheet(
                f"color: {Colors.PROFIT}; font-family: {Fonts.MONO}; font-size: 10px;"
            )
        elif net < 0:
            net_label.setStyleSheet(
                f"color: {Colors.LOSS}; font-family: {Fonts.MONO}; font-size: 10px;"
            )

        self._summary_labels["est_cost"].setText(
            f"${stats.get('total_estimated_cost', 0):,.2f}"
        )
        self._summary_labels["turnover"].setText(
            f"{stats.get('turnover', 0):.1%}"
        )

    def _on_send_to_blotter(self):
        """Convert orders to blotter-compatible dicts and emit signal."""
        from datetime import datetime
        blotter_trades = []
        for order in self._orders:
            blotter_trades.append({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "symbol": order.get("symbol", ""),
                "side": order.get("side", ""),
                "quantity": order.get("quantity", 0),
                "price": order.get("price", 0),
                "cost": order.get("estimated_cost", 0),
                "weight_after": order.get("weight_after", 0),
            })
        self.send_to_blotter.emit(blotter_trades)
