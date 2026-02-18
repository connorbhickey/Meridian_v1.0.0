"""Dockable RISK BUDGET panel — risk contribution chart, budget editor, ERC toggle."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QProgressBar,
    QSplitter, QTableWidget, QTableWidgetItem, QHeaderView,
    QWidget, QLabel, QCheckBox,
)
import pyqtgraph as pg
import numpy as np

from portopt.gui.panels.base_panel import BasePanel
from portopt.constants import Colors, Fonts


def _pg_color(hex_color: str) -> QColor:
    return QColor(hex_color)


class RiskBudgetPanel(BasePanel):
    panel_id = "risk_budget"
    panel_title = "RISK BUDGET"

    run_requested = Signal(object)  # dict with risk budgets or "erc" flag

    def __init__(self, parent=None):
        super().__init__(parent)
        self._symbols: list[str] = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header
        header = QHBoxLayout()
        header.setSpacing(8)

        title = QLabel("Risk Contribution Analysis & Budgeting")
        title.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-family: {Fonts.SANS}; "
            f"font-size: 10px; font-weight: bold;"
        )
        header.addWidget(title)
        header.addStretch()

        self._erc_check = QCheckBox("Equal Risk Contribution")
        self._erc_check.setChecked(True)
        self._erc_check.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-family: {Fonts.SANS}; font-size: 10px;"
        )
        self._erc_check.toggled.connect(self._on_erc_toggled)
        header.addWidget(self._erc_check)

        self._run_btn = QPushButton("OPTIMIZE")
        self._run_btn.setFixedHeight(28)
        self._run_btn.setFixedWidth(120)
        self._run_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT_DIM};
                color: {Colors.ACCENT};
                border: 1px solid {Colors.ACCENT};
                border-radius: 4px;
                font-family: {Fonts.SANS};
                font-size: 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT};
                color: {Colors.BG_PRIMARY};
            }}
            QPushButton:disabled {{
                background: {Colors.BG_INPUT};
                color: {Colors.TEXT_MUTED};
                border-color: {Colors.BORDER};
            }}
        """)
        self._run_btn.clicked.connect(self._on_run)
        header.addWidget(self._run_btn)

        layout.addLayout(header)

        self._progress = QProgressBar()
        self._progress.setFixedHeight(4)
        self._progress.setTextVisible(False)
        self._progress.setRange(0, 0)
        self._progress.hide()
        self._progress.setStyleSheet(f"""
            QProgressBar {{ background: {Colors.BG_INPUT}; border: none; }}
            QProgressBar::chunk {{ background: {Colors.ACCENT}; }}
        """)
        layout.addWidget(self._progress)

        splitter = QSplitter(Qt.Horizontal)

        # ── Risk contribution chart (stacked bar: target vs actual) ──
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)
        chart_layout.setContentsMargins(0, 0, 0, 0)

        self._rc_plot = pg.PlotWidget(title="Risk Contributions (Target vs Actual)")
        self._rc_plot.setBackground(Colors.BG_SECONDARY)
        self._rc_plot.showGrid(x=False, y=True, alpha=0.15)
        self._rc_plot.setLabel("left", "Risk Contribution (%)", color=Colors.TEXT_SECONDARY)
        chart_layout.addWidget(self._rc_plot)

        splitter.addWidget(chart_widget)

        # ── Budget editor table ──────────────────────────────────────
        self._budget_table = QTableWidget(0, 4)
        self._budget_table.setHorizontalHeaderLabels([
            "Symbol", "Target %", "Actual %", "Weight %",
        ])
        for i in range(4):
            self._budget_table.horizontalHeader().setSectionResizeMode(
                i, QHeaderView.Stretch
            )
        self._budget_table.verticalHeader().hide()
        self._budget_table.setStyleSheet(f"""
            QTableWidget {{
                background: {Colors.BG_SECONDARY};
                color: {Colors.TEXT_PRIMARY};
                gridline-color: {Colors.BORDER};
                font-family: {Fonts.MONO};
                font-size: 10px;
                border: 1px solid {Colors.BORDER};
            }}
            QHeaderView::section {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_MUTED};
                border: 1px solid {Colors.BORDER};
                padding: 4px;
                font-family: {Fonts.SANS};
                font-size: 9px;
                font-weight: bold;
            }}
            QTableWidget::item:selected {{
                background: {Colors.ACCENT_DIM};
            }}
        """)
        splitter.addWidget(self._budget_table)
        splitter.setSizes([400, 300])

        layout.addWidget(splitter)
        self.content_layout.addLayout(layout)

    # ── Public API ────────────────────────────────────────────────────

    def set_symbols(self, symbols: list[str]):
        """Set available symbols and populate budget table with equal budgets."""
        self._symbols = symbols
        n = len(symbols)
        if n == 0:
            return

        equal_pct = 100.0 / n
        self._budget_table.setRowCount(n)
        for i, sym in enumerate(symbols):
            self._budget_table.setItem(i, 0, QTableWidgetItem(sym))

            target = QTableWidgetItem(f"{equal_pct:.1f}")
            target.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._budget_table.setItem(i, 1, target)

            actual = QTableWidgetItem("—")
            actual.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._budget_table.setItem(i, 2, actual)

            weight = QTableWidgetItem("—")
            weight.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._budget_table.setItem(i, 3, weight)

        # Make target column editable, others read-only
        self._budget_table.setEditTriggers(QTableWidget.DoubleClicked)

    def set_running(self, running: bool):
        self._run_btn.setEnabled(not running)
        self._progress.setVisible(running)

    def set_result(self, result, risk_contributions=None):
        """Update table and chart with optimization result.

        Args:
            result: OptimizationResult with weights and metadata.
            risk_contributions: list[RiskContribution] or None.
        """
        actual_rc = result.metadata.get("actual_risk_contributions", {})

        # Update table
        for i in range(self._budget_table.rowCount()):
            sym_item = self._budget_table.item(i, 0)
            if not sym_item:
                continue
            sym = sym_item.text()

            # Actual RC %
            rc_val = actual_rc.get(sym, 0.0) * 100
            rc_item = QTableWidgetItem(f"{rc_val:.1f}")
            rc_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._budget_table.setItem(i, 2, rc_item)

            # Weight %
            w = result.weights.get(sym, 0.0) * 100
            w_item = QTableWidgetItem(f"{w:.1f}")
            w_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._budget_table.setItem(i, 3, w_item)

        # Update chart
        self._update_rc_chart(actual_rc)

    def _update_rc_chart(self, actual_rc: dict[str, float]):
        self._rc_plot.clear()
        if not actual_rc:
            return

        symbols = sorted(actual_rc.keys())
        n = len(symbols)
        x_pos = np.arange(n)

        # Target (from table)
        targets = []
        for sym in symbols:
            target_val = 1.0 / n  # default equal
            for i in range(self._budget_table.rowCount()):
                sym_item = self._budget_table.item(i, 0)
                if sym_item and sym_item.text() == sym:
                    tgt_item = self._budget_table.item(i, 1)
                    if tgt_item:
                        try:
                            target_val = float(tgt_item.text()) / 100
                        except ValueError:
                            pass
                    break
            targets.append(target_val * 100)

        actuals = [actual_rc.get(s, 0.0) * 100 for s in symbols]

        # Target bars (dimmer)
        bar_target = pg.BarGraphItem(
            x=x_pos - 0.15, height=targets, width=0.25,
            brush=pg.mkBrush(QColor(Colors.TEXT_MUTED)),
            pen=pg.mkPen(None),
            name="Target",
        )
        self._rc_plot.addItem(bar_target)

        # Actual bars (accent)
        colors = [
            _pg_color(Colors.ACCENT) if abs(a - t) < 5 else _pg_color(Colors.WARNING)
            for a, t in zip(actuals, targets)
        ]
        bar_actual = pg.BarGraphItem(
            x=x_pos + 0.15, height=actuals, width=0.25,
            brushes=[pg.mkBrush(c) for c in colors],
            pens=[pg.mkPen(None)] * n,
            name="Actual",
        )
        self._rc_plot.addItem(bar_actual)

        bottom = self._rc_plot.getAxis("bottom")
        bottom.setTicks([list(zip(x_pos, symbols))])

    # ── Internal ──────────────────────────────────────────────────────

    def _on_erc_toggled(self, checked: bool):
        """When ERC is checked, make target column non-editable."""
        if checked:
            n = self._budget_table.rowCount()
            if n > 0:
                equal_pct = 100.0 / n
                for i in range(n):
                    item = QTableWidgetItem(f"{equal_pct:.1f}")
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    self._budget_table.setItem(i, 1, item)

    def _on_run(self):
        """Emit run config."""
        is_erc = self._erc_check.isChecked()

        budgets = {}
        for i in range(self._budget_table.rowCount()):
            sym_item = self._budget_table.item(i, 0)
            tgt_item = self._budget_table.item(i, 1)
            if sym_item and tgt_item:
                try:
                    budgets[sym_item.text()] = float(tgt_item.text()) / 100
                except ValueError:
                    pass

        self.run_requested.emit({
            "erc": is_erc,
            "budgets": budgets,
        })
